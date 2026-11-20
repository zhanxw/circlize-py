"""Chord diagrams from adjacency matrices or data frames, and calc_gap."""

from __future__ import annotations

import math
from typing import Any, Sequence
import warnings

import numpy as np
import pandas as pd

from ._colors import add_transparency, color_to_hex, is_na_like, rand_color, resolve_color
from ._link import circos_link
from ._low_level import circos_axis, circos_rect, circos_text
from ._plot import circos_initialize, circos_track_plot_region
from ._state import (
    circos_par,
    get_all_track_index,
    get_cell_meta_data,
    get_most_inside_radius,
    normalize_kwargs,
    set_current_cell,
)
from ._utils import convert_height, mm_h, mm_y, validate_data_frame


def _parse_pre_allocate_tracks(pre_allocate_tracks: Any) -> list[dict[str, Any]]:
    base = {
        "ylim": [0.0, 1.0],
        "track_height": float(circos_par.get_value("track.height")),
        "bg_col": None,
        "bg_border": None,
        "bg_lty": 1,
        "bg_lwd": 1.0,
        "track_margin": circos_par.get_value("track.margin"),
    }
    if isinstance(pre_allocate_tracks, (int, np.integer)):
        return [dict(base) for _ in range(int(pre_allocate_tracks))]
    if isinstance(pre_allocate_tracks, dict):
        nkw = normalize_kwargs(pre_allocate_tracks)
        lt = dict(base)
        for k in base:
            if k in nkw:
                lt[k] = nkw[k]
        return [lt]
    if isinstance(pre_allocate_tracks, (list, tuple)):
        if all(isinstance(item, dict) for item in pre_allocate_tracks):
            out = []
            for item in pre_allocate_tracks:
                nkw = normalize_kwargs(item)
                lt = dict(base)
                for k in base:
                    if k in nkw:
                        lt[k] = nkw[k]
                out.append(lt)
            return out
        raise ValueError("Wrong `pre_allocate_tracks` list format.")
    raise ValueError("Wrong `pre_allocate_tracks` value.")


def _normalize_to_mat(value: Any, rn: list[str], cn: list[str], default: Any) -> np.ndarray:
    mat = np.full((len(rn), len(cn)), default, dtype=object)
    if isinstance(value, pd.DataFrame):
        if value.shape[1] == 3:
            c0 = value.iloc[:, 0].astype(str).tolist()
            c1 = value.iloc[:, 1].astype(str).tolist()
            c2 = value.iloc[:, 2].tolist()
            rmap = {r: i for i, r in enumerate(rn)}
            cmap = {c: j for j, c in enumerate(cn)}
            for r_k, c_k, v in zip(c0, c1, c2):
                if r_k in rmap and c_k in cmap:
                    mat[rmap[r_k], cmap[c_k]] = v
            return mat
        if value.shape == (len(rn), len(cn)):
            return value.to_numpy(dtype=object)
        raise ValueError("If `value` is a DataFrame, it should have three columns or match matrix shape.")

    if isinstance(value, (str, bytes, int, float, np.number, bool, np.bool_)) or value is None:
        mat[:, :] = value
        return mat

    arr = np.asarray(value, dtype=object)
    if arr.ndim == 1 and len(arr) == len(rn) * len(cn):
        # Fill column-major like R
        return arr.reshape((len(cn), len(rn))).T.copy()
    if arr.ndim == 2 and arr.shape == (len(rn), len(cn)):
        return arr.copy()
    raise ValueError("Cannot broadcast value to link matrix shape.")


def _mat2df(mat: np.ndarray, rn: list[str], cn: list[str]) -> pd.DataFrame:
    nr, nc = mat.shape
    # Column-major order matching R's mat2df
    rn_col = rn * nc
    ri_col = list(range(nr)) * nc
    cn_col = [c for c in cn for _ in range(nr)]
    ci_col = [j for j in range(nc) for _ in range(nr)]
    vals = mat.T.ravel()
    return pd.DataFrame({"rn": rn_col, "cn": cn_col, "ri": ri_col, "ci": ci_col, "value": vals})


def _apply_transparency_to_colors(cols: Sequence[Any], transparency: Any) -> list[str]:
    out: list[str] = []
    if transparency is None or (np.isscalar(transparency) and is_na_like(transparency)):
        for c in cols:
            hx = color_to_hex(c, include_alpha=True)
            out.append(hx if hx is not None else "#FFFFFF00")
        return out

    if not np.isscalar(transparency) and len(transparency) == len(cols):
        for c, tr in zip(cols, transparency):
            if is_na_like(tr):
                hx = color_to_hex(c, include_alpha=True)
                out.append(hx if hx is not None else "#FFFFFF00")
            else:
                hx = add_transparency(c, float(tr))
                out.append(hx if hx is not None else "#FFFFFF00")
        return out

    tr_f = max(0.0, min(1.0, float(transparency)))
    hex_list = [color_to_hex(c, include_alpha=True) for c in cols]
    all_opaque = all((h is not None and h.endswith("FF")) for h in hex_list)
    for h in hex_list:
        if h is None:
            out.append("#FFFFFF00")
        elif all_opaque:
            out.append(add_transparency(h[:7], tr_f))
        else:
            out.append(h)
    return out


def _convert_h_from_canvas_to_data(h: float, sector_index: str, track_index: int) -> float:
    yplot = get_cell_meta_data("yplot", sector_index=sector_index, track_index=track_index)
    cy = get_cell_meta_data("cell.ylim", sector_index=sector_index, track_index=track_index)
    return float(h) / abs(float(yplot[1] - yplot[0])) * float(cy[1] - cy[0])


def chord_diagram(
    x: Any,
    grid_col: Any = None,
    grid_border: Any = None,
    transparency: float = 0.5,
    col: Any = None,
    row_col: Any = None,
    column_col: Any = None,
    order: Sequence[Any] | None = None,
    directional: Any = 0,
    xmax: dict[str, float] | pd.Series | None = None,
    symmetric: bool = False,
    keep_diagonal: bool = False,
    direction_type: str | Sequence[str] = "diffHeight",
    diff_height: float | None = None,
    link_target_prop: bool = True,
    target_prop_height: float | None = None,
    reduce: float = 1e-5,
    self_link: int = 2,
    pre_allocate_tracks: Any = None,
    annotation_track: str | Sequence[str] | None = ("name", "grid", "axis"),
    annotation_track_height: Any = None,
    link_border: Any = None,
    link_lwd: Any = 1.0,
    link_lty: Any = 1,
    link_auto: bool = True,
    link_sort: Any = "default",
    link_decreasing: Any = True,
    link_arr_length: Any = None,
    link_arr_width: Any = None,
    link_arr_type: Any = "triangle",
    link_arr_lty: Any = 1,
    link_arr_lwd: Any = 1.0,
    link_arr_col: Any = "black",
    link_largest_ontop: bool = False,
    link_visible: Any = True,
    link_rank: Any = None,
    link_zindex: Any = None,
    link_overlap: bool = False,
    scale: bool = False,
    group: dict[str, Any] | pd.Series | None = None,
    big_gap: float = 10.0,
    small_gap: float = 1.0,
    plot: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Plot a Chord diagram from an adjacency matrix (ndarray/DataFrame) or adjacency list DataFrame."""
    nkw = normalize_kwargs(kwargs)
    grid_col = nkw.pop("grid_col", grid_col)
    grid_border = nkw.pop("grid_border", grid_border)
    row_col = nkw.pop("row_col", row_col)
    column_col = nkw.pop("column_col", column_col)
    keep_diagonal = bool(nkw.pop("keep_diagonal", keep_diagonal))
    direction_type = nkw.pop("direction_type", direction_type)
    diff_height = nkw.pop("diff_height", diff_height)
    link_target_prop = bool(nkw.pop("link_target_prop", link_target_prop))
    target_prop_height = nkw.pop("target_prop_height", target_prop_height)
    self_link = int(nkw.pop("self_link", self_link))
    pre_allocate_tracks = nkw.pop("pre_allocate_tracks", pre_allocate_tracks)
    annotation_track = nkw.pop("annotation_track", annotation_track)
    annotation_track_height = nkw.pop("annotation_track_height", annotation_track_height)
    link_border = nkw.pop("link_border", link_border)
    link_lwd = nkw.pop("link_lwd", link_lwd)
    link_lty = nkw.pop("link_lty", link_lty)
    link_auto = bool(nkw.pop("link_auto", link_auto))
    link_sort = nkw.pop("link_sort", link_sort)
    link_decreasing = nkw.pop("link_decreasing", link_decreasing)
    link_arr_length = nkw.pop("link_arr_length", link_arr_length)
    link_arr_width = nkw.pop("link_arr_width", link_arr_width)
    link_arr_type = nkw.pop("link_arr_type", link_arr_type)
    link_arr_lty = nkw.pop("link_arr_lty", link_arr_lty)
    link_arr_lwd = nkw.pop("link_arr_lwd", link_arr_lwd)
    link_arr_col = nkw.pop("link_arr_col", link_arr_col)
    link_largest_ontop = bool(nkw.pop("link_largest_ontop", link_largest_ontop))
    link_visible = nkw.pop("link_visible", link_visible)
    link_rank = nkw.pop("link_rank", link_rank)
    link_zindex = nkw.pop("link_zindex", link_zindex)
    link_overlap = bool(nkw.pop("link_overlap", link_overlap))
    big_gap = float(nkw.pop("big_gap", big_gap))
    small_gap = float(nkw.pop("small_gap", small_gap))

    common = dict(
        grid_col=grid_col,
        grid_border=grid_border,
        transparency=transparency,
        col=col,
        order=order,
        directional=directional,
        xmax=xmax,
        direction_type=direction_type,
        diff_height=diff_height,
        link_target_prop=link_target_prop,
        target_prop_height=target_prop_height,
        reduce=reduce,
        self_link=self_link,
        pre_allocate_tracks=pre_allocate_tracks,
        annotation_track=annotation_track,
        annotation_track_height=annotation_track_height,
        link_border=link_border,
        link_lwd=link_lwd,
        link_lty=link_lty,
        link_auto=link_auto,
        link_sort=link_sort,
        link_decreasing=link_decreasing,
        link_arr_length=link_arr_length,
        link_arr_width=link_arr_width,
        link_arr_type=link_arr_type,
        link_arr_lty=link_arr_lty,
        link_arr_lwd=link_arr_lwd,
        link_arr_col=link_arr_col,
        link_largest_ontop=link_largest_ontop,
        link_visible=link_visible,
        link_rank=link_rank,
        link_zindex=link_zindex,
        link_overlap=link_overlap,
        scale=scale,
        group=group,
        big_gap=big_gap,
        small_gap=small_gap,
        plot=plot,
    )

    if isinstance(x, np.ndarray):
        if x.ndim != 2:
            raise ValueError("Matrix input `x` must be 2D.")
        if x.dtype.kind in ("U", "S", "O") and not all(isinstance(v, (int, float, np.number)) for v in x.ravel()):
            df = pd.DataFrame(x)
            return chord_diagram_from_data_frame(df, **common, **nkw)
        return chord_diagram_from_matrix(
            x,
            row_col=row_col,
            column_col=column_col,
            symmetric=symmetric,
            keep_diagonal=keep_diagonal,
            **common,
            **nkw,
        )

    df = validate_data_frame(x)
    all_num = all(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)
    has_explicit_names = not isinstance(df.index, pd.RangeIndex) and not isinstance(df.columns, pd.RangeIndex)
    if all_num and (df.shape[1] > 3 or has_explicit_names):
        if df.shape[1] > 3 and not has_explicit_names:
            warnings.warn("It seems your input data is an adjacency matrix; converting to matrix.", UserWarning, stacklevel=2)
        return chord_diagram_from_matrix(
            df,
            row_col=row_col,
            column_col=column_col,
            symmetric=symmetric,
            keep_diagonal=keep_diagonal,
            **common,
            **nkw,
        )
    return chord_diagram_from_data_frame(df, **common, **nkw)


chordDiagram = chord_diagram


def chord_diagram_from_matrix(
    mat: Any,
    grid_col: Any = None,
    grid_border: Any = None,
    transparency: float = 0.5,
    col: Any = None,
    row_col: Any = None,
    column_col: Any = None,
    order: Sequence[Any] | None = None,
    directional: Any = 0,
    direction_type: str | Sequence[str] = "diffHeight",
    diff_height: float | None = None,
    link_target_prop: bool = True,
    target_prop_height: float | None = None,
    reduce: float = 1e-5,
    xmax: dict[str, float] | pd.Series | None = None,
    self_link: int = 2,
    symmetric: bool = False,
    keep_diagonal: bool = False,
    pre_allocate_tracks: Any = None,
    annotation_track: str | Sequence[str] | None = ("name", "grid", "axis"),
    annotation_track_height: Any = None,
    link_border: Any = None,
    link_lwd: Any = 1.0,
    link_lty: Any = 1,
    link_auto: bool = True,
    link_sort: Any = "default",
    link_decreasing: Any = True,
    link_arr_length: Any = None,
    link_arr_width: Any = None,
    link_arr_type: Any = "triangle",
    link_arr_lty: Any = 1,
    link_arr_lwd: Any = 1.0,
    link_arr_col: Any = "black",
    link_largest_ontop: bool = False,
    link_visible: Any = True,
    link_rank: Any = None,
    link_zindex: Any = None,
    link_overlap: bool = False,
    scale: bool = False,
    group: dict[str, Any] | pd.Series | None = None,
    big_gap: float = 10.0,
    small_gap: float = 1.0,
    plot: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Plot Chord diagram from an adjacency matrix (2D ndarray or DataFrame)."""
    nkw = normalize_kwargs(kwargs)
    if isinstance(mat, pd.DataFrame):
        rn = [str(r) for r in mat.index] if not isinstance(mat.index, pd.RangeIndex) else None
        cn = [str(c) for c in mat.columns] if not isinstance(mat.columns, pd.RangeIndex) else None
        arr = mat.to_numpy(dtype=float).copy()
    else:
        arr = np.asarray(mat, dtype=float).copy()
        rn = None
        cn = None

    if arr.ndim != 2:
        raise ValueError("`mat` must be a 2D matrix.")

    if isinstance(direction_type, (list, tuple)) and len(direction_type) == 2 and set(direction_type) == {"diffHeight", "arrows"}:
        direction_type = "diffHeight+arrows"

    if link_zindex is None:
        link_zindex = 1

    if symmetric:
        if arr.shape[0] != arr.shape[1]:
            raise ValueError("`mat` should be a square matrix when `symmetric=True`.")
        n_m = arr.shape[0]
        for _ in range(min(10, n_m * n_m)):
            ir, ic = int(np.random.randint(0, n_m)), int(np.random.randint(0, n_m))
            if abs(arr[ir, ic] - arr[ic, ir]) > 1e-8 and not (np.isnan(arr[ir, ic]) and np.isnan(arr[ic, ir])):
                raise ValueError("Is `mat` really a symmetric matrix?")
        if rn is None and cn is None:
            rn = [f"S{i + 1}" for i in range(n_m)]
            cn = [f"S{i + 1}" for i in range(n_m)]
        elif rn is None:
            rn = list(cn)
        elif cn is None:
            cn = list(rn)
        if set(rn) != set(cn):
            raise ValueError("Since `symmetric=True`, row names and column names should match.")
        k_diag = 0 if not keep_diagonal else 1
        arr[np.triu_indices(n_m, k=k_diag)] = 0.0

    arr[np.isnan(arr)] = 0.0

    if order is not None and (rn is None or cn is None):
        raise ValueError("Since `order` is specified, your matrix should have row names and column names.")
    if rn is None:
        rn = [f"R{i + 1}" for i in range(arr.shape[0])]
    if cn is None:
        cn = [f"C{j + 1}" for j in range(arr.shape[1])]

    if order is not None:
        all_names = set(rn) | set(cn)
        ord_s = [str(o) for o in order]
        if set(ord_s) != all_names:
            raise ValueError("Elements in `order` should match union of row and column names.")

    rs = pd.Series(np.sum(np.abs(arr), axis=1), index=rn)
    cs = pd.Series(np.sum(np.abs(arr), axis=0), index=cn)
    nn = list(dict.fromkeys(rn + cn))
    xlim_s = pd.Series(0.0, index=nn)
    xlim_s[rs.index] += rs
    xlim_s[cs.index] += cs
    for s in set(rn) & set(cn):
        ri_s = rn.index(s)
        ci_s = cn.index(s)
        xlim_s[s] -= abs(arr[ri_s, ci_s])

    tot_sum = float(xlim_s.sum())
    keep_index = set(xlim_s.index[(xlim_s / max(tot_sum, 1e-30)) >= reduce].tolist())

    ri = [i for i, r in enumerate(rn) if r in keep_index]
    ci = [j for j, c in enumerate(cn) if c in keep_index]

    while True:
        sub = arr[np.ix_(ri, ci)]
        ri_zero = [ri[k] for k in range(len(ri)) if np.sum(np.abs(sub[k, :])) == 0]
        ci_zero = [ci[k] for k in range(len(ci)) if np.sum(np.abs(sub[:, k])) == 0]
        if not ri_zero and not ci_zero:
            break
        ri = [i for i in ri if i not in ri_zero]
        ci = [j for j in ci if j not in ci_zero]

    un = list(dict.fromkeys(rn + cn))
    rn_red = [rn[i] for i in ri]
    cn_red = [cn[j] for j in ci]
    nn_red = list(dict.fromkeys(rn_red + cn_red))
    if len(ri) + len(ci) < arr.shape[0] + arr.shape[1]:
        if isinstance(grid_col, (list, tuple, np.ndarray)) and len(grid_col) == len(un):
            keep_set = set(nn_red)
            grid_col = [grid_col[k] for k, name in enumerate(un) if name in keep_set]

    arr = arr[np.ix_(ri, ci)]
    rn, cn = rn_red, cn_red
    if isinstance(col, np.ndarray) and col.ndim == 2:
        col = col[np.ix_(ri, ci)]
    elif isinstance(col, pd.DataFrame) and col.shape == (len(rs), len(cs)):
        col = col.iloc[ri, ci]
    if isinstance(row_col, (list, tuple, np.ndarray)) and len(row_col) > 1:
        row_col = [row_col[i] for i in ri]
    if isinstance(column_col, (list, tuple, np.ndarray)) and len(column_col) > 1:
        column_col = [column_col[j] for j in ci]

    factors = [str(o) for o in order] if order is not None else nn_red
    factors = [f for f in factors if f in set(nn_red)]
    n_f = len(factors)

    # Resolve grid.col
    if grid_col is None:
        gcol_map = dict(zip(factors, rand_color(n_f)))
    elif isinstance(grid_col, (dict, pd.Series)):
        g_dict = {str(k): v for k, v in dict(grid_col).items()}
        unnamed = [f for f in factors if f not in g_dict]
        if unnamed:
            for u_name, u_col in zip(unnamed, rand_color(len(unnamed))):
                g_dict[u_name] = u_col
        gcol_map = {f: g_dict[f] for f in factors}
    elif isinstance(grid_col, (str, int, float, np.number)):
        gcol_map = {f: grid_col for f in factors}
    else:
        g_list = list(grid_col)
        if len(g_list) == 1:
            gcol_map = {f: g_list[0] for f in factors}
        elif len(g_list) == n_f:
            gcol_map = dict(zip(factors, g_list))
        else:
            raise ValueError("`grid_col` length should be 1 or match number of sectors, or be a dict/Series.")

    # Resolve link color matrix
    if col is not None:
        if callable(col):
            col_mat = np.asarray(col(arr), dtype=object)
        else:
            col_mat = _normalize_to_mat(col, rn, cn, "#FFFFFF00")
    elif row_col is not None:
        rc = [row_col] * len(rn) if isinstance(row_col, (str, int, float)) or len(row_col) == 1 else list(row_col)
        if len(rc) == 1:
            rc = rc * len(rn)
        col_mat = np.repeat(np.asarray(rc, dtype=object)[:, None], len(cn), axis=1)
    elif column_col is not None:
        cc = [column_col] * len(cn) if isinstance(column_col, (str, int, float)) or len(column_col) == 1 else list(column_col)
        if len(cc) == 1:
            cc = cc * len(cn)
        col_mat = np.repeat(np.asarray(cc, dtype=object)[None, :], len(rn), axis=0)
    else:
        col_mat = np.repeat(np.asarray([gcol_map[r] for r in rn], dtype=object)[:, None], len(cn), axis=1)

    flat_cols = _apply_transparency_to_colors(col_mat.T.ravel(), transparency)
    col_mat = np.asarray(flat_cols, dtype=object).reshape((len(cn), len(rn))).T

    if link_arr_length is None:
        link_arr_length = 0.02 if link_arr_type == "big.arrow" else 0.4
    if link_arr_width is None:
        link_arr_width = np.asarray(link_arr_length, dtype=float) / 2.0 if np.isscalar(link_arr_length) else np.asarray(link_arr_length, dtype=float) / 2.0

    l_border = _normalize_to_mat(link_border, rn, cn, None)
    l_lwd = _normalize_to_mat(link_lwd, rn, cn, 1.0)
    l_lty = _normalize_to_mat(link_lty, rn, cn, 1)
    l_alen = _normalize_to_mat(link_arr_length, rn, cn, 0.4)
    l_awid = _normalize_to_mat(link_arr_width, rn, cn, 0.2)
    l_atyp = _normalize_to_mat(link_arr_type, rn, cn, "triangle")
    l_alty = _normalize_to_mat(link_arr_lty, rn, cn, 1)
    l_alwd = _normalize_to_mat(link_arr_lwd, rn, cn, 1.0)
    l_acol = _normalize_to_mat(link_arr_col, rn, cn, None)
    l_vis = _normalize_to_mat(link_visible, rn, cn, True)
    l_z = _normalize_to_mat(link_zindex, rn, cn, 1)
    l_dir = _normalize_to_mat(directional, rn, cn, 0)
    l_dtyp = _normalize_to_mat(direction_type, rn, cn, "diffHeight")

    df_m = _mat2df(arr, rn, cn)
    ri_idx = df_m["ri"].to_numpy(dtype=int)
    ci_idx = df_m["ci"].to_numpy(dtype=int)

    return chord_diagram_from_data_frame(
        df_m[["rn", "cn", "value"]],
        grid_col=gcol_map,
        grid_border=grid_border,
        transparency=None,
        col=col_mat[ri_idx, ci_idx].tolist(),
        order=order,
        xmax=xmax,
        directional=l_dir[ri_idx, ci_idx].tolist(),
        direction_type=l_dtyp[ri_idx, ci_idx].tolist(),
        diff_height=diff_height,
        link_target_prop=link_target_prop,
        target_prop_height=target_prop_height,
        reduce=0.0,
        self_link=self_link,
        pre_allocate_tracks=pre_allocate_tracks,
        annotation_track=annotation_track,
        annotation_track_height=annotation_track_height,
        link_auto=link_auto,
        link_sort=link_sort,
        link_decreasing=link_decreasing,
        link_border=l_border[ri_idx, ci_idx].tolist(),
        link_lwd=l_lwd[ri_idx, ci_idx].tolist(),
        link_lty=l_lty[ri_idx, ci_idx].tolist(),
        link_arr_length=l_alen[ri_idx, ci_idx].tolist(),
        link_arr_width=l_awid[ri_idx, ci_idx].tolist(),
        link_arr_type=l_atyp[ri_idx, ci_idx].tolist(),
        link_arr_lwd=l_alwd[ri_idx, ci_idx].tolist(),
        link_arr_lty=l_alty[ri_idx, ci_idx].tolist(),
        link_arr_col=l_acol[ri_idx, ci_idx].tolist(),
        link_largest_ontop=link_largest_ontop,
        link_visible=l_vis[ri_idx, ci_idx].tolist(),
        link_rank=link_rank,
        link_zindex=l_z[ri_idx, ci_idx].tolist(),
        link_overlap=link_overlap,
        scale=scale,
        group=group,
        big_gap=big_gap,
        small_gap=small_gap,
        plot=plot,
        **nkw,
    )


chordDiagramFromMatrix = chord_diagram_from_matrix


def chord_diagram_from_data_frame(
    df: pd.DataFrame,
    grid_col: Any = None,
    grid_border: Any = None,
    transparency: float | None = 0.5,
    col: Any = None,
    order: Sequence[Any] | None = None,
    directional: Any = 0,
    xmax: dict[str, float] | pd.Series | None = None,
    direction_type: Any = "diffHeight",
    diff_height: float | None = None,
    link_target_prop: bool = True,
    target_prop_height: float | None = None,
    reduce: float = 1e-5,
    self_link: int = 2,
    pre_allocate_tracks: Any = None,
    annotation_track: str | Sequence[str] | None = ("name", "grid", "axis"),
    annotation_track_height: Any = None,
    link_border: Any = None,
    link_lwd: Any = 1.0,
    link_lty: Any = 1,
    link_auto: bool = True,
    link_sort: Any = "default",
    link_decreasing: Any = True,
    link_arr_length: Any = None,
    link_arr_width: Any = None,
    link_arr_type: Any = "triangle",
    link_arr_lty: Any = 1,
    link_arr_lwd: Any = 1.0,
    link_arr_col: Any = "black",
    link_largest_ontop: bool = False,
    link_visible: Any = True,
    link_rank: Any = None,
    link_zindex: Any = None,
    link_overlap: bool = False,
    scale: bool = False,
    group: dict[str, Any] | pd.Series | None = None,
    big_gap: float = 10.0,
    small_gap: float = 1.0,
    plot: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Plot a Chord diagram from an adjacency list DataFrame."""
    nkw = normalize_kwargs(kwargs)
    if link_rank is not None:
        raise ValueError("`link.rank` is removed; use `link.zindex` instead.")

    if diff_height is None:
        diff_height = float(convert_height(2.0, "mm"))
    if target_prop_height is None:
        target_prop_height = float(mm_h(1.0))
    if annotation_track is None:
        annotation_track = []
    elif isinstance(annotation_track, str):
        annotation_track = [annotation_track]
    else:
        annotation_track = list(annotation_track)
    if annotation_track_height is None:
        annotation_track_height = [float(v) for v in convert_height([3.0, 2.0], "mm")]
    else:
        annotation_track_height = [float(annotation_track_height)] if np.isscalar(annotation_track_height) else [float(v) for v in annotation_track_height]

    df_in = validate_data_frame(df)
    if df_in.shape[1] < 2:
        raise ValueError("`df` should have at least two columns.")

    if isinstance(direction_type, (list, tuple)) and len(direction_type) == 2 and set(direction_type) == {"diffHeight", "arrows"}:
        direction_type = "diffHeight+arrows"

    if link_arr_length is None:
        link_arr_length = 0.02 if link_arr_type == "big.arrow" else 0.4
    if link_arr_width is None:
        link_arr_width = float(link_arr_length) / 2.0 if np.isscalar(link_arr_length) else np.asarray(link_arr_length, dtype=float) / 2.0

    rn_col = df_in.iloc[:, 0].astype(str).tolist()
    cn_col = df_in.iloc[:, 1].astype(str).tolist()
    nr = len(rn_col)

    num_cols = [j for j in range(2, df_in.shape[1]) if pd.api.types.is_numeric_dtype(df_in.iloc[:, j])]
    if len(num_cols) == 0:
        v1 = np.ones(nr, dtype=float)
        v2 = np.ones(nr, dtype=float)
    elif len(num_cols) == 1:
        v1 = df_in.iloc[:, num_cols[0]].to_numpy(dtype=float)
        v2 = v1.copy()
    else:
        v1 = df_in.iloc[:, num_cols[0]].to_numpy(dtype=float)
        v2 = df_in.iloc[:, num_cols[1]].to_numpy(dtype=float)
        if circos_par.get_value("message"):
            warnings.warn(
                "More than one numeric column found; using the first two for unequal link end widths.",
                UserWarning,
                stacklevel=2,
            )

    if scale:
        s1_out = np.zeros(nr, dtype=float)
        s2_out = np.zeros(nr, dtype=float)
        rn_s = pd.Series(rn_col)
        cn_s = pd.Series(cn_col)
        for nm in set(rn_col):
            tot = float(np.sum(np.abs(v1[rn_s == nm])) + np.sum(np.abs(v1[cn_s == nm])))
            if tot > 0:
                s1_out[rn_s == nm] = v1[rn_s == nm] / tot
        for nm in set(cn_col):
            tot = float(np.sum(np.abs(v1[cn_s == nm])) + np.sum(np.abs(v1[rn_s == nm])))
            if tot > 0:
                s2_out[cn_s == nm] = v1[cn_s == nm] / tot
        v1, v2 = s1_out, s2_out

    work = pd.DataFrame({"rn": rn_col, "cn": cn_col, "value1": v1, "value2": v2})
    if link_zindex is None:
        link_zindex = list(range(1, nr + 1))

    cate = list(dict.fromkeys(rn_col + cn_col))
    if order is not None:
        ord_list = list(order)
        if all(isinstance(o, (int, np.integer)) for o in ord_list):
            if set(ord_list) == set(range(1, len(cate) + 1)):
                cate = [cate[o - 1] for o in ord_list]
            elif set(ord_list) == set(range(len(cate))):
                cate = [cate[o] for o in ord_list]
            else:
                raise ValueError("Numeric `order` must be a permutation of sectors.")
        else:
            ord_s = [str(o) for o in ord_list]
            if set(ord_s) != set(cate):
                raise ValueError("`order` should contain names of all sectors.")
            cate = ord_s

    n_c = len(cate)
    if grid_col is None:
        gcol_map = dict(zip(cate, rand_color(n_c)))
    elif isinstance(grid_col, (dict, pd.Series)):
        g_dict = {str(k): v for k, v in dict(grid_col).items()}
        unnamed = [c for c in cate if c not in g_dict]
        if unnamed:
            for uc, uv in zip(unnamed, rand_color(len(unnamed))):
                g_dict[uc] = uv
        gcol_map = {c: g_dict[c] for c in cate}
    elif isinstance(grid_col, (str, int, float, np.number)):
        gcol_map = {c: grid_col for c in cate}
    else:
        g_list = list(grid_col)
        if len(g_list) == 1:
            gcol_map = {c: g_list[0] for c in cate}
        elif len(g_list) == n_c:
            gcol_map = dict(zip(cate, g_list))
        else:
            raise ValueError("`grid_col` length should be 1 or number of sectors, or a dict/Series.")

    if col is None:
        link_cols = [gcol_map[r] for r in work["rn"]]
    elif callable(col):
        link_cols = list(col(np.maximum(work["value1"].to_numpy(), work["value2"].to_numpy())))
    elif isinstance(col, (str, int, float, np.number)):
        link_cols = [col] * nr
    else:
        c_list = list(col)
        link_cols = [c_list[i % len(c_list)] for i in range(nr)]

    link_cols = _apply_transparency_to_colors(link_cols, transparency)

    def _norm_vec(val: Any, default: Any) -> list[Any]:
        if isinstance(val, pd.DataFrame):
            y_map = {(str(a), str(b)): v for a, b, v in zip(val.iloc[:, 0], val.iloc[:, 1], val.iloc[:, 2])}
            return [y_map.get((r, c), default) for r, c in zip(work["rn"], work["cn"])]
        if isinstance(val, (str, bytes, int, float, np.number, bool, np.bool_)) or val is None:
            return [val] * nr
        seq = list(val)
        if len(seq) == 1:
            return seq * nr
        return [seq[i % len(seq)] for i in range(nr)]

    l_border = _norm_vec(link_border, None)
    l_lwd = [float(v) for v in _norm_vec(link_lwd, 1.0)]
    l_lty = _norm_vec(link_lty, 1)
    l_alen = [float(v) for v in _norm_vec(link_arr_length, 0.4)]
    l_awid = [float(v) for v in _norm_vec(link_arr_width, 0.2)]
    l_atyp = [str(v) for v in _norm_vec(link_arr_type, "triangle")]
    l_alty = _norm_vec(link_arr_lty, 1)
    l_alwd = [float(v) for v in _norm_vec(link_arr_lwd, 1.0)]
    l_acol = _norm_vec(link_arr_col, None)
    l_vis = [bool(v) for v in _norm_vec(link_visible, True)]
    l_z = [float(v) for v in _norm_vec(link_zindex, 1)]
    l_dir = [int(v) for v in _norm_vec(directional, 0)]
    l_dtyp = [str(v) for v in _norm_vec(direction_type, "diffHeight")]

    # Iteratively reduce tiny sectors
    onr = len(work)
    red = float(reduce)
    while True:
        xsum = {c: 0.0 for c in cate}
        for i in range(len(work)):
            r_k = work.iloc[i]["rn"]
            c_k = work.iloc[i]["cn"]
            vv1 = abs(float(work.iloc[i]["value1"]))
            vv2 = abs(float(work.iloc[i]["value2"]))
            if r_k == c_k:
                xsum[r_k] += vv1
                if self_link == 2:
                    xsum[r_k] += vv2
            else:
                xsum[r_k] += vv1
                xsum[c_k] += vv2
        tot = sum(xsum.values())
        keep = {c for c, val in xsum.items() if (val / max(tot, 1e-30)) >= red}
        mask = [((r in keep) and (c in keep)) for r, c in zip(work["rn"], work["cn"])]
        cate = [c for c in cate if c in keep]
        work = work[mask].reset_index(drop=True)
        gcol_map = {k: v for k, v in gcol_map.items() if k in keep}
        link_cols = [v for v, m in zip(link_cols, mask) if m]
        l_border = [v for v, m in zip(l_border, mask) if m]
        l_lwd = [v for v, m in zip(l_lwd, mask) if m]
        l_lty = [v for v, m in zip(l_lty, mask) if m]
        l_alen = [v for v, m in zip(l_alen, mask) if m]
        l_awid = [v for v, m in zip(l_awid, mask) if m]
        l_atyp = [v for v, m in zip(l_atyp, mask) if m]
        l_alwd = [v for v, m in zip(l_alwd, mask) if m]
        l_alty = [v for v, m in zip(l_alty, mask) if m]
        l_acol = [v for v, m in zip(l_acol, mask) if m]
        l_vis = [v for v, m in zip(l_vis, mask) if m]
        l_z = [v for v, m in zip(l_z, mask) if m]
        l_dir = [v for v, m in zip(l_dir, mask) if m]
        l_dtyp = [v for v, m in zip(l_dtyp, mask) if m]
        if len(work) == onr:
            break
        onr = len(work)
        red = 1e-10

    nr = len(work)
    xsum = {c: 0.0 for c in cate}
    for i in range(nr):
        r_k = work.iloc[i]["rn"]
        c_k = work.iloc[i]["cn"]
        vv1 = abs(float(work.iloc[i]["value1"]))
        vv2 = abs(float(work.iloc[i]["value2"]))
        if r_k == c_k:
            xsum[r_k] += vv1
            if self_link == 2:
                xsum[r_k] += vv2
        else:
            xsum[r_k] += vv1
            xsum[c_k] += vv2

    gap_after: dict[str, float] | list[float] | None = None
    if group is not None:
        g_dict = {str(k): v for k, v in dict(group).items()}
        if not set(cate).issubset(set(g_dict.keys())):
            raise ValueError("Keys in `group` should cover all sector names.")
        grp_filtered = [(s, g_dict[s]) for s in g_dict if s in set(cate)]
        grp_order: dict[Any, list[str]] = {}
        for s, g_lbl in grp_filtered:
            grp_order.setdefault(g_lbl, []).append(s)
        # Preserve sorted group keys if strings/numbers like R's split()
        sorted_gkeys = sorted(grp_order.keys(), key=lambda g: str(g))
        sn_by_group = [s for gk in sorted_gkeys for s in grp_order[gk]]
        cate = [s for s in sn_by_group if s in set(cate)]
        xsum = {s: xsum[s] for s in cate}
        gap_after = {}
        for gk in sorted_gkeys:
            members = grp_order[gk]
            for j, s in enumerate(members):
                gap_after[s] = big_gap if j == len(members) - 1 else small_gap

    work["o1"] = 0
    work["o2"] = 0
    work["x1"] = 0.0
    work["x2"] = 0.0

    ls_pair = [link_sort, link_sort] if not isinstance(link_sort, (list, tuple)) else list(link_sort)[:2]
    ld_pair = [bool(link_decreasing), bool(link_decreasing)] if not isinstance(link_decreasing, (list, tuple)) else [bool(link_decreasing[0]), bool(link_decreasing[1])]

    def _simple_order(vals: Sequence[float], do_sort: bool, rev: bool) -> list[int]:
        if do_sort:
            od = np.argsort(vals, kind="mergesort").tolist()
        else:
            od = list(range(len(vals)))
        if rev:
            od = od[::-1]
        return od

    # Root 1 (rn)
    max_o1: dict[str, int] = {}
    sum_1: dict[str, float] = {}
    for nm in list(dict.fromkeys(work["rn"].tolist())):
        idx = np.where(work["rn"].to_numpy() == nm)[0]
        vals1 = np.abs(work["value1"].to_numpy()[idx])
        sum_1[nm] = float(np.sum(vals1))
        if ls_pair[0] == "default":
            fa = cate + cate
            pos_nm = fa.index(nm)
            fa_sub = fa[pos_nm : pos_nm + len(cate)]
            rank_map = {s: r for r, s in enumerate(fa_sub)}
            cns = work["cn"].to_numpy()[idx]
            od = sorted(range(len(idx)), key=lambda k: rank_map[cns[k]], reverse=True)
        elif ls_pair[0] == "asis":
            od = list(range(len(idx)))
        elif ls_pair[0] is False:
            od = _simple_order(vals1, False, ld_pair[0])
        else:
            od = _simple_order(vals1, bool(ls_pair[0]), ld_pair[0])

        od_1based = [k + 1 for k in od]
        work.loc[idx, "o1"] = od_1based
        max_o1[nm] = max(od_1based)

        csum = np.cumsum(vals1[od])
        x1_vals = np.empty(len(idx), dtype=float)
        x1_vals[od] = csum
        work.loc[idx, "x1"] = x1_vals

        if self_link == 1:
            for row_i in idx:
                if work.loc[row_i, "rn"] == work.loc[row_i, "cn"]:
                    work.loc[row_i, "x2"] = float(work.loc[row_i, "x1"]) + abs(float(work.loc[row_i, "value1"])) * 1e-6

    # Root 2 (cn)
    for nm in list(dict.fromkeys(work["cn"].tolist())):
        idx = np.where(work["cn"].to_numpy() == nm)[0]
        vals2 = np.abs(work["value2"].to_numpy()[idx])
        if ls_pair[1] == "default":
            fa = cate + cate
            pos_nm = fa.index(nm)
            fa_sub = fa[pos_nm : pos_nm + len(cate)]
            rank_map = {s: r for r, s in enumerate(fa_sub)}
            rns = work["rn"].to_numpy()[idx]
            seen_cnt: dict[str, int] = {}
            dup_ord = []
            for r_val in rns:
                seen_cnt[r_val] = seen_cnt.get(r_val, 0) + 1
                dup_ord.append(seen_cnt[r_val])
            od = sorted(range(len(idx)), key=lambda k: (rank_map[rns[k]], dup_ord[k]), reverse=True)
        elif ls_pair[1] == "asis":
            od = list(range(len(idx)))
        elif ls_pair[1] is False:
            od = _simple_order(vals2, False, ld_pair[1])
        else:
            od = _simple_order(vals2, bool(ls_pair[1]), ld_pair[1])

        if nm in max_o1:
            is_non_self = (work["rn"].to_numpy()[idx] != nm) if self_link == 1 else np.ones(len(idx), dtype=bool)
            non_self_idx = idx[is_non_self]
            if len(non_self_idx) > 0:
                if self_link == 1:
                    # Re-rank od subset for non-self links
                    sub_pos = [k for k in od if is_non_self[k]]
                    # Map to 0..len-1 in order of non_self_idx
                    orig_to_sub = {orig_k: s_k for s_k, orig_k in enumerate(np.where(is_non_self)[0])}
                    od_sub = [orig_to_sub[k] for k in sub_pos]
                else:
                    od_sub = od
                od_1based = [k + 1 for k in od_sub]
                work.loc[non_self_idx, "o2"] = [v + max_o1[nm] for v in od_1based]
                vals2_sub = np.abs(work["value2"].to_numpy()[non_self_idx])
                csum = np.cumsum(vals2_sub[od_sub]) + sum_1[nm]
                x2_vals = np.empty(len(non_self_idx), dtype=float)
                x2_vals[od_sub] = csum
                work.loc[non_self_idx, "x2"] = x2_vals
        else:
            od_1based = [k + 1 for k in od]
            work.loc[idx, "o2"] = od_1based
            csum = np.cumsum(vals2[od])
            x2_vals = np.empty(len(idx), dtype=float)
            x2_vals[od] = csum
            work.loc[idx, "x2"] = x2_vals

    if self_link == 1:
        self_mask = work["rn"] == work["cn"]
        pmin = np.minimum(work.loc[self_mask, "x1"], work.loc[self_mask, "x2"])
        work.loc[self_mask, "x1"] = pmin
        work.loc[self_mask, "x2"] = pmin

    if xmax is not None:
        xm_dict = {str(k): float(v) for k, v in dict(xmax).items()}
        for k, v in xm_dict.items():
            if k in xsum and v > xsum[k]:
                xsum[k] = v

    if link_overlap:
        if not any(d != 0 for d in l_dir):
            warnings.warn("`link.overlap` should be used with directional = 1 or -1.", UserWarning, stacklevel=2)
        x1_max = {c: 0.0 for c in cate}
        x2_max = {c: 0.0 for c in cate}
        for c in cate:
            m1 = work["rn"] == c
            if np.any(m1):
                x1_max[c] = float(work.loc[m1, "x1"].max())
            m2 = work["cn"] == c
            if np.any(m2):
                x2_max[c] = float(work.loc[m2, "x2"].max()) - x1_max[c]
        work["x2"] = work["x2"] - work["cn"].map(x1_max)
        xsum = {c: max(x1_max[c], x2_max[c]) for c in cate}

    if not plot:
        return work

    o_cell_padding = circos_par.get_value("cell.padding")
    o_start_degree = circos_par.get_value("start.degree")
    o_gap_after = circos_par.get_value("gap.after")
    o_pow = circos_par.get_value("points.overflow.warning")

    circos_par.set_value("cell_padding", [0.0, 0.0, 0.0, 0.0], _suppress_warn=True)
    circos_par.set_value("points_overflow_warning", False, _suppress_warn=True)

    try:
        if gap_after is not None:
            circos_par.set_value("gap_after", gap_after, _suppress_warn=True)

        cur_gap = circos_par.get_value("gap.after")
        if isinstance(cur_gap, (int, float)) and cur_gap == 1.0:
            rn_set = set(work["rn"])
            cn_set = set(work["cn"])
            if len(rn_set & cn_set) == 0 and len(cate) >= 2:
                ind1 = [i for i, c in enumerate(cate) if c in rn_set]
                ind2 = [i for i, c in enumerate(cate) if c in cn_set]
                if ind1 and ind2 and (max(ind1) < min(ind2) or max(ind2) < min(ind1)):
                    n_df1 = len(rn_set)
                    n_df2 = len(cn_set)
                    s1 = float(np.sum(np.abs(work["value1"])))
                    s2 = float(np.sum(np.abs(work["value2"])))
                    if cate[0] in cn_set:
                        n_df1, n_df2 = n_df2, n_df1
                        s1, s2 = s2, s1
                    n_sec = n_df1 + n_df2
                    d1 = (360.0 - small_gap * (n_sec - 2) - big_gap * 2.0) * (s1 / max(s1 + s2, 1e-30)) + small_gap * (n_df1 - 1)
                    sd0 = float(circos_par.get_value("start.degree"))
                    if circos_par.get_value("clock.wise"):
                        start_deg = sd0 - (180.0 - d1) / 2.0
                    else:
                        start_deg = sd0 + (180.0 - d1) / 2.0
                    g_vec = [small_gap] * (n_df1 - 1) + [big_gap] + [small_gap] * (n_df2 - 1) + [big_gap]
                    circos_par.set_value("start_degree", start_deg, _suppress_warn=True)
                    circos_par.set_value("gap_after", g_vec, _suppress_warn=True)

        xlim_mat = np.column_stack([np.zeros(len(cate), dtype=float), np.asarray([xsum[c] for c in cate], dtype=float)])
        circos_initialize(sectors=pd.Categorical(cate, categories=cate, ordered=True), xlim=xlim_mat)

        if pre_allocate_tracks is not None:
            pa = _parse_pre_allocate_tracks(pre_allocate_tracks)
            for va in pa:
                circos_track_plot_region(
                    ylim=va["ylim"],
                    track_height=va["track_height"],
                    bg_col=va["bg_col"],
                    bg_border=va["bg_border"],
                    bg_lty=va["bg_lty"],
                    bg_lwd=va["bg_lwd"],
                    track_margin=va["track_margin"],
                )

        if "name" in annotation_track:
            h_idx = annotation_track.index("name")
            th = annotation_track_height[h_idx] if h_idx < len(annotation_track_height) else annotation_track_height[0]

            def _panel_name(x: Any, y: Any) -> None:
                xlim_c = get_cell_meta_data("xlim")
                si_c = get_cell_meta_data("sector.index")
                circos_text(float(np.mean(xlim_c)), 0.9, labels=si_c, cex=0.8, facing="inside", nice_facing=True, adj=(0.5, 0.0))

            circos_track_plot_region(ylim=[0, 1], bg_border=None, panel_fun=_panel_name, track_height=th)

        if "grid" in annotation_track:
            h_idx = annotation_track.index("grid")
            th = annotation_track_height[h_idx] if h_idx < len(annotation_track_height) else annotation_track_height[-1]

            def _panel_grid(x: Any, y: Any) -> None:
                xlim_c = get_cell_meta_data("xlim")
                si_c = get_cell_meta_data("sector.index")
                b_col = gcol_map[si_c] if grid_border is None else grid_border
                circos_rect(xlim_c[0], 0.0, xlim_c[1], 1.0, col=gcol_map[si_c], border=b_col)
                if "axis" in annotation_track:
                    if scale:
                        circos_axis("top", labels=lambda vals: [f"{int(round(v * 100))}%" for v in vals], labels_cex=0.5)
                    else:
                        circos_axis("top", labels_cex=0.5)

            circos_track_plot_region(ylim=[0, 1], bg_border=None, panel_fun=_panel_grid, track_height=th)

        rou = get_most_inside_radius()
        rou1 = np.empty(nr, dtype=float)
        rou2 = np.empty(nr, dtype=float)
        dh = float(diff_height)
        for i in range(nr):
            if l_dir[i] != 0 and "diffHeight" in l_dtyp[i]:
                if l_dir[i] == 1:
                    rou1[i], rou2[i] = (rou - dh, rou) if dh > 0 else (rou, rou + dh)
                elif l_dir[i] == -1:
                    rou1[i], rou2[i] = (rou, rou - dh) if dh > 0 else (rou + dh, rou)
                else:
                    rou1[i], rou2[i] = (rou - dh, rou - dh) if dh > 0 else (rou + dh, rou + dh)
            else:
                rou1[i] = rou
                rou2[i] = rou

        if link_largest_ontop:
            link_order = np.argsort(np.maximum(np.abs(work["value1"].to_numpy()), np.abs(work["value2"].to_numpy())), kind="mergesort")
        else:
            link_order = np.argsort(np.asarray(l_z, dtype=float), kind="mergesort")

        sum_v1 = float(np.sum(np.abs(work["value1"])))
        sum_v2 = float(np.sum(np.abs(work["value2"])))
        for k in link_order:
            vv1 = abs(float(work.iloc[k]["value1"]))
            vv2 = abs(float(work.iloc[k]["value2"]))
            if (vv1 / max(sum_v1, 1e-30) < 1e-6) and (vv2 / max(sum_v2, 1e-30) < 1e-6):
                continue
            if l_vis[k] and link_cols[k] != "#FFFFFF00":
                p1_end = [float(work.iloc[k]["x1"]) - vv1, float(work.iloc[k]["x1"])]
                p2_end = [float(work.iloc[k]["x2"]) - vv2, float(work.iloc[k]["x2"])]
                if "arrows" in l_dtyp[k]:
                    circos_link(
                        work.iloc[k]["rn"],
                        p1_end,
                        work.iloc[k]["cn"],
                        p2_end,
                        directional=l_dir[k],
                        col=link_cols[k],
                        rou1=float(rou1[k]),
                        rou2=float(rou2[k]),
                        border=l_border[k],
                        lwd=l_lwd[k],
                        lty=l_lty[k],
                        arr_length=l_alen[k],
                        arr_width=l_awid[k],
                        arr_type=l_atyp[k],
                        arr_col=l_acol[k],
                        arr_lty=l_alty[k],
                        arr_lwd=l_alwd[k],
                        **nkw,
                    )
                else:
                    circos_link(
                        work.iloc[k]["rn"],
                        p1_end,
                        work.iloc[k]["cn"],
                        p2_end,
                        directional=0,
                        col=link_cols[k],
                        rou1=float(rou1[k]),
                        rou2=float(rou2[k]),
                        border=l_border[k],
                        lwd=l_lwd[k],
                        lty=l_lty[k],
                        **nkw,
                    )

        work["col"] = link_cols

        if link_target_prop and all("diffHeight" in dt for dt in l_dtyp) and dh > 0:
            tracks = get_all_track_index()
            if tracks:
                last_track = tracks[-1]
                if all(d in (1, 2) for d in l_dir):
                    for i in range(nr):
                        vv1 = abs(float(work.iloc[i]["value1"]))
                        if vv1 > 0 and l_vis[i] and link_cols[i] != "#FFFFFF00":
                            si = str(work.iloc[i]["rn"])
                            set_current_cell(si, last_track)
                            bar_h = _convert_h_from_canvas_to_data(min(float(target_prop_height), dh), si, last_track)
                            tm_val = float(get_cell_meta_data("track.margin", track_index=last_track)[0]) + float(circos_par.get_value("track.margin")[1])
                            rect_y = _convert_h_from_canvas_to_data(tm_val, si, last_track)
                            x1_i = float(work.iloc[i]["x1"])
                            circos_rect(x1_i, -rect_y, x1_i - vv1, -rect_y - bar_h, col=gcol_map[str(work.iloc[i]["cn"])], border=None)
                if all(d in (-1, 2) for d in l_dir):
                    for i in range(nr):
                        vv1 = abs(float(work.iloc[i]["value1"]))
                        vv2 = abs(float(work.iloc[i]["value2"]))
                        if vv1 > 0 and l_vis[i] and link_cols[i] != "#FFFFFF00":
                            si = str(work.iloc[i]["cn"])
                            set_current_cell(si, last_track)
                            bar_h = _convert_h_from_canvas_to_data(min(float(target_prop_height), dh), si, last_track)
                            off_y = float(mm_y(1.0))
                            x2_i = float(work.iloc[i]["x2"])
                            circos_rect(x2_i, -off_y, x2_i - vv2, -off_y - bar_h, col=gcol_map[str(work.iloc[i]["rn"])], border=None)

        return work
    finally:
        circos_par.set_value("cell_padding", o_cell_padding, _suppress_warn=True)
        circos_par.set_value("start_degree", o_start_degree, _suppress_warn=True)
        circos_par.set_value("gap_after", o_gap_after, _suppress_warn=True)
        circos_par.set_value("points_overflow_warning", o_pow, _suppress_warn=True)


chordDiagramFromDataFrame = chord_diagram_from_data_frame


def calc_gap(x1: Any, x2: Any, big_gap: float = 10.0, small_gap: float = 1.0, **kwargs: Any) -> float:
    """Calculate big.gap for the second two-group Chord diagram so sectors share scale with the first."""
    nkw = normalize_kwargs(kwargs)
    big_gap = float(nkw.get("big_gap", big_gap))
    small_gap = float(nkw.get("small_gap", small_gap))

    def _stats(obj: Any) -> tuple[float, int, int]:
        if isinstance(obj, np.ndarray):
            arr = np.asarray(obj, dtype=float)
            return float(np.sum(np.abs(arr))), arr.shape[0], arr.shape[1]
        df = pd.DataFrame(obj)
        all_num = all(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)
        has_explicit_names = not isinstance(df.index, pd.RangeIndex) and not isinstance(df.columns, pd.RangeIndex)
        if all_num and (df.shape[1] != 3 or has_explicit_names):
            arr = df.to_numpy(dtype=float)
            return float(np.sum(np.abs(arr))), arr.shape[0], arr.shape[1]
        return float(np.sum(np.abs(df.iloc[:, 2].to_numpy(dtype=float)))), int(df.iloc[:, 0].nunique()), int(df.iloc[:, 1].nunique())

    sum1, n1_1, n2_1 = _stats(x1)
    sum_gap1 = small_gap * (n1_1 - 1) + big_gap + small_gap * (n2_1 - 1) + big_gap
    sum2, n1_2, n2_2 = _stats(x2)
    sum_gap2 = small_gap * (n1_2 + n2_2 - 2)

    if sum1 < sum2:
        raise ValueError("Sum of `x1` should be larger than sum of `x2`.")
    percent = sum2 / sum1
    blank_deg = (360.0 - sum_gap1) * (1.0 - percent)
    return float((blank_deg - sum_gap2) / 2.0)
