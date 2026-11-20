"""Circular heatmaps (circos.heatmap, circos.heatmap.initialize, link, get.x)."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from ._canvas import strwidth_canvas
from ._colors import color_to_hex, is_na_like
from ._dendrogram import (
    DendrogramNode,
    HClust,
    adjust_dend_by_x,
    as_dendrogram,
    circos_dendrogram,
    cut_dendrogram,
    cutree,
    hclust_from_data,
    reorder_dendrogram,
)
from ._link import circos_link
from ._low_level import circos_rect, circos_text
from ._plot import circos_initialize, circos_track_plot_region
from ._state import (
    add_sector_meta_data,
    circos_par,
    get_all_sector_index,
    get_cell_meta_data,
    is_circos_initialized,
    normalize_kwargs,
)
from ._utils import recycle_to_length


def _is_heatmap_cached() -> bool:
    env = circos_par.get_value("__tempenv__")
    return env.get("circos.heatmap.initialized") is True


def _format_heatmap_matrix(mat: Any) -> tuple[np.ndarray, list[str] | None, list[str] | None, bool]:
    """Return (2D array, rownames or None, colnames or None, is_char)."""
    if isinstance(mat, pd.DataFrame):
        rownames = [str(r) for r in mat.index] if not isinstance(mat.index, pd.RangeIndex) or not list(mat.index) == list(range(len(mat))) else None
        colnames = [str(c) for c in mat.columns]
        arr = mat.to_numpy()
    else:
        arr = np.asarray(mat)
        rownames = None
        colnames = None
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("Input `mat` should be a 1D vector or 2D matrix/DataFrame.")
    is_char = arr.dtype.kind in ("U", "S") or (arr.dtype == object and any(isinstance(v, str) for v in arr.ravel()))
    return arr, rownames, colnames, is_char


def _format_heatmap_input(mat_arr: np.ndarray, split: Any) -> tuple[dict[str, np.ndarray], dict[str, list[int]], list[str]]:
    nr = mat_arr.shape[0]
    cached_levels: list[str] | None = None
    if split is None and _is_heatmap_cached():
        env = circos_par.get_value("__tempenv__")
        split = env.get("circos.heatmap.split")
        cached_levels = env.get("circos.heatmap.split_levels")

    if split is None:
        mat_list = {"mat": mat_arr}
        subset_list = {"mat": list(range(nr))}
        split_vec = ["mat"] * nr
        return mat_list, subset_list, split_vec

    if isinstance(split, pd.Categorical):
        s_vals = [str(v) for v in split]
        present = set(s_vals)
        levels = [str(c) for c in split.categories if str(c) in present]
    elif isinstance(split, pd.Series) and isinstance(split.dtype, pd.CategoricalDtype):
        s_vals = [str(v) for v in split]
        present = set(s_vals)
        levels = [str(c) for c in split.cat.categories if str(c) in present]
    else:
        s_vals = [str(v) for v in split]
        if cached_levels is not None:
            present = set(s_vals)
            levels = [str(c) for c in cached_levels if str(c) in present]
        else:
            levels = list(dict.fromkeys(s_vals))

    mat_list: dict[str, np.ndarray] = {}
    subset_list: dict[str, list[int]] = {}
    s_arr = np.asarray(s_vals, dtype=object)
    for lv in levels:
        idx = np.where(s_arr == lv)[0].tolist()
        mat_list[lv] = mat_arr[idx, :]
        subset_list[lv] = idx
    return mat_list, subset_list, s_vals


def _default_dend_callback(dend: DendrogramNode, m: np.ndarray, si: str) -> DendrogramNode:
    return reorder_dendrogram(dend, np.nanmean(np.asarray(m, dtype=float), axis=1))


def _invoke_dend_callback(cb: Callable[..., DendrogramNode], dend: DendrogramNode, m: np.ndarray, si: str) -> DendrogramNode:
    n_p = len(inspect.signature(cb).parameters)
    if n_p >= 3:
        return cb(dend, m, si)
    if n_p == 2:
        return cb(dend, m)
    return cb(dend)


def circos_heatmap_initialize(
    mat: Any,
    split: Any = None,
    cluster: Any = True,
    clustering_method: str = "complete",
    distance_method: str = "euclidean",
    dend_callback: Callable[..., DendrogramNode] = _default_dend_callback,
    cell_width: float | Sequence[float] = 1.0,
    **kwargs: Any,
) -> None:
    """Initialize circular layout and optional row clustering for `circos_heatmap`."""
    nkw = normalize_kwargs(kwargs)
    clustering_method = nkw.get("clustering_method", clustering_method)
    distance_method = nkw.get("distance_method", distance_method)
    dend_callback = nkw.get("dend_callback", dend_callback)
    cell_width = nkw.get("cell_width", cell_width)

    arr, rownames, _, is_char = _format_heatmap_matrix(mat)
    nr = arr.shape[0]
    cw_arr = np.asarray(recycle_to_length(cell_width, nr), dtype=float)
    cw_arr = cw_arr / float(np.sum(cw_arr)) * nr

    if cluster is None or (isinstance(cluster, float) and np.isnan(cluster)):
        cluster = False

    cluster_is_dend = False
    dend_obj: DendrogramNode | None = None
    if not isinstance(cluster, (bool, np.bool_)):
        dend_obj = as_dendrogram(cluster)
        cluster_is_dend = True
    elif bool(cluster) and split is not None and np.isscalar(split) and isinstance(split, (int, np.integer)):
        hc = hclust_from_data(arr, distance_method=distance_method, clustering_method=clustering_method, labels=rownames)
        dend_obj = as_dendrogram(hc)
        cluster_is_dend = True

    dend_list: dict[str, DendrogramNode] | None = None
    if cluster_is_dend and dend_obj is not None:
        if split is None:
            split2: Any = ["group"] * nr
            dend_list = {"group": dend_obj}
        else:
            if not (np.isscalar(split) and isinstance(split, (int, np.integer))):
                raise ValueError("When `cluster` is a dendrogram/hclust object, `split` can only be a single integer.")
            k = int(split)
            ct = cutree(dend_obj, k)
            lbls = dend_obj.labels()
            lbl_to_idx = {l: i for i, l in enumerate(rownames if rownames is not None else [str(j + 1) for j in range(nr)])}
            split2 = [f"group{ct[lbl]}" for lbl in (rownames if rownames is not None else [str(j + 1) for j in range(nr)])]
            subs = cut_dendrogram(dend_obj, k)
            dend_list = {}
            for sub in subs:
                first_lbl = sub.labels()[0]
                grp_name = f"group{ct[first_lbl]}"
                # Re-index leaf_index locally 0..sub.members-1 using rank of original leaf indices
                orig_idxs = sub.leaf_indices()
                ranks = np.argsort(np.argsort(orig_idxs, kind="mergesort"), kind="mergesort")
                rank_map = {orig_idxs[p]: int(ranks[p]) for p in range(len(orig_idxs))}

                def _remap(nd: DendrogramNode) -> None:
                    if nd.leaf:
                        nd.leaf_index = rank_map[int(nd.leaf_index)]
                    for c in nd.children:
                        _remap(c)

                _remap(sub)
                dend_list[grp_name] = sub
    else:
        split2 = ["group"] * nr if split is None else split

    mat_list, subset_list, split_vec = _format_heatmap_input(arr, split2)
    cw_list = {k: cw_arr[subset_list[k]] for k in mat_list}
    if dend_list is not None:
        for nm in list(mat_list.keys()):
            dend_list[nm] = _invoke_dend_callback(dend_callback, dend_list[nm], mat_list[nm], nm)

    cp = [0.0, 0.0, 0.0, 0.0] if circos_par.get_value("cell.padding") == [0.02, 1.0, 0.02, 1.0] else circos_par.get_value("cell.padding")
    tm = [0.02, 0.0] if circos_par.get_value("track.margin") == [0.01, 0.01] else circos_par.get_value("track.margin")
    gd = 2.0 if circos_par.get_value("gap.degree") == 1.0 else circos_par.get_value("gap.degree")
    circos_par.set_value("cell_padding", cp, _suppress_warn=True)
    circos_par.set_value("track_margin", tm, _suppress_warn=True)
    circos_par.set_value("gap_degree", gd, _suppress_warn=True)

    sec_names = list(mat_list.keys())
    xlim_mat = np.column_stack([np.zeros(len(sec_names)), np.array([float(np.sum(cw_list[k])) for k in sec_names])])
    circos_initialize(sectors=pd.Categorical(sec_names, categories=sec_names, ordered=True), xlim=xlim_mat)

    env = circos_par.get_value("__tempenv__")
    if cluster_is_dend and dend_list is not None:
        for se in get_all_sector_index():
            dend = dend_list[se]
            od = dend.leaf_indices()
            cw = cw_list[se][od]
            od2 = np.argsort(od, kind="mergesort")
            cmid = np.cumsum(cw) - cw / 2.0
            add_sector_meta_data("cell_width", cw[od2], sector_index=se)
            add_sector_meta_data("cell_middle", cmid[od2], sector_index=se)
            dend_adj = adjust_dend_by_x(dend, cmid)
            add_sector_meta_data("row_dend", dend_adj, sector_index=se)
            add_sector_meta_data("dend", dend_adj, sector_index=se)
            add_sector_meta_data("row_order", od, sector_index=se)
            add_sector_meta_data("order", od, sector_index=se)
            add_sector_meta_data("subset", subset_list[se], sector_index=se)
        env["circos.heatmap.cluster"] = True
    else:
        if is_char:
            cluster = False
        if bool(cluster):
            dend_list = {}
            for nm, m in mat_list.items():
                hc = hclust_from_data(m, distance_method=distance_method, clustering_method=clustering_method)
                d = as_dendrogram(hc)
                dend_list[nm] = _invoke_dend_callback(dend_callback, d, m, nm)
            for se in get_all_sector_index():
                dend = dend_list[se]
                od = dend.leaf_indices()
                cw = cw_list[se][od]
                od2 = np.argsort(od, kind="mergesort")
                cmid = np.cumsum(cw) - cw / 2.0
                add_sector_meta_data("cell_width", cw[od2], sector_index=se)
                add_sector_meta_data("cell_middle", cmid[od2], sector_index=se)
                dend_adj = adjust_dend_by_x(dend, cmid)
                add_sector_meta_data("row_dend", dend_adj, sector_index=se)
                add_sector_meta_data("dend", dend_adj, sector_index=se)
                add_sector_meta_data("row_order", od, sector_index=se)
                add_sector_meta_data("order", od, sector_index=se)
                add_sector_meta_data("subset", subset_list[se], sector_index=se)
            env["circos.heatmap.cluster"] = True
        else:
            for se in get_all_sector_index():
                nr_s = mat_list[se].shape[0]
                cw = cw_list[se]
                add_sector_meta_data("row_order", list(range(nr_s)), sector_index=se)
                add_sector_meta_data("order", list(range(nr_s)), sector_index=se)
                add_sector_meta_data("subset", subset_list[se], sector_index=se)
                add_sector_meta_data("cell_width", cw, sector_index=se)
                add_sector_meta_data("cell_middle", np.cumsum(cw) - cw / 2.0, sector_index=se)
            env["circos.heatmap.cluster"] = False

    env["circos.heatmap.split"] = split_vec
    env["circos.heatmap.split_levels"] = sec_names
    env["circos.heatmap.rownames"] = rownames
    env["circos.heatmap.initialized"] = True


def _validate_heatmap(mat_list: dict[str, np.ndarray]) -> None:
    env = circos_par.get_value("__tempenv__")
    smeta = env.get("sector.meta.data", {})
    sec_order = get_all_sector_index()
    if list(mat_list.keys()) != sec_order:
        raise ValueError(
            "The sector order in `mat` differs from the cached heatmap layout. "
            "Provide matching sectors or call `circos.clear()` first."
        )
    sizes_cached = [len(smeta[s]["row_order"]) for s in sec_order]
    sizes_curr = [mat_list[s].shape[0] for s in sec_order]
    if sizes_cached != sizes_curr:
        raise ValueError(
            "The number of rows in each group differs from the cached heatmap layout. "
            "Provide a matrix with the same row count or call `circos.clear()` first."
        )


def circos_heatmap_fn(
    mat: Any,
    split: Any = None,
    col: Any = None,
    na_col: Any = "grey",
    cell_border: Any = None,
    cell_lty: Any = 1,
    cell_lwd: float = 1.0,
    bg_border: Any = None,
    bg_lty: Any = 1,
    bg_lwd: float = 1.0,
    ignore_white: bool | None = None,
    cluster: Any = True,
    clustering_method: str = "complete",
    distance_method: str = "euclidean",
    dend_callback: Callable[..., DendrogramNode] = _default_dend_callback,
    dend_side: str = "none",
    dend_track_height: float = 0.1,
    rownames_side: str = "none",
    rownames_cex: float | Sequence[float] = 0.5,
    rownames_font: Any = 1,
    rownames_col: Any = "black",
    show_sector_labels: bool = False,
    cell_width: float | Sequence[float] = 1.0,
    **kwargs: Any,
) -> None:
    """Draw a circular heatmap track (with optional dendrograms, row names, and multi-track sharing)."""
    nkw = normalize_kwargs(kwargs)
    na_col = nkw.pop("na_col", na_col)
    cell_border = nkw.pop("cell_border", cell_border)
    cell_lty = nkw.pop("cell_lty", cell_lty)
    cell_lwd = float(nkw.pop("cell_lwd", cell_lwd))
    bg_border = nkw.pop("bg_border", bg_border)
    bg_lty = nkw.pop("bg_lty", bg_lty)
    bg_lwd = float(nkw.pop("bg_lwd", bg_lwd))
    ignore_white = nkw.pop("ignore_white", ignore_white)
    clustering_method = nkw.pop("clustering_method", clustering_method)
    distance_method = nkw.pop("distance_method", distance_method)
    dend_callback = nkw.pop("dend_callback", dend_callback)
    dend_side = nkw.pop("dend_side", dend_side)
    dend_track_height = float(nkw.pop("dend_track_height", dend_track_height))
    rownames_side = nkw.pop("rownames_side", rownames_side)
    rownames_cex = nkw.pop("rownames_cex", rownames_cex)
    rownames_font = nkw.pop("rownames_font", rownames_font)
    rownames_col = nkw.pop("rownames_col", rownames_col)
    show_sector_labels = bool(nkw.pop("show_sector_labels", show_sector_labels))
    cell_width = nkw.pop("cell_width", cell_width)

    if col is None:
        raise ValueError("You should provide `col` (a ColorRamp2 function or a dict mapping categories to colors).")
    if ignore_white is None:
        ignore_white = is_na_like(cell_border)

    arr, rownames, _, _ = _format_heatmap_matrix(mat)
    env = circos_par.get_value("__tempenv__")

    if not is_circos_initialized():
        env["circos.heatmap.split"] = None
        env["circos.heatmap.split_levels"] = None
        circos_heatmap_initialize(
            mat,
            split=split,
            cluster=cluster,
            clustering_method=clustering_method,
            distance_method=distance_method,
            dend_callback=dend_callback,
            cell_width=cell_width,
        )

    mat_list, _, _ = _format_heatmap_input(arr, None)
    _validate_heatmap(mat_list)

    if dend_side not in ("none", "outside", "inside"):
        raise ValueError("`dend_side` must be 'none', 'outside', or 'inside'.")
    if rownames_side not in ("none", "outside", "inside"):
        raise ValueError("`rownames_side` must be 'none', 'outside', or 'inside'.")
    if dend_side == rownames_side and dend_side in ("inside", "outside"):
        raise ValueError("Dendrograms and row names cannot be on the same side.")

    if not env.get("circos.heatmap.cluster", False):
        dend_side = "none"

    smeta = env["sector.meta.data"]
    sec_names = get_all_sector_index()

    if dend_side == "outside":
        dend_list = [smeta[s]["row_dend"] for s in sec_names]
        max_h = max(float(d.height) for d in dend_list)
        circos_track_plot_region(
            ylim=[0.0, max_h],
            bg_border=None,
            track_height=dend_track_height,
            panel_fun=lambda x, y: circos_dendrogram(
                dend_list[get_cell_meta_data("sector.numeric.index") - 1],
                max_height=max_h,
                facing="inside",
                use_x_attr=True,
            ),
        )

    nr_tot = arr.shape[0]
    rn_cexs = [float(v) for v in recycle_to_length(rownames_cex, nr_tot)]
    rn_fonts = recycle_to_length(rownames_font, nr_tot)
    rn_cols = recycle_to_length(rownames_col, nr_tot)

    rn_track_h = 0.05
    if rownames is not None:
        rn_track_h = max(
            strwidth_canvas(rownames[idx], cex=rn_cexs[idx], font=rn_fonts[idx])
            for s in sec_names
            for idx in smeta[s]["subset"]
        )

    if rownames_side == "outside" and rownames is not None:
        def _rn_out_panel(x: Any, y: Any) -> None:
            si = get_cell_meta_data("sector.index")
            od = smeta[si]["row_order"]
            sub_idx = smeta[si]["subset"]
            cmid = smeta[si]["cell_middle"]
            circos_text(
                cmid[od],
                np.zeros(len(od)),
                [rownames[sub_idx[i]] for i in od],
                cex=[rn_cexs[sub_idx[i]] for i in od],
                font=[rn_fonts[sub_idx[i]] for i in od],
                col=[rn_cols[sub_idx[i]] for i in od],
                facing="clockwise",
                nice_facing=True,
                adj=(0.0, 0.5),
            )

        circos_track_plot_region(ylim=[0.0, 1.0], bg_border=None, track_height=rn_track_h, panel_fun=_rn_out_panel)

    if callable(col):
        def _col_fun(m: np.ndarray) -> np.ndarray:
            res = col(m)
            out = np.empty(m.shape, dtype=object)
            na_hx = color_to_hex(na_col, include_alpha=True) or "#BEBEBEFF"
            for r in range(m.shape[0]):
                for c in range(m.shape[1]):
                    v = res[r, c]
                    out[r, c] = na_hx if v is None else (color_to_hex(v, include_alpha=True) or na_hx)
            return out
    elif isinstance(col, (dict, pd.Series)):
        cmap = {str(k): (color_to_hex(v, include_alpha=True) or "#BEBEBEFF") for k, v in dict(col).items()}
        na_hx = color_to_hex(na_col, include_alpha=True) or "#BEBEBEFF"

        def _col_fun(m: np.ndarray) -> np.ndarray:
            out = np.empty(m.shape, dtype=object)
            for r in range(m.shape[0]):
                for c in range(m.shape[1]):
                    val = m[r, c]
                    out[r, c] = na_hx if is_na_like(val) else cmap.get(str(val), na_hx)
            return out
    else:
        raise ValueError("`col` should be a callable ColorRamp2 or a dict mapping category values to colors.")

    nc_mat = arr.shape[1]

    def _heatmap_panel(x: Any, y: Any) -> None:
        si = get_cell_meta_data("sector.index")
        m = mat_list[si]
        od = smeta[si]["row_order"]
        m2 = m[od, :]
        col_mat = _col_fun(m2)
        nr_s, nc_s = m2.shape
        cmid = np.asarray(smeta[si]["cell_middle"], dtype=float)[od]
        cwid = np.asarray(smeta[si]["cell_width"], dtype=float)[od]

        if ignore_white and not is_na_like(cell_border):
            if np.any(np.char.startswith(col_mat.astype(str), "#FFFFFF")):
                raise ValueError("`ignore_white` must be False when `cell_border` is set.")

        for j in range(nc_s):
            col_j = col_mat[:, j]
            if ignore_white:
                keep = [not str(c).upper().startswith("#FFFFFF") for c in col_j]
                if not any(keep):
                    continue
                k_idx = np.where(keep)[0]
                c_draw = [col_j[k] for k in k_idx]
                b_draw = c_draw if is_na_like(cell_border) else cell_border
                circos_rect(
                    cmid[k_idx] - cwid[k_idx] / 2.0,
                    np.repeat(nc_s - j - 1.0, len(k_idx)),
                    cmid[k_idx] + cwid[k_idx] / 2.0,
                    np.repeat(nc_s - j, len(k_idx)),
                    border=b_draw,
                    lty=cell_lty,
                    lwd=cell_lwd,
                    col=c_draw,
                )
            else:
                b_draw = list(col_j) if is_na_like(cell_border) else cell_border
                circos_rect(
                    cmid - cwid / 2.0,
                    np.repeat(nc_s - j - 1.0, nr_s),
                    cmid + cwid / 2.0,
                    np.repeat(nc_s - j, nr_s),
                    border=b_draw,
                    col=list(col_j),
                    lty=cell_lty,
                    lwd=cell_lwd,
                )

        if not is_na_like(bg_border):
            cx = get_cell_meta_data("cell.xlim")
            cy = get_cell_meta_data("cell.ylim")
            circos_rect(cx[0], cy[0], cx[1], cy[1], border=bg_border, lwd=bg_lwd, lty=bg_lty, col=None)

    circos_track_plot_region(ylim=[0.0, float(nc_mat)], bg_border=None, panel_fun=_heatmap_panel, **nkw)

    if dend_side == "inside":
        dend_list = [smeta[s]["row_dend"] for s in sec_names]
        max_h = max(float(d.height) for d in dend_list)
        circos_track_plot_region(
            ylim=[0.0, max_h],
            bg_border=None,
            track_height=dend_track_height,
            panel_fun=lambda x, y: circos_dendrogram(
                dend_list[get_cell_meta_data("sector.numeric.index") - 1],
                max_height=max_h,
                facing="outside",
                use_x_attr=True,
            ),
        )

    if rownames_side == "inside" and rownames is not None:
        def _rn_in_panel(x: Any, y: Any) -> None:
            si = get_cell_meta_data("sector.index")
            od = smeta[si]["row_order"]
            sub_idx = smeta[si]["subset"]
            cmid = smeta[si]["cell_middle"]
            circos_text(
                cmid[od],
                np.ones(len(od)),
                [rownames[sub_idx[i]] for i in od],
                cex=[rn_cexs[sub_idx[i]] for i in od],
                font=[rn_fonts[sub_idx[i]] for i in od],
                col=[rn_cols[sub_idx[i]] for i in od],
                facing="clockwise",
                nice_facing=True,
                adj=(1.0, 0.5),
            )

        circos_track_plot_region(ylim=[0.0, 1.0], bg_border=None, track_height=rn_track_h, panel_fun=_rn_in_panel)

    if show_sector_labels:
        from ._utils import convert_y

        def _sec_lbl_panel(x: Any, y: Any) -> None:
            xc = get_cell_meta_data("xcenter")
            cy = get_cell_meta_data("cell.ylim")
            si = get_cell_meta_data("sector.index")
            circos_text(xc, cy[1] + float(convert_y(2.0, "mm")), si, facing="bending.inside", cex=0.8, adj=(0.5, 0.0), nice_facing=True)

        circos_track_plot_region(track_index=1, panel_fun=_sec_lbl_panel, bg_border=None)


def _resolve_row_index(row: int | str, rownames: list[str] | None, n_rows: int) -> int:
    if isinstance(row, str):
        if rownames is None or row not in rownames:
            raise ValueError(f"Cannot find row name {row!r}.")
        return rownames.index(row)
    idx = int(row)
    if idx < 0 or idx >= n_rows:
        raise ValueError(f"Row index {idx} out of bounds [0, {n_rows}).")
    return idx


def circos_heatmap_link(row_from: int | str, row_to: int | str, **kwargs: Any) -> None:
    """Draw a link between two matrix rows in the current circular heatmap."""
    env = circos_par.get_value("__tempenv__")
    split = env.get("circos.heatmap.split")
    rownames = env.get("circos.heatmap.rownames")
    if split is None:
        raise RuntimeError("No circular heatmap has been initialized.")

    r1 = _resolve_row_index(row_from, rownames, len(split))
    r2 = _resolve_row_index(row_to, rownames, len(split))
    g1 = split[r1]
    g2 = split[r2]

    sub1 = get_cell_meta_data("subset", sector_index=g1)
    ord1 = get_cell_meta_data("row_order", sector_index=g1)
    x1 = [sub1[i] for i in ord1].index(r1) + 0.5

    sub2 = get_cell_meta_data("subset", sector_index=g2)
    ord2 = get_cell_meta_data("row_order", sector_index=g2)
    x2 = [sub2[i] for i in ord2].index(r2) + 0.5

    circos_link(g1, x1, g2, x2, **kwargs)


def circos_heatmap_get_x(row_ind: Sequence[int | str]) -> pd.DataFrame:
    """Return a DataFrame (sector, x, row_ind) with x-positions of heatmap rows."""
    env = circos_par.get_value("__tempenv__")
    split = env.get("circos.heatmap.split")
    rownames = env.get("circos.heatmap.rownames")
    if split is None:
        raise RuntimeError("No circular heatmap has been initialized.")

    rows = [_resolve_row_index(r, rownames, len(split)) for r in row_ind]
    by_sec: dict[str, list[int]] = {}
    for r in rows:
        by_sec.setdefault(split[r], []).append(r)

    out_sec: list[str] = []
    out_x: list[float] = []
    out_r: list[int] = []
    for sec, r_list in by_sec.items():
        sub = get_cell_meta_data("subset", sector_index=sec)
        ord_s = get_cell_meta_data("row_order", sector_index=sec)
        ordered_orig = [sub[i] for i in ord_s]
        for r in r_list:
            pos = ordered_orig.index(r)
            out_sec.append(sec)
            out_x.append(pos + 0.5)
            out_r.append(r)

    return pd.DataFrame({"sector": out_sec, "x": out_x, "row_ind": out_r})


class _HeatmapGetProxy:
    def x(self, row_ind: Sequence[int | str]) -> pd.DataFrame:
        return circos_heatmap_get_x(row_ind)


class _CircosHeatmapAPI:
    """Callable object supporting both circos.heatmap(...) and circos.heatmap.initialize/link/get.x."""

    get = _HeatmapGetProxy()

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return circos_heatmap_fn(*args, **kwargs)

    def initialize(self, *args: Any, **kwargs: Any) -> None:
        return circos_heatmap_initialize(*args, **kwargs)

    def link(self, *args: Any, **kwargs: Any) -> None:
        return circos_heatmap_link(*args, **kwargs)


circos_heatmap = _CircosHeatmapAPI()
