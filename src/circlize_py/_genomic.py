"""Genomic circular plots: ideograms, tracks, points/lines/rect/text/links, density, rainfall, heatmaps, and labels."""

from __future__ import annotations

import inspect
import math
import warnings
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from ._canvas import strheight_canvas, strwidth_canvas
from ._colors import color_ramp2
from ._genomic_utils import (
    _UNSET,
    cytoband_col,
    read_chrom_info,
    read_cytoband,
    usable_chromosomes,
    validate_region,
)
from ._link import circos_link
from ._low_level import (
    circos_axis,
    circos_lines,
    circos_points,
    circos_rect,
    circos_segments,
    circos_text,
    default_major_by,
)
from ._plot import circos_initialize, circos_track_plot_region
from ._regions import order_region, overlap_region, reduce_region, sort_region
from ._state import (
    _STATE,
    circos_par,
    get_all_sector_index,
    get_cell_meta_data,
    get_current_chromosome,
    get_current_sector_index,
    get_current_track_index,
    get_most_inside_radius,
    get_sector_data,
    is_circos_initialized,
    normalize_kwargs,
)
from ._utils import (
    as_degree,
    circlize,
    cm_h,
    convert_length,
    convert_y,
    recycle_to_length,
    reverse_circlize,
    validate_data_frame,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_data_frame_list(data: Any) -> bool:
    return (
        isinstance(data, (list, tuple))
        and not isinstance(data, pd.DataFrame)
        and len(data) > 0
        and all(isinstance(d, pd.DataFrame) for d in data)
    )


def _normalize_to_data_frame(data: Any, sort: bool = False) -> pd.DataFrame | list[pd.DataFrame]:
    """Normalize BED input into DataFrame or list of DataFrames filtered to initialized chromosomes."""
    if isinstance(data, (list, tuple)) and not isinstance(data, pd.DataFrame):
        if len(data) > 0 and isinstance(data[0], (pd.DataFrame, dict, list, tuple, np.ndarray)):
            # Check whether this is a list of BED data frames or a single BED passed as list
            if isinstance(data[0], pd.DataFrame) or (
                isinstance(data[0], (list, tuple)) and len(data[0]) > 0 and isinstance(data[0][0], pd.DataFrame)
            ):
                pass
            if all(isinstance(item, pd.DataFrame) for item in data):
                out = [_normalize_to_data_frame(item, sort=sort) for item in data]
                return out  # type: ignore[return-value]

    df = validate_data_frame(data)
    if df.shape[1] < 2:
        raise ValueError("BED data must have at least 2 or 3 columns.")
    if df.shape[1] == 2:
        df = pd.DataFrame({"chr": df.iloc[:, 0].astype(str), "start": df.iloc[:, 1], "end": df.iloc[:, 1]})
        df["start"] = pd.to_numeric(df["start"], errors="raise").astype(float)
        df["end"] = pd.to_numeric(df["end"], errors="raise").astype(float)
    else:
        col_names = list(df.columns)
        parts = [
            df.iloc[:, 0].astype(str).rename(col_names[0]),
            pd.to_numeric(df.iloc[:, 1], errors="raise").astype(float).rename(col_names[1]),
            pd.to_numeric(df.iloc[:, 2], errors="raise").astype(float).rename(col_names[2]),
        ] + [df.iloc[:, k] for k in range(3, df.shape[1])]
        df = pd.concat(parts, axis=1)

    if is_circos_initialized():
        sectors = get_all_sector_index()
        sec_set = set(sectors)
        df = df[df.iloc[:, 0].isin(sec_set)].copy()
        if sort and len(df) > 0:
            sec_rank = {s: i for i, s in enumerate(sectors)}
            ranks = df.iloc[:, 0].map(sec_rank).to_numpy()
            od = np.lexsort((df.iloc[:, 2].to_numpy(), df.iloc[:, 1].to_numpy(), ranks))
            df = df.iloc[od].reset_index(drop=True)
    elif sort and len(df) > 0:
        od = np.lexsort((df.iloc[:, 2].to_numpy(), df.iloc[:, 1].to_numpy()))
        df = df.iloc[od].reset_index(drop=True)

    return df.reset_index(drop=True)


def _resolve_col_indices(
    df: pd.DataFrame,
    col_spec: Any = None,
    min_pos: int = 0,
    numeric_only: bool = False,
) -> list[int]:
    """Resolve column selector supporting column names, None (cols >= min_pos), 0-based ints, and 1-based fallback."""
    ncols = df.shape[1]
    if col_spec is None:
        cols = list(range(min_pos, ncols))
        if numeric_only:
            cols = [c for c in cols if pd.api.types.is_numeric_dtype(df.iloc[:, c])]
        return cols

    if isinstance(col_spec, str):
        if col_spec not in df.columns:
            raise KeyError(f"Column {col_spec!r} not found in DataFrame columns {list(df.columns)}.")
        return [int(df.columns.get_loc(col_spec))]

    if isinstance(col_spec, (int, np.integer)):
        idx = int(col_spec)
        if idx == ncols and ncols >= 1:
            idx = idx - 1
        if idx < 0 or idx >= ncols:
            raise IndexError(f"Column index {col_spec} out of bounds for {ncols} columns.")
        return [idx]

    items = list(col_spec)
    if not items:
        return []
    if all(isinstance(x, (bool, np.bool_)) for x in items) and len(items) == ncols:
        return [i for i, b in enumerate(items) if b]
    if all(isinstance(x, str) for x in items):
        out: list[int] = []
        for s in items:
            if s not in df.columns:
                raise KeyError(f"Column {s!r} not found in DataFrame columns {list(df.columns)}.")
            out.append(int(df.columns.get_loc(s)))
        return out

    ints = [int(x) for x in items]
    if any(x == ncols for x in ints) and all(1 <= x <= ncols for x in ints):
        ints = [x - 1 for x in ints]
    for idx in ints:
        if idx < 0 or idx >= ncols:
            raise IndexError(f"Column index {idx} out of bounds for {ncols} columns.")
    return ints


def _normalize_graphical_param(value: Any, nc: int, nr: int, name: str = "param") -> Any:
    """Broadcast graphical attribute across multiple columns (nc > 1) or rows (nc == 1)."""
    if value is None or isinstance(value, (str, bool, int, float, np.number)):
        if nc <= 1:
            return value
        return [value] * nc

    vals = list(value)
    if len(vals) == 1:
        if nc <= 1:
            return vals[0]
        return [vals[0]] * nc

    if nc > 1:
        if len(vals) == nc:
            return vals
        return recycle_to_length(vals, nc)

    if len(vals) == nr:
        return vals
    return recycle_to_length(vals, nr)


def _call_genomic_panel_fun(
    panel_fun: Callable[..., Any],
    region: pd.DataFrame,
    value: pd.DataFrame,
    param: dict[str, Any],
) -> Any:
    prev_param = _STATE.current_genomic_param
    _STATE.current_genomic_param = param
    try:
        sig = inspect.signature(panel_fun)
        accepts_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if accepts_varkw or "_param" in sig.parameters:
            return panel_fun(region, value, _param=param)
        return panel_fun(region, value)
    finally:
        _STATE.current_genomic_param = prev_param


def _strip_chr_col(region: Any) -> pd.DataFrame:
    df = validate_data_frame(region)
    if df.shape[1] > 2 and not pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
        return df.iloc[:, 1:3].copy().reset_index(drop=True)
    return df.iloc[:, :2].copy().reset_index(drop=True)


def _coerce_value_df(value: Any, nr: int) -> tuple[pd.DataFrame, list[int] | None]:
    if value is None:
        return pd.DataFrame(index=range(nr)), None
    if isinstance(value, pd.DataFrame):
        return value.copy().reset_index(drop=True), None
    if isinstance(value, (str, int, float, bool, np.number)):
        return pd.DataFrame({"value": [value] * nr}), [0]
    arr = np.asarray(value)
    if arr.ndim == 1:
        if len(arr) == 1:
            return pd.DataFrame({"value": np.repeat(arr[0], nr)}), [0]
        if len(arr) == nr:
            return pd.DataFrame({"value": arr}), [0]
        raise ValueError(f"1D `value` length ({len(arr)}) does not match nrow(region) ({nr}).")
    if arr.ndim == 2:
        return pd.DataFrame(arr), None
    raise ValueError("`value` must be convertible to a DataFrame.")


def get_i(*args: Any, **kwargs: Any) -> int:
    """Return the 1-based index of the current genomic data layer inside `circos_genomic_track`."""
    if "_param" in kwargs and isinstance(kwargs["_param"], dict) and "i" in kwargs["_param"]:
        return int(kwargs["_param"]["i"])
    for a in args:
        if isinstance(a, dict) and "i" in a:
            return int(a["i"])
    if _STATE.current_genomic_param and "i" in _STATE.current_genomic_param:
        return int(_STATE.current_genomic_param["i"])
    return 1


getI = get_i


# ---------------------------------------------------------------------------
# Initialization & Ideogram
# ---------------------------------------------------------------------------

def circos_initialize_with_ideogram(
    cytoband: Any = None,
    species: str | None = None,
    sort_chr: bool = True,
    draw_chr_prefix: bool = False,
    chromosome_index: Any = _UNSET,
    major_by: float | None = None,
    plot_type: str | Sequence[str] | None = ("ideogram", "axis", "labels"),
    track_height: float | None = None,
    ideogram_height: float | None = None,
    ax: Any = None,
    **kwargs: Any,
) -> None:
    """Initialize a circular genomic layout from cytoband or chromInfo data."""
    nkw = normalize_kwargs(kwargs)
    if "sector_names" in nkw:
        raise ValueError("Use `chromosome_index` instead of `sector_names` in `circos_initialize_with_ideogram`.")

    sort_chr = bool(nkw.pop("sort_chr", sort_chr))
    draw_chr_prefix = bool(nkw.pop("draw_chr_prefix", draw_chr_prefix))
    chromosome_index = nkw.pop("chromosome_index", chromosome_index)
    major_by = nkw.pop("major_by", major_by)
    plot_type = nkw.pop("plot_type", plot_type)
    track_height = nkw.pop("track_height", track_height)
    ideogram_height = nkw.pop("ideogram_height", ideogram_height)
    ax = nkw.pop("ax", ax)

    if plot_type is None:
        pt_list: list[str] = []
    elif isinstance(plot_type, str):
        pt_list = [plot_type]
    else:
        pt_list = [str(p) for p in plot_type]

    cb_data = None
    try:
        cb_data = read_cytoband(
            cytoband=cytoband,
            species=species,
            chromosome_index=chromosome_index,
            sort_chr=sort_chr,
        )
    except Exception as exc:
        if species is not None:
            ci_arg = chromosome_index
            cb_data = read_chrom_info(species=species, chromosome_index=ci_arg, sort_chr=sort_chr)
            pt_list = [p for p in pt_list if p != "ideogram"]
            if ci_arg is _UNSET or ci_arg is None:
                full_ci = read_chrom_info(species=species)
                sorted_lens = sorted(full_ci["chr.len"].items(), key=lambda kv: kv[1], reverse=True)
                cutoff = None
                for k in range(len(sorted_lens) - 1):
                    if sorted_lens[k + 1][1] > 0 and (sorted_lens[k][1] / sorted_lens[k + 1][1]) > 5.0:
                        cutoff = sorted_lens[k][1]
                        break
                if cutoff is not None:
                    keep_chrs = [ch for ch in full_ci["chromosome"] if full_ci["chr.len"][ch] >= cutoff]
                    cb_data = read_chrom_info(species=species, chromosome_index=keep_chrs, sort_chr=sort_chr)
        else:
            raise exc

    df = cb_data["df"].copy()
    chroms = list(cb_data["chromosome"])
    if chromosome_index is _UNSET or chromosome_index is None:
        effective_index = chroms
    else:
        req = [chromosome_index] if isinstance(chromosome_index, str) else [str(c) for c in chromosome_index]
        present = set(chroms)
        effective_index = [c for c in req if c in present]

    # Order df rows by effective_index
    rank = {c: i for i, c in enumerate(effective_index)}
    df = df[df.iloc[:, 0].isin(rank)].copy()
    df["_rank"] = df.iloc[:, 0].map(rank)
    df = df.sort_values("_rank", kind="mergesort").drop(columns=["_rank"]).reset_index(drop=True)

    sn = list(effective_index) if draw_chr_prefix else [c.replace("chr", "") for c in effective_index]

    o_pad = list(circos_par.get_value("cell.padding"))
    circos_par.set_value("cell_padding", [o_pad[0], 0.0, o_pad[2], 0.0], _suppress_warn=True)

    # Note: circos_genomic_initialize ignores "ideogram" in plotType until we add ideogram track below
    circos_genomic_initialize(
        df,
        sector_names=sn,
        major_by=major_by,
        plot_type=pt_list,
        track_height=track_height,
        ax=ax,
        **nkw,
    )

    if "ideogram" in pt_list:
        h_ideo = float(convert_length(2.0, "mm")) if ideogram_height is None else float(ideogram_height)
        circos_genomic_ideogram(cytoband=df, track_height=h_ideo)


def circos_genomic_initialize(
    data: Any,
    sector_names: Sequence[str] | None = None,
    major_by: float | None = None,
    plot_type: str | Sequence[str] | None = ("axis", "labels"),
    tick_labels_start_from_zero: bool = True,
    axis_labels_cex: float = 0.4,
    labels_cex: float = 0.8,
    track_height: float | None = None,
    ax: Any = None,
    **kwargs: Any,
) -> None:
    """Initialize a circular layout from a BED DataFrame defining sector ranges."""
    nkw = normalize_kwargs(kwargs)
    sector_names = nkw.pop("sector_names", sector_names)
    major_by = nkw.pop("major_by", major_by)
    plot_type = nkw.pop("plot_type", plot_type)
    tick_labels_start_from_zero = bool(nkw.pop("tick_labels_start_from_zero", tick_labels_start_from_zero))
    axis_labels_cex = float(nkw.pop("axis_labels_cex", axis_labels_cex))
    labels_cex = float(nkw.pop("labels_cex", labels_cex))
    track_height = nkw.pop("track_height", track_height)
    ax = nkw.pop("ax", ax)

    df = validate_data_frame(data)
    validate_region(df, check_chr=False)
    if df.shape[1] < 3:
        raise ValueError("`data` in `circos_genomic_initialize` must have at least 3 columns (chr, start, end).")

    chrs = df.iloc[:, 0].astype(str)
    fa = list(dict.fromkeys(chrs.tolist()))
    if sector_names is not None:
        sn_list = [str(s) for s in sector_names]
        if len(sn_list) != len(fa):
            raise ValueError("Length of `sector_names` and number of unique chromosomes differ.")
    else:
        sn_list = list(fa)
    sector_name_map = dict(zip(fa, sn_list))

    grouped_min = df.assign(_chr=chrs, _s=pd.to_numeric(df.iloc[:, 1])).groupby("_chr", sort=False)["_s"].min()
    grouped_max = df.assign(_chr=chrs, _e=pd.to_numeric(df.iloc[:, 2])).groupby("_chr", sort=False)["_e"].max()
    xlim = np.column_stack([[float(grouped_min[c]) for c in fa], [float(grouped_max[c]) for c in fa]])

    op = list(circos_par.get_value("cell.padding"))
    ow = bool(circos_par.get_value("points.overflow.warning"))
    circos_par.set_value("cell_padding", [0.0, 0.0, 0.0, 0.0], _suppress_warn=True)
    circos_par.set_value("points_overflow_warning", False, _suppress_warn=True)

    circos_initialize(sectors=fa, xlim=xlim, ax=ax, **nkw)
    if circos_par.get_value("ring"):
        op = [op[0], 0.0, op[2], 0.0]
        ow = False

    if plot_type is None:
        pt_list: list[str] = []
    elif isinstance(plot_type, str):
        pt_list = [plot_type]
    else:
        pt_list = [str(p) for p in plot_type]

    if track_height is None:
        if ("axis" in pt_list) and ("labels" in pt_list):
            track_height = (
                float(convert_length(1.5, "mm"))
                + float(strheight_canvas("0", cex=axis_labels_cex))
                + float(convert_length(0.5, "mm"))
                + float(strheight_canvas("chr", cex=labels_cex))
            )
        elif "labels" in pt_list:
            track_height = float(strheight_canvas("chr", cex=labels_cex))
        elif "axis" in pt_list:
            track_height = float(convert_length(1.5, "mm")) + float(strheight_canvas("0", cex=axis_labels_cex))
        else:
            track_height = float(convert_length(3.0, "mm"))

    if ("axis" in pt_list) or ("labels" in pt_list):
        def _init_panel(region: pd.DataFrame, value: pd.DataFrame, **pkw: Any) -> None:
            si = get_current_sector_index()
            cx = get_cell_meta_data("xlim", sector_index=si)
            if ("axis" in pt_list) and ("labels" in pt_list):
                circos_genomic_axis(
                    h="bottom",
                    major_by=major_by,
                    tick_labels_start_from_zero=tick_labels_start_from_zero,
                    labels_cex=axis_labels_cex,
                )
                y_pos = (
                    float(convert_y(1.5, "mm"))
                    + float(convert_y(strheight_canvas("chr", cex=axis_labels_cex), unit="canvas"))
                    + float(convert_y(0.5, "mm"))
                )
                circos_text(
                    0.5 * (float(cx[0]) + float(cx[1])),
                    y_pos,
                    sector_name_map[si],
                    cex=labels_cex,
                    adj=[0.5, 0.0],
                    nice_facing=True,
                )
            elif "labels" in pt_list:
                circos_text(
                    0.5 * (float(cx[0]) + float(cx[1])),
                    0.0,
                    sector_name_map[si],
                    cex=labels_cex,
                    adj=[0.5, 0.0],
                    nice_facing=True,
                )
            elif "axis" in pt_list:
                circos_genomic_axis(
                    h="bottom",
                    major_by=major_by,
                    tick_labels_start_from_zero=tick_labels_start_from_zero,
                    labels_cex=axis_labels_cex,
                )

        circos_genomic_track_plot_region(
            df,
            ylim=[0.0, 1.0],
            bg_border=None,
            track_height=track_height,
            panel_fun=_init_panel,
        )

    circos_par.set_value("cell_padding", op, _suppress_warn=True)
    circos_par.set_value("points_overflow_warning", ow, _suppress_warn=True)


def circos_initialize_circular_genome(
    name: str,
    genome_size: float,
    plot_type: str | Sequence[str] | None = "axis",
    **kwargs: Any,
) -> None:
    """Initialize a single-sector ring circular genome layout."""
    nkw = normalize_kwargs(kwargs)
    plot_type = nkw.pop("plot_type", plot_type)
    nkw.pop("ring", None)
    df = pd.DataFrame({"chr": [str(name)], "start": [0.0], "end": [float(genome_size)]})
    circos_genomic_initialize(df, ring=True, plot_type=plot_type, **nkw)


def circos_genomic_ideogram(
    cytoband: Any = None,
    species: str | None = None,
    track_height: float | None = None,
    track_margin: Sequence[float] | None = None,
    **kwargs: Any,
) -> None:
    """Add an ideogram track displaying chromosome bands and centromeres."""
    nkw = normalize_kwargs(kwargs)
    track_height = nkw.pop("track_height", track_height)
    track_margin = nkw.pop("track_margin", track_margin)
    if track_height is None:
        track_height = float(convert_length(2.0, "mm"))
    if track_margin is None:
        track_margin = circos_par.get_value("track.margin")

    chr_index = get_all_sector_index()
    cb = read_cytoband(cytoband=cytoband, species=species, chromosome_index=chr_index)
    df = cb["df"]
    cols_all = cytoband_col(df.iloc[:, 4])
    if all(c == "#FFFFFF" for c in cols_all):
        warnings.warn("Cannot map colors to cytobands. The ideogram will not be drawn.", UserWarning, stacklevel=2)
        return

    def _ideo_panel(region: pd.DataFrame, value: pd.DataFrame, **pkw: Any) -> None:
        stains = value.iloc[:, 1] if value.shape[1] >= 2 else value.iloc[:, 0]
        cols = cytoband_col(stains)
        circos_genomic_rect(region, value, ybottom=0.0, ytop=1.0, col=cols, border=None)
        cx = get_cell_meta_data("xlim")
        circos_rect(float(cx[0]), 0.0, float(cx[1]), 1.0, col=None, border="black")

    circos_genomic_track_plot_region(
        df,
        ylim=[0.0, 1.0],
        bg_border=None,
        track_height=float(track_height),
        cell_padding=[0.0, 0.0, 0.0, 0.0],
        track_margin=track_margin,
        panel_fun=_ideo_panel,
    )


def circos_genomic_axis(
    h: str = "top",
    major_at: Sequence[float] | None = None,
    labels: Sequence[str] | None = None,
    major_by: float | None = None,
    tick_labels_start_from_zero: bool = True,
    labels_cex: float = 0.4,
    sector_index: str | None = None,
    track_index: int | None = None,
    **kwargs: Any,
) -> None:
    """Draw formatted genomic coordinate axes (bp/kb/Mb)."""
    nkw = normalize_kwargs(kwargs)
    major_at = nkw.pop("major_at", major_at)
    major_by = nkw.pop("major_by", major_by)
    tick_labels_start_from_zero = bool(nkw.pop("tick_labels_start_from_zero", tick_labels_start_from_zero))
    labels_cex = float(nkw.pop("labels_cex", labels_cex))
    sector_index = nkw.pop("sector_index", sector_index)
    track_index = nkw.pop("track_index", track_index)

    if h not in ("top", "bottom"):
        raise ValueError("`h` must be 'top' or 'bottom'.")
    if sector_index is None:
        sector_index = get_current_sector_index()
    if track_index is None:
        track_index = get_current_track_index()

    xlim = get_cell_meta_data("xlim", sector_index=sector_index, track_index=track_index)
    x0, x1 = float(xlim[0]), float(xlim[1])

    if major_at is not None and labels is not None and len(major_at) != len(labels):
        raise ValueError("Length of `major_at` and `labels` must match.")
    if major_at is None and labels is not None:
        raise ValueError("If `labels` is set, `major_at` must also be set.")

    if major_by is None:
        major_by = float(default_major_by(sector_index=sector_index, track_index=track_index))
    else:
        major_by = float(major_by)

    def _fmt_val(v: float) -> str:
        return f"{int(round(v))}" if abs(v - round(v)) < 1e-9 else f"{v:g}"

    if tick_labels_start_from_zero:
        offset = x0
        if major_at is None:
            pts = np.arange(x0, x1 + 1e-9 * max(1.0, major_by), major_by)
            if len(pts) == 0:
                pts = np.array([x0])
            pts = np.append(pts, pts[-1] + major_by)
        else:
            pts = np.asarray(major_at, dtype=float)

        if labels is None:
            if major_by >= 1e6:
                tick_labels = [f"{_fmt_val((v - offset) / 1e6)}Mb" for v in pts]
            elif major_by >= 1e3:
                tick_labels = [f"{_fmt_val((v - offset) / 1e3)}kb" for v in pts]
            else:
                tick_labels = [f"{_fmt_val(v - offset)}bp" for v in pts]
        else:
            tick_labels = [str(l) for l in labels]
    else:
        if major_at is None:
            start_tick = math.floor(x0 / major_by) * major_by
            pts = np.arange(start_tick, x1 + 1e-9 * max(1.0, major_by), major_by)
            if len(pts) == 0:
                pts = np.array([start_tick])
            pts = np.append(pts, pts[-1] + major_by)
        else:
            pts = np.asarray(major_at, dtype=float)

        if labels is None:
            if major_by >= 1e6:
                tick_labels = [f"{_fmt_val(v / 1e6)}MB" for v in pts]
            elif major_by >= 1e3:
                tick_labels = [f"{_fmt_val(v / 1e3)}KB" for v in pts]
            else:
                tick_labels = [f"{_fmt_val(v)}bp" for v in pts]
        else:
            tick_labels = [str(l) for l in labels]

    circos_axis(
        h=h,
        major_at=pts,
        labels=tick_labels,
        labels_cex=labels_cex,
        sector_index=sector_index,
        track_index=track_index,
        **nkw,
    )


# ---------------------------------------------------------------------------
# High-level genomic track & low-level genomic wrappers
# ---------------------------------------------------------------------------

def circos_genomic_track_plot_region(
    data: Any = None,
    ylim: Sequence[float] | None = None,
    stack: bool = False,
    numeric_column: Any = None,
    jitter: float = 0.0,
    panel_fun: Callable[..., Any] | None = None,
    track_height: float | None = None,
    track_margin: Sequence[float] | None = None,
    cell_padding: Sequence[float] | None = None,
    bg_col: Any = None,
    bg_border: Any = "black",
    bg_lty: Any = 1,
    bg_lwd: Any = 1.0,
    **kwargs: Any,
) -> None:
    """Create a genomic track and invoke `panel_fun(region, value, ...)` per chromosome cell."""
    nkw = normalize_kwargs(kwargs)
    ylim = nkw.pop("ylim", ylim)
    stack = bool(nkw.pop("stack", stack))
    numeric_column = nkw.pop("numeric_column", numeric_column)
    jitter = float(nkw.pop("jitter", jitter))
    panel_fun = nkw.pop("panel_fun", panel_fun)
    track_height = nkw.pop("track_height", track_height)
    track_margin = nkw.pop("track_margin", track_margin)
    cell_padding = nkw.pop("cell_padding", cell_padding)
    bg_col = nkw.pop("bg_col", bg_col)
    bg_border = nkw.pop("bg_border", bg_border)
    bg_lty = nkw.pop("bg_lty", bg_lty)
    bg_lwd = nkw.pop("bg_lwd", bg_lwd)

    if data is None:
        secs = get_all_sector_index()
        data = pd.DataFrame({"chr": secs, "start": [0.0] * len(secs), "end": [0.0] * len(secs)})

    if panel_fun is None:
        panel_fun = lambda region, value, **kw: None

    norm = _normalize_to_data_frame(data, sort=False)
    is_list = isinstance(norm, list)
    ring_mode = bool(circos_par.get_value("ring"))

    def _adjust_ring(d: pd.DataFrame) -> pd.DataFrame:
        if ring_mode and len(d) > 0:
            sdata = get_sector_data()
            span = float(sdata["max.value"] - sdata["min.value"])
            mask = d.iloc[:, 1] > d.iloc[:, 2]
            if mask.any():
                d = d.copy()
                d.loc[mask, d.columns[1]] = d.loc[mask, d.columns[1]] - span
        validate_region(d, check_chr=False)
        return d

    if is_list:
        data_list = [_adjust_ring(d) for d in norm]
        n_df = len(data_list)
        if numeric_column is not None:
            if isinstance(numeric_column, (list, tuple, np.ndarray)) and len(numeric_column) == n_df:
                num_specs = list(numeric_column)
            else:
                num_specs = [numeric_column] * n_df
            resolved_num_cols: list[list[int]] = []
            for d, spec in zip(data_list, num_specs):
                cols = _resolve_col_indices(d, spec, min_pos=3)
                val_cols = [c - 3 for c in cols if c >= 3]
                if len(cols) != len(val_cols):
                    raise ValueError("`numeric_column` must refer to BED value columns (>= index 3 / col 4 in R).")
                for c in val_cols:
                    if not pd.api.types.is_numeric_dtype(d.iloc[:, c + 3]):
                        raise ValueError("Selected `numeric_column` is not numeric.")
                resolved_num_cols.append(val_cols)
        else:
            resolved_num_cols = []
            for d in data_list:
                ncs = [i for i, c in enumerate(d.columns[3:]) if pd.api.types.is_numeric_dtype(d[c])]
                resolved_num_cols.append([ncs[0]] if ncs else [])
    else:
        df_single = _adjust_ring(norm)
        if numeric_column is not None:
            cols = _resolve_col_indices(df_single, numeric_column, min_pos=3)
            val_cols = [c - 3 for c in cols if c >= 3]
            if len(cols) != len(val_cols):
                raise ValueError("`numeric_column` must refer to BED value columns (>= index 3 / col 4 in R).")
            for c in val_cols:
                if not pd.api.types.is_numeric_dtype(df_single.iloc[:, c + 3]):
                    raise ValueError("Selected `numeric_column` is not numeric.")
            single_num_cols = val_cols
        else:
            single_num_cols = [i for i, c in enumerate(df_single.columns[3:]) if pd.api.types.is_numeric_dtype(df_single[c])]

    if stack:
        if is_list:
            n_stack = len(data_list)
            eff_ylim = [0.5, n_stack + 0.5] if ylim is None else [float(ylim[0]), float(ylim[1])]

            def _track_panel(x: Any = None, y: Any = None) -> None:
                ch = get_current_chromosome()
                for idx_layer, (d, ncs) in enumerate(zip(data_list, resolved_num_cols), start=1):
                    sub = d[d.iloc[:, 0] == ch]
                    if len(sub) == 0:
                        continue
                    reg = sub.iloc[:, 1:3].copy().reset_index(drop=True)
                    val = sub.iloc[:, 3:].copy().reset_index(drop=True)
                    param: dict[str, Any] = {"i": idx_layer, "stack": True, "jitter": jitter}
                    if ncs:
                        param["numeric_column"] = list(ncs)
                    _call_genomic_panel_fun(panel_fun, reg, val, param)
        else:
            n_num = len(single_num_cols)
            all_val_idx = list(range(max(0, df_single.shape[1] - 3)))
            non_num = [j for j in all_val_idx if j not in single_num_cols]
            if n_num == 0:
                eff_ylim = [0.5, 1.5] if ylim is None else [float(ylim[0]), float(ylim[1])]

                def _track_panel(x: Any = None, y: Any = None) -> None:
                    ch = get_current_chromosome()
                    sub = df_single[df_single.iloc[:, 0] == ch]
                    if len(sub) == 0:
                        return
                    reg = sub.iloc[:, 1:3].copy().reset_index(drop=True)
                    val_all = sub.iloc[:, 3:].copy().reset_index(drop=True)
                    val = val_all.iloc[:, non_num].copy() if non_num else pd.DataFrame(index=range(len(reg)))
                    param = {"i": 1, "stack": True, "jitter": jitter}
                    _call_genomic_panel_fun(panel_fun, reg, val, param)
            else:
                eff_ylim = [0.5, n_num + 0.5] if ylim is None else [float(ylim[0]), float(ylim[1])]

                def _track_panel(x: Any = None, y: Any = None) -> None:
                    ch = get_current_chromosome()
                    sub = df_single[df_single.iloc[:, 0] == ch]
                    if len(sub) == 0:
                        return
                    reg = sub.iloc[:, 1:3].copy().reset_index(drop=True)
                    val_all = sub.iloc[:, 3:].copy().reset_index(drop=True)
                    for idx_layer, nc_idx in enumerate(single_num_cols, start=1):
                        sel = [nc_idx] + non_num
                        val = val_all.iloc[:, sel].copy()
                        param = {"i": idx_layer, "stack": True, "numeric_column": [0], "jitter": jitter}
                        _call_genomic_panel_fun(panel_fun, reg, val, param)
    else:
        if ylim is None:
            vals_all: list[np.ndarray] = []
            if is_list:
                for d, ncs in zip(data_list, resolved_num_cols):
                    if not ncs:
                        raise ValueError("No numeric column found to determine `ylim`; please pass `ylim` explicitly.")
                    for c in ncs:
                        arr = pd.to_numeric(d.iloc[:, c + 3], errors="coerce").to_numpy(dtype=float)
                        arr = arr[np.isfinite(arr)]
                        if len(arr):
                            vals_all.append(arr)
            else:
                if not single_num_cols:
                    raise ValueError("No numeric column found to determine `ylim`; please pass `ylim` explicitly.")
                for c in single_num_cols:
                    arr = pd.to_numeric(df_single.iloc[:, c + 3], errors="coerce").to_numpy(dtype=float)
                    arr = arr[np.isfinite(arr)]
                    if len(arr):
                        vals_all.append(arr)

            if not vals_all:
                raise ValueError("Cannot infer `ylim` from empty numeric data.")
            cat = np.concatenate(vals_all)
            ymin, ymax = float(np.min(cat)), float(np.max(cat))
            if ymin == ymax:
                raise ValueError("All data values are identical; please set `ylim` explicitly.")
            eff_ylim = [ymin, ymax]
        else:
            eff_ylim = [float(ylim[0]), float(ylim[1])]

        if is_list:
            def _track_panel(x: Any = None, y: Any = None) -> None:
                ch = get_current_chromosome()
                for idx_layer, (d, ncs) in enumerate(zip(data_list, resolved_num_cols), start=1):
                    sub = d[d.iloc[:, 0] == ch]
                    if len(sub) == 0:
                        continue
                    reg = sub.iloc[:, 1:3].copy().reset_index(drop=True)
                    val = sub.iloc[:, 3:].copy().reset_index(drop=True)
                    param: dict[str, Any] = {"i": idx_layer}
                    if ncs:
                        param["numeric_column"] = list(ncs)
                    _call_genomic_panel_fun(panel_fun, reg, val, param)
        else:
            def _track_panel(x: Any = None, y: Any = None) -> None:
                ch = get_current_chromosome()
                sub = df_single[df_single.iloc[:, 0] == ch]
                if len(sub) == 0:
                    return
                reg = sub.iloc[:, 1:3].copy().reset_index(drop=True)
                val = sub.iloc[:, 3:].copy().reset_index(drop=True)
                param: dict[str, Any] = {"i": 1}
                if single_num_cols:
                    param["numeric_column"] = list(single_num_cols)
                _call_genomic_panel_fun(panel_fun, reg, val, param)

    circos_track_plot_region(
        ylim=eff_ylim,
        track_height=track_height,
        track_margin=track_margin,
        cell_padding=cell_padding,
        bg_col=bg_col,
        bg_border=bg_border,
        bg_lty=bg_lty,
        bg_lwd=bg_lwd,
        panel_fun=_track_panel,
        **nkw,
    )


circos_genomic_track = circos_genomic_track_plot_region


def circos_genomic_points(
    region: Any,
    value: Any = None,
    numeric_column: Any = None,
    sector_index: str | None = None,
    track_index: int | None = None,
    pos_transform: Callable[..., pd.DataFrame] | None = None,
    pch: Any = 1,
    col: Any = "black",
    cex: Any = 1.0,
    bg: Any = None,
    **kwargs: Any,
) -> None:
    """Plot genomic points at region midpoints."""
    nkw = normalize_kwargs(kwargs)
    numeric_column = nkw.pop("numeric_column", numeric_column)
    sector_index = nkw.pop("sector_index", sector_index)
    track_index = nkw.pop("track_index", track_index)
    pos_transform = nkw.pop("pos_transform", pos_transform)
    if sector_index is None:
        sector_index = get_current_sector_index()
    if track_index is None:
        track_index = get_current_track_index()

    reg = _strip_chr_col(region)
    nr = len(reg)
    if nr == 0:
        return

    param = nkw.pop("_param", None) or _STATE.current_genomic_param or {}
    val_df, forced_nc = _coerce_value_df(value, nr)

    if param.get("stack") and numeric_column is None:
        j = float(param.get("jitter") or 0.0)
        y_vals = np.full(nr, float(param.get("i", 1)), dtype=float)
        if j != 0.0:
            y_vals = y_vals + (np.random.default_rng().random(nr) - 0.5) * abs(j)
        val_df = pd.DataFrame({"hline": y_vals})
        num_cols = [0]
    else:
        if forced_nc is not None and numeric_column is None:
            num_cols = forced_nc
        elif numeric_column is None and param.get("numeric_column"):
            num_cols = [int(c) for c in param["numeric_column"] if 0 <= int(c) < val_df.shape[1]]
        elif numeric_column is not None:
            num_cols = _resolve_col_indices(val_df, numeric_column, min_pos=0)
        else:
            num_cols = [i for i, c in enumerate(val_df.columns) if pd.api.types.is_numeric_dtype(val_df[c])]
            if not num_cols:
                raise ValueError("Cannot find numeric column in `value`.")

    if pos_transform is not None:
        reg = _strip_chr_col(pos_transform(reg))

    xs = 0.5 * (pd.to_numeric(reg.iloc[:, 0]).to_numpy(dtype=float) + pd.to_numeric(reg.iloc[:, 1]).to_numpy(dtype=float))
    nc = len(num_cols)
    pch_n = _normalize_graphical_param(pch, nc, nr, "pch")
    col_n = _normalize_graphical_param(col, nc, nr, "col")
    cex_n = _normalize_graphical_param(cex, nc, nr, "cex")
    bg_n = _normalize_graphical_param(bg, nc, nr, "bg")

    if nc == 1:
        ys = pd.to_numeric(val_df.iloc[:, num_cols[0]], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(xs) & np.isfinite(ys)
        if np.any(mask):
            c_arg = [col_n[k] for k in np.where(mask)[0]] if isinstance(col_n, list) else col_n
            p_arg = [pch_n[k] for k in np.where(mask)[0]] if isinstance(pch_n, list) else pch_n
            s_arg = [cex_n[k] for k in np.where(mask)[0]] if isinstance(cex_n, list) else cex_n
            b_arg = [bg_n[k] for k in np.where(mask)[0]] if isinstance(bg_n, list) else bg_n
            circos_points(
                xs[mask],
                ys[mask],
                pch=p_arg,
                col=c_arg,
                cex=s_arg,
                bg=b_arg,
                sector_index=sector_index,
                track_index=track_index,
            )
    else:
        for k, col_idx in enumerate(num_cols):
            ys = pd.to_numeric(val_df.iloc[:, col_idx], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(xs) & np.isfinite(ys)
            if np.any(mask):
                circos_points(
                    xs[mask],
                    ys[mask],
                    pch=pch_n[k],
                    col=col_n[k],
                    cex=cex_n[k],
                    bg=bg_n[k],
                    sector_index=sector_index,
                    track_index=track_index,
                )


def circos_genomic_lines(
    region: Any,
    value: Any = None,
    numeric_column: Any = None,
    sector_index: str | None = None,
    track_index: int | None = None,
    pos_transform: Callable[..., pd.DataFrame] | None = None,
    col: Any = None,
    lwd: Any = 1.0,
    lty: Any = 1,
    type: Any = "l",
    area: Any = False,
    area_baseline: Any = None,
    border: Any = "black",
    baseline: Any = "bottom",
    pt_col: Any = "black",
    cex: Any = 1.0,
    pch: Any = 1,
    **kwargs: Any,
) -> None:
    """Plot genomic lines, segments, or filled area curves."""
    nkw = normalize_kwargs(kwargs)
    numeric_column = nkw.pop("numeric_column", numeric_column)
    sector_index = nkw.pop("sector_index", sector_index)
    track_index = nkw.pop("track_index", track_index)
    pos_transform = nkw.pop("pos_transform", pos_transform)
    area_baseline = nkw.pop("area_baseline", area_baseline)
    pt_col = nkw.pop("pt_col", pt_col)
    if area_baseline is not None:
        baseline = area_baseline

    if sector_index is None:
        sector_index = get_current_sector_index()
    if track_index is None:
        track_index = get_current_track_index()

    if col is None:
        is_area = bool(area[0]) if isinstance(area, (list, tuple, np.ndarray)) else bool(area)
        col = "grey" if is_area else "black"

    reg = _strip_chr_col(region)
    nr = len(reg)
    if nr == 0:
        return

    param = nkw.pop("_param", None) or _STATE.current_genomic_param or {}
    val_df, forced_nc = _coerce_value_df(value, nr)

    if param.get("stack") and numeric_column is None:
        val_df = pd.DataFrame({"hline": np.full(nr, float(param.get("i", 1)))})
        num_cols = [0]
        type = "segment"
    else:
        if forced_nc is not None and numeric_column is None:
            num_cols = forced_nc
        elif numeric_column is None and param.get("numeric_column"):
            num_cols = [int(c) for c in param["numeric_column"] if 0 <= int(c) < val_df.shape[1]]
        elif numeric_column is not None:
            num_cols = _resolve_col_indices(val_df, numeric_column, min_pos=0)
        else:
            num_cols = [i for i, c in enumerate(val_df.columns) if pd.api.types.is_numeric_dtype(val_df[c])]
            if not num_cols:
                raise ValueError("Cannot find numeric column in `value`.")

    if pos_transform is not None:
        reg = _strip_chr_col(pos_transform(reg))

    x0 = pd.to_numeric(reg.iloc[:, 0]).to_numpy(dtype=float)
    x1 = pd.to_numeric(reg.iloc[:, 1]).to_numpy(dtype=float)
    xmid = 0.5 * (x0 + x1)

    nc = len(num_cols)
    col_n = _normalize_graphical_param(col, nc, 1, "col")
    lwd_n = _normalize_graphical_param(lwd, nc, 1, "lwd")
    lty_n = _normalize_graphical_param(lty, nc, 1, "lty")
    pt_col_n = _normalize_graphical_param(pt_col, nc, 1, "pt_col")
    cex_n = _normalize_graphical_param(cex, nc, 1, "cex")
    pch_n = _normalize_graphical_param(pch, nc, 1, "pch")
    type_n = _normalize_graphical_param(type, nc, 1, "type")
    area_n = _normalize_graphical_param(area, nc, 1, "area")
    base_n = _normalize_graphical_param(baseline, nc, 1, "baseline")
    border_n = _normalize_graphical_param(border, nc, 1, "border")

    if nc == 1:
        ys = pd.to_numeric(val_df.iloc[:, num_cols[0]], errors="coerce").to_numpy(dtype=float)
        if nkw.get("hline") is not None or type_n == "segment":
            circos_segments(
                x0,
                ys,
                x1,
                ys,
                col=col_n,
                lwd=lwd_n,
                lty=lty_n,
                sector_index=sector_index,
                track_index=track_index,
            )
        else:
            circos_lines(
                xmid,
                ys,
                col=col_n,
                lwd=lwd_n,
                lty=lty_n,
                type=type_n,
                area=area_n,
                baseline=base_n,
                border=border_n,
                pt_col=pt_col_n,
                cex=cex_n,
                pch=pch_n,
                sector_index=sector_index,
                track_index=track_index,
            )
    else:
        for k, col_idx in enumerate(num_cols):
            ys = pd.to_numeric(val_df.iloc[:, col_idx], errors="coerce").to_numpy(dtype=float)
            if type_n[k] == "segment":
                circos_segments(
                    x0,
                    ys,
                    x1,
                    ys,
                    col=col_n[k],
                    lwd=lwd_n[k],
                    lty=lty_n[k],
                    sector_index=sector_index,
                    track_index=track_index,
                )
            else:
                circos_lines(
                    xmid,
                    ys,
                    col=col_n[k],
                    lwd=lwd_n[k],
                    lty=lty_n[k],
                    type=type_n[k],
                    area=area_n[k],
                    baseline=base_n[k],
                    border=border_n[k],
                    pt_col=pt_col_n[k],
                    cex=cex_n[k],
                    pch=pch_n[k],
                    sector_index=sector_index,
                    track_index=track_index,
                )


def circos_genomic_rect(
    region: Any,
    value: Any = None,
    ytop: Any = None,
    ybottom: Any = None,
    ytop_column: Any = None,
    ybottom_column: Any = None,
    sector_index: str | None = None,
    track_index: int | None = None,
    pos_transform: Callable[..., pd.DataFrame] | None = None,
    col: Any = None,
    border: Any = "black",
    lty: Any = 1,
    lwd: Any = 1.0,
    **kwargs: Any,
) -> None:
    """Plot genomic rectangles in chromosomal coordinates."""
    nkw = normalize_kwargs(kwargs)
    ytop_column = nkw.pop("ytop_column", ytop_column)
    ybottom_column = nkw.pop("ybottom_column", ybottom_column)
    sector_index = nkw.pop("sector_index", sector_index)
    track_index = nkw.pop("track_index", track_index)
    pos_transform = nkw.pop("pos_transform", pos_transform)

    if sector_index is None:
        sector_index = get_current_sector_index()
    if track_index is None:
        track_index = get_current_track_index()

    reg = _strip_chr_col(region)
    nr = len(reg)
    if nr == 0:
        return

    param = nkw.pop("_param", None) or _STATE.current_genomic_param or {}
    if param.get("stack"):
        i_val = float(param.get("i", 1))
        if ytop is None and ytop_column is None:
            ytop = i_val + 0.5
        if ybottom is None and ybottom_column is None:
            ybottom = i_val - 0.5

    val_df, _ = _coerce_value_df(value, nr)
    ylim = get_cell_meta_data("ylim", sector_index=sector_index, track_index=track_index)

    if ytop is not None:
        yt_arr = np.repeat(float(ytop), nr) if np.isscalar(ytop) else np.asarray(ytop, dtype=float)
    elif ytop_column is not None:
        tc = _resolve_col_indices(val_df, ytop_column, min_pos=0)
        if len(tc) != 1:
            raise ValueError("Only one `ytop_column` is allowed.")
        yt_arr = pd.to_numeric(val_df.iloc[:, tc[0]]).to_numpy(dtype=float)
    else:
        yt_arr = np.repeat(float(ylim[1]), nr)

    if ybottom is not None:
        yb_arr = np.repeat(float(ybottom), nr) if np.isscalar(ybottom) else np.asarray(ybottom, dtype=float)
    elif ybottom_column is not None:
        bc = _resolve_col_indices(val_df, ybottom_column, min_pos=0)
        if len(bc) != 1:
            raise ValueError("Only one `ybottom_column` is allowed.")
        yb_arr = pd.to_numeric(val_df.iloc[:, bc[0]]).to_numpy(dtype=float)
    else:
        yb_arr = np.repeat(float(ylim[0]), nr)

    if pos_transform is not None:
        reg = _strip_chr_col(pos_transform(reg))

    xleft = pd.to_numeric(reg.iloc[:, 0]).to_numpy(dtype=float)
    xright = pd.to_numeric(reg.iloc[:, 1]).to_numpy(dtype=float)

    col_n = _normalize_graphical_param(col, 1, nr, "col")
    border_n = _normalize_graphical_param(border, 1, nr, "border")
    lty_n = _normalize_graphical_param(lty, 1, nr, "lty")
    lwd_n = _normalize_graphical_param(lwd, 1, nr, "lwd")

    circos_rect(
        xleft,
        yb_arr,
        xright,
        yt_arr,
        sector_index=sector_index,
        track_index=track_index,
        col=col_n,
        border=border_n,
        lty=lty_n,
        lwd=lwd_n,
    )


def circos_genomic_text(
    region: Any,
    value: Any = None,
    y: Any = None,
    labels: Any = None,
    labels_column: Any = None,
    numeric_column: Any = None,
    sector_index: str | None = None,
    track_index: int | None = None,
    pos_transform: Callable[..., pd.DataFrame] | None = None,
    direction: str | None = None,
    facing: str = "inside",
    nice_facing: bool = False,
    adj: Sequence[float] = (0.5, 0.5),
    cex: Any = 1.0,
    col: Any = "black",
    font: Any = 1,
    padding: float = 0.0,
    extend: float | Sequence[float] = 0.0,
    align_to: str = "region",
    **kwargs: Any,
) -> None:
    """Plot genomic text labels with optional `posTransform_text` anti-overlap repositioning."""
    nkw = normalize_kwargs(kwargs)
    labels_column = nkw.pop("labels_column", labels_column)
    numeric_column = nkw.pop("numeric_column", numeric_column)
    sector_index = nkw.pop("sector_index", sector_index)
    track_index = nkw.pop("track_index", track_index)
    pos_transform = nkw.pop("pos_transform", pos_transform)
    nice_facing = bool(nkw.pop("nice_facing", nice_facing))
    align_to = nkw.pop("align_to", align_to)
    if direction is not None:
        warnings.warn("`direction` is deprecated; use `facing` instead.", DeprecationWarning, stacklevel=2)
        facing = direction

    if sector_index is None:
        sector_index = get_current_sector_index()
    if track_index is None:
        track_index = get_current_track_index()

    reg = _strip_chr_col(region)
    nr = len(reg)
    if nr == 0:
        return

    param = nkw.pop("_param", None) or _STATE.current_genomic_param or {}
    val_df, forced_nc = _coerce_value_df(value, nr)

    if param.get("stack") and numeric_column is None and y is None:
        y_vals = np.full(nr, float(param.get("i", 1)), dtype=float)
        val_df = val_df.copy()
        val_df["_stack_y"] = y_vals
        num_cols = [val_df.shape[1] - 1]
    else:
        num_cols = None

    if y is not None:
        y_arr = np.repeat(float(y), nr) if np.isscalar(y) else np.asarray(y, dtype=float)
        val_df = val_df.copy()
        val_df["_y_col"] = y_arr
        num_cols = [val_df.shape[1] - 1]

    if labels is None and labels_column is None:
        raise ValueError("Specify either `labels` or `labels_column`.")

    if labels is not None:
        lbl_arr = [str(labels)] * nr if isinstance(labels, str) else [str(v) for v in labels]
    else:
        lc = _resolve_col_indices(val_df, labels_column, min_pos=0)
        if len(lc) != 1:
            raise ValueError("`labels_column` must resolve to a single column.")
        lbl_arr = [str(v) for v in val_df.iloc[:, lc[0]].tolist()]

    if num_cols is None:
        if forced_nc is not None and numeric_column is None:
            num_cols = forced_nc
        elif numeric_column is None and param.get("numeric_column"):
            num_cols = [int(c) for c in param["numeric_column"] if 0 <= int(c) < val_df.shape[1]]
        elif numeric_column is not None:
            num_cols = _resolve_col_indices(val_df, numeric_column, min_pos=0)
        else:
            num_cols = [i for i, c in enumerate(val_df.columns) if pd.api.types.is_numeric_dtype(val_df[c])]
            if not num_cols:
                raise ValueError("Cannot find numeric column for text y coordinates.")
            num_cols = [num_cols[0]]

    if len(num_cols) > 1:
        raise ValueError("Only one numeric column is allowed for `circos_genomic_text`.")

    y_vec = pd.to_numeric(val_df.iloc[:, num_cols[0]]).to_numpy(dtype=float)

    if pos_transform is not None:
        if pos_transform in (pos_transform_text, posTransform_text):
            if facing not in ("clockwise", "reverse.clockwise"):
                raise ValueError("`facing` must be 'clockwise' or 'reverse.clockwise' when using `posTransform_text`.")
            reg = _strip_chr_col(
                pos_transform_text(
                    reg,
                    y=y_vec,
                    labels=lbl_arr,
                    cex=cex,
                    font=font,
                    sector_index=sector_index,
                    track_index=track_index,
                    padding=padding,
                    extend=extend,
                    align_to=align_to,
                )
            )
        else:
            reg = _strip_chr_col(pos_transform(reg))

    xs = 0.5 * (pd.to_numeric(reg.iloc[:, 0]).to_numpy(dtype=float) + pd.to_numeric(reg.iloc[:, 1]).to_numpy(dtype=float))
    col_n = _normalize_graphical_param(col, 1, nr, "col")
    cex_n = _normalize_graphical_param(cex, 1, nr, "cex")
    font_n = _normalize_graphical_param(font, 1, nr, "font")

    circos_text(
        xs,
        y_vec,
        lbl_arr,
        facing=facing,
        nice_facing=nice_facing,
        adj=adj,
        cex=cex_n,
        col=col_n,
        font=font_n,
        sector_index=sector_index,
        track_index=track_index,
    )


def circos_genomic_link(
    region1: Any,
    region2: Any,
    rou: Any = None,
    rou1: Any = None,
    rou2: Any = None,
    col: Any = "black",
    lwd: Any = 1.0,
    lty: Any = 1,
    border: Any = None,
    **kwargs: Any,
) -> None:
    """Draw chord links between paired genomic BED intervals."""
    nkw = normalize_kwargs(kwargs)
    r1 = _normalize_to_data_frame(region1, sort=False)
    r2 = _normalize_to_data_frame(region2, sort=False)
    if isinstance(r1, list) or isinstance(r2, list):
        raise ValueError("`region1` and `region2` cannot be lists of BED tables.")

    if circos_par.get_value("ring"):
        span = float(get_sector_data()["max.value"] - get_sector_data()["min.value"])
        m1 = r1.iloc[:, 1] > r1.iloc[:, 2]
        if m1.any():
            r1.loc[m1, r1.columns[1]] -= span
        m2 = r2.iloc[:, 1] > r2.iloc[:, 2]
        if m2.any():
            r2.loc[m2, r2.columns[1]] -= span

    if len(r1) != len(r2):
        raise ValueError(
            f"Number of rows in `region1` ({len(r1)}) and `region2` ({len(r2)}) differ; "
            "check chromosomes match initialized sectors."
        )
    nr = len(r1)
    if nr == 0:
        return

    if rou is None:
        rou = get_most_inside_radius()
    if rou1 is None:
        rou1 = rou
    if rou2 is None:
        rou2 = rou
    if border is None:
        border = col

    r1_v = recycle_to_length(rou1, nr)
    r2_v = recycle_to_length(rou2, nr)
    col_v = recycle_to_length(col, nr)
    lwd_v = recycle_to_length(lwd, nr)
    lty_v = recycle_to_length(lty, nr)
    border_v = recycle_to_length(border, nr)

    for i in range(nr):
        s1 = str(r1.iloc[i, 0])
        a1, b1 = float(r1.iloc[i, 1]), float(r1.iloc[i, 2])
        p1 = a1 if a1 == b1 else [a1, b1]

        s2 = str(r2.iloc[i, 0])
        a2, b2 = float(r2.iloc[i, 1]), float(r2.iloc[i, 2])
        p2 = a2 if a2 == b2 else [a2, b2]

        circos_link(
            s1,
            p1,
            s2,
            p2,
            rou1=float(r1_v[i]),
            rou2=float(r2_v[i]),
            col=col_v[i],
            lwd=float(lwd_v[i]),
            lty=lty_v[i],
            border=border_v[i],
            **nkw,
        )


# ---------------------------------------------------------------------------
# Position transforms & alignment
# ---------------------------------------------------------------------------

def smart_align(x1: Sequence[float], x2: Sequence[float], xlim: Sequence[float]) -> pd.DataFrame:
    """Resolve overlapping 1D intervals [x1, x2] within [xlim[0], xlim[1]]."""
    a1 = np.asarray(x1, dtype=float).copy()
    a2 = np.asarray(x2, dtype=float).copy()
    n = len(a1)
    if n == 0:
        return pd.DataFrame({"start": pd.Series(dtype=float), "end": pd.Series(dtype=float)})
    if len(a2) != n:
        raise ValueError("`x1` and `x2` must have the same length.")

    lo, hi = float(xlim[0]), float(xlim[1])
    xrange = hi - lo

    ncluster_before = -1
    ncluster = n
    while ncluster_before != ncluster:
        ncluster_before = ncluster
        clusters = np.zeros(n, dtype=int)
        c_id = 1
        clusters[0] = c_id
        for i in range(1, n):
            if a1[i] <= a2[i - 1]:
                clusters[i] = c_id
            else:
                c_id += 1
                clusters[i] = c_id

        ncluster = len(np.unique(clusters))
        if ncluster_before == ncluster:
            break

        new_x1 = np.zeros(n, dtype=float)
        new_x2 = np.zeros(n, dtype=float)
        for cid in np.unique(clusters):
            idx = np.where(clusters == cid)[0]
            lens = a2[idx] - a1[idx]
            total_len = float(np.sum(lens))
            mid = 0.5 * (float(np.min(a1[idx])) + float(np.max(a2[idx])))
            if total_len > xrange:
                tp = np.concatenate([[0.0], np.cumsum(lens)]) + 0.5 * (lo + hi) - 0.5 * total_len
            elif mid - 0.5 * total_len < lo:
                tp = np.concatenate([[0.0], np.cumsum(lens)]) + lo
            elif mid + 0.5 * total_len > hi:
                tp = hi - total_len + np.concatenate([[0.0], np.cumsum(lens)])
            else:
                tp = np.concatenate([[0.0], np.cumsum(lens)]) + mid - 0.5 * total_len
            new_x1[idx] = tp[:-1]
            new_x2[idx] = tp[1:]
        a1 = new_x1
        a2 = new_x2

    return pd.DataFrame({"start": a1, "end": a2})


smartAlign = smart_align


def pos_transform_default(region: Any, **kwargs: Any) -> pd.DataFrame:
    """Evenly distribute regions across the full sector x-range."""
    reg = _strip_chr_col(region)
    xlim = get_cell_meta_data("xlim")
    seg = np.linspace(float(xlim[0]), float(xlim[1]), len(reg) + 1)
    return pd.DataFrame({"start": seg[:-1], "end": seg[1:]})


posTransform_default = pos_transform_default


def pos_transform_text(
    region: Any,
    y: Any,
    labels: Any,
    cex: Any = 1.0,
    font: Any = 1,
    sector_index: str | None = None,
    track_index: int | None = None,
    padding: float = 0.0,
    extend: float | Sequence[float] = 0.0,
    align_to: str = "region",
    **kwargs: Any,
) -> pd.DataFrame:
    """Reposition intervals using text height in data coordinates to prevent overlaps."""
    nkw = normalize_kwargs(kwargs)
    sector_index = nkw.pop("sector_index", sector_index)
    track_index = nkw.pop("track_index", track_index)
    if sector_index is None:
        sector_index = get_cell_meta_data("sector.index")
    if track_index is None:
        track_index = get_cell_meta_data("track.index")

    reg = _strip_chr_col(region)
    nr = len(reg)
    if nr == 0:
        return pd.DataFrame({"start": pd.Series(dtype=float), "end": pd.Series(dtype=float)})

    y_arr = np.repeat(float(y), nr) if np.isscalar(y) else np.asarray(y, dtype=float)
    lbl_list = [str(labels)] * nr if isinstance(labels, str) else [str(v) for v in labels]

    ext = [float(extend), float(extend)] if np.isscalar(extend) else [float(extend[0]), float(extend[1])]

    s0 = pd.to_numeric(reg.iloc[:, 0]).to_numpy(dtype=float)
    e0 = pd.to_numeric(reg.iloc[:, 1]).to_numpy(dtype=float)
    od = np.argsort(s0 + e0, kind="mergesort")
    od_back = np.empty(nr, dtype=int)
    od_back[od] = np.arange(nr)

    s_ord = s0[od]
    e_ord = e0[od]
    y_ord = y_arr[od]
    lbl_ord = [lbl_list[i] for i in od]
    cex_v = [float(v) for v in recycle_to_length(cex, nr)]
    font_v = recycle_to_length(font, nr)
    cex_ord = [cex_v[i] for i in od]
    font_ord = [font_v[i] for i in od]

    text_h = np.array(
        [strheight_canvas(lbl_ord[i], cex=cex_ord[i], font=font_ord[i]) * (1.0 + float(padding)) for i in range(nr)],
        dtype=float,
    )
    mids = 0.5 * (s_ord + e_ord)
    d = circlize(mids, y_ord, sector_index=sector_index, track_index=track_index)
    delta_deg = as_degree(np.arctan(0.5 * text_h / np.maximum(d[:, 1], 1e-9)))
    alpha1 = d[:, 0] + delta_deg
    alpha2 = d[:, 0] - delta_deg

    x1 = reverse_circlize(alpha1, d[:, 1], sector_index=sector_index, track_index=track_index)[:, 0]
    x2 = reverse_circlize(alpha2, d[:, 1], sector_index=sector_index, track_index=track_index)[:, 0]

    # Note: depending on xaxis_clock_wise / clock_wise, ensure x1 <= x2 for smartAlign
    x_lo = np.minimum(x1, x2)
    x_hi = np.maximum(x1, x2)

    cx = get_cell_meta_data("cell.xlim", sector_index=sector_index, track_index=track_index)
    xr = float(cx[1] - cx[0])
    xlim = [float(cx[0]) - ext[0] * xr, float(cx[1]) + ext[1] * xr]
    xrange = xlim[1] - xlim[0]

    too_wide = (x_hi - x_lo) >= xrange
    x_lo[too_wide] = xlim[0]
    x_hi[too_wide] = xlim[1]

    left_out = x_lo < xlim[0]
    x_hi[left_out] = x_hi[left_out] + (xlim[0] - x_lo[left_out])
    x_lo[left_out] = xlim[0]

    right_out = x_hi > xlim[1]
    x_lo[right_out] = x_lo[right_out] - (x_hi[right_out] - xlim[1])
    x_hi[right_out] = xlim[1]

    aligned = smart_align(x_lo, x_hi, xlim=xlim)
    return aligned.iloc[od_back].reset_index(drop=True)


posTransform_text = pos_transform_text


def circos_genomic_pos_transform_lines(
    data: Any,
    track_height: float = 0.1,
    pos_transform: Callable[..., pd.DataFrame] | None = None,
    horizontal_line: str = "none",
    track_margin: Sequence[float] = (0.0, 0.0),
    direction: str = "inside",
    col: Any = "black",
    lwd: Any = 1.0,
    lty: Any = 1,
    **kwargs: Any,
) -> None:
    """Draw 3-segment orthogonal/elbow connector lines showing transformed genomic positions."""
    nkw = normalize_kwargs(kwargs)
    track_height = float(nkw.pop("track_height", track_height))
    pos_transform = nkw.pop("pos_transform", pos_transform)
    horizontal_line = str(nkw.pop("horizontal_line", horizontal_line))
    track_margin = nkw.pop("track_margin", track_margin)

    if horizontal_line not in ("none", "top", "bottom", "both"):
        raise ValueError("`horizontal_line` must be one of 'none', 'top', 'bottom', 'both'.")
    if direction not in ("inside", "outside"):
        raise ValueError("`direction` must be 'inside' or 'outside'.")

    df = _normalize_to_data_frame(data, sort=False)
    if isinstance(df, list):
        raise ValueError("`data` cannot be a list of region tables.")
    nr = len(df)
    col_v = recycle_to_length(col, nr)
    lwd_v = [float(v) for v in recycle_to_length(lwd, nr)]
    lty_v = recycle_to_length(lty, nr)

    o_tm = list(circos_par.get_value("track.margin"))
    circos_par.set_value("track_margin", [float(track_margin[0]), float(track_margin[1])], _suppress_warn=True)

    try:
        def _pt_panel(region: pd.DataFrame, value: pd.DataFrame, **pkw: Any) -> None:
            ch = get_current_chromosome()
            mask = (df.iloc[:, 0] == ch).to_numpy()
            if not np.any(mask):
                return
            idx_rows = np.where(mask)[0]
            c_sub = [col_v[k] for k in idx_rows]
            w_sub = [lwd_v[k] for k in idx_rows]
            t_sub = [lty_v[k] for k in idx_rows]

            if pos_transform is not None:
                sig = inspect.signature(pos_transform)
                npos = sum(
                    1
                    for p in sig.parameters.values()
                    if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                    and p.default is inspect.Parameter.empty
                )
                if npos >= 2:
                    reg_new = _strip_chr_col(pos_transform(region, value))
                else:
                    reg_new = _strip_chr_col(pos_transform(region))
            else:
                reg_new = region

            r_nr = len(region)
            x_old0 = pd.to_numeric(region.iloc[:, 0]).to_numpy(dtype=float)
            x_old1 = pd.to_numeric(region.iloc[:, 1]).to_numpy(dtype=float)
            x_new0 = pd.to_numeric(reg_new.iloc[:, 0]).to_numpy(dtype=float)
            x_new1 = pd.to_numeric(reg_new.iloc[:, 1]).to_numpy(dtype=float)

            mid = 0.5 * (x_old0 + x_old1)
            mid_new = 0.5 * (x_new0 + x_new1)

            if horizontal_line in ("both", "bottom"):
                circos_segments(x_old0, np.zeros(r_nr), x_old1, np.zeros(r_nr), col=c_sub, lwd=w_sub, lty=t_sub)
            if horizontal_line in ("both", "top"):
                circos_segments(x_old0, np.ones(r_nr), x_old1, np.ones(r_nr), col=c_sub, lwd=w_sub, lty=t_sub)

            if direction == "inside":
                circos_segments(mid, np.ones(r_nr), mid, np.full(r_nr, 2.0 / 3.0), col=c_sub, lwd=w_sub, lty=t_sub)
                circos_segments(
                    mid, np.full(r_nr, 2.0 / 3.0), mid_new, np.full(r_nr, 1.0 / 3.0), col=c_sub, lwd=w_sub, lty=t_sub
                )
                circos_segments(mid_new, np.full(r_nr, 1.0 / 3.0), mid_new, np.zeros(r_nr), col=c_sub, lwd=w_sub, lty=t_sub)
            else:
                circos_segments(mid, np.zeros(r_nr), mid, np.full(r_nr, 1.0 / 3.0), col=c_sub, lwd=w_sub, lty=t_sub)
                circos_segments(
                    mid, np.full(r_nr, 1.0 / 3.0), mid_new, np.full(r_nr, 2.0 / 3.0), col=c_sub, lwd=w_sub, lty=t_sub
                )
                circos_segments(mid_new, np.full(r_nr, 2.0 / 3.0), mid_new, np.ones(r_nr), col=c_sub, lwd=w_sub, lty=t_sub)

        circos_genomic_track_plot_region(
            df,
            ylim=[0.0, 1.0],
            bg_border=None,
            track_height=track_height,
            panel_fun=_pt_panel,
            **nkw,
        )
    finally:
        circos_par.set_value("track_margin", o_tm, _suppress_warn=True)


# ---------------------------------------------------------------------------
# Genomic density & rainfall
# ---------------------------------------------------------------------------

def genomic_density(
    region: Any,
    window_size: float = 1e7,
    n_window: int | None = None,
    overlap: bool = True,
    count_by: str = "percent",
    chr_len: Any = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Calculate sliding-window genomic feature density (fraction or count per window)."""
    nkw = normalize_kwargs(kwargs)
    window_size = float(nkw.pop("window_size", window_size))
    n_window = nkw.pop("n_window", n_window)
    chr_len = nkw.pop("chr_len", chr_len)

    if count_by not in ("percent", "number"):
        raise ValueError("`count_by` must be 'percent' or 'number'.")

    df = validate_data_frame(region)
    if len(df) == 0:
        return pd.DataFrame(columns=["chr", "start", "end", "value"] if df.shape[1] >= 3 else ["start", "end", "value"])

    first_is_chr = not pd.api.types.is_numeric_dtype(df.iloc[:, 0])
    numeric_small_chr = (not first_is_chr) and (df.shape[1] >= 3) and (float(df.iloc[:, 0].max()) < 100)

    if first_is_chr or numeric_small_chr:
        validate_region(df, start_col=1, end_col=2, check_chr=first_is_chr)
        chrs_col = df.iloc[:, 0].astype(str) if first_is_chr else df.iloc[:, 0]
        all_chr = list(dict.fromkeys(chrs_col.tolist()))

        chr_len_map: dict[str, float] = {}
        if chr_len is not None:
            if isinstance(chr_len, dict):
                chr_len_map = {str(k): float(v) for k, v in chr_len.items()}
            elif np.isscalar(chr_len) and len(all_chr) == 1:
                chr_len_map = {str(all_chr[0]): float(chr_len)}
            elif len(chr_len) == len(all_chr):
                chr_len_map = {str(c): float(v) for c, v in zip(all_chr, chr_len)}

        frames: list[pd.DataFrame] = []
        for ch in all_chr:
            sub = df[chrs_col == ch].iloc[:, 1:3]
            max_rg: float | None = None
            if chr_len is None:
                if is_circos_initialized() and str(ch) in set(get_all_sector_index()):
                    max_rg = float(get_sector_data(sector_index=str(ch))["max.data"])
            elif str(ch) in chr_len_map:
                max_rg = chr_len_map[str(ch)]

            res = genomic_density(
                sub,
                window_size=window_size,
                n_window=n_window,
                overlap=overlap,
                count_by=count_by,
                chr_len=max_rg,
            )
            res.insert(0, "chr", ch)
            frames.append(res)
        return pd.concat(frames, ignore_index=True)

    validate_region(df, start_col=0, end_col=1, check_chr=False)
    reg2 = reduce_region(sort_region(df.iloc[:, :2]))
    max_data_end = float(reg2["end"].max())
    max_pos = max(float(chr_len), max_data_end) if chr_len is not None else max_data_end

    if overlap:
        if n_window is None:
            step = window_size / 2.0
            b = np.arange(1.0, max_pos + 1e-12, step)
            s = b[:-2] if len(b) >= 2 else np.array([], dtype=float)
            e = s + window_size - 1.0
        else:
            b = np.linspace(1.0, max_pos, int(2 * n_window - 1))
            s = b[:-2] if len(b) >= 2 else np.array([], dtype=float)
            e = s + (b[2] - b[0] - 1.0) if len(s) > 0 else np.array([], dtype=float)
    else:
        if n_window is None:
            b = np.arange(1.0, max_pos + 1e-12, window_size)
            s = b[:-1] if len(b) >= 1 else np.array([], dtype=float)
            e = s + window_size - 1.0
        else:
            b = np.linspace(1.0, max_pos, int(n_window))
            s = b[:-1] if len(b) >= 1 else np.array([], dtype=float)
            e = s + (b[1] - b[0]) if len(s) > 0 else np.array([], dtype=float)

    s_int = s.astype(int)
    e_int = e.astype(int)
    if len(s_int) == 0:
        return pd.DataFrame({"start": pd.Series(dtype=int), "end": pd.Series(dtype=int), "value": pd.Series(dtype=float)})

    windows = pd.DataFrame({"start": s_int, "end": e_int})
    op = overlap_region(windows, reg2, count_by=count_by)  # type: ignore[arg-type]
    return pd.DataFrame({"start": s_int, "end": e_int, "value": op})


genomicDensity = genomic_density


def circos_genomic_density(
    data: Any,
    ylim_force: bool = False,
    window_size: float | None = None,
    overlap: bool = True,
    count_by: str = "percent",
    col: Any = None,
    lwd: Any = 1.0,
    lty: Any = 1,
    type: Any = "l",
    area: Any = True,
    area_baseline: Any = None,
    baseline: Any = 0.0,
    border: Any = None,
    **kwargs: Any,
) -> None:
    """Add a circular genomic density track for one or more BED DataFrames."""
    nkw = normalize_kwargs(kwargs)
    ylim_force = bool(nkw.pop("ylim_force", ylim_force))
    window_size = nkw.pop("window_size", window_size)
    area_baseline = nkw.pop("area_baseline", area_baseline)
    if area_baseline is not None:
        baseline = area_baseline

    if col is None:
        is_area = bool(area[0]) if isinstance(area, (list, tuple, np.ndarray)) else bool(area)
        col = "grey" if is_area else "black"

    norm = _normalize_to_data_frame(data, sort=False)
    data_list = norm if isinstance(norm, list) else [norm]
    ndf = len(data_list)

    col_v = recycle_to_length(col, ndf)
    lwd_v = recycle_to_length(lwd, ndf)
    lty_v = recycle_to_length(lty, ndf)
    type_v = recycle_to_length(type, ndf)
    area_v = recycle_to_length(area, ndf)
    base_v = recycle_to_length(baseline, ndf)
    border_v = [None] * ndf if border is None else recycle_to_length(border, ndf)

    if window_size is None:
        total_range = sum(float(get_cell_meta_data("xrange", sector_index=si)) for si in get_all_sector_index())
        digits = len(str(max(1, int(round(total_range)))))
        window_size = (10.0**digits) / 1000.0
    else:
        window_size = float(window_size)

    dens_list = [
        genomic_density(d, window_size=window_size, overlap=overlap, count_by=count_by)
        for d in data_list
    ]
    if ylim_force and count_by == "percent":
        ymax = 1.0
    else:
        max_vals = [float(d.iloc[:, 3].max()) for d in dens_list if len(d) > 0]
        ymax = max(max_vals) if max_vals else 1.0
        if ymax <= 0.0:
            ymax = 1.0

    if ndf == 1:
        def _dens_panel_1(region: pd.DataFrame, value: pd.DataFrame, **pkw: Any) -> None:
            circos_genomic_lines(
                region,
                value,
                col=col_v[0],
                lwd=lwd_v[0],
                lty=lty_v[0],
                type=type_v[0],
                border=border_v[0],
                area=area_v[0],
                baseline=base_v[0],
            )

        circos_genomic_track_plot_region(dens_list[0], ylim=[0.0, ymax], panel_fun=_dens_panel_1, **nkw)
    else:
        def _dens_panel_k(region: pd.DataFrame, value: pd.DataFrame, **pkw: Any) -> None:
            idx = get_i(**pkw) - 1
            circos_genomic_lines(
                region,
                value,
                col=col_v[idx],
                lwd=lwd_v[idx],
                lty=lty_v[idx],
                type=type_v[idx],
                border=border_v[idx],
                area=area_v[idx],
                baseline=base_v[idx],
            )

        circos_genomic_track_plot_region(dens_list, ylim=[0.0, ymax], panel_fun=_dens_panel_k, **nkw)


def rainfall_transform(
    region: Any,
    mode: str = "min",
    normalize_to_width: bool = False,
) -> pd.DataFrame:
    """Compute inter-feature distances between neighboring genomic intervals."""
    if mode not in ("min", "max", "mean", "left", "right"):
        raise ValueError("`mode` must be one of 'min', 'max', 'mean', 'left', 'right'.")

    df = validate_data_frame(region)
    if len(df) == 0:
        cols = ["chr", "start", "end", "dist"] if df.shape[1] >= 3 else ["start", "end", "dist"]
        return pd.DataFrame(columns=cols)

    first_is_chr = not pd.api.types.is_numeric_dtype(df.iloc[:, 0])
    numeric_small_chr = (not first_is_chr) and (df.shape[1] >= 3) and (float(df.iloc[:, 0].max()) < 100)

    if first_is_chr or numeric_small_chr:
        out = df.iloc[:, :3].copy()
        out.columns = ["chr", "start", "end"]
        out["chr"] = out["chr"].astype(str)
        out["start"] = pd.to_numeric(out["start"], errors="raise").astype(float)
        out["end"] = pd.to_numeric(out["end"], errors="raise").astype(float)
        dists = np.full(len(out), np.nan, dtype=float)
        for ch in dict.fromkeys(out["chr"].tolist()):
            mask = (out["chr"] == ch).to_numpy()
            sub_res = rainfall_transform(out.loc[mask, ["start", "end"]], mode=mode, normalize_to_width=False)
            dists[mask] = sub_res["dist"].to_numpy(dtype=float)
        out["dist"] = dists
        if normalize_to_width:
            out["dist"] = out["dist"] / (out["end"] - out["start"])
        return out

    reg_bk = df.iloc[:, :2].copy().reset_index(drop=True)
    reg_bk.columns = ["start", "end"]
    reg_bk["start"] = pd.to_numeric(reg_bk["start"], errors="raise").astype(float)
    reg_bk["end"] = pd.to_numeric(reg_bk["end"], errors="raise").astype(float)

    n = len(reg_bk)
    if n < 2:
        reg_bk["dist"] = np.nan
        return reg_bk

    od = order_region(reg_bk)
    reg_ord = reg_bk.iloc[od].reset_index(drop=True)
    s = reg_ord["start"].to_numpy(dtype=float)
    e = reg_ord["end"].to_numpy(dtype=float)

    dist = np.zeros(n, dtype=float)
    dist[0] = s[1] - e[0]
    dist[n - 1] = s[n - 1] - e[n - 2]
    if n > 2:
        d_right = s[2:n] - e[1 : (n - 1)]
        d_left = s[1 : (n - 1)] - e[0 : (n - 2)]
        if mode == "min":
            dist[1 : (n - 1)] = np.minimum(d_right, d_left)
        elif mode == "max":
            dist[1 : (n - 1)] = np.maximum(d_right, d_left)
        elif mode == "mean":
            dist[1 : (n - 1)] = 0.5 * (d_right + d_left)
        elif mode == "left":
            dist[1 : (n - 1)] = d_left
        elif mode == "right":
            dist[1 : (n - 1)] = d_right

    dist = np.where(dist < 0.0, 0.0, dist)
    if mode == "left":
        dist[0] = np.nan
    elif mode == "right":
        dist[n - 1] = np.nan

    out_dist = np.empty(n, dtype=float)
    out_dist[od] = dist
    reg_bk["dist"] = out_dist
    if normalize_to_width:
        reg_bk["dist"] = reg_bk["dist"] / (reg_bk["end"] - reg_bk["start"])
    return reg_bk


rainfallTransform = rainfall_transform


def circos_genomic_rainfall(
    data: Any,
    mode: str = "min",
    ylim: Sequence[float] | None = None,
    col: Any = "black",
    pch: Any = 1,
    cex: Any = 1.0,
    normalize_to_width: bool = False,
    **kwargs: Any,
) -> None:
    """Plot genomic rainfall track showing log10 inter-variant/inter-DMR distances."""
    nkw = normalize_kwargs(kwargs)
    norm = _normalize_to_data_frame(data, sort=False)
    data_list = norm if isinstance(norm, list) else [norm]
    for d in data_list:
        validate_region(d, check_chr=True)

    ndf = len(data_list)
    col_v = recycle_to_length(col, ndf)
    pch_v = recycle_to_length(pch, ndf)
    cex_v = recycle_to_length(cex, ndf)

    if ylim is None:
        eff_ylim = [-10.0, 10.0] if normalize_to_width else [0.0, 9.0]
    else:
        eff_ylim = [float(ylim[0]), float(ylim[1])]

    def _rf_panel(region: pd.DataFrame, value: pd.DataFrame, **pkw: Any) -> None:
        df_rf = rainfall_transform(region, mode=mode, normalize_to_width=normalize_to_width)
        idx = get_i(**pkw) - 1
        dvals = df_rf["dist"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            yvals = np.log10(dvals) if normalize_to_width else np.log10(dvals + 1.0)
        circos_genomic_points(
            df_rf[["start", "end"]],
            pd.DataFrame({"y": yvals}),
            col=col_v[idx],
            cex=cex_v[idx],
            pch=pch_v[idx],
        )

    circos_genomic_track_plot_region(data_list if ndf > 1 else data_list[0], ylim=eff_ylim, panel_fun=_rf_panel, **nkw)


# ---------------------------------------------------------------------------
# High-level genomic heatmap & non-overlapping labels
# ---------------------------------------------------------------------------

def circos_genomic_heatmap(
    bed: Any,
    col: Any = None,
    na_col: str = "grey",
    numeric_column: Any = None,
    border: Any = None,
    border_lwd: float = 1.0,
    border_lty: Any = 1,
    connection_height: float | None = None,
    line_col: Any = "black",
    line_lwd: Any = 1.0,
    line_lty: Any = 1,
    heatmap_height: float = 0.15,
    side: str = "inside",
    track_margin: Sequence[float] | None = None,
    **kwargs: Any,
) -> None:
    """Draw a stacked genomic heatmap track with optional positional connection lines."""
    nkw = normalize_kwargs(kwargs)
    numeric_column = nkw.pop("numeric_column", numeric_column)
    if connection_height is None and "connection_height" not in kwargs:
        connection_height = float(convert_length(5.0, "mm"))
    if track_margin is None:
        track_margin = circos_par.get_value("track.margin")

    if side not in ("inside", "outside"):
        raise ValueError("`side` must be 'inside' or 'outside'.")

    df = _normalize_to_data_frame(bed, sort=False)
    if isinstance(df, list):
        raise ValueError("`bed` cannot be a list of DataFrames.")
    validate_region(df, check_chr=True)

    num_cols = _resolve_col_indices(df, numeric_column, min_pos=3, numeric_only=True)
    if not num_cols:
        raise ValueError("No numeric columns found in `bed`.")

    mat = df.iloc[:, num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    bed_clean = pd.concat([df.iloc[:, :3].reset_index(drop=True), pd.DataFrame(mat)], axis=1)

    if col is None:
        vmin = float(np.nanmin(mat))
        vmax = float(np.nanmax(mat))
        breaks = np.linspace(vmin, vmax, 3) if vmin < vmax else np.array([vmin - 1.0, vmin, vmin + 1.0])
        col = color_ramp2(breaks, ["blue", "#EEEEEE", "red"])

    if callable(col):
        col_mat = np.asarray(col(mat), dtype=object)
        if col_mat.shape != mat.shape:
            col_mat = col_mat.reshape(mat.shape)
    else:
        col_mat = np.asarray(col, dtype=object)
        if col_mat.shape != mat.shape:
            raise ValueError("`col` matrix shape must match selected numeric columns.")

    # Fill NA colors
    na_mask = np.isnan(mat) | pd.isna(col_mat)
    col_mat = col_mat.copy()
    col_mat[na_mask] = na_col

    if border is None:
        border_mat = col_mat.copy()
    elif np.isscalar(border):
        border_mat = np.full(mat.shape, border, dtype=object)
    else:
        border_mat = np.asarray(border, dtype=object).reshape(mat.shape)

    def _hm_panel(region: pd.DataFrame, value: pd.DataFrame, **pkw: Any) -> None:
        si = get_current_sector_index()
        mask = (bed_clean.iloc[:, 0] == si).to_numpy()
        layer_idx = get_i(**pkw) - 1
        c_slice = col_mat[mask, layer_idx].tolist()
        b_slice = border_mat[mask, layer_idx].tolist()
        circos_genomic_rect(
            region,
            value,
            col=c_slice,
            border=b_slice,
            lwd=border_lwd,
            lty=border_lty,
            pos_transform=pos_transform_default,
            _param=pkw.get("_param"),
        )

    if side == "inside":
        if connection_height is not None:
            circos_genomic_pos_transform_lines(
                bed_clean,
                pos_transform=pos_transform_default,
                horizontal_line="top",
                track_height=float(connection_height),
                track_margin=[float(convert_length(1.0, "mm")), float(track_margin[1])],
                cell_padding=[0.0, 0.0, 0.0, 0.0],
                col=line_col,
                lwd=line_lwd,
                lty=line_lty,
            )
        circos_genomic_track_plot_region(
            bed_clean,
            stack=True,
            bg_border=None,
            track_height=float(heatmap_height),
            track_margin=[float(track_margin[0]), 0.0],
            cell_padding=[0.0, 0.0, 0.0, 0.0],
            panel_fun=_hm_panel,
        )
    else:
        circos_genomic_track_plot_region(
            bed_clean,
            stack=True,
            bg_border=None,
            track_height=float(heatmap_height),
            track_margin=[0.0, float(track_margin[1])],
            cell_padding=[0.0, 0.0, 0.0, 0.0],
            panel_fun=_hm_panel,
        )
        if connection_height is not None:
            circos_genomic_pos_transform_lines(
                bed_clean,
                pos_transform=pos_transform_default,
                direction="outside",
                horizontal_line="bottom",
                track_height=float(connection_height),
                track_margin=[float(track_margin[1]), float(convert_length(1.0, "mm"))],
                cell_padding=[0.0, 0.0, 0.0, 0.0],
                col=line_col,
                lwd=line_lwd,
                lty=line_lty,
            )


def circos_genomic_labels(
    bed: Any,
    labels: Sequence[str] | None = None,
    labels_column: Any = None,
    facing: str = "clockwise",
    nice_facing: bool = True,
    col: Any = "black",
    cex: Any = 0.8,
    font: Any = 1,
    padding: float = 0.4,
    connection_height: float | None = None,
    line_col: Any = "black",
    line_lwd: Any = 1.0,
    line_lty: Any = 1,
    labels_height: float | None = None,
    side: str = "inside",
    labels_side: str | None = None,
    track_margin: Sequence[float] | None = None,
    **kwargs: Any,
) -> None:
    """Add radial genomic label track with non-overlapping cross-sector alignment and elbow connectors."""
    nkw = normalize_kwargs(kwargs)
    labels_column = nkw.pop("labels_column", labels_column)
    nice_facing = bool(nkw.pop("nice_facing", nice_facing))
    labels_side = nkw.pop("labels_side", labels_side)
    if labels_side is None:
        labels_side = side
    if connection_height is None:
        connection_height = float(convert_length(5.0, "mm"))
    if track_margin is None:
        track_margin = circos_par.get_value("track.margin")

    if facing not in ("clockwise", "reverse.clockwise"):
        raise ValueError("`facing` can only be 'clockwise' or 'reverse.clockwise'.")
    if labels_side not in ("inside", "outside"):
        raise ValueError("`side` / `labels_side` must be 'inside' or 'outside'.")

    df = _normalize_to_data_frame(bed, sort=False)
    if isinstance(df, list):
        raise ValueError("`bed` cannot be a list of DataFrames.")
    validate_region(df, check_chr=True)

    if labels is None:
        if labels_column is None:
            if df.shape[1] < 4:
                raise ValueError("Specify `labels` or `labels_column` when `bed` has fewer than 4 columns.")
            labels_column = 3
        lc = _resolve_col_indices(df, labels_column, min_pos=0)
        lbl_list = [str(v) for v in df.iloc[:, lc[0]].tolist()]
    else:
        lbl_list = [str(labels)] * len(df) if isinstance(labels, str) else [str(v) for v in labels]

    if labels_height is None:
        c_max = float(cex) if np.isscalar(cex) else float(max(cex))
        f_first = font if np.isscalar(font) else list(font)[0]
        max_w = max((strwidth_canvas(s, cex=c_max, font=f_first) for s in lbl_list), default=0.05)
        labels_height = min(float(cm_h(1.5)), max_w)

    nr = len(df)
    if nr == 0:
        return

    all_sectors = get_all_sector_index()
    sec_rank = {s: i for i, s in enumerate(all_sectors)}
    df4 = pd.DataFrame(
        {
            "chr": df.iloc[:, 0].astype(str).tolist(),
            "start": pd.to_numeric(df.iloc[:, 1]).to_numpy(dtype=float),
            "end": pd.to_numeric(df.iloc[:, 2]).to_numpy(dtype=float),
            "label": lbl_list,
        }
    )

    od = sorted(range(nr), key=lambda k: (sec_rank[df4.iloc[k, 0]], df4.iloc[k, 1]))
    df4 = df4.iloc[od].reset_index(drop=True)

    col_v = [recycle_to_length(col, nr)[k] for k in od]
    cex_v = [float(recycle_to_length(cex, nr)[k]) for k in od]
    font_v = [recycle_to_length(font, nr)[k] for k in od]
    lc_v = [recycle_to_length(line_col, nr)[k] for k in od]
    lw_v = [float(recycle_to_length(line_lwd, nr)[k]) for k in od]
    lt_v = [recycle_to_length(line_lty, nr)[k] for k in od]

    orig_xcw = bool(circos_par.get_value("xaxis.clock.wise"))
    if not orig_xcw:
        for idx in range(nr):
            sdata = get_sector_data(df4.loc[idx, "chr"])
            s_min, s_max = float(sdata["min.value"]), float(sdata["max.value"])
            a = s_max - df4.loc[idx, "start"] + s_min
            b = s_max - df4.loc[idx, "end"] + s_min
            df4.loc[idx, "start"] = b
            df4.loc[idx, "end"] = a
        circos_par.set_value("xaxis_clock_wise", True, _suppress_warn=True)

    op_warn = bool(circos_par.get_value("points.overflow.warning"))
    circos_par.set_value("points_overflow_warning", False, _suppress_warn=True)

    try:
        # Map all regions across chromosomes into a single extended virtual sector all_sectors[0]
        chr0 = all_sectors[0]
        sdata0 = get_sector_data(chr0)
        sd0 = float(sdata0["start.degree"])
        ed0 = float(sdata0["end.degree"])

        rho: list[float] = []
        for cr in dict.fromkeys(df4["chr"].tolist()):
            sub = df4[df4["chr"] == cr]
            th = circlize(sub["start"].to_numpy(), np.ones(len(sub)), sector_index=cr)[:, 0]
            rho.extend(th.tolist())

        if len(rho) == 0:
            anchor = (0.5 * (sd0 + ed0) + 180.0) % 360.0
        elif len(rho) == 1:
            anchor = (rho[0] + 180.0) % 360.0
        else:
            rho_s = np.sort(np.asarray(rho, dtype=float) % 360.0)
            rho_loop = np.append(rho_s, rho_s[0] + 360.0)
            i_max = int(np.argmax(np.diff(rho_loop)))
            anchor = (0.5 * (rho_loop[i_max] + rho_loop[i_max + 1])) % 360.0

        chr_width = sd0 - ed0
        s1 = sd0 % 360.0
        s2 = ed0 % 360.0
        if s1 < s2:
            s1 += 360.0
        if s2 <= anchor <= s1:
            if s1 - s2 > 180.0:
                ext1 = (abs(s1 - anchor) % 360.0) / chr_width
                ext2 = -(abs(s2 - anchor) % 360.0) / chr_width
            else:
                ext1 = 0.0
                ext2 = (360.0 - chr_width) / chr_width
        else:
            ext1 = ((anchor - s1) % 360.0) / chr_width
            ext2 = ((s2 - anchor) % 360.0) / chr_width
        extend = [abs(float(ext1)), abs(float(ext2))]
        new_chr_range = [sd0 + chr_width * extend[0], ed0 - chr_width * extend[1]]

        frames2: list[pd.DataFrame] = []
        for cr in dict.fromkeys(df4["chr"].tolist()):
            sub = df4[df4["chr"] == cr].copy()
            if cr != chr0:
                dfx1 = circlize(sub["start"].to_numpy(), np.ones(len(sub)), sector_index=cr)
                dfx2 = circlize(sub["end"].to_numpy(), np.ones(len(sub)), sector_index=cr)
                deg_diff = dfx2[:, 0] - dfx1[:, 0]
                out_mask = (dfx1[:, 0] > new_chr_range[0]) | (dfx1[:, 0] < new_chr_range[1])
                if np.any(out_mask):
                    dfx1[out_mask, 0] = (dfx1[out_mask, 0] - new_chr_range[1]) % 360.0 + new_chr_range[1]
                x1_new = reverse_circlize(dfx1[:, 0], dfx1[:, 1], sector_index=chr0)[:, 0]
                dfx2[:, 0] = dfx1[:, 0] + deg_diff
                x2_new = reverse_circlize(dfx2[:, 0], dfx2[:, 1], sector_index=chr0)[:, 0]
                sub["start"] = x1_new
                sub["end"] = x2_new
            frames2.append(sub)
        bed2 = pd.concat(frames2, ignore_index=True)
        bed2["chr"] = chr0

        if labels_side == "inside":
            circos_genomic_track_plot_region(
                bed2,
                ylim=[0.0, 1.0],
                track_margin=[float(convert_length(0.5, "mm")), float(track_margin[1])],
                cell_padding=[0.0, 0.0, 0.0, 0.0],
                track_height=float(connection_height),
                bg_border=None,
            )
            i_track = get_current_track_index()

            def _lbl_inside(region: pd.DataFrame, value: pd.DataFrame, **pkw: Any) -> None:
                circos_genomic_text(
                    region,
                    value,
                    y=1.0,
                    labels_column=0,
                    facing=facing,
                    adj=[1.0 if facing == "clockwise" else 0.0, 0.5],
                    pos_transform=pos_transform_text,
                    col=col_v,
                    cex=cex_v,
                    font=font_v,
                    nice_facing=nice_facing,
                    padding=padding,
                    extend=extend,
                )

            circos_genomic_track_plot_region(
                bed2,
                ylim=[0.0, 1.0],
                track_height=float(labels_height),
                bg_border=None,
                track_margin=[float(track_margin[0]), 0.0],
                cell_padding=[0.0, 0.0, 0.0, 0.0],
                panel_fun=_lbl_inside,
            )

            circos_genomic_pos_transform_lines(
                bed2,
                pos_transform=lambda region, value: pos_transform_text(
                    region,
                    y=1.0,
                    labels=value.iloc[:, 0].tolist(),
                    cex=cex_v,
                    font=font_v,
                    track_index=i_track + 1,
                    padding=padding,
                    extend=extend,
                ),
                direction="inside",
                horizontal_line="top",
                track_index=i_track,
                cell_padding=[0.0, 0.0, 0.0, 0.0],
                col=lc_v,
                lwd=lw_v,
                lty=lt_v,
            )
        else:
            def _lbl_outside(region: pd.DataFrame, value: pd.DataFrame, **pkw: Any) -> None:
                circos_genomic_text(
                    region,
                    value,
                    y=0.0,
                    labels_column=0,
                    facing=facing,
                    adj=[1.0 if facing == "reverse.clockwise" else 0.0, 0.5],
                    pos_transform=pos_transform_text,
                    col=col_v,
                    cex=cex_v,
                    font=font_v,
                    nice_facing=nice_facing,
                    padding=padding,
                    extend=extend,
                )

            circos_genomic_track_plot_region(
                bed2,
                ylim=[0.0, 1.0],
                track_height=float(labels_height),
                bg_border=None,
                track_margin=[0.0, float(track_margin[1])],
                cell_padding=[0.0, 0.0, 0.0, 0.0],
                panel_fun=_lbl_outside,
            )
            i_track = get_current_track_index()

            circos_genomic_pos_transform_lines(
                bed2,
                pos_transform=lambda region, value: pos_transform_text(
                    region,
                    y=0.0,
                    labels=value.iloc[:, 0].tolist(),
                    cex=cex_v,
                    font=font_v,
                    track_index=i_track,
                    padding=padding,
                    extend=extend,
                ),
                direction="outside",
                horizontal_line="bottom",
                col=lc_v,
                lwd=lw_v,
                lty=lt_v,
                track_height=float(connection_height),
                track_margin=[float(track_margin[0]), float(convert_length(0.5, "mm"))],
                cell_padding=[0.0, 0.0, 0.0, 0.0],
            )
    finally:
        circos_par.set_value("points_overflow_warning", op_warn, _suppress_warn=True)
        if not orig_xcw:
            circos_par.set_value("xaxis_clock_wise", False, _suppress_warn=True)
