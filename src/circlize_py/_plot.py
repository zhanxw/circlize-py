"""Circular layout initialization, tracks, update, draw.sector, and highlight.sector."""

from __future__ import annotations

import inspect
import math
import warnings
from typing import Any, Callable, Sequence

from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Polygon as MplPolygon
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd

from ._canvas import ensure_canvas, parse_unit
from ._colors import is_na_like, resolve_color, resolve_lty
from ._state import (
    _STATE,
    circos_par,
    get_all_sector_index,
    get_all_track_index,
    get_cell_meta_data,
    get_current_track_index,
    get_sector_data,
    is_circos_initialized,
    normalize_kwargs,
    set_cell_data,
    set_current_cell,
    set_current_sector_index,
    set_current_track_index,
)
from ._utils import (
    as_radian,
    check_track_position,
    polar2Cartesian,
    recycle_with_levels,
    uy,
)


class AsIs:
    """Wrapper equivalent to R's I() for literal padding values in highlight.sector."""

    def __init__(self, value: Any) -> None:
        self.value = value


def I(x: Any) -> AsIs:
    return AsIs(x)


def continuous_index_segment(x: Sequence[int], n: int | None = None, loop: bool = False) -> list[list[int]]:
    arr = list(x)
    if len(arr) <= 1:
        return [arr] if arr else []
    breaks = [0] + [i + 1 for i in range(len(arr) - 1) if arr[i + 1] - arr[i] > 1] + [len(arr)]
    lt = [arr[breaks[i] : breaks[i + 1]] for i in range(len(breaks) - 1)]
    if loop and len(lt) > 1 and n is not None:
        if lt[0][0] == 1 and lt[-1][-1] == n:
            lt[0] = lt[-1] + lt[0]
            lt = lt[:-1]
    return lt


def _normalize_factors(factors: Any, preserve_unique_order: bool = True) -> tuple[list[str], list[str]]:
    """Convert factors (Categorical, Series, ndarray, list) to (str_values, ordered_levels)."""
    if isinstance(factors, pd.Categorical):
        vals = [str(v) for v in factors]
        present = set(vals)
        levels = [str(c) for c in factors.categories if str(c) in present]
        return vals, levels

    if isinstance(factors, pd.Series) and isinstance(factors.dtype, pd.CategoricalDtype):
        vals = [str(v) for v in factors]
        present = set(vals)
        levels = [str(c) for c in factors.cat.categories if str(c) in present]
        return vals, levels

    raw = list(factors)
    if raw and all(isinstance(v, (int, float, np.number)) and not isinstance(v, (bool, np.bool_)) for v in raw):
        warnings.warn("Your `sectors` is numeric, it will be converted to characters internally.", UserWarning, stacklevel=3)

    vals = [str(v) for v in raw]
    if any(v == "" for v in vals):
        raise ValueError("`sectors` cannot contain empty strings.")

    uniq = list(dict.fromkeys(vals))
    if preserve_unique_order and len(vals) == len(uniq):
        levels = uniq
    else:
        levels = sorted(uniq)
    return vals, levels


def circos_initialize(
    sectors: Any = None,
    x: Any = None,
    xlim: Any = None,
    sector_width: Sequence[float] | float | None = None,
    factors: Any = None,
    ring: bool = False,
    ax: Any = None,
    **kwargs: Any,
) -> None:
    """Initialize the circular layout with sectors and x-ranges."""
    nkw = normalize_kwargs(kwargs)
    if "sector_width" in nkw:
        sector_width = nkw["sector_width"]
    if "factors" in nkw and factors is None:
        factors = nkw["factors"]
    if sectors is None and factors is not None:
        sectors = factors

    _STATE.reset_layout(keep_ax=(_STATE.canvas_overlay_scale is not None))

    if sectors is None:
        if isinstance(xlim, pd.DataFrame) and not isinstance(xlim.index, pd.RangeIndex):
            sectors = list(xlim.index)
        else:
            raise ValueError("Since `sectors` is not specified, row names (index) of `xlim` DataFrame are required.")

    vals, le = _normalize_factors(sectors, preserve_unique_order=True)
    n_sector = len(le)

    if ring:
        if n_sector != 1:
            raise ValueError("There should be only one sector under 'ring' mode.")
        circos_par.set_value("ring", True, _suppress_warn=True)
        circos_par.set_value("gap_degree", 0.0, _suppress_warn=True)
        cp = circos_par.get_value("cell.padding")
        circos_par.set_value("cell_padding", [cp[0], 0.0, cp[2], 0.0], _suppress_warn=True)
        circos_par.set_value("points_overflow_warning", False, _suppress_warn=True)
    else:
        circos_par.set_value("ring", False, _suppress_warn=True)

    # Compute min_value and max_value for each sector level
    if xlim is not None:
        if isinstance(xlim, pd.DataFrame):
            if xlim.shape[0] != n_sector or xlim.shape[1] != 2:
                raise ValueError("`xlim` DataFrame must have rows equal to number of sectors and 2 columns.")
            if not set(le) - set(xlim.index.astype(str)):
                xlim = xlim.copy()
                xlim.index = xlim.index.astype(str)
                xlim = xlim.loc[le]
            arr = xlim.to_numpy(dtype=float)
            min_value = arr[:, 0]
            max_value = arr[:, 1]
        else:
            arr = np.asarray(xlim, dtype=float)
            if arr.ndim == 1:
                if len(arr) != 2:
                    raise ValueError("Since `xlim` is 1D vector, it should have length of 2.")
                min_value = np.repeat(arr[0], n_sector)
                max_value = np.repeat(arr[1], n_sector)
            elif arr.ndim == 2:
                if arr.shape[0] != n_sector or arr.shape[1] != 2:
                    raise ValueError("Since `xlim` is a 2D matrix, it should have shape (n_sectors, 2).")
                min_value = arr[:, 0]
                max_value = arr[:, 1]
            else:
                raise ValueError("Invalid `xlim` dimensions.")
    elif x is not None:
        x_arr = np.asarray(x, dtype=float).ravel()
        if len(x_arr) != len(vals):
            raise ValueError("Length of `x` and length of `sectors` differ.")
        min_value = np.empty(n_sector, dtype=float)
        max_value = np.empty(n_sector, dtype=float)
        val_arr = np.asarray(vals, dtype=object)
        for i, lv in enumerate(le):
            sub = x_arr[val_arr == lv]
            min_value[i] = float(np.nanmin(sub))
            max_value[i] = float(np.nanmax(sub))
    else:
        raise ValueError("You should specify either `x` or `xlim`.")

    cell_padding = circos_par.get_value("cell.padding")
    sector_range = max_value - min_value

    gap_cfg = circos_par.get_value("gap.degree")
    if isinstance(gap_cfg, dict):
        if set(le).issubset(set(gap_cfg.keys())):
            gap_degree = np.array([float(gap_cfg[lv]) for lv in le], dtype=float)
        else:
            raise ValueError("Named `gap.degree` does not cover all sectors.")
    elif isinstance(gap_cfg, (list, tuple, np.ndarray)):
        gap_arr = np.asarray(gap_cfg, dtype=float).ravel()
        if len(gap_arr) == 1:
            gap_degree = np.repeat(gap_arr[0], n_sector)
        elif len(gap_arr) == n_sector:
            gap_degree = gap_arr
        else:
            raise ValueError("Since `gap.degree` has length > 1, it must match number of sectors.")
    else:
        gap_degree = np.repeat(float(gap_cfg), n_sector)

    if 360.0 - float(np.sum(gap_degree)) <= 0:
        raise ValueError("Maybe your `gap.degree` is too large so that there is no space to allocate sectors.")

    start_degree = float(circos_par.get_value("start.degree"))
    clock_wise = bool(circos_par.get_value("clock.wise"))

    s_deg = np.empty(n_sector, dtype=float)
    e_deg = np.empty(n_sector, dtype=float)

    if sector_width is None:
        unit = (360.0 - float(np.sum(gap_degree))) / float(np.sum(sector_range))
        for i in range(n_sector):
            if sector_range[i] == 0:
                raise ValueError(
                    f"Range of the sector ('{le[i]}') cannot be 0. "
                    "You might need to set `cell.padding = [0.02, 0, 0.02, 0]`."
                )
            if clock_wise:
                s_deg[i] = start_degree if i == 0 else e_deg[i - 1] - gap_degree[i - 1]
                e_deg[i] = s_deg[i] - sector_range[i] * unit
            else:
                e_deg[i] = start_degree if i == 0 else s_deg[i - 1] + gap_degree[i - 1]
                s_deg[i] = e_deg[i] + sector_range[i] * unit
    else:
        if np.isscalar(sector_width):
            sw = np.repeat(float(sector_width), n_sector)
        else:
            sw = np.asarray(sector_width, dtype=float).ravel()
            if len(sw) != n_sector:
                raise ValueError("Length of `sector.width` must be 1 or match the number of sectors.")
        deg_per_sector = (360.0 - float(np.sum(gap_degree))) * (sw / float(np.sum(sw)))
        if np.any(deg_per_sector <= 0):
            raise ValueError("Maybe your `gap.degree` is too large so that there is no space to allocate sectors.")
        for i in range(n_sector):
            if sector_range[i] == 0:
                raise ValueError(f"Range of the sector ({le[i]}) cannot be 0.")
            if clock_wise:
                s_deg[i] = start_degree if i == 0 else e_deg[i - 1] - gap_degree[i - 1]
                e_deg[i] = s_deg[i] - deg_per_sector[i]
            else:
                e_deg[i] = start_degree if i == 0 else s_deg[i - 1] + gap_degree[i - 1]
                s_deg[i] = e_deg[i] + deg_per_sector[i]

    if clock_wise:
        s_deg = s_deg + 360.0
        e_deg = e_deg + 360.0

    if np.any(cell_padding[1] + cell_padding[3] >= (s_deg - e_deg)):
        raise ValueError(
            "Summation of cell padding on x-direction is larger than the width for some sectors. "
            "Set e.g. `circos.par(cell_padding=[0.02, 0, 0.02, 0])`."
        )

    eff_deg = s_deg - e_deg - cell_padding[1] - cell_padding[3]
    real_min = min_value - cell_padding[1] / eff_deg * sector_range
    real_max = max_value + cell_padding[3] / eff_deg * sector_range

    _STATE.sector_order = list(le)
    _STATE.sector_data = {}
    _STATE.cell_data = {}
    for i, lv in enumerate(le):
        _STATE.sector_data[lv] = {
            "min.value": float(real_min[i]),
            "max.value": float(real_max[i]),
            "start.degree": float(s_deg[i]),
            "end.degree": float(e_deg[i]),
            "min.data": float(min_value[i]),
            "max.data": float(max_value[i]),
        }
        _STATE.cell_data[lv] = {}
    _STATE.current_sector_index = le[0]
    _STATE.current_track_index = 0

    ensure_canvas(ax=ax, initialize=True)


def _create_plot_region(
    track_start: float,
    track_height: float,
    sector_index: str,
    track_index: int,
    ylim: Sequence[float],
    bg_col: Any = None,
    bg_border: Any = "black",
    bg_lty: Any = 1,
    bg_lwd: float = 1.0,
    radius: Any = None,
) -> None:
    sdata = get_sector_data(sector_index)
    cell_xlim = [sdata["min.value"], sdata["max.value"]]
    xlim = [sdata["min.data"], sdata["max.data"]]
    cell_padding = circos_par.get_value("cell.padding")

    if cell_padding[0] + cell_padding[2] >= track_height:
        raise ValueError("Summation of cell padding on y-direction is larger than the height of the cells.")
    y0, y1 = float(ylim[0]), float(ylim[1])
    if y0 == y1:
        raise ValueError("Range of `ylim` should be different (ylim[0] != ylim[1]).")

    yl0 = y0 - (y1 - y0) * cell_padding[0] / track_height
    yl1 = y1 + (y1 - y0) * cell_padding[2] / track_height

    set_cell_data(
        sector_index=sector_index,
        track_index=track_index,
        xlim=xlim,
        ylim=[y0, y1],
        cell_xlim=cell_xlim,
        cell_ylim=[yl0, yl1],
        track_start=float(track_start),
        track_height=float(track_height),
        track_margin=circos_par.get_value("track.margin"),
        cell_padding=list(cell_padding),
        bg_col=bg_col,
        bg_border=bg_border,
        bg_lty=bg_lty,
        bg_lwd=bg_lwd,
    )
    set_current_sector_index(sector_index)

    if circos_par.get_value("ring"):
        highlight_sector(
            sector_index=sector_index,
            track_index=track_index,
            col=bg_col,
            border=bg_border,
            lty=bg_lty,
            lwd=bg_lwd,
        )
    else:
        from ._low_level import circos_rect

        circos_rect(
            cell_xlim[0],
            yl0,
            cell_xlim[1],
            yl1,
            sector_index=sector_index,
            track_index=track_index,
            col=bg_col,
            border=bg_border,
            lty=bg_lty,
            lwd=bg_lwd,
            radius=radius,
        )


def _call_panel_fun(panel_fun: Callable[..., Any], nx: np.ndarray | None, ny: np.ndarray | None) -> None:
    sig = inspect.signature(panel_fun)
    params = list(sig.parameters.values())
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
    pos_params = [p for p in params if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    if has_varargs or len(pos_params) >= 2:
        panel_fun(nx, ny)
    elif len(pos_params) == 1:
        panel_fun(nx)
    else:
        panel_fun()


def circos_track_plot_region(
    sectors: Any = None,
    x: Any = None,
    y: Any = None,
    ylim: Any = None,
    force_ylim: bool = True,
    track_index: int | None = None,
    track_height: float | None = None,
    track_margin: Sequence[float] | None = None,
    cell_padding: Sequence[float] | None = None,
    bg_col: Any = None,
    bg_border: Any = "black",
    bg_lty: Any = 1,
    bg_lwd: float = 1.0,
    panel_fun: Callable[..., Any] | None = None,
    radius: Any = None,
    factors: Any = None,
    **kwargs: Any,
) -> None:
    """Create plotting regions for a whole track (or update an existing track)."""
    if not is_circos_initialized():
        raise RuntimeError("Your circular plot has not been initialized yet!")

    nkw = normalize_kwargs(kwargs)
    if sectors is None and factors is not None:
        sectors = factors
    if "factors" in nkw and sectors is None:
        sectors = nkw["factors"]
    ylim = nkw.get("ylim", ylim)
    force_ylim = nkw.get("force_ylim", force_ylim)
    track_index = nkw.get("track_index", track_index)
    track_height = nkw.get("track_height", track_height)
    track_margin = nkw.get("track_margin", track_margin)
    cell_padding = nkw.get("cell_padding", cell_padding)
    bg_col = nkw.get("bg_col", bg_col)
    bg_border = nkw.get("bg_border", bg_border)
    bg_lty = nkw.get("bg_lty", bg_lty)
    bg_lwd = nkw.get("bg_lwd", bg_lwd)
    panel_fun = nkw.get("panel_fun", panel_fun)
    radius = nkw.get("radius", radius)

    if callable(sectors) or callable(x) or callable(y) or callable(ylim):
        raise TypeError("The panel function should be set explicitly with `panel_fun=...`.")

    if track_height is None:
        track_height = float(circos_par.get_value("track.height"))
    if track_margin is None:
        track_margin = circos_par.get_value("track.margin")
    if cell_padding is None:
        cell_padding = circos_par.get_value("cell.padding")

    o_track_margin = circos_par.get_value("track.margin")
    o_cell_padding = circos_par.get_value("cell.padding")
    circos_par.set_value("track_margin", track_margin, _suppress_warn=True)
    circos_par.set_value("cell_padding", cell_padding, _suppress_warn=True)

    try:
        all_le = get_all_sector_index()
        if sectors is None:
            factor_vals = list(all_le)
            le = list(all_le)
        else:
            factor_vals, le = _normalize_factors(sectors, preserve_unique_order=False)
            # Reorder le to match all_le order if not Categorical
            if not isinstance(sectors, (pd.Categorical, pd.Series)):
                le_set = set(le)
                le = [s for s in all_le if s in le_set]

        if y is not None:
            y_arr = np.asarray(y, dtype=float).ravel()
            if len(y_arr) != len(factor_vals):
                raise ValueError("Length of `y` and length of `sectors` differ.")
        else:
            y_arr = None

        if x is not None:
            x_arr = np.asarray(x, dtype=float).ravel()
            if len(x_arr) != len(factor_vals):
                raise ValueError("Length of `x` and length of `sectors` differ.")
        else:
            x_arr = None

        bad_sec = set(le) - set(all_le)
        if bad_sec:
            raise ValueError(f"Cannot find these categories in existing sectors: {sorted(bad_sec)}")

        tracks = get_all_track_index()
        last_track_index = tracks[-1] if tracks else 0
        flag_create_new = False
        if track_index is None:
            set_current_track_index(last_track_index + 1)
            track_index = get_current_track_index()
            flag_create_new = True
        elif int(track_index) == last_track_index + 1:
            track_index = int(track_index)
            set_current_track_index(track_index)
            flag_create_new = True
        else:
            track_index = int(track_index)
            if track_index > last_track_index + 1:
                raise ValueError(f"Wrong track index: it should be no more than {last_track_index + 1}.")
            ref_sec = factor_vals[0] if factor_vals else all_le[0]
            track_height = get_cell_meta_data("track.height", sector_index=ref_sec, track_index=track_index)
            circos_par.set_value("track_margin", get_cell_meta_data("track.margin", sector_index=ref_sec, track_index=track_index), _suppress_warn=True)
            circos_par.set_value("cell_padding", get_cell_meta_data("cell.padding", sector_index=ref_sec, track_index=track_index), _suppress_warn=True)
            if ylim is None and y_arr is None:
                ylim = np.vstack([get_cell_meta_data("ylim", sector_index=sid, track_index=track_index) for sid in all_le])
            set_current_track_index(track_index)

        nlevel = len(all_le)
        bg_col_l = recycle_with_levels(bg_col, all_le)
        bg_border_l = recycle_with_levels(bg_border, all_le)
        bg_lty_l = recycle_with_levels(bg_lty, all_le)
        bg_lwd_l = recycle_with_levels(bg_lwd, all_le)

        if ylim is None:
            if y_arr is None:
                raise ValueError("You have to specify either `y` or `ylim`.")
            if force_ylim:
                yr = [float(np.nanmin(y_arr)), float(np.nanmax(y_arr))]
                ylim_mat = np.tile(yr, (nlevel, 1))
            else:
                val_arr = np.asarray(factor_vals, dtype=object)
                ylim_mat = np.empty((nlevel, 2), dtype=float)
                last_yr = [float(np.nanmin(y_arr)), float(np.nanmax(y_arr))]
                for i, sid in enumerate(all_le):
                    sub = y_arr[val_arr == sid]
                    if len(sub) > 0:
                        last_yr = [float(np.nanmin(sub)), float(np.nanmax(sub))]
                    ylim_mat[i] = last_yr
        else:
            y_in = np.asarray(ylim, dtype=float)
            if y_in.ndim == 1 and len(y_in) == 2:
                ylim_mat = np.tile(y_in, (nlevel, 1))
            elif y_in.ndim == 2 and y_in.shape == (nlevel, 2):
                ylim_mat = y_in
            else:
                raise ValueError("Wrong `ylim` format. Expected [ymin, ymax] or shape (n_sectors, 2).")

        if flag_create_new:
            if track_index == 1:
                track_start = 1.0 - float(circos_par.get_value("track.margin")[1])
            else:
                prev_b = get_cell_meta_data("cell.bottom.radius", track_index=track_index - 1)
                prev_m = get_cell_meta_data("track.margin", track_index=track_index - 1)[0]
                curr_m = circos_par.get_value("track.margin")[1]
                track_start = float(prev_b - prev_m - curr_m)
            check_track_position(track_index, track_start, float(track_height))
        else:
            track_start = get_cell_meta_data("cell.top.radius", track_index=track_index)

        val_arr = np.asarray(factor_vals, dtype=object)
        le_set = set(le)
        for i, sid in enumerate(all_le):
            _create_plot_region(
                track_start=track_start,
                track_height=float(track_height),
                sector_index=sid,
                track_index=track_index,
                ylim=ylim_mat[i],
                bg_col=bg_col_l[i],
                bg_border=bg_border_l[i],
                bg_lty=bg_lty_l[i],
                bg_lwd=float(bg_lwd_l[i]),
                radius=radius,
            )
            if sid in le_set and panel_fun is not None:
                mask = val_arr == sid
                nx = x_arr[mask] if x_arr is not None else None
                ny = y_arr[mask] if y_arr is not None else None
                cur_t = get_current_track_index()
                _call_panel_fun(panel_fun, nx, ny)
                if get_current_track_index() != cur_t:
                    raise RuntimeError("`panel_fun` should not change the current track index.")
    finally:
        circos_par.set_value("track_margin", o_track_margin, _suppress_warn=True)
        circos_par.set_value("cell_padding", o_cell_padding, _suppress_warn=True)


def circos_track(*args: Any, **kwargs: Any) -> None:
    """Shortcut alias for `circos_track_plot_region`."""
    return circos_track_plot_region(*args, **kwargs)


def circos_update_plot_region(
    sector_index: str | None = None,
    track_index: int | None = None,
    bg_col: Any = None,
    bg_border: Any = "black",
    bg_lty: Any = 1,
    bg_lwd: float = 1.0,
    **kwargs: Any,
) -> None:
    """Erase and redraw background for an existing cell, setting it as current."""
    nkw = normalize_kwargs(kwargs)
    sector_index = nkw.get("sector_index", sector_index)
    track_index = nkw.get("track_index", track_index)
    bg_col = nkw.get("bg_col", bg_col)
    bg_border = nkw.get("bg_border", bg_border)
    bg_lty = nkw.get("bg_lty", bg_lty)
    bg_lwd = nkw.get("bg_lwd", bg_lwd)

    if sector_index is None:
        sector_index = get_cell_meta_data("sector.index")
    if track_index is None:
        track_index = get_cell_meta_data("track.index")

    from ._state import has_cell

    if not has_cell(sector_index, track_index):
        raise ValueError("You can only update an existing cell.")

    cx = get_cell_meta_data("cell.xlim", sector_index=sector_index, track_index=track_index)
    cy = get_cell_meta_data("cell.ylim", sector_index=sector_index, track_index=track_index)
    set_current_cell(sector_index, track_index)

    from ._low_level import circos_rect

    lwd_old = get_cell_meta_data("bg.lwd", sector_index=sector_index, track_index=track_index)
    circos_rect(cx[0], cy[0], cx[1], cy[1], col="white", border="white", lty=1, lwd=lwd_old)
    circos_rect(cx[0], cy[0], cx[1], cy[1], col=bg_col, border=bg_border, lty=bg_lty, lwd=bg_lwd)


def circos_update(*args: Any, **kwargs: Any) -> None:
    """Shortcut alias for `circos_update_plot_region`."""
    return circos_update_plot_region(*args, **kwargs)


def draw_sector(
    start_degree: float = 0.0,
    end_degree: float = 360.0,
    rou1: float = 1.0,
    rou2: float | None = None,
    center: Sequence[float] = (0.0, 0.0),
    clock_wise: bool = True,
    col: Any = None,
    border: Any = "black",
    lwd: float = 1.0,
    lty: Any = 1,
    ax: Any = None,
    **kwargs: Any,
) -> None:
    """Draw a sector, ring, or circular wedge in polar coordinates."""
    nkw = normalize_kwargs(kwargs)
    start_degree = float(nkw.get("start_degree", start_degree))
    end_degree = float(nkw.get("end_degree", end_degree))
    rou1 = float(nkw.get("rou1", rou1))
    rou2 = nkw.get("rou2", rou2)
    clock_wise = bool(nkw.get("clock_wise", clock_wise))

    target_ax = ensure_canvas(ax=ax, initialize=False)
    scale = _STATE.canvas_overlay_scale if _STATE.canvas_overlay_scale is not None else 1.0
    cx = float(center[0]) * scale
    cy = float(center[1]) * scale

    def is_circular(s: float, e: float) -> bool:
        return abs((e - s) % 360.0) < 1e-12 and abs(e - s) > 1e-12

    def degree_diff(s: float, e: float, cw: bool) -> float:
        if is_circular(s, e):
            return 360.0
        s_m = s % 360.0
        e_m = e % 360.0
        return (s_m - e_m) % 360.0 if cw else (e_m - s_m) % 360.0

    def degree_seq(s: float, e: float, cw: bool, length_out: int) -> np.ndarray:
        if is_circular(s, e):
            return np.linspace(0.0, 360.0, length_out)
        s_m = s % 360.0
        e_m = e % 360.0
        if cw:
            if s_m < e_m:
                s_m += 360.0
            return np.linspace(s_m, e_m, length_out)
        if s_m > e_m:
            s_m -= 360.0
        return np.linspace(s_m, e_m, length_out)

    seg_len = 2.0 * math.pi / float(circos_par.get_value("unit.circle.segments"))
    l1 = as_radian(degree_diff(start_degree, end_degree, clock_wise)) * rou1
    ncut1 = max(2, int(math.floor(l1 / seg_len)))
    d1 = np.column_stack([degree_seq(start_degree, end_degree, clock_wise, ncut1), np.repeat(rou1, ncut1)])

    fc = resolve_color(col)
    ec = resolve_color(border)
    ls = resolve_lty(lty)

    if rou2 is None:
        m1 = polar2Cartesian(d1)
        if not is_circular(start_degree, end_degree):
            m1 = np.vstack([m1, [0.0, 0.0]])
        m1[:, 0] += cx
        m1[:, 1] += cy
        patch = MplPolygon(m1, closed=True, facecolor=fc, edgecolor=ec, linewidth=lwd, linestyle=ls)
        target_ax.add_patch(patch)
    else:
        rou2_f = float(rou2)
        l2 = as_radian(degree_diff(start_degree, end_degree, clock_wise)) * rou2_f
        ncut2 = max(2, int(math.floor(l2 / seg_len)))
        d2 = np.column_stack([degree_seq(end_degree, start_degree, not clock_wise, ncut2), np.repeat(rou2_f, ncut2)])

        m1 = polar2Cartesian(d1)
        m2 = polar2Cartesian(d2)
        m1[:, 0] += cx
        m1[:, 1] += cy
        m2[:, 0] += cx
        m2[:, 1] += cy

        if is_circular(start_degree, end_degree):
            # Opposite winding for outer & inner ring so nonzero rule carves the hole seamlessly
            outer = m1
            inner = m2[::-1]
            if rou1 < rou2_f:
                outer, inner = inner, outer
            else:
                inner = m2[::-1]
            # Ensure outer is CCW (0->360) and inner is CW (360->0)
            verts = np.vstack([outer, outer[0], inner, inner[0]])
            codes = (
                [MplPath.MOVETO]
                + [MplPath.LINETO] * (len(outer) - 1)
                + [MplPath.CLOSEPOLY]
                + [MplPath.MOVETO]
                + [MplPath.LINETO] * (len(inner) - 1)
                + [MplPath.CLOSEPOLY]
            )
            path = MplPath(verts, codes)
            if fc != "none":
                target_ax.add_patch(PathPatch(path, facecolor=fc, edgecolor="none"))
            if ec != "none" and ls != "none":
                target_ax.add_line(Line2D(m1[:, 0], m1[:, 1], color=ec, linewidth=lwd, linestyle=ls))
                target_ax.add_line(Line2D(m2[:, 0], m2[:, 1], color=ec, linewidth=lwd, linestyle=ls))
        else:
            m = np.vstack([m1, m2])
            patch = MplPolygon(m, closed=True, facecolor=fc, edgecolor=ec, linewidth=lwd, linestyle=ls)
            target_ax.add_patch(patch)


def highlight_sector(
    sector_index: str | Sequence[str],
    track_index: int | Sequence[int] | None = None,
    col: Any = "#FF000040",
    border: Any = None,
    lwd: float = 1.0,
    lty: Any = 1,
    padding: Sequence[float] | AsIs = (0.0, 0.0, 0.0, 0.0),
    text: str | None = None,
    text_col: Any = "black",
    text_vjust: float | str = 0.5,
    **kwargs: Any,
) -> None:
    """Highlight specified sectors and tracks with colored regions and optional text."""
    nkw = normalize_kwargs(kwargs)
    track_index = nkw.pop("track_index", track_index)
    text_col = nkw.pop("text_col", text_col)
    text_vjust = nkw.pop("text_vjust", text_vjust)

    sectors = get_all_sector_index()
    sec_in = [sector_index] if isinstance(sector_index, str) else [str(s) for s in sector_index]
    if not set(sec_in).issubset(set(sectors)):
        raise ValueError("`sector_index` contains index that does not belong to available sectors.")

    tracks = get_all_track_index()
    if track_index is None:
        trk_in = list(tracks)
    elif isinstance(track_index, (int, np.integer)):
        trk_in = [int(track_index)]
    else:
        trk_in = [int(t) for t in track_index]
    if not set(trk_in).issubset(set(tracks)):
        raise ValueError("`track_index` contains index that does not belong to available tracks.")

    y_offset = None
    if isinstance(text_vjust, str):
        y_offset = parse_unit(text_vjust)
        text_vjust_f = 0.5
    else:
        text_vjust_f = float(text_vjust)

    if isinstance(padding, AsIs):
        if len(sec_in) > 1:
            raise ValueError("`padding` can be set with I()/AsIs only when there is one single sector.")
        pval = [float(v) for v in padding.value]
        cx = get_cell_meta_data("cell.xlim", sector_index=sec_in[0])
        cy = get_cell_meta_data("cell.ylim", sector_index=sec_in[0])
        pad = [
            pval[0] / float(cy[1] - cy[0]),
            pval[1] / float(cx[1] - cx[0]),
            pval[2] / float(cy[1] - cy[0]),
            pval[3] / float(cx[1] - cx[0]),
        ]
    else:
        pad = [float(v) for v in padding]

    from ._low_level import circos_text
    from ._utils import reverse_circlize

    def _draw_text_if_any(sd: float, ed: float, r1: float, r2: float) -> None:
        if text is None:
            return
        pos = reverse_circlize((sd + ed) / 2.0 + (180.0 if sd < ed else 0.0), (r1 + r2) / 2.0)
        op_warn = circos_par.get_value("points.overflow.warning")
        circos_par.set_value("points_overflow_warning", False, _suppress_warn=True)
        try:
            if y_offset is None:
                circos_text(pos[0, 0], pos[0, 1], text, adj=[0.5, text_vjust_f], col=text_col, **nkw)
            else:
                dy = uy(y_offset[0], unit=y_offset[1])
                circos_text(pos[0, 0], pos[0, 1] + dy, text, adj=[0.5, 0.5], col=text_col, **nkw)
        finally:
            circos_par.set_value("points_overflow_warning", op_warn, _suppress_warn=True)

    trk_sorted = sorted(set(trk_in))
    ts = continuous_index_segment(trk_sorted)

    if set(sectors).issubset(set(sec_in)):
        for tvec in ts:
            sd, ed = 0.0, 360.0
            rou1 = get_cell_meta_data("cell.top.radius", sectors[0], tvec[0])
            rou2 = get_cell_meta_data("cell.bottom.radius", sectors[0], tvec[-1])
            d2 = rou1 - rou2
            rou1 = rou1 + d2 * pad[2]
            rou2 = rou2 - d2 * pad[0]
            draw_sector(start_degree=sd, end_degree=ed, rou1=rou1, rou2=rou2, col=col, border=border, lwd=lwd, lty=lty)
            _draw_text_if_any(sd, ed, rou1, rou2)
    else:
        sec_num = [i + 1 for i, s in enumerate(sectors) if s in set(sec_in)]
        ss = continuous_index_segment(sec_num, n=len(sectors), loop=True)
        clock_wise = bool(circos_par.get_value("clock.wise"))
        for s_idx_vec in ss:
            svec = [sectors[k - 1] for k in s_idx_vec]
            for tvec in ts:
                if clock_wise:
                    sd = get_sector_data(svec[0])["start.degree"]
                    ed = get_sector_data(svec[-1])["end.degree"]
                else:
                    ed = get_sector_data(svec[0])["end.degree"]
                    sd = get_sector_data(svec[-1])["start.degree"]
                rou1 = get_cell_meta_data("cell.top.radius", svec[0], tvec[0])
                rou2 = get_cell_meta_data("cell.bottom.radius", svec[0], tvec[-1])
                d1 = abs(ed - sd)
                d2 = rou1 - rou2
                sd = sd + d1 * pad[1]
                ed = ed - d1 * pad[3]
                rou1 = rou1 + d2 * pad[2]
                rou2 = rou2 - d2 * pad[0]
                draw_sector(start_degree=sd, end_degree=ed, rou1=rou1, rou2=rou2, col=col, border=border, lwd=lwd, lty=lty)
                _draw_text_if_any(sd, ed, rou1, rou2)


def highlight_chromosome(*args: Any, **kwargs: Any) -> None:
    """Shortcut alias for `highlight_sector`."""
    return highlight_sector(*args, **kwargs)


def set_track_gap(gap: float = 0.02) -> None:
    """Set bottom/top gap between tracks."""
    circos_par.set_value("track_margin", [0.0, float(gap)], _suppress_warn=True)
