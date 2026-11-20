"""Low-level and track-level circular graphics primitives."""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence
import warnings

from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Affine2D
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import gaussian_kde

from ._canvas import (
    BASE_FONTSIZE_PT,
    ensure_canvas,
    get_canvas_pt_per_inch,
    parse_unit,
    strheight_canvas,
    strheight_inches,
    strwidth_canvas,
    strwidth_inches,
)
from ._colors import (
    is_na_like,
    resolve_color,
    resolve_colors,
    resolve_font,
    resolve_lty,
    resolve_pch,
)
from ._plot import _normalize_factors
from ._state import (
    _STATE,
    circos_par,
    get_all_sector_index,
    get_cell_meta_data,
    get_current_sector_index,
    get_current_track_index,
    get_sector_data,
    has_cell,
    normalize_kwargs,
    set_current_cell,
    set_current_sector_index,
)
from ._utils import (
    DegreeValue,
    as_degree,
    check_points_position,
    circlize,
    circos_roundrect_pos,
    circos_str_ascent,
    circos_str_descent,
    circos_strheight,
    circos_strwidth,
    cm_h,
    convert_length,
    convert_x,
    lines_expand,
    mm_x,
    mm_y,
    polar2Cartesian,
    recycle_to_length,
    recycle_with_factors,
    recycle_with_levels,
    refer_to_one_sector,
    reverse_circlize,
    stack_text,
)


def _split_nan_paths(xy: np.ndarray) -> list[np.ndarray]:
    """Split a (N, 2) coordinate array at NaN rows into contiguous sub-paths."""
    if len(xy) == 0:
        return []
    bad = np.isnan(xy[:, 0]) | np.isnan(xy[:, 1])
    if not np.any(bad):
        return [xy]
    paths: list[np.ndarray] = []
    start = 0
    for idx in np.where(bad)[0]:
        if idx > start:
            paths.append(xy[start:idx])
        start = idx + 1
    if start < len(xy):
        paths.append(xy[start:])
    return paths


def _extract_xy_args(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    if y is None:
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, 0].ravel(), arr[:, 1].ravel()
        raise ValueError("If `y` is not specified, `x` must be a 2-column array.")
    x_arr = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    y_arr = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    if len(x_arr) == 1 and len(y_arr) > 1:
        x_arr = np.repeat(x_arr, len(y_arr))
    elif len(y_arr) == 1 and len(x_arr) > 1:
        y_arr = np.repeat(y_arr, len(x_arr))
    if len(x_arr) != len(y_arr):
        raise ValueError("Length of `x` and `y` differ.")
    return x_arr, y_arr


# ---------------------------------------------------------------------------
# Points
# ---------------------------------------------------------------------------

def circos_points(
    x: Any,
    y: Any = None,
    sector_index: str | None = None,
    track_index: int | None = None,
    pch: Any = 1,
    col: Any = "black",
    cex: float | Sequence[float] = 1.0,
    bg: Any = None,
    **kwargs: Any,
) -> None:
    """Add points to a plotting region."""
    nkw = normalize_kwargs(kwargs)
    sector_index = nkw.get("sector_index", sector_index)
    track_index = nkw.get("track_index", track_index)
    if sector_index is None:
        sector_index = get_current_sector_index()
    if track_index is None:
        track_index = get_current_track_index()

    if not has_cell(sector_index, track_index):
        raise RuntimeError("'circos.points' can only be used after the plotting region has been created.")

    x_arr, y_arr = _extract_xy_args(x, y)
    n = len(x_arr)
    if n == 0:
        return

    check_points_position(x_arr, y_arr, sector_index, track_index)
    d = circlize(x_arr, y_arr, sector_index=sector_index, track_index=track_index)
    cart = polar2Cartesian(d)
    ax = ensure_canvas(initialize=False)

    pchs = recycle_to_length(pch, n)
    cols = resolve_colors(col, n, default="black")
    cexs = [float(v) for v in recycle_to_length(cex, n)]
    bgs = resolve_colors(bg, n, default="none")

    # Group by (marker, mode) for fast batched scatter rendering
    groups: dict[tuple[str, str], list[int]] = {}
    for i in range(n):
        m, mode = resolve_pch(pchs[i])
        groups.setdefault((m, mode), []).append(i)

    for (marker, mode), idxs in groups.items():
        pts = cart[idxs]
        sz_base = 4.5 if mode == "filled_small" else 6.0
        sizes = [(cexs[i] * sz_base) ** 2 for i in idxs]
        c_sub = [cols[i] for i in idxs]
        bg_sub = [bgs[i] for i in idxs]
        if mode == "open":
            ax.scatter(pts[:, 0], pts[:, 1], s=sizes, marker=marker, facecolors="none", edgecolors=c_sub, linewidths=0.9)
        elif mode == "line_only":
            ax.scatter(pts[:, 0], pts[:, 1], s=sizes, marker=marker, c=c_sub, linewidths=1.0)
        elif mode == "bg_filled":
            ax.scatter(pts[:, 0], pts[:, 1], s=sizes, marker=marker, facecolors=bg_sub, edgecolors=c_sub, linewidths=0.9)
        else:  # filled / filled_small
            ax.scatter(pts[:, 0], pts[:, 1], s=sizes, marker=marker, facecolors=c_sub, edgecolors="none")


def circos_track_points(
    sectors: Any,
    x: Any,
    y: Any,
    track_index: int | None = None,
    pch: Any = 1,
    col: Any = "black",
    cex: Any = 1.0,
    bg: Any = None,
    factors: Any = None,
    **kwargs: Any,
) -> None:
    """Add points across multiple sectors on the same track."""
    nkw = normalize_kwargs(kwargs)
    track_index = nkw.get("track_index", track_index)
    if sectors is None and factors is not None:
        sectors = factors
    if "factors" in nkw and sectors is None:
        sectors = nkw["factors"]
    if track_index is None:
        track_index = get_current_track_index()

    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    f_vals, le = _normalize_factors(sectors, preserve_unique_order=False)
    if len(x_arr) != len(f_vals) or len(y_arr) != len(f_vals):
        raise ValueError("Length of data and length of sectors differ.")

    all_sec = set(get_all_sector_index())
    bad = set(le) - all_sec
    if bad:
        raise ValueError(f"Cannot find these categories in existing sectors: {sorted(bad)}")

    pchs = recycle_with_factors(pch, f_vals, le)
    cols = recycle_with_factors(col, f_vals, le)
    cexs = recycle_with_factors(cex, f_vals, le)
    bgs = recycle_with_factors(bg, f_vals, le)

    f_arr = np.asarray(f_vals, dtype=object)
    for lv in le:
        idx = np.where(f_arr == lv)[0]
        circos_points(
            x_arr[idx],
            y_arr[idx],
            sector_index=lv,
            track_index=track_index,
            pch=[pchs[i] for i in idx],
            col=[cols[i] for i in idx],
            cex=[cexs[i] for i in idx],
            bg=[bgs[i] for i in idx],
        )


# ---------------------------------------------------------------------------
# Lines, Segments, Rectangles, Polygons, Triangles
# ---------------------------------------------------------------------------

def circos_lines(
    x: Any,
    y: Any = None,
    sector_index: str | None = None,
    track_index: int | None = None,
    col: Any = None,
    lwd: float | Sequence[float] = 1.0,
    lty: Any = 1,
    type: str = "l",
    straight: bool = False,
    area: bool = False,
    area_baseline: Any = None,
    border: Any = "black",
    baseline: Any = "bottom",
    pt_col: Any = "black",
    cex: Any = 1.0,
    pch: Any = 1,
    **kwargs: Any,
) -> None:
    """Add lines or filled areas to a cell."""
    nkw = normalize_kwargs(kwargs)
    sector_index = nkw.get("sector_index", sector_index)
    track_index = nkw.get("track_index", track_index)
    area_baseline = nkw.get("area_baseline", area_baseline)
    pt_col = nkw.get("pt_col", pt_col)

    if sector_index is None:
        sector_index = get_current_sector_index()
    if track_index is None:
        track_index = get_current_track_index()

    if not has_cell(sector_index, track_index):
        raise RuntimeError("'circos.lines' can only be used after the plotting region has been created.")

    if area_baseline is not None:
        baseline = area_baseline
        warnings.warn("`area.baseline` is deprecated, please use `baseline` instead.", DeprecationWarning, stacklevel=2)

    if col is None:
        col = "grey" if area else "black"

    x_arr, y_arr = _extract_xy_args(x, y)
    n = len(x_arr)
    if n == 0:
        return

    if circos_par.get_value("ring") and n > 1:
        diff_mask = np.append(x_arr[:-1] - x_arr[1:] > 1e-10, False)
        if np.any(diff_mask):
            x_arr = x_arr.copy()
            x_arr[diff_mask] = x_arr[diff_mask] - get_cell_meta_data("xrange", sector_index, track_index)

    if baseline == "bottom":
        base_val = float(get_cell_meta_data("ylim", sector_index, track_index)[0])
    elif baseline == "top":
        base_val = float(get_cell_meta_data("ylim", sector_index, track_index)[1])
    else:
        base_val = float(baseline)

    if type == "o":
        circos_points(x_arr, y_arr, sector_index=sector_index, track_index=track_index, col=pt_col, cex=cex, pch=pch)
        circos_lines(x_arr, y_arr, sector_index=sector_index, track_index=track_index, col=col, lwd=lwd, lty=lty, area=area, border=border, baseline=base_val)
        return
    if type == "h":
        cols = recycle_to_length(col, n)
        lwds = recycle_to_length(lwd, n)
        ltys = recycle_to_length(lty, n)
        for i in range(n):
            circos_lines(
                [x_arr[i], x_arr[i]],
                [base_val, y_arr[i]],
                sector_index=sector_index,
                track_index=track_index,
                col=cols[i],
                lwd=lwds[i],
                lty=ltys[i],
                straight=True,
            )
        return
    if type == "s":
        parts = []
        for i in range(1, n):
            parts.append(lines_expand([x_arr[i - 1], x_arr[i]], [y_arr[i - 1], y_arr[i - 1]], sector_index, track_index))
            parts.append(np.array([[x_arr[i], y_arr[i - 1]], [x_arr[i], y_arr[i]]], dtype=float))
        d = np.vstack(parts) if parts else np.column_stack([x_arr, y_arr])
        if area:
            d = np.vstack([d, [d[-1, 0], base_val], [d[0, 0], base_val]])
            circos_polygon(d[:, 0], d[:, 1], sector_index=sector_index, track_index=track_index, col=col, border=border, lwd=lwd, lty=lty)
        else:
            circos_lines(d[:, 0], d[:, 1], sector_index=sector_index, track_index=track_index, col=col, lwd=lwd, lty=lty, straight=True)
        return

    check_points_position(x_arr, y_arr, sector_index, track_index)
    d = np.column_stack([x_arr, y_arr]) if straight else lines_expand(x_arr, y_arr, sector_index, track_index)

    if area:
        d = np.vstack([d, [d[-1, 0], base_val], [d[0, 0], base_val]])
        circos_polygon(d[:, 0], d[:, 1], sector_index=sector_index, track_index=track_index, col=col, border=border, lwd=lwd, lty=lty)
        return

    d2 = circlize(d[:, 0], d[:, 1], sector_index=sector_index, track_index=track_index)
    cart = polar2Cartesian(d2)
    ax = ensure_canvas(initialize=False)
    c_val = resolve_color(col, default="black")
    ls = resolve_lty(lty)
    if ls == "none" or c_val == "none":
        return
    lw = float(lwd[0] if isinstance(lwd, (list, tuple, np.ndarray)) else lwd)
    for seg in _split_nan_paths(cart):
        ax.add_line(Line2D(seg[:, 0], seg[:, 1], color=c_val, linewidth=lw, linestyle=ls, solid_capstyle="butt"))


def circos_track_lines(
    sectors: Any,
    x: Any,
    y: Any,
    track_index: int | None = None,
    col: Any = "black",
    lwd: Any = 1.0,
    lty: Any = 1,
    type: str = "l",
    straight: bool = False,
    area: Any = False,
    area_baseline: Any = None,
    border: Any = "black",
    baseline: Any = "bottom",
    pt_col: Any = "black",
    cex: Any = 1.0,
    pch: Any = 1,
    factors: Any = None,
    **kwargs: Any,
) -> None:
    """Add lines across multiple sectors on the same track."""
    nkw = normalize_kwargs(kwargs)
    track_index = nkw.get("track_index", track_index)
    area_baseline = nkw.get("area_baseline", area_baseline)
    pt_col = nkw.get("pt_col", pt_col)
    if sectors is None and factors is not None:
        sectors = factors
    if "factors" in nkw and sectors is None:
        sectors = nkw["factors"]
    if track_index is None:
        track_index = get_current_track_index()

    if area_baseline is not None:
        baseline = area_baseline

    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    f_vals, le = _normalize_factors(sectors, preserve_unique_order=False)
    if len(x_arr) != len(f_vals) or len(y_arr) != len(f_vals):
        raise ValueError("Length of data and length of sectors differ.")

    bad = set(le) - set(get_all_sector_index())
    if bad:
        raise ValueError(f"Cannot find these categories in existing sectors: {sorted(bad)}")

    cols = recycle_with_factors(col, f_vals, le)
    lwds = recycle_with_factors(lwd, f_vals, le)
    ltys = recycle_with_factors(lty, f_vals, le)
    pt_cols = recycle_with_factors(pt_col, f_vals, le)
    cexs = recycle_with_factors(cex, f_vals, le)
    pchs = recycle_with_factors(pch, f_vals, le)

    areas = recycle_with_levels(area, le)
    baselines = recycle_with_levels(baseline, le)
    borders = recycle_with_levels(border, le)

    f_arr = np.asarray(f_vals, dtype=object)
    for k, lv in enumerate(le):
        idx = np.where(f_arr == lv)[0]
        circos_lines(
            x_arr[idx],
            y_arr[idx],
            sector_index=lv,
            track_index=track_index,
            col=cols[idx[0]],
            lwd=lwds[idx[0]],
            lty=ltys[idx[0]],
            area=bool(areas[k]),
            border=borders[k],
            baseline=baselines[k],
            pt_col=[pt_cols[i] for i in idx],
            cex=[cexs[i] for i in idx],
            pch=[pchs[i] for i in idx],
            type=type,
            straight=straight,
        )


def circos_polygon(
    x: Any,
    y: Any = None,
    sector_index: str | None = None,
    track_index: int | None = None,
    col: Any = None,
    border: Any = "black",
    lwd: float = 1.0,
    lty: Any = 1,
    **kwargs: Any,
) -> None:
    """Draw polygon(s) in a cell (supports NaN separators for multiple polygons)."""
    nkw = normalize_kwargs(kwargs)
    sector_index = nkw.get("sector_index", sector_index)
    track_index = nkw.get("track_index", track_index)
    if sector_index is None:
        sector_index = get_current_sector_index()
    if track_index is None:
        track_index = get_current_track_index()

    if not has_cell(sector_index, track_index):
        raise RuntimeError("'circos.polygon' can only be used after the plotting region has been created.")

    x_arr, y_arr = _extract_xy_args(x, y)
    if len(x_arr) == 0:
        return

    check_points_position(x_arr, y_arr, sector_index, track_index)
    d = lines_expand(x_arr, y_arr, sector_index, track_index)
    d2 = circlize(d, sector_index=sector_index, track_index=track_index)
    cart = polar2Cartesian(d2)
    polys = _split_nan_paths(cart)
    if not polys:
        return

    ax = ensure_canvas(initialize=False)
    is_col_list = isinstance(col, (list, tuple, np.ndarray, pd.Series)) and not (
        isinstance(col, tuple) and len(col) in (3, 4) and all(isinstance(v, (int, float, np.number)) for v in col)
    )
    is_brd_list = isinstance(border, (list, tuple, np.ndarray, pd.Series)) and not (
        isinstance(border, tuple) and len(border) in (3, 4) and all(isinstance(v, (int, float, np.number)) for v in border)
    )

    if len(polys) == 1 and not is_col_list and not is_brd_list:
        fc = resolve_color(col)
        ec = resolve_color(border)
        ls = resolve_lty(lty)
        if ls == "none":
            ec = "none"
        ax.add_patch(MplPolygon(polys[0], closed=True, facecolor=fc, edgecolor=ec, linewidth=float(lwd), linestyle=ls))
    else:
        np_cnt = len(polys)
        fcs = resolve_colors(col, np_cnt)
        ecs = resolve_colors(border, np_cnt)
        ls = resolve_lty(lty)
        if ls == "none":
            ecs = ["none"] * np_cnt
        coll = PolyCollection(polys, facecolors=fcs, edgecolors=ecs, linewidths=float(lwd), linestyles=ls)
        ax.add_collection(coll)


def _rotate_points(x: np.ndarray, y: np.ndarray, cx: np.ndarray, cy: np.ndarray, theta_rad: float) -> tuple[np.ndarray, np.ndarray]:
    x2 = x - cx
    y2 = y - cy
    rho = np.hypot(x2, y2)
    alpha = np.arctan2(y2, x2)
    beta = alpha + theta_rad
    return rho * np.cos(beta) + cx, rho * np.sin(beta) + cy


def circos_rect(
    xleft: Any = None,
    ybottom: Any = None,
    xright: Any = None,
    ytop: Any = None,
    sector_index: str | None = None,
    track_index: int | None = None,
    radius: Any = None,
    rot: float = 0.0,
    col: Any = None,
    border: Any = "black",
    lwd: float = 1.0,
    lty: Any = 1,
    **kwargs: Any,
) -> None:
    """Draw rectangle-like grid(s) in polar coordinates (vectorized via PolyCollection)."""
    nkw = normalize_kwargs(kwargs)
    sector_index = nkw.get("sector_index", sector_index)
    track_index = nkw.get("track_index", track_index)
    if sector_index is None:
        sector_index = get_current_sector_index()
    if track_index is None:
        track_index = get_current_track_index()

    if not has_cell(sector_index, track_index):
        raise RuntimeError("'circos.rect' can only be used after the plotting region has been created.")

    if xleft is None and ybottom is None and xright is None and ytop is None:
        cx = get_cell_meta_data("cell.xlim", sector_index=sector_index, track_index=track_index)
        cy = get_cell_meta_data("cell.ylim", sector_index=sector_index, track_index=track_index)
        xleft, ybottom, xright, ytop = cx[0], cy[0], cx[1], cy[1]

    xl = np.atleast_1d(np.asarray(xleft, dtype=float)).ravel()
    yb = np.atleast_1d(np.asarray(ybottom, dtype=float)).ravel()
    xr = np.atleast_1d(np.asarray(xright, dtype=float)).ravel()
    yt = np.atleast_1d(np.asarray(ytop, dtype=float)).ravel()
    np_cnt = max(len(xl), len(yb), len(xr), len(yt))
    if np_cnt == 0:
        return
    if len(xl) == 1:
        xl = np.repeat(xl, np_cnt)
    if len(yb) == 1:
        yb = np.repeat(yb, np_cnt)
    if len(xr) == 1:
        xr = np.repeat(xr, np_cnt)
    if len(yt) == 1:
        yt = np.repeat(yt, np_cnt)
    if not (len(xl) == len(yb) == len(xr) == len(yt)):
        raise ValueError("xleft, ybottom, xright, ytop should have the same length.")

    if circos_par.get_value("ring"):
        mask = xl > xr
        if np.any(mask):
            xl = xl.copy()
            xl[mask] = xl[mask] - get_cell_meta_data("xrange", sector_index, track_index)

    if radius is not None:
        xs: list[float] = []
        ys: list[float] = []
        for i in range(np_cnt):
            px, py = circos_roundrect_pos(xl[i], yb[i], xr[i], yt[i], radius=radius, sector_index=sector_index, track_index=track_index)
            xs.extend(px.tolist())
            ys.extend(py.tolist())
            if i < np_cnt - 1:
                xs.append(math.nan)
                ys.append(math.nan)
        circos_polygon(xs, ys, sector_index=sector_index, track_index=track_index, col=col, border=border, lwd=lwd, lty=lty)
        return

    if rot != 0.0:
        cx_r = (xl + xr) / 2.0
        cy_r = (yb + yt) / 2.0
        rad = rot / 180.0 * math.pi
        q1x, q1y = _rotate_points(xl, yb, cx_r, cy_r, rad)
        q2x, q2y = _rotate_points(xr, yb, cx_r, cy_r, rad)
        q3x, q3y = _rotate_points(xr, yt, cx_r, cy_r, rad)
        q4x, q4y = _rotate_points(xl, yt, cx_r, cy_r, rad)
        xs = []
        ys = []
        for i in range(np_cnt):
            xs.extend([q1x[i], q2x[i], q3x[i], q4x[i], q1x[i]])
            ys.extend([q1y[i], q2y[i], q3y[i], q4y[i], q1y[i]])
            if i < np_cnt - 1:
                xs.append(math.nan)
                ys.append(math.nan)
        circos_polygon(xs, ys, sector_index=sector_index, track_index=track_index, col=col, border=border, lwd=lwd, lty=lty)
        return

    # Fast vectorized arc-rectangle path generation for rot == 0 and radius is None
    check_points_position(np.concatenate([xl, xr]), np.concatenate([yb, yt]), sector_index, track_index)
    sdata = get_sector_data(sector_index)
    deg_span = abs(sdata["start.degree"] - sdata["end.degree"])
    val_span = sdata["max.value"] - sdata["min.value"]
    seg_rad = 2.0 * math.pi / float(circos_par.get_value("unit.circle.segments"))

    # Pre-check ncut per rectangle from x-span in radians
    arc_rad = np.abs(xr - xl) / val_span * deg_span / 180.0 * math.pi
    ncuts = np.floor(arc_rad / seg_rad).astype(int)

    polys: list[np.ndarray] = []
    # Fast path if all ncuts == 0 (tiny cells like high-res rasters/heatmaps)
    if np.all(ncuts == 0):
        all_x = np.column_stack([xl, xl, xr, xr]).ravel()
        all_y = np.column_stack([yb, yt, yt, yb]).ravel()
        d2 = circlize(all_x, all_y, sector_index=sector_index, track_index=track_index)
        cart = polar2Cartesian(d2).reshape(np_cnt, 4, 2)
        polys = [cart[i] for i in range(np_cnt)]
    else:
        for i in range(np_cnt):
            nc = int(ncuts[i])
            if nc <= 0:
                rx = np.array([xl[i], xl[i], xr[i], xr[i]], dtype=float)
                ry = np.array([yb[i], yt[i], yt[i], yb[i]], dtype=float)
            else:
                ts = np.linspace(0.0, 1.0, nc + 2)
                top_x = xl[i] + (xr[i] - xl[i]) * ts
                top_y = np.repeat(yt[i], nc + 2)
                bot_x = xr[i] + (xl[i] - xr[i]) * ts
                bot_y = np.repeat(yb[i], nc + 2)
                rx = np.concatenate([top_x, bot_x])
                ry = np.concatenate([top_y, bot_y])
            d2 = circlize(rx, ry, sector_index=sector_index, track_index=track_index)
            polys.append(polar2Cartesian(d2))

    ax = ensure_canvas(initialize=False)
    fcs = resolve_colors(col, np_cnt)
    ecs = resolve_colors(border, np_cnt)
    ls = resolve_lty(lty)
    if ls == "none":
        ecs = ["none"] * np_cnt
    # Skip drawing completely invisible rectangles
    if all(fc == "none" for fc in fcs) and all(ec == "none" for ec in ecs):
        return
    coll = PolyCollection(polys, facecolors=fcs, edgecolors=ecs, linewidths=float(lwd), linestyles=ls)
    ax.add_collection(coll)


def circos_triangle(
    x1: Any,
    y1: Any,
    x2: Any,
    y2: Any,
    x3: Any,
    y3: Any,
    **kwargs: Any,
) -> None:
    """Draw triangle(s) in a cell."""
    a1 = np.atleast_1d(np.asarray(x1, dtype=float)).ravel()
    b1 = np.atleast_1d(np.asarray(y1, dtype=float)).ravel()
    a2 = np.atleast_1d(np.asarray(x2, dtype=float)).ravel()
    b2 = np.atleast_1d(np.asarray(y2, dtype=float)).ravel()
    a3 = np.atleast_1d(np.asarray(x3, dtype=float)).ravel()
    b3 = np.atleast_1d(np.asarray(y3, dtype=float)).ravel()
    n = max(len(a1), len(b1), len(a2), len(b2), len(a3), len(b3))
    if len(a1) == 1: a1 = np.repeat(a1, n)
    if len(b1) == 1: b1 = np.repeat(b1, n)
    if len(a2) == 1: a2 = np.repeat(a2, n)
    if len(b2) == 1: b2 = np.repeat(b2, n)
    if len(a3) == 1: a3 = np.repeat(a3, n)
    if len(b3) == 1: b3 = np.repeat(b3, n)

    xs: list[float] = []
    ys: list[float] = []
    for i in range(n):
        xs.extend([a1[i], a2[i], a3[i], a1[i]])
        ys.extend([b1[i], b2[i], b3[i], b1[i]])
        if i < n - 1:
            xs.append(math.nan)
            ys.append(math.nan)
    circos_polygon(xs, ys, **kwargs)


def circos_segments(
    x0: Any,
    y0: Any,
    x1: Any,
    y1: Any,
    sector_index: str | None = None,
    track_index: int | None = None,
    straight: bool | Sequence[bool] = False,
    col: Any = "black",
    lwd: Any = 1.0,
    lty: Any = 1,
    **kwargs: Any,
) -> None:
    """Draw segments between pairs of points (x0, y0) -> (x1, y1)."""
    nkw = normalize_kwargs(kwargs)
    sector_index = nkw.get("sector_index", sector_index)
    track_index = nkw.get("track_index", track_index)
    if sector_index is None:
        sector_index = get_current_sector_index()
    if track_index is None:
        track_index = get_current_track_index()

    if not has_cell(sector_index, track_index):
        raise RuntimeError("'circos.segments' can only be used after the plotting region has been created.")

    a0 = np.atleast_1d(np.asarray(x0, dtype=float)).ravel()
    b0 = np.atleast_1d(np.asarray(y0, dtype=float)).ravel()
    a1 = np.atleast_1d(np.asarray(x1, dtype=float)).ravel()
    b1 = np.atleast_1d(np.asarray(y1, dtype=float)).ravel()
    n = max(len(a0), len(b0), len(a1), len(b1))
    if n == 0:
        return
    if len(a0) == 1: a0 = np.repeat(a0, n)
    if len(b0) == 1: b0 = np.repeat(b0, n)
    if len(a1) == 1: a1 = np.repeat(a1, n)
    if len(b1) == 1: b1 = np.repeat(b1, n)
    if not (len(a0) == len(b0) == len(a1) == len(b1)):
        raise ValueError("x0, y0, x1, y1 should have the same length.")

    if circos_par.get_value("ring"):
        mask = (a0 - a1) > 1e-10
        if np.any(mask):
            a0 = a0.copy()
            a0[mask] = a0[mask] - get_cell_meta_data("xrange", sector_index, track_index)

    straights = [bool(v) for v in recycle_to_length(straight, n)]
    cols = resolve_colors(col, n, default="black")
    lwds = [float(v) for v in recycle_to_length(lwd, n)]
    ltys = [resolve_lty(v) for v in recycle_to_length(lty, n)]

    segs: list[np.ndarray] = []
    seg_cols: list[Any] = []
    seg_lwds: list[float] = []
    seg_ltys: list[Any] = []

    for i in range(n):
        if ltys[i] == "none" or cols[i] == "none":
            continue
        if straights[i]:
            d = np.array([[a0[i], b0[i]], [a1[i], b1[i]]], dtype=float)
        else:
            d = lines_expand([a0[i], a1[i]], [b0[i], b1[i]], sector_index, track_index)
        d2 = circlize(d[:, 0], d[:, 1], sector_index=sector_index, track_index=track_index)
        segs.append(polar2Cartesian(d2))
        seg_cols.append(cols[i])
        seg_lwds.append(lwds[i])
        seg_ltys.append(ltys[i])

    if segs:
        ax = ensure_canvas(initialize=False)
        coll = LineCollection(segs, colors=seg_cols, linewidths=seg_lwds, linestyles=seg_ltys, capstyle="butt")
        ax.add_collection(coll)


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def _ha_va_from_adj(ax_adj: float, ay_adj: float) -> tuple[str, str, bool]:
    ha_map = {0.0: "left", 0.5: "center", 1.0: "right"}
    va_map = {0.0: "bottom", 0.5: "center", 1.0: "top"}
    if ax_adj in ha_map and ay_adj in va_map:
        return ha_map[ax_adj], va_map[ay_adj], False
    return "center", "center", True


def circos_text(
    x: Any,
    y: Any = None,
    labels: Any = None,
    sector_index: str | None = None,
    track_index: int | None = None,
    direction: str | None = None,
    facing: str = "inside",
    nice_facing: bool = False,
    adj: Any = (0.5, 0.5),
    cex: Any = 1.0,
    col: Any = "black",
    font: Any = 1,
    family: str = "sans-serif",
    **kwargs: Any,
) -> None:
    """Draw text in a cell with radial/tangential/curved facing and niceFacing support."""
    nkw = normalize_kwargs(kwargs)
    sector_index = nkw.pop("sector_index", sector_index)
    track_index = nkw.pop("track_index", track_index)
    nice_facing = bool(nkw.pop("nice_facing", nice_facing))

    if sector_index is None:
        sector_index = get_current_sector_index()
    if track_index is None:
        track_index = get_current_track_index()

    if not has_cell(sector_index, track_index):
        raise RuntimeError("'circos.text' can only be used after the plotting region has been created.")

    if labels is None:
        raise ValueError("`labels` must be specified in `circos_text`.")

    x_arr, y_arr = _extract_xy_args(x, y)
    n = len(x_arr)
    if n == 0:
        return

    if isinstance(labels, (str, bytes, int, float, np.number)):
        lbl_list = [str(labels)] * n
    else:
        lbl_list = [str(v) for v in labels]
        if len(lbl_list) == 1 and n > 1:
            lbl_list = lbl_list * n

    cexs = [float(v) for v in recycle_to_length(cex, n)]
    cols = resolve_colors(col, n, default="black")
    fonts = recycle_to_length(font, n)

    if direction is not None:
        warnings.warn("`direction` is deprecated, please use `facing` instead.", DeprecationWarning, stacklevel=2)
        dmap = {
            "default": "inside",
            "default2": "outside",
            "vertical_left": "reverse.clockwise",
            "vertical_right": "clockwise",
            "horizontal": "downward",
            "arc": "bending.inside",
        }
        if direction not in dmap:
            raise ValueError("Wrong `direction` value, please use `facing` instead.")
        facing = dmap[direction]

    valid_facings = ("inside", "outside", "reverse.clockwise", "clockwise", "downward", "bending", "bending.inside", "bending.outside")
    if facing not in valid_facings:
        raise ValueError(f"Invalid `facing` {facing!r}. Expected one of {valid_facings}.")
    if facing == "bending":
        facing = "bending.inside"

    d = circlize(x_arr, y_arr, sector_index=sector_index, track_index=track_index)

    # Support degree() list adj
    if isinstance(adj, (list, tuple)) and len(adj) == 2 and any(isinstance(a, DegreeValue) for a in adj):
        ax_val = adj[0]
        ay_val = adj[1]
        lbl_w = np.array([strwidth_canvas(lbl_list[i], cex=cexs[i], font=fonts[i], family=family) for i in range(n)])
        lbl_h = np.array([strheight_canvas(lbl_list[i], cex=cexs[i], font=fonts[i], family=family) for i in range(n)])
        if facing == "clockwise":
            if not isinstance(ay_val, DegreeValue):
                raise ValueError("Second item in `adj` should be wrapped by `degree()` if facing is clockwise.")
            rou_off = -(float(ax_val) - 0.5) * lbl_w
            theta_off = float(ay_val)
        elif facing == "reverse.clockwise":
            if not isinstance(ay_val, DegreeValue):
                raise ValueError("Second item in `adj` should be wrapped by `degree()` if facing is reverse.clockwise.")
            rou_off = (float(ax_val) - 0.5) * lbl_w
            theta_off = float(ay_val)
        elif facing == "inside":
            if not isinstance(ax_val, DegreeValue):
                raise ValueError("First item in `adj` should be wrapped by `degree()` if facing is inside.")
            rou_off = -(float(ay_val) - 0.5) * lbl_h
            theta_off = float(ax_val)
        elif facing == "outside":
            if not isinstance(ax_val, DegreeValue):
                raise ValueError("First item in `adj` should be wrapped by `degree()` if facing is outside.")
            rou_off = (float(ay_val) - 0.5) * lbl_h
            theta_off = float(ax_val)
        else:
            rou_off = np.zeros(n)
            theta_off = 0.0

        if facing in ("clockwise", "reverse.clockwise", "inside", "outside"):
            d2 = np.array(d, dtype=float)
            d2[:, 1] = d2[:, 1] + rou_off
            d2[:, 0] = d2[:, 0] + theta_off
            dd = reverse_circlize(d2, sector_index=sector_index, track_index=track_index)
            x_arr = dd[:, 0]
            y_arr = dd[:, 1]
            adj_pair = (0.5, 0.5)
            d = circlize(x_arr, y_arr, sector_index=sector_index, track_index=track_index)
    else:
        if np.isscalar(adj):
            adj_pair = (float(adj), float(adj))
        else:
            adj_pair = (float(adj[0]), float(adj[1]))

    check_points_position(x_arr, y_arr, sector_index, track_index)

    if nice_facing and facing in ("clockwise", "reverse.clockwise", "inside", "outside", "bending.inside", "bending.outside"):
        deg_m = d[:, 0] % 360.0
        if facing in ("clockwise", "reverse.clockwise"):
            l1 = (deg_m >= 90.0) & (deg_m < 270.0)
            l2 = ~l1
            if facing == "reverse.clockwise":
                adj1, adj2 = adj_pair, (1.0 - adj_pair[0], 1.0 - adj_pair[1])
            else:
                adj1, adj2 = (1.0 - adj_pair[0], 1.0 - adj_pair[1]), adj_pair
            f1, f2 = "reverse.clockwise", "clockwise"
        else:
            l1 = (deg_m > 0.0) & (deg_m < 180.0)
            l2 = ~l1
            if facing in ("inside", "bending.inside"):
                adj1, adj2 = adj_pair, (1.0 - adj_pair[0], 1.0 - adj_pair[1])
            else:
                adj1, adj2 = (1.0 - adj_pair[0], 1.0 - adj_pair[1]), adj_pair
            if facing.startswith("bending"):
                f1, f2 = "bending.inside", "bending.outside"
            else:
                f1, f2 = "inside", "outside"

        if np.any(l1):
            idx1 = np.where(l1)[0]
            circos_text(
                x_arr[idx1],
                y_arr[idx1],
                [lbl_list[i] for i in idx1],
                sector_index=sector_index,
                track_index=track_index,
                facing=f1,
                nice_facing=False,
                adj=adj1,
                cex=[cexs[i] for i in idx1],
                col=[cols[i] for i in idx1],
                font=[fonts[i] for i in idx1],
                family=family,
            )
        if np.any(l2):
            idx2 = np.where(l2)[0]
            circos_text(
                x_arr[idx2],
                y_arr[idx2],
                [lbl_list[i] for i in idx2],
                sector_index=sector_index,
                track_index=track_index,
                facing=f2,
                nice_facing=False,
                adj=adj2,
                cex=[cexs[i] for i in idx2],
                col=[cols[i] for i in idx2],
                font=[fonts[i] for i in idx2],
                family=family,
            )
        return

    if facing in ("bending.inside", "bending.outside"):
        chars_list = [list(s) for s in lbl_list]
        if facing == "bending.outside":
            chars_list = [list(reversed(c)) for c in chars_list]
            adj_pair = (1.0 - adj_pair[0], 1.0 - adj_pair[1])

        strw_list = [[strwidth_canvas(ch, cex=cexs[i], font=fonts[i], family=family) for ch in chars_list[i]] for i in range(n)]
        strh_list = [[strheight_canvas(ch, cex=cexs[i], font=fonts[i], family=family) for ch in chars_list[i]] for i in range(n)]

        for i in range(n):
            if not chars_list[i]:
                continue
            r_base = float(d[i, 1])
            alpha_off = sum(strw_list[i]) * adj_pair[0] / max(r_base, 1e-6) * 180.0 / math.pi
            rou_off = -strh_list[i][0] * adj_pair[1]
            alpha = float(d[i, 0]) + alpha_off
            rou = r_base + rou_off
            thetas = np.empty(len(chars_list[i]), dtype=float)
            for j, cw in enumerate(strw_list[i]):
                ratio1 = max(-1.0, min(1.0, cw / 2.0 / max(r_base, 1e-6)))
                ratio2 = max(-1.0, min(1.0, cw / max(r_base, 1e-6)))
                thetas[j] = alpha - math.asin(ratio1) * 180.0 / math.pi
                alpha -= math.asin(ratio2) * 180.0 / math.pi
            dr = reverse_circlize(thetas, np.repeat(rou, len(thetas)), sector_index=sector_index, track_index=track_index)
            if facing == "bending.inside":
                circos_text(
                    dr[:, 0], dr[:, 1], chars_list[i],
                    sector_index=sector_index, track_index=track_index,
                    cex=cexs[i], col=cols[i], font=fonts[i], family=family,
                    facing="inside", adj=(0.5, 0.0),
                )
            else:
                circos_text(
                    dr[:, 0], dr[:, 1], chars_list[i],
                    sector_index=sector_index, track_index=track_index,
                    cex=cexs[i], col=cols[i], font=fonts[i], family=family,
                    facing="outside", adj=(0.5, 1.0),
                )
        return

    srt = d[:, 0] - 90.0
    if facing == "reverse.clockwise":
        srt = srt - 90.0
    elif facing == "clockwise":
        srt = srt + 90.0
    elif facing == "downward":
        srt = np.zeros_like(srt)
    elif facing == "outside":
        srt = srt + 180.0

    m = polar2Cartesian(d)
    ax = ensure_canvas(initialize=False)
    ha, va, custom_offset = _ha_va_from_adj(adj_pair[0], adj_pair[1])
    scale = _STATE.canvas_overlay_scale if _STATE.canvas_overlay_scale is not None else 1.0

    for i in range(n):
        if cols[i] == "none":
            continue
        weight, style = resolve_font(fonts[i])
        px, py = float(m[i, 0]), float(m[i, 1])
        rot_deg = float(srt[i])
        if custom_offset:
            w_c = strwidth_canvas(lbl_list[i], cex=cexs[i], font=fonts[i], family=family, ax=ax) * scale
            h_c = strheight_canvas(lbl_list[i], cex=cexs[i], font=fonts[i], family=family, ax=ax) * scale
            u = (0.5 - adj_pair[0]) * w_c
            v = (0.5 - adj_pair[1]) * h_c
            rad = math.radians(rot_deg)
            px += u * math.cos(rad) - v * math.sin(rad)
            py += u * math.sin(rad) + v * math.cos(rad)

        ax.text(
            px,
            py,
            lbl_list[i],
            fontsize=cexs[i] * BASE_FONTSIZE_PT,
            color=cols[i],
            fontweight=weight,
            fontstyle=style,
            fontfamily=family,
            rotation=rot_deg,
            rotation_mode="anchor",
            ha=ha,
            va=va,
            clip_on=False,
        )


def circos_track_text(
    sectors: Any,
    x: Any,
    y: Any,
    labels: Any,
    track_index: int | None = None,
    direction: str | None = None,
    facing: str = "inside",
    nice_facing: bool = False,
    adj: Any = (0.5, 0.5),
    cex: Any = 1.0,
    col: Any = "black",
    font: Any = 1,
    factors: Any = None,
    **kwargs: Any,
) -> None:
    """Draw text in cells across multiple sectors on a track."""
    nkw = normalize_kwargs(kwargs)
    track_index = nkw.get("track_index", track_index)
    nice_facing = bool(nkw.get("nice_facing", nice_facing))
    if sectors is None and factors is not None:
        sectors = factors
    if "factors" in nkw and sectors is None:
        sectors = nkw["factors"]
    if track_index is None:
        track_index = get_current_track_index()

    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    f_vals, le = _normalize_factors(sectors, preserve_unique_order=False)
    if len(x_arr) != len(f_vals) or len(y_arr) != len(f_vals):
        raise ValueError("Length of data and length of sectors differ.")
    bad = set(le) - set(get_all_sector_index())
    if bad:
        raise ValueError(f"Cannot find these categories in existing sectors: {sorted(bad)}")

    lbls = recycle_to_length(labels, len(f_vals))
    cexs = recycle_with_factors(cex, f_vals, le)
    cols = recycle_with_factors(col, f_vals, le)
    fonts = recycle_with_factors(font, f_vals, le)

    f_arr = np.asarray(f_vals, dtype=object)
    for lv in le:
        idx = np.where(f_arr == lv)[0]
        circos_text(
            x_arr[idx],
            y_arr[idx],
            [lbls[i] for i in idx],
            sector_index=lv,
            track_index=track_index,
            direction=direction,
            facing=facing,
            nice_facing=nice_facing,
            adj=adj,
            cex=[cexs[i] for i in idx],
            col=[cols[i] for i in idx],
            font=[fonts[i] for i in idx],
        )


# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------

def default_major_by(
    sector_index: str | None = None,
    track_index: int | None = None,
) -> float:
    if sector_index is None:
        sector_index = get_current_sector_index()
    if track_index is None:
        track_index = get_current_track_index()
    d = float(circos_par.get_value("major.by.degree"))
    sd = float(get_cell_meta_data("cell.start.degree", sector_index, track_index))
    br = float(get_cell_meta_data("cell.bottom.radius", sector_index, track_index))
    tm = reverse_circlize([sd, sd - d], [br, br], sector_index=sector_index, track_index=track_index)
    major_by = abs(float(tm[0, 0]) - float(tm[1, 0]))
    if major_by <= 0:
        return 1.0
    exp_val = int(math.floor(math.log10(major_by)))
    return float(round(major_by, -exp_val))


def circos_axis(
    h: str | float = "top",
    major_at: Sequence[float] | None = None,
    labels: Any = True,
    major_tick: bool = True,
    sector_index: str | None = None,
    track_index: int | None = None,
    labels_font: Any = 1,
    labels_cex: float = 1.0,
    labels_facing: str = "inside",
    labels_direction: str | None = None,
    labels_nice_facing: bool = True,
    direction: str = "outside",
    minor_ticks: int = 4,
    major_tick_length: float | None = None,
    major_tick_percentage: float | None = None,
    lwd: float = 1.0,
    col: Any = "black",
    labels_col: Any = "black",
    labels_pos_adjust: bool | Sequence[bool] = True,
    **kwargs: Any,
) -> None:
    """Draw x-axis along the top, bottom, or specified y position of a cell."""
    nkw = normalize_kwargs(kwargs)
    major_at = nkw.get("major_at", major_at)
    major_tick = nkw.get("major_tick", major_tick)
    sector_index = nkw.get("sector_index", sector_index)
    track_index = nkw.get("track_index", track_index)
    labels_font = nkw.get("labels_font", labels_font)
    labels_cex = float(nkw.get("labels_cex", labels_cex))
    labels_facing = nkw.get("labels_facing", labels_facing)
    labels_direction = nkw.get("labels_direction", labels_direction)
    labels_nice_facing = bool(nkw.get("labels_nice_facing", labels_nice_facing))
    minor_ticks = int(nkw.get("minor_ticks", minor_ticks))
    major_tick_length = nkw.get("major_tick_length", major_tick_length)
    labels_col = nkw.get("labels_col", labels_col)
    labels_pos_adjust = nkw.get("labels_pos_adjust", labels_pos_adjust)

    if sector_index is None:
        sector_index = get_current_sector_index()
    if track_index is None:
        track_index = get_current_track_index()

    os_idx = get_current_sector_index()
    ot_idx = get_current_track_index()
    set_current_cell(sector_index, track_index)

    try:
        if major_tick_length is None:
            major_tick_length = float(mm_y(1.0, sector_index, track_index))
        if major_tick_percentage is not None:
            warnings.warn("`major.tick.percentage` is not used any more, please use `major.tick.length`.", UserWarning, stacklevel=2)

        if labels_direction is not None:
            dmap = {
                "default": "inside",
                "default2": "outside",
                "vertical_left": "reverse.clockwise",
                "vertical_right": "clockwise",
                "horizontal": "downward",
                "arc": "bending",
            }
            labels_facing = dmap.get(labels_direction, labels_facing)
            warnings.warn("`labels.direction` is deprecated, please use `labels.facing` instead.", DeprecationWarning, stacklevel=2)

        if direction not in ("outside", "inside"):
            raise ValueError("`direction` should be 'outside' or 'inside'.")

        xlim = get_cell_meta_data("xlim", sector_index, track_index)
        if h == "top":
            h_val = float(get_cell_meta_data("cell.ylim", sector_index, track_index)[1])
        elif h == "bottom":
            h_val = float(get_cell_meta_data("cell.ylim", sector_index, track_index)[0])
        else:
            h_val = float(h)

        if major_at is None:
            mb = default_major_by(sector_index, track_index)
            start_at = math.floor(xlim[0] / mb) * mb
            major_arr = np.arange(start_at, xlim[1] + 1e-12 * abs(mb), mb, dtype=float)
            if len(major_arr) == 0:
                major_arr = np.array([xlim[0], xlim[1]], dtype=float)
            major_arr = np.append(major_arr, major_arr[-1] + mb)
        else:
            major_arr = np.asarray(major_at, dtype=float).ravel()

        minor_list: list[float] = []
        if minor_ticks != 0 and len(major_arr) > 1:
            ks = np.arange(1, minor_ticks + 1, dtype=float) / (minor_ticks + 1.0)
            for i in range(1, len(major_arr)):
                minor_list.extend((ks * (major_arr[i] - major_arr[i - 1]) + major_arr[i - 1]).tolist())
        minor_arr = np.asarray(minor_list, dtype=float)

        line_x0 = max(float(major_arr[0]), float(xlim[0])) if len(major_arr) else float(xlim[0])
        line_x1 = min(float(major_arr[-1]), float(xlim[1])) if len(major_arr) else float(xlim[1])
        circos_lines([line_x0, line_x1], [h_val, h_val], sector_index=sector_index, track_index=track_index, lwd=lwd, col=col)

        op_warn = circos_par.get_value("points.overflow.warning")
        circos_par.set_value("points_overflow_warning", False, _suppress_warn=True)
        try:
            l_mask = (major_arr >= xlim[0] - 1e-12) & (major_arr <= xlim[1] + 1e-12)
            sign = 1.0 if direction == "outside" else -1.0
            if major_tick and np.any(l_mask):
                mx = major_arr[l_mask]
                circos_segments(
                    mx,
                    np.repeat(h_val, len(mx)),
                    mx,
                    np.repeat(h_val + float(major_tick_length) * sign, len(mx)),
                    straight=True,
                    sector_index=sector_index,
                    track_index=track_index,
                    lwd=lwd,
                    col=col,
                )

            if direction == "outside":
                adj_map = {
                    "inside": (0.5, 0.0),
                    "outside": (0.5, 1.0),
                    "reverse.clockwise": (1.0, 0.5),
                    "clockwise": (0.0, 0.5),
                    "downward": (0.5, 0.5),
                }
                labels_adj = adj_map.get(labels_facing, (0.5, 0.0))
            else:
                adj_map = {
                    "inside": (0.5, 1.0),
                    "outside": (0.5, 0.0),
                    "reverse.clockwise": (0.0, 0.5),
                    "clockwise": (1.0, 0.5),
                    "downward": (0.5, 0.5),
                }
                labels_adj = adj_map.get(labels_facing, (0.5, 1.0))

            def _format_num(v: float) -> str:
                if abs(v - round(v)) < 1e-9:
                    return str(int(round(v)))
                return f"{v:g}"

            def _add_axis_labels(xs_pos: np.ndarray, ys_pos: np.ndarray, lbls: list[str]) -> None:
                nn = len(xs_pos)
                if nn == 0:
                    return
                cx_lim = get_cell_meta_data("cell.xlim", sector_index, track_index)
                first_w = float(convert_x(strwidth_inches(lbls[0], cex=labels_cex, font=labels_font), "inches", sector_index, track_index, h=h_val))
                first_h = float(convert_x(strheight_inches(lbls[0], cex=labels_cex, font=labels_font), "inches", sector_index, track_index, h=h_val))
                last_w = float(convert_x(strwidth_inches(lbls[-1], cex=labels_cex, font=labels_font), "inches", sector_index, track_index, h=h_val))
                last_h = float(convert_x(strheight_inches(lbls[-1], cex=labels_cex, font=labels_font), "inches", sector_index, track_index, h=h_val))

                if labels_facing in ("reverse.clockwise", "clockwise"):
                    off_first = first_h / 2.0 - (xs_pos[0] - cx_lim[0])
                    off_last = last_h / 2.0 - abs(xs_pos[-1] - cx_lim[1])
                else:
                    off_first = first_w / 2.0 - (xs_pos[0] - cx_lim[0])
                    off_last = last_w / 2.0 - abs(xs_pos[-1] - cx_lim[1])

                lpa = [bool(labels_pos_adjust)] * 2 if isinstance(labels_pos_adjust, (bool, np.bool_)) else [bool(labels_pos_adjust[0]), bool(labels_pos_adjust[1])]
                if not lpa[0]:
                    off_first = 0.0
                if not lpa[1]:
                    off_last = 0.0

                adj_xs = xs_pos.copy()
                if nn == 1:
                    if off_first > 0:
                        adj_xs[0] += off_first
                else:
                    if off_first > 0:
                        adj_xs[0] += off_first
                    if off_last > 0:
                        adj_xs[-1] -= off_last

                circos_text(
                    adj_xs,
                    ys_pos,
                    lbls,
                    sector_index=sector_index,
                    track_index=track_index,
                    facing=labels_facing,
                    nice_facing=labels_nice_facing,
                    adj=labels_adj,
                    cex=labels_cex,
                    col=labels_col,
                    font=labels_font,
                )

            mx = major_arr[l_mask]
            y_lbl = np.repeat(h_val + (float(major_tick_length) + float(mm_y(0.5, sector_index, track_index))) * sign, len(mx))

            if isinstance(labels, (bool, np.bool_)):
                if bool(labels):
                    _add_axis_labels(mx, y_lbl, [_format_num(v) for v in mx])
            elif callable(labels):
                res_lbls = labels(mx)
                _add_axis_labels(mx, y_lbl, [str(v) for v in res_lbls])
            elif labels is not None:
                lbl_seq = [str(v) for v in labels] if not isinstance(labels, str) else [labels]
                if len(lbl_seq) == len(major_arr):
                    lbl_sub = [lbl_seq[i] for i, m in enumerate(l_mask) if m]
                elif len(lbl_seq) == len(mx):
                    lbl_sub = lbl_seq
                else:
                    lbl_sub = [lbl_seq[i % len(lbl_seq)] for i in range(len(mx))]
                if any(s != "" for s in lbl_sub):
                    _add_axis_labels(mx, y_lbl, lbl_sub)

            if major_tick and len(minor_arr):
                m_mask = (minor_arr >= xlim[0] - 1e-12) & (minor_arr <= xlim[1] + 1e-12)
                if np.any(m_mask):
                    mnx = minor_arr[m_mask]
                    circos_segments(
                        mnx,
                        np.repeat(h_val, len(mnx)),
                        mnx,
                        np.repeat(h_val + float(major_tick_length) / 2.0 * sign, len(mnx)),
                        straight=True,
                        sector_index=sector_index,
                        track_index=track_index,
                        lwd=lwd,
                        col=col,
                    )
        finally:
            circos_par.set_value("points_overflow_warning", op_warn, _suppress_warn=True)
    finally:
        if os_idx is not None:
            set_current_cell(os_idx, ot_idx)


circos_xaxis = circos_axis


def circos_yaxis(
    side: str = "left",
    at: Sequence[float] | None = None,
    labels: Any = True,
    tick: bool = True,
    sector_index: str | None = None,
    track_index: int | None = None,
    labels_font: Any = 1,
    labels_cex: float = 1.0,
    labels_nice_facing: bool = True,
    tick_length: float | None = None,
    lwd: float = 1.0,
    col: Any = "black",
    labels_col: Any = "black",
    **kwargs: Any,
) -> None:
    """Draw y-axis on the left or right boundary of a cell."""
    nkw = normalize_kwargs(kwargs)
    sector_index = nkw.get("sector_index", sector_index)
    track_index = nkw.get("track_index", track_index)
    labels_font = nkw.get("labels_font", labels_font)
    labels_cex = float(nkw.get("labels_cex", labels_cex))
    labels_nice_facing = bool(nkw.get("labels_nice_facing", labels_nice_facing))
    tick_length = nkw.get("tick_length", tick_length)
    labels_col = nkw.get("labels_col", labels_col)

    if sector_index is None:
        sector_index = get_current_sector_index()
    if track_index is None:
        track_index = get_current_track_index()

    if side not in ("left", "right"):
        raise ValueError("`side` must be 'left' or 'right'.")

    if tick_length is None:
        tick_length = float(convert_x(1.0, "mm", sector_index, track_index))

    ylim = get_cell_meta_data("ylim", sector_index, track_index)
    v = float(get_cell_meta_data("cell.xlim", sector_index, track_index)[0 if side == "left" else 1])

    if at is None:
        loc = MaxNLocator(nbins=5)
        at_arr = np.asarray(loc.tick_values(float(ylim[0]), float(ylim[1])), dtype=float)
        if callable(labels):
            lbl_vals = [str(s) for s in labels(at_arr)]
        elif isinstance(labels, (bool, np.bool_)) and not bool(labels):
            lbl_vals = None
        else:
            lbl_vals = [f"{val:g}" for val in at_arr]
    else:
        at_arr = np.asarray(at, dtype=float).ravel()
        if callable(labels):
            lbl_vals = [str(s) for s in labels(at_arr)]
        elif isinstance(labels, (bool, np.bool_)):
            lbl_vals = [f"{val:g}" for val in at_arr] if bool(labels) else None
        else:
            lbl_vals = [str(s) for s in labels]

    cy = get_cell_meta_data("cell.ylim", sector_index, track_index)
    circos_lines([v, v], [cy[0], cy[1]], sector_index=sector_index, track_index=track_index, lwd=lwd, col=col)

    op_warn = circos_par.get_value("points.overflow.warning")
    circos_par.set_value("points_overflow_warning", False, _suppress_warn=True)
    try:
        mask = (at_arr >= ylim[0] - 1e-12) & (at_arr <= ylim[1] + 1e-12)
        at_sub = at_arr[mask]
        sign = 1.0 if side == "right" else -1.0
        if tick and len(at_sub):
            circos_segments(
                np.repeat(v, len(at_sub)),
                at_sub,
                np.repeat(v + float(tick_length) * sign, len(at_sub)),
                at_sub,
                straight=True,
                sector_index=sector_index,
                track_index=track_index,
                lwd=lwd,
                col=col,
            )

        xcw = bool(circos_par.get_value("xaxis.clock.wise"))
        if side == "left":
            labels_adj = (1.0, 0.5) if xcw else (0.0, 0.5)
        else:
            labels_adj = (0.0, 0.5) if xcw else (1.0, 0.5)

        if lbl_vals is not None and len(at_sub):
            lbl_sub = [lbl_vals[i] for i, m in enumerate(mask) if m]
            x_txt = np.repeat(v + (float(tick_length) + float(convert_x(0.5, "mm", sector_index, track_index))) * sign, len(at_sub))
            circos_text(
                x_txt,
                at_sub,
                lbl_sub,
                adj=labels_adj,
                font=labels_font,
                cex=labels_cex,
                sector_index=sector_index,
                track_index=track_index,
                facing="inside",
                nice_facing=labels_nice_facing,
                col=labels_col,
            )
    finally:
        circos_par.set_value("points_overflow_warning", op_warn, _suppress_warn=True)


# ---------------------------------------------------------------------------
# Histogram, Barplot, Boxplot, Violin, Arrow, Raster, Connect, Labels, StackedText
# ---------------------------------------------------------------------------

def _sturges_nbins(n: int) -> int:
    return max(1, int(math.ceil(math.log2(max(n, 1)) + 1.0)))


def circos_track_hist(
    sectors: Any,
    x: Any,
    track_height: float | None = None,
    track_index: int | None = None,
    ylim: Any = None,
    force_ylim: bool = True,
    col: Any = None,
    border: Any = "black",
    lty: Any = 1,
    lwd: float = 1.0,
    bg_col: Any = None,
    bg_border: Any = "black",
    bg_lty: Any = 1,
    bg_lwd: float = 1.0,
    breaks: str | int | Sequence[float] = "Sturges",
    include_lowest: bool = True,
    right: bool = True,
    draw_density: bool = False,
    bin_size: float | None = None,
    area: bool = False,
    factors: Any = None,
    **kwargs: Any,
) -> None:
    """Draw histograms or density step lines across a whole track."""
    from ._plot import circos_track_plot_region

    nkw = normalize_kwargs(kwargs)
    track_height = nkw.get("track_height", track_height)
    track_index = nkw.get("track_index", track_index)
    force_ylim = nkw.get("force_ylim", force_ylim)
    draw_density = nkw.get("draw_density", draw_density)
    bin_size = nkw.get("bin_size", bin_size)
    if sectors is None and factors is not None:
        sectors = factors
    if "factors" in nkw and sectors is None:
        sectors = nkw["factors"]
    if track_height is None:
        track_height = float(circos_par.get_value("track.height"))
    if col is None:
        col = "black" if draw_density else None

    x_arr = np.asarray(x, dtype=float).ravel()
    f_vals, le = _normalize_factors(sectors, preserve_unique_order=False)
    if len(x_arr) != len(f_vals):
        raise ValueError("Length of `x` and `sectors` differ.")
    bad = set(le) - set(get_all_sector_index())
    if bad:
        raise ValueError(f"Cannot find these categories in existing sectors: {sorted(bad)}")

    f_arr = np.asarray(f_vals, dtype=object)
    xx: list[float] = []
    yy: list[float] = []
    fa: list[str] = []

    for lv in le:
        nx = x_arr[f_arr == lv]
        if bin_size is not None:
            mn, mx = float(np.min(nx)), float(np.max(nx))
            bks = np.arange(mn, mx + 1e-12, float(bin_size))
            if len(bks) == 0 or bks[-1] < mx:
                bks = np.append(bks, (bks[-1] if len(bks) else mn) + float(bin_size))
            counts, edges = np.histogram(nx, bins=bks)
        elif isinstance(breaks, str):
            nb = _sturges_nbins(len(nx))
            counts, edges = np.histogram(nx, bins=nb)
        else:
            counts, edges = np.histogram(nx, bins=breaks)

        if draw_density:
            widths = np.diff(edges)
            vals = counts.astype(float) / (max(len(nx), 1) * np.where(widths > 0, widths, 1.0))
        else:
            vals = counts.astype(float)

        xx.extend(edges.tolist())
        yy.extend([0.0] + vals.tolist())
        fa.extend([lv] * len(edges))

    circos_track_plot_region(
        sectors=fa,
        y=yy,
        track_height=track_height,
        track_index=track_index,
        ylim=ylim,
        force_ylim=force_ylim,
        bg_col=bg_col,
        bg_border=bg_border,
        bg_lty=bg_lty,
        bg_lwd=bg_lwd,
    )
    t_idx = get_current_track_index()

    xx_a = np.asarray(xx, dtype=float)
    yy_a = np.asarray(yy, dtype=float)
    fa_a = np.asarray(fa, dtype=object)

    keep = np.zeros(len(xx_a), dtype=bool)
    for lv in le:
        xlim = get_cell_meta_data("xlim", sector_index=lv, track_index=t_idx)
        m = fa_a == lv
        keep[m] = (xx_a[m] >= xlim[0] - 1e-12) & (xx_a[m] <= xlim[1] + 1e-12)
    xx_a, yy_a, fa_a = xx_a[keep], yy_a[keep], fa_a[keep]

    if draw_density:
        circos_track_lines(sectors=fa_a, x=xx_a, y=yy_a, track_index=t_idx, col=col, lty=lty, lwd=lwd, area=area, border=border)
    else:
        cols = recycle_with_levels(col, le)
        borders = recycle_with_levels(border, le)
        ltys = recycle_with_levels(lty, le)
        lwds = recycle_with_levels(lwd, le)
        for i, lv in enumerate(le):
            m = fa_a == lv
            nx = xx_a[m].copy()
            ny = yy_a[m]
            cx = get_cell_meta_data("cell.xlim", lv, t_idx)
            nx = np.clip(nx, cx[0], cx[1])
            for j in range(1, len(nx)):
                circos_rect(nx[j - 1], 0.0, nx[j], ny[j], sector_index=lv, track_index=t_idx, col=cols[i], border=borders[i], lty=ltys[i], lwd=lwds[i])


def circos_barplot(
    value: Any,
    pos: Sequence[float],
    bar_width: float | Sequence[float] = 0.6,
    col: Any = None,
    border: Any = "black",
    lwd: Any = 1.0,
    lty: Any = 1,
    sector_index: str | None = None,
    track_index: int | None = None,
    **kwargs: Any,
) -> None:
    """Draw simple or stacked barplots in the current cell."""
    nkw = normalize_kwargs(kwargs)
    sector_index = nkw.get("sector_index", sector_index)
    track_index = nkw.get("track_index", track_index)
    if sector_index is None:
        sector_index = get_current_sector_index()
    if track_index is None:
        track_index = get_current_track_index()

    os_idx = get_current_sector_index()
    ot_idx = get_current_track_index()
    set_current_cell(sector_index, track_index)
    try:
        pos_arr = np.atleast_1d(np.asarray(pos, dtype=float)).ravel()
        if isinstance(value, pd.DataFrame):
            val_arr = value.to_numpy(dtype=float)
        else:
            val_arr = np.asarray(value, dtype=float)

        if val_arr.ndim == 2:
            nr, nc = val_arr.shape
            if nr != len(pos_arr):
                raise ValueError("Number of rows of `value` must match length of `pos`.")
            if np.any(val_arr < 0):
                raise ValueError("`value` should all be positive if it is a 2D matrix.")
            cols = recycle_to_length(col, nc)
            borders = recycle_to_length(border, nc)
            lwds = recycle_to_length(lwd, nc)
            ltys = recycle_to_length(lty, nc)
            bws = np.asarray(recycle_to_length(bar_width, nr), dtype=float)
            cum = np.zeros(nr, dtype=float)
            for i in range(nc):
                nxt = cum + val_arr[:, i]
                circos_rect(pos_arr - bws / 2.0, cum, pos_arr + bws / 2.0, nxt, col=cols[i], border=borders[i], lwd=lwds[i], lty=ltys[i])
                cum = nxt
        elif val_arr.ndim == 1:
            if len(val_arr) != len(pos_arr):
                raise ValueError("`value` and `pos` should have the same length.")
            bw = np.asarray(recycle_to_length(bar_width, len(pos_arr)), dtype=float)
            circos_rect(pos_arr - bw / 2.0, 0.0, pos_arr + bw / 2.0, val_arr, col=col, border=border, lwd=lwd, lty=lty)
        else:
            raise ValueError("`value` should be a 1D vector or 2D matrix.")
    finally:
        if os_idx is not None:
            set_current_cell(os_idx, ot_idx)


def _tukey_boxplot_stats(vals: np.ndarray) -> tuple[float, float, float, float, float]:
    """Compute (lower_whisker, q1, median, q3, upper_whisker) using 1.5*IQR rule."""
    clean = vals[~np.isnan(vals)]
    q1, med, q3 = np.percentile(clean, [25.0, 50.0, 75.0])
    iqr = q3 - q1
    lo_fence = q1 - 1.5 * iqr
    hi_fence = q3 + 1.5 * iqr
    in_lo = clean[clean >= lo_fence]
    in_hi = clean[clean <= hi_fence]
    w_lo = float(np.min(in_lo)) if len(in_lo) else float(q1)
    w_hi = float(np.max(in_hi)) if len(in_hi) else float(q3)
    return (w_lo, float(q1), float(med), float(q3), w_hi)


def circos_boxplot(
    value: Any,
    pos: float | Sequence[float],
    outline: bool = True,
    box_width: float | Sequence[float] = 0.6,
    col: Any = None,
    border: Any = "black",
    lwd: Any = 1.0,
    lty: Any = 1,
    cex: Any = 1.0,
    pch: Any = 1,
    pt_col: Any = "black",
    sector_index: str | None = None,
    track_index: int | None = None,
    **kwargs: Any,
) -> None:
    """Draw boxplots (from vector, 2D matrix by columns, or list of vectors) in a cell."""
    nkw = normalize_kwargs(kwargs)
    sector_index = nkw.get("sector_index", sector_index)
    track_index = nkw.get("track_index", track_index)
    pt_col = nkw.get("pt_col", pt_col)
    if sector_index is None:
        sector_index = get_current_sector_index()
    if track_index is None:
        track_index = get_current_track_index()

    os_idx = get_current_sector_index()
    ot_idx = get_current_track_index()
    set_current_cell(sector_index, track_index)
    try:
        def _single(v: np.ndarray, p: float, bw: float, c_i: Any, b_i: Any, lw_i: float, lt_i: Any, cx_i: float, pch_i: Any, pc_i: Any) -> None:
            w_lo, q1, med, q3, w_hi = _tukey_boxplot_stats(v)
            circos_rect(p - 0.5 * bw, q1, p + 0.5 * bw, q3, col=c_i, border=b_i, lty=lt_i, lwd=lw_i)
            circos_segments(
                [p - 0.5 * bw, p, p, p - 0.5 * bw, p - 0.5 * bw],
                [w_hi, w_hi, w_lo, w_lo, med],
                [p + 0.5 * bw, p, p, p + 0.5 * bw, p + 0.5 * bw],
                [w_hi, q3, q1, w_lo, med],
                col=b_i,
                lty=lt_i,
                lwd=lw_i,
            )
            if outline:
                out = v[(v > w_hi) | (v < w_lo)]
                if len(out):
                    circos_points(np.repeat(p, len(out)), out, cex=cx_i, col=pc_i, pch=pch_i)

        if isinstance(value, pd.DataFrame):
            series_list = [value.iloc[:, j].to_numpy(dtype=float) for j in range(value.shape[1])]
        elif isinstance(value, np.ndarray) and value.ndim == 2:
            series_list = [value[:, j].astype(float) for j in range(value.shape[1])]
        elif isinstance(value, list) and len(value) > 0 and not np.isscalar(value[0]):
            series_list = [np.asarray(v, dtype=float).ravel() for v in value]
        else:
            v_arr = np.asarray(value, dtype=float).ravel()
            _single(v_arr, float(pos), float(box_width), col, border, float(lwd), lty, float(cex), pch, pt_col)
            return

        n = len(series_list)
        pos_arr = np.atleast_1d(np.asarray(pos, dtype=float)).ravel()
        if len(pos_arr) != n:
            raise ValueError("Length of `pos` should equal number of boxes.")
        bws = [float(v) for v in recycle_to_length(box_width, n)]
        cols = recycle_to_length(col, n)
        borders = recycle_to_length(border, n)
        lwds = [float(v) for v in recycle_to_length(lwd, n)]
        ltys = recycle_to_length(lty, n)
        cexs = [float(v) for v in recycle_to_length(cex, n)]
        pchs = recycle_to_length(pch, n)
        pcols = recycle_to_length(pt_col, n)
        for i in range(n):
            _single(series_list[i], float(pos_arr[i]), bws[i], cols[i], borders[i], lwds[i], ltys[i], cexs[i], pchs[i], pcols[i])
    finally:
        if os_idx is not None:
            set_current_cell(os_idx, ot_idx)


def circos_violin(
    value: Any,
    pos: float | Sequence[float],
    violin_width: float | Sequence[float] = 0.8,
    col: Any = None,
    border: Any = "black",
    lwd: Any = 1.0,
    lty: Any = 1,
    show_quantile: bool = True,
    pt_col: Any = "black",
    cex: Any = 1.0,
    pch: Any = 16,
    max_density: float | None = None,
    sector_index: str | None = None,
    track_index: int | None = None,
    **kwargs: Any,
) -> None:
    """Draw violin plots in a cell."""
    nkw = normalize_kwargs(kwargs)
    sector_index = nkw.get("sector_index", sector_index)
    track_index = nkw.get("track_index", track_index)
    pt_col = nkw.get("pt_col", pt_col)
    if sector_index is None:
        sector_index = get_current_sector_index()
    if track_index is None:
        track_index = get_current_track_index()

    os_idx = get_current_sector_index()
    ot_idx = get_current_track_index()
    set_current_cell(sector_index, track_index)
    try:
        def _kde_vals(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            clean = arr[~np.isnan(arr)]
            mn, mx = float(np.min(clean)), float(np.max(clean))
            grid = np.linspace(mn, mx, 128)
            if mx - mn < 1e-12 or np.std(clean) < 1e-12:
                dens = np.ones_like(grid)
            else:
                kde = gaussian_kde(clean)
                dens = kde(grid)
            return grid, dens, clean

        def _single(grid: np.ndarray, dens: np.ndarray, clean: np.ndarray, p: float, vw: float, c_i: Any, b_i: Any, lw_i: float, lt_i: Any, md: float, pc_i: Any, cx_i: float, pch_i: Any) -> None:
            dx = dens / max(md, 1e-12) * (vw / 2.0)
            y_poly = np.concatenate([grid, grid[::-1]])
            x_poly = np.concatenate([-dx + p, (dx + p)[::-1]])
            circos_polygon(x_poly, y_poly, border=b_i, col=c_i, lwd=lw_i, lty=lt_i)
            if show_quantile:
                w_lo, q1, med, q3, w_hi = _tukey_boxplot_stats(clean)
                circos_lines([p, p], [w_lo, q1])
                circos_lines([p, p], [q3, w_hi])
                circos_points([p], [med], cex=cx_i, col=pc_i, pch=pch_i)

        if isinstance(value, pd.DataFrame):
            series_list = [value.iloc[:, j].to_numpy(dtype=float) for j in range(value.shape[1])]
        elif isinstance(value, np.ndarray) and value.ndim == 2:
            series_list = [value[:, j].astype(float) for j in range(value.shape[1])]
        elif isinstance(value, list) and len(value) > 0 and not np.isscalar(value[0]):
            series_list = [np.asarray(v, dtype=float).ravel() for v in value]
        else:
            v_arr = np.asarray(value, dtype=float).ravel()
            grid, dens, clean = _kde_vals(v_arr)
            md = float(np.max(dens)) if max_density is None else float(max_density)
            _single(grid, dens, clean, float(pos), float(violin_width), col, border, float(lwd), lty, md, pt_col, float(cex), pch)
            return

        n = len(series_list)
        pos_arr = np.atleast_1d(np.asarray(pos, dtype=float)).ravel()
        if len(pos_arr) != n:
            raise ValueError("Length of `pos` should match number of violins.")
        kdes = [_kde_vals(s) for s in series_list]
        md = max(float(np.max(d)) for _, d, _ in kdes) if max_density is None else float(max_density)
        vws = [float(v) for v in recycle_to_length(violin_width, n)]
        cols = recycle_to_length(col, n)
        borders = recycle_to_length(border, n)
        lwds = [float(v) for v in recycle_to_length(lwd, n)]
        ltys = recycle_to_length(lty, n)
        pcols = recycle_to_length(pt_col, n)
        cexs = [float(v) for v in recycle_to_length(cex, n)]
        pchs = recycle_to_length(pch, n)
        for i in range(n):
            grid, dens, clean = kdes[i]
            _single(grid, dens, clean, float(pos_arr[i]), vws[i], cols[i], borders[i], lwds[i], ltys[i], md, pcols[i], cexs[i], pchs[i])
    finally:
        if os_idx is not None:
            set_current_cell(os_idx, ot_idx)


def circos_arrow(
    x1: float,
    x2: float,
    y: float | None = None,
    width: float | None = None,
    sector_index: str | None = None,
    track_index: int | None = None,
    arrow_head_length: float | None = None,
    arrow_head_width: float | None = None,
    arrow_position: str = "end",
    tail: str = "normal",
    border: Any = "black",
    col: Any = "#FFCCCC",
    lty: Any = 1,
    lwd: float = 1.0,
    **kwargs: Any,
) -> None:
    """Draw a circular arrow parallel to the track."""
    nkw = normalize_kwargs(kwargs)
    sector_index = nkw.pop("sector_index", sector_index)
    track_index = nkw.pop("track_index", track_index)
    if sector_index is None:
        sector_index = get_current_sector_index()
    if track_index is None:
        track_index = get_current_track_index()

    os_idx = get_current_sector_index()
    ot_idx = get_current_track_index()
    set_current_cell(sector_index, track_index)
    try:
        if y is None:
            y = float(get_cell_meta_data("ycenter", sector_index, track_index))
        if width is None:
            width = float(get_cell_meta_data("yrange", sector_index, track_index)) / 2.0
        if arrow_head_length is None:
            arrow_head_length = float(mm_x(5.0, sector_index, track_index))
        if arrow_head_width is None:
            arrow_head_width = float(width) * 2.0

        x1_f, x2_f, y_f = float(x1), float(x2), float(y)
        w_f = float(width)
        ahl = float(arrow_head_length)
        ahw = float(arrow_head_width)

        if circos_par.get_value("ring"):
            if x1_f > x2_f:
                x1_f -= float(get_cell_meta_data("xrange", sector_index, track_index))
        elif x2_f <= x1_f:
            x1_f, x2_f = x2_f, x1_f
            arrow_position = "start" if arrow_position == "end" else "end"

        if abs(x2_f - x1_f - ahl) < 1e-6 or ahl > (x2_f - x1_f):
            raise ValueError("Arrow head is too long (longer than the arrow itself).")

        if arrow_position == "end":
            head = np.array([[x2_f - ahl, y_f + ahw / 2.0], [x2_f, y_f], [x2_f - ahl, y_f - ahw / 2.0]])
            if tail == "normal":
                body1 = lines_expand([x2_f - ahl, x1_f], [y_f - w_f / 2.0, y_f - w_f / 2.0], sector_index, track_index)
                body2 = lines_expand([x1_f, x2_f - ahl], [y_f + w_f / 2.0, y_f + w_f / 2.0], sector_index, track_index)
            else:
                body1 = lines_expand([x2_f - ahl, x1_f], [y_f - w_f / 2.0, y_f], sector_index, track_index)
                body2 = lines_expand([x1_f, x2_f - ahl], [y_f, y_f + w_f / 2.0], sector_index, track_index)
        else:
            head = np.array([[x1_f + ahl, y_f + ahw / 2.0], [x1_f, y_f], [x1_f + ahl, y_f - ahw / 2.0]])
            if tail == "normal":
                body1 = lines_expand([x1_f + ahl, x2_f], [y_f - w_f / 2.0, y_f - w_f / 2.0], sector_index, track_index)
                body2 = lines_expand([x2_f, x1_f + ahl], [y_f + w_f / 2.0, y_f + w_f / 2.0], sector_index, track_index)
            else:
                body1 = lines_expand([x1_f + ahl, x2_f], [y_f - w_f / 2.0, y_f], sector_index, track_index)
                body2 = lines_expand([x2_f, x1_f + ahl], [y_f, y_f + w_f / 2.0], sector_index, track_index)

        coor = np.vstack([body1, body2, head, body1[0:1]])
        d2 = circlize(coor[:, 0], coor[:, 1], sector_index=sector_index, track_index=track_index)
        cart = polar2Cartesian(d2)
        ax = ensure_canvas(initialize=False)
        ax.add_patch(
            MplPolygon(
                cart,
                closed=True,
                facecolor=resolve_color(col),
                edgecolor=resolve_color(border),
                linewidth=float(lwd),
                linestyle=resolve_lty(lty),
            )
        )
    finally:
        if os_idx is not None:
            set_current_cell(os_idx, ot_idx)


def _resize_image_nn(arr: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    h0, w0 = arr.shape[0], arr.shape[1]
    new_h = max(1, int(round(new_h)))
    new_w = max(1, int(round(new_w)))
    if new_h == h0 and new_w == w0:
        return arr
    r_idx = np.clip(np.ceil(np.arange(1, new_h + 1) * (h0 / new_h)).astype(int) - 1, 0, h0 - 1)
    c_idx = np.clip(np.ceil(np.arange(1, new_w + 1) * (w0 / new_w)).astype(int) - 1, 0, w0 - 1)
    return arr[np.ix_(r_idx, c_idx)]


def _load_raster_array(image: Any) -> np.ndarray:
    if isinstance(image, str):
        arr = np.asarray(Image.open(image))
    elif isinstance(image, Image.Image):
        arr = np.asarray(image)
    else:
        arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.dtype != np.uint8:
        arr_f = np.asarray(arr, dtype=float)
        if np.nanmax(arr_f) <= 1.0:
            arr_u8 = np.clip(np.rint(arr_f * 255.0), 0, 255).astype(np.uint8)
        else:
            arr_u8 = np.clip(np.rint(arr_f), 0, 255).astype(np.uint8)
    else:
        arr_u8 = arr
    if arr_u8.shape[2] == 3:
        alpha = np.full((arr_u8.shape[0], arr_u8.shape[1], 1), 255, dtype=np.uint8)
        arr_u8 = np.concatenate([arr_u8, alpha], axis=-1)
    return arr_u8


def circos_raster(
    image: Any,
    x: float,
    y: float,
    width: float | str | None = None,
    height: float | str | None = None,
    facing: str = "inside",
    nice_facing: bool = False,
    sector_index: str | None = None,
    track_index: int | None = None,
    scaling: float = 1.0,
    **kwargs: Any,
) -> None:
    """Add a raster image (array, PIL Image, or file path) inside a cell."""
    nkw = normalize_kwargs(kwargs)
    nice_facing = bool(nkw.get("nice_facing", nice_facing))
    sector_index = nkw.get("sector_index", sector_index)
    track_index = nkw.get("track_index", track_index)
    if sector_index is None:
        sector_index = get_current_sector_index()
    if track_index is None:
        track_index = get_current_track_index()

    arr = _load_raster_array(image)
    asp = arr.shape[0] / float(arr.shape[1])

    if facing in ("bending.inside", "bending.outside"):
        if width is None or height is None:
            raise ValueError("`width` and `height` are mandatory when `facing` is 'bending.inside' or 'bending.outside'.")
        w_f = float(width)
        h_f = float(height)
        if nice_facing:
            deg = float(circlize(x, y, sector_index=sector_index, track_index=track_index)[0, 0]) % 360.0
            if 180.0 < deg < 360.0:
                facing = "bending.outside" if facing == "bending.inside" else "bending.inside"

        arr_s = _resize_image_nn(arr, arr.shape[0] * scaling, arr.shape[1] * scaling)
        nr, nc = arr_s.shape[0], arr_s.shape[1]
        rr, cc = np.meshgrid(np.arange(1, nr + 1), np.arange(1, nc + 1), indexing="ij")
        if facing == "bending.inside":
            yv = 1.0 - (rr - 0.5) / nr
            xv = (cc - 0.5) / nc
        else:
            yv = (rr - 0.5) / nr
            xv = 1.0 - (cc - 0.5) / nc
        xv = float(x) - w_f / 2.0 + xv.ravel() * w_f
        yv = float(y) - h_f / 2.0 + yv.ravel() * h_f
        flat_rgba = arr_s.reshape(-1, 4)
        # Filter transparent white pixels (#FFFFFF00) or zero-alpha pixels
        mask = flat_rgba[:, 3] > 0
        if not np.any(mask):
            return
        hex_cols = [f"#{r:02X}{g:02X}{b:02X}{a:02X}" for r, g, b, a in flat_rgba[mask]]
        op_warn = circos_par.get_value("points.overflow.warning")
        circos_par.set_value("points_overflow_warning", False, _suppress_warn=True)
        try:
            dx = w_f / nc / 2.0
            dy = h_f / nr / 2.0
            circos_rect(
                xv[mask] - dx,
                yv[mask] - dy,
                xv[mask] + dx,
                yv[mask] + dy,
                col=hex_cols,
                border=hex_cols,
                sector_index=sector_index,
                track_index=track_index,
            )
        finally:
            circos_par.set_value("points_overflow_warning", op_warn, _suppress_warn=True)
        return

    if width is None and height is None:
        raise ValueError("At least one of `width` and `height` should be specified.")

    w_native: float | None = None
    h_native: float | None = None
    if width is not None:
        w_val, w_unit = parse_unit(width) if isinstance(width, str) else (float(width), "cm")
        w_native = float(convert_length(w_val, w_unit))
    if height is not None:
        h_val, h_unit = parse_unit(height) if isinstance(height, str) else (float(height), "cm")
        h_native = float(convert_length(h_val, h_unit))
    if w_native is None and h_native is not None:
        w_native = h_native / asp
    if h_native is None and w_native is not None:
        h_native = w_native * asp

    arr_s = _resize_image_nn(arr, arr.shape[0] * scaling, arr.shape[1] * scaling)
    df1 = circlize(x, y, sector_index=sector_index, track_index=track_index)
    cart = polar2Cartesian(df1)
    x0, y0 = float(cart[0, 0]), float(cart[0, 1])
    scale = _STATE.canvas_overlay_scale if _STATE.canvas_overlay_scale is not None else 1.0
    w_nat = float(w_native) * scale
    h_nat = float(h_native) * scale

    if nice_facing and facing != "downward":
        deg = float(df1[0, 0]) % 360.0
        if facing in ("inside", "outside"):
            if 180.0 < deg < 360.0:
                facing = "outside" if facing == "inside" else "inside"
        elif facing == "clockwise":
            if 90.0 < deg < 270.0:
                facing = "reverse.clockwise"
        elif facing == "reverse.clockwise":
            if deg < 90.0 or deg > 270.0:
                facing = "clockwise"

    if facing == "downward":
        theta = 0.0
    elif facing == "inside":
        theta = float(df1[0, 0]) - 90.0
    elif facing == "outside":
        theta = float(df1[0, 0]) - 90.0 + 180.0
    elif facing == "clockwise":
        theta = float(df1[0, 0])
    elif facing == "reverse.clockwise":
        theta = float(df1[0, 0]) + 180.0
    else:
        raise ValueError(f"Unsupported `facing`: {facing!r}")

    ax = ensure_canvas(initialize=False)
    im = ax.imshow(
        arr_s,
        extent=(-w_nat / 2.0, w_nat / 2.0, -h_nat / 2.0, h_nat / 2.0),
        origin="upper",
        zorder=2,
    )
    trans = Affine2D().rotate_deg(theta).translate(x0, y0) + ax.transData
    im.set_transform(trans)


def _get_cubic_bezier_points(x1: float, y1: float, x2: float, y2: float, xlim: Sequence[float], ylim: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    x1n = (x1 - xlim[0]) / (xlim[1] - xlim[0])
    x2n = (x2 - xlim[0]) / (xlim[1] - xlim[0])
    y1n = (y1 - ylim[0]) / (ylim[1] - ylim[0])
    y2n = (y2 - ylim[0]) / (ylim[1] - ylim[0])
    r = 0.6 - abs(x2n - x1n) / 2.0
    p = np.array([[0.0, 0.0], [0.0, r], [1.0, r], [1.0, 1.0]], dtype=float)
    t = np.linspace(0.0, 1.0, 50)[:, None]
    pt = (
        ((1.0 - t) ** 3) * p[0]
        + 3.0 * ((1.0 - t) ** 2) * t * p[1]
        + 3.0 * (1.0 - t) * (t ** 2) * p[2]
        + (t ** 3) * p[3]
    )
    xs = (pt[:, 0] * (x2n - x1n) + x1n) * (xlim[1] - xlim[0]) + xlim[0]
    ys = (pt[:, 1] * (y2n - y1n) + y1n) * (ylim[1] - ylim[0]) + ylim[0]
    return xs, ys


def circos_connect(
    x0: Any,
    y0: Any,
    x1: Any,
    y1: Any,
    sector_index: str | None = None,
    track_index: int | None = None,
    type: str = "normal",
    segments_ratio: Sequence[float] = (1.0, 1.0, 1.0),
    col: Any = "black",
    border: Any = "black",
    lwd: Any = 1.0,
    lty: Any = 1,
    **kwargs: Any,
) -> None:
    """Draw connecting lines or ribbons between two sets of points/intervals inside a cell."""
    nkw = normalize_kwargs(kwargs)
    sector_index = nkw.get("sector_index", sector_index)
    track_index = nkw.get("track_index", track_index)
    segments_ratio = nkw.get("segments_ratio", segments_ratio)
    if sector_index is None:
        sector_index = get_current_sector_index()
    if track_index is None:
        track_index = get_current_track_index()

    if type not in ("normal", "segments", "bezier"):
        raise ValueError("`type` must be 'normal', 'segments', or 'bezier'.")

    def _to_2d(val: Any) -> np.ndarray:
        arr = np.asarray(val, dtype=float)
        if arr.ndim == 0:
            return arr.reshape(1, 1)
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        return arr[:, :2]

    x0_m = _to_2d(x0)
    x1_m = _to_2d(x1)
    y0_a = np.atleast_1d(np.asarray(y0, dtype=float)).ravel()
    y1_a = np.atleast_1d(np.asarray(y1, dtype=float)).ravel()
    n = max(x0_m.shape[0], len(y0_a), x1_m.shape[0], len(y1_a))
    if x0_m.shape[0] == 1 and n > 1:
        x0_m = np.repeat(x0_m, n, axis=0)
    if x1_m.shape[0] == 1 and n > 1:
        x1_m = np.repeat(x1_m, n, axis=0)
    if len(y0_a) == 1 and n > 1:
        y0_a = np.repeat(y0_a, n)
    if len(y1_a) == 1 and n > 1:
        y1_a = np.repeat(y1_a, n)

    cols = recycle_to_length(col, n)
    borders = recycle_to_length(border, n)
    lwds = [float(v) for v in recycle_to_length(lwd, n)]
    ltys = recycle_to_length(lty, n)

    if x0_m.shape[1] == 1 and x1_m.shape[1] == 1:
        if type == "normal":
            circos_segments(x0_m[:, 0], y0_a, x1_m[:, 0], y1_a, sector_index=sector_index, track_index=track_index, col=cols, lwd=lwds, lty=ltys)
        elif type == "segments":
            sr = np.asarray(recycle_to_length(segments_ratio, 3), dtype=float)[:3]
            sr = sr / np.sum(sr)
            for i in range(n):
                h = y1_a[i] - y0_a[i]
                circos_segments(x0_m[i, 0], y0_a[i], x0_m[i, 0], y0_a[i] + h * sr[0], col=cols[i], lwd=lwds[i], lty=ltys[i], straight=True, sector_index=sector_index, track_index=track_index)
                circos_segments(x0_m[i, 0], y0_a[i] + h * sr[0], x1_m[i, 0], y0_a[i] + h * (sr[0] + sr[1]), col=cols[i], lwd=lwds[i], lty=ltys[i], sector_index=sector_index, track_index=track_index)
                circos_segments(x1_m[i, 0], y0_a[i] + h * (sr[0] + sr[1]), x1_m[i, 0], y1_a[i], col=cols[i], lwd=lwds[i], lty=ltys[i], straight=True, sector_index=sector_index, track_index=track_index)
        else:
            xlim = get_cell_meta_data("xlim", sector_index=sector_index, track_index=track_index)
            ylim = get_cell_meta_data("ylim", sector_index=sector_index, track_index=track_index)
            for i in range(n):
                bx, by = _get_cubic_bezier_points(x0_m[i, 0], y0_a[i], x1_m[i, 0], y1_a[i], xlim, ylim)
                circos_lines(bx, by, col=cols[i], lwd=lwds[i], lty=ltys[i], sector_index=sector_index, track_index=track_index)
    else:
        if type == "segments":
            raise ValueError("'segments' is only allowed for single-line connections.")
        xlim = get_cell_meta_data("xlim", sector_index=sector_index, track_index=track_index)
        ylim = get_cell_meta_data("ylim", sector_index=sector_index, track_index=track_index)
        for i in range(n):
            if type == "normal":
                if x0_m.shape[1] == 2 and x1_m.shape[1] == 2:
                    px = [x0_m[i, 0], x0_m[i, 1], x1_m[i, 1], x1_m[i, 0], x0_m[i, 0]]
                    py = [y0_a[i], y0_a[i], y1_a[i], y1_a[i], y0_a[i]]
                elif x0_m.shape[1] == 2:
                    px = [x0_m[i, 0], x0_m[i, 1], x1_m[i, 0], x0_m[i, 0]]
                    py = [y0_a[i], y0_a[i], y1_a[i], y0_a[i]]
                else:
                    px = [x0_m[i, 0], x1_m[i, 1], x1_m[i, 0], x0_m[i, 0]]
                    py = [y0_a[i], y1_a[i], y1_a[i], y0_a[i]]
                circos_polygon(px, py, col=cols[i], border=borders[i], lwd=lwds[i], lty=ltys[i], sector_index=sector_index, track_index=track_index)
            else:
                x0r = x0_m[i, 1] if x0_m.shape[1] == 2 else x0_m[i, 0]
                x1r = x1_m[i, 1] if x1_m.shape[1] == 2 else x1_m[i, 0]
                bx1, by1 = _get_cubic_bezier_points(x0r, y0_a[i], x1r, y1_a[i], xlim, ylim)
                bx2, by2 = _get_cubic_bezier_points(x1_m[i, 0], y1_a[i], x0_m[i, 0], y0_a[i], xlim, ylim)
                px = np.concatenate([bx1, bx2, [bx1[0]]])
                py = np.concatenate([by1, by2, [by1[0]]])
                circos_polygon(px, py, col=cols[i], border=borders[i], lwd=lwds[i], lty=ltys[i], sector_index=sector_index, track_index=track_index)


def circos_labels(
    sectors: Any = None,
    x: Any = None,
    labels: Any = None,
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
    """Add a label track with automatic non-overlapping label alignment."""
    from ._genomic import circos_genomic_labels

    nkw = normalize_kwargs(kwargs)
    nice_facing = bool(nkw.get("nice_facing", nice_facing))
    connection_height = nkw.get("connection_height", connection_height)
    labels_height = nkw.get("labels_height", labels_height)
    labels_side = nkw.get("labels_side", labels_side)
    track_margin = nkw.get("track_margin", track_margin)

    if sectors is None and x is None:
        env = circos_par.get_value("__tempenv__")
        if env.get("circos.heatmap.initialized") is True:
            smeta = env["sector.meta.data"]
            subset = [idx for k in get_all_sector_index() for idx in smeta[k]["subset"]]
            x = [xm for k in get_all_sector_index() for xm in smeta[k]["cell_middle"]]
            labels_list = list(labels)
            labels = [labels_list[i] for i in subset]
            sectors = [k for k in get_all_sector_index() for _ in smeta[k]["subset"]]

    bed = pd.DataFrame({"sectors": [str(s) for s in sectors], "start": np.asarray(x, dtype=float), "end": np.asarray(x, dtype=float)})
    if labels_height is None:
        max_w = max(strwidth_canvas(str(l), cex=float(cex if np.isscalar(cex) else max(cex)), font=font if np.isscalar(font) else font[0]) for l in labels) if labels else 0.05
        labels_height = min(float(cm_h(1.5)), max_w)
    if connection_height is None:
        connection_height = float(convert_length(5.0, "mm"))

    circos_genomic_labels(
        bed,
        labels=labels,
        facing=facing,
        nice_facing=nice_facing,
        col=col,
        cex=cex,
        font=font,
        padding=padding,
        connection_height=connection_height,
        line_col=line_col,
        line_lwd=line_lwd,
        line_lty=line_lty,
        labels_height=labels_height,
        side=side,
        labels_side=labels_side or side,
        track_margin=track_margin,
    )


def circos_stacked_text(
    sectors: Any,
    x: Any,
    text: Sequence[str],
    col: Any = "black",
    font: Any = 1,
    cex: Any = 1.0,
    family: str = "sans-serif",
    bg_border: Any = "black",
    bg_col: Any = "#FF8080",
    nice_facing: bool = False,
    side: str = "outside",
    **kwargs: Any,
) -> None:
    """Add a stacked text label track."""
    from ._plot import circos_track_plot_region

    nkw = normalize_kwargs(kwargs)
    bg_border = nkw.get("bg_border", bg_border)
    bg_col = nkw.get("bg_col", bg_col)
    if nice_facing:
        raise NotImplementedError("`nice_facing` is not supported in `circos_stacked_text`.")

    x_arr = np.asarray(x, dtype=float).ravel()
    n = len(x_arr)
    bg_borders = recycle_to_length(bg_border, n)
    bg_cols = recycle_to_length(bg_col, n)
    cols = recycle_to_length(col, n)
    fonts = recycle_to_length(font, n)
    cexs = [float(v) for v in recycle_to_length(cex, n)]
    txt_list = [str(t) for t in text]

    op_warn = circos_par.get_value("points.overflow.warning")
    circos_par.set_value("points_overflow_warning", False, _suppress_warn=True)
    try:
        circos_track_plot_region(ylim=[0, 1], bg_border=None)
        df = refer_to_one_sector(sectors, x_arr)
        set_current_sector_index(str(df.iloc[0, 0]))
        ord_idx = df["order"].to_numpy(dtype=int)
        x_mapped = df["x"].to_numpy(dtype=float)
        txt_ord = [txt_list[i] for i in ord_idx]
        cols_ord = [cols[i] for i in ord_idx]
        fonts_ord = [fonts[i] for i in ord_idx]
        cexs_ord = [cexs[i] for i in ord_idx]
        bgc_ord = [bg_cols[i] for i in ord_idx]
        bgb_ord = [bg_borders[i] for i in ord_idx]

        th = circos_strheight(txt_ord, cex=cexs_ord, font=fonts_ord, family=family)
        ta = circos_str_ascent(txt_ord, cex=cexs_ord, font=fonts_ord, family=family)
        td = circos_str_descent(txt_ord, cex=cexs_ord, font=fonts_ord, family=family)
        rows = stack_text(x_mapped, txt_ord, margin=0.2, cex=cexs_ord, font=fonts_ord, family=family)

        y_start = float(mm_y(4.5)) if side == "outside" else 1.0 - float(mm_y(4.5))
        y = y_start
        for r_idx in rows:
            idx = np.asarray(r_idx, dtype=int)
            if side == "outside":
                circos_segments(x_mapped[idx], np.zeros(len(idx)), x_mapped[idx], np.repeat(y, len(idx)), col=[bgb_ord[j] for j in idx])
                y = y + float(mm_y(0.5)) + float(np.max(td[idx] + ta[idx])) + float(mm_y(1.0))
            else:
                circos_segments(x_mapped[idx], np.ones(len(idx)), x_mapped[idx], np.repeat(y, len(idx)), col=[bgb_ord[j] for j in idx])
                y = y - float(mm_y(0.5)) - float(np.max(td[idx] + ta[idx])) - float(mm_y(1.0))

        y = y_start
        for r_idx in rows:
            idx = np.asarray(r_idx, dtype=int)
            tw = circos_strwidth(txt_ord, h=y, cex=cexs_ord, font=fonts_ord, family=family)
            if side == "outside":
                circos_rect(
                    x_mapped[idx] - tw[idx] / 2.0 - float(mm_x(0.5)),
                    np.repeat(y, len(idx)),
                    x_mapped[idx] + tw[idx] / 2.0 + float(mm_x(0.5)),
                    y + float(mm_y(0.5)) + td[idx] + ta[idx] + float(mm_y(0.5)),
                    col=[bgc_ord[j] for j in idx],
                    border=[bgb_ord[j] for j in idx],
                )
                circos_text(
                    x_mapped[idx],
                    y + float(mm_y(0.5)) + td[idx] + th[idx] / 2.0,
                    [txt_ord[j] for j in idx],
                    facing="bending.inside",
                    col=[cols_ord[j] for j in idx],
                    cex=[cexs_ord[j] for j in idx],
                    font=[fonts_ord[j] for j in idx],
                    family=family,
                )
                y = y + float(mm_y(0.5)) + float(np.max(td[idx] + ta[idx])) + float(mm_y(1.0))
            else:
                circos_rect(
                    x_mapped[idx] - tw[idx] / 2.0 - float(mm_x(0.5)),
                    np.repeat(y, len(idx)),
                    x_mapped[idx] + tw[idx] / 2.0 + float(mm_x(0.5)),
                    y - float(mm_y(0.5)) - td[idx] - ta[idx] - float(mm_y(0.5)),
                    col=[bgc_ord[j] for j in idx],
                    border=[bgb_ord[j] for j in idx],
                )
                circos_text(
                    x_mapped[idx],
                    y - float(mm_y(0.5)) - (ta[idx] - th[idx] / 2.0),
                    [txt_ord[j] for j in idx],
                    facing="bending.inside",
                    col=[cols_ord[j] for j in idx],
                    cex=[cexs_ord[j] for j in idx],
                    font=[fonts_ord[j] for j in idx],
                    family=family,
                )
                y = y - float(mm_y(0.5)) - float(np.max(td[idx] + ta[idx])) - float(mm_y(1.0))
    finally:
        circos_par.set_value("points_overflow_warning", op_warn, _suppress_warn=True)
