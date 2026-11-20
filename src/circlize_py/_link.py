"""Quadratic Bezier links, ribbons, arrowheads, and evenly spaced link helpers."""

from __future__ import annotations

import math
from typing import Any, Sequence
import warnings

from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import pandas as pd

from ._canvas import ensure_canvas
from ._colors import resolve_color, resolve_lty
from ._state import (
    _STATE,
    circos_par,
    get_all_sector_index,
    get_cell_meta_data,
    get_most_inside_radius,
    get_sector_data,
    is_circos_initialized,
    normalize_kwargs,
)
from ._utils import as_radian, circlize, convert_length


def _scale_xy(arr: np.ndarray) -> np.ndarray:
    scale = _STATE.canvas_overlay_scale if _STATE.canvas_overlay_scale is not None else 1.0
    if scale == 1.0:
        return arr
    return arr * scale


def degree_diff(theta1: float, theta2: float) -> float:
    t1 = theta1 % 360.0
    t2 = theta2 % 360.0
    d = abs(t2 - t1)
    return (d - 180.0) if d > 180.0 else d


def degree_diff2(theta1: float, theta2: float) -> float:
    return (theta2 - theta1) % 360.0


def line_degree(x0: float, y0: float, x1: float, y1: float) -> float:
    return math.degrees(math.atan2(y1 - y0, x1 - x0)) % 360.0


def arc_points(theta1: float, theta2: float, rou: float) -> np.ndarray:
    """Sample arc points from theta1 to theta2 counter-clockwise (matching R arc.points)."""
    t1 = theta1 % 360.0
    t2 = theta2 % 360.0
    t_diff = (t2 - t1) % 360.0
    length = as_radian(t_diff) * rou
    seg = 2.0 * math.pi / float(circos_par.get_value("unit.circle.segments"))
    ncut = max(2, int(math.floor(length / seg)))
    k = np.arange(1, ncut, dtype=float)
    degs = t1 + k * t_diff / (ncut - 1.0)
    rads = np.deg2rad(degs)
    return np.column_stack([rou * np.cos(rads), rou * np.sin(rads)])


def arc_midpoint(theta1: float, theta2: float, rou: float) -> np.ndarray:
    t1 = theta1 % 360.0
    t2 = theta2 % 360.0
    if t2 < t1:
        t2 += 360.0
    theta = (t1 + t2) / 2.0
    r = as_radian(theta)
    return np.array([[rou * math.cos(r), rou * math.sin(r)]], dtype=float)


def quadratic_bezier_length(p0: Sequence[float], p1: Sequence[float], p2: Sequence[float], w: float = 1.0) -> float:
    n = 50
    t = np.linspace(0.0, 1.0, n)
    omt = 1.0 - t
    denom = omt ** 2 + 2.0 * t * omt * w + t ** 2
    x = (omt ** 2 * p0[0] + 2.0 * t * omt * p1[0] * w + t ** 2 * p2[0]) / denom
    y = (omt ** 2 * p0[1] + 2.0 * t * omt * p1[1] * w + t ** 2 * p2[1]) / denom
    return float(np.sum(np.hypot(np.diff(x), np.diff(y))))


def quadratic_bezier(p0: Sequence[float], p1: Sequence[float], p2: Sequence[float], w: float = 1.0) -> np.ndarray:
    seg = 2.0 * math.pi / float(circos_par.get_value("unit.circle.segments"))
    ncut = int(math.floor(quadratic_bezier_length(p0, p1, p2, w=w) / seg))
    ncut = max(3, ncut)
    if ncut % 2 == 0:
        ncut += 1
    t = np.linspace(0.0, 1.0, ncut)
    omt = 1.0 - t
    denom = omt ** 2 + 2.0 * t * omt * w + t ** 2
    x = (omt ** 2 * p0[0] + 2.0 * t * omt * p1[0] * w + t ** 2 * p2[0]) / denom
    y = (omt ** 2 * p0[1] + 2.0 * t * omt * p1[1] * w + t ** 2 * p2[1]) / denom
    return np.column_stack([x, y])


def get_quadratic_points(
    theta1: float,
    theta2: float,
    rou1: float,
    rou2: float,
    h: float | None = None,
    h_ratio: float = 0.5,
    w: float = 1.0,
) -> np.ndarray:
    """Compute rational quadratic Bezier points connecting (theta1, rou1) and (theta2, rou2)."""
    t1 = float(theta1) % 360.0
    t2 = float(theta2) % 360.0
    if abs(abs(t2 - t1) - 180.0) < 1e-3:
        t1 += 1e-3

    if abs(t2 - t1) > 180.0:
        theta_mid = ((t2 + t1) / 2.0 - 180.0) % 360.0
    else:
        theta_mid = ((t2 + t1) / 2.0) % 360.0

    rou_min = min(float(rou1), float(rou2))
    x1 = rou_min * math.cos(as_radian(t1))
    y1 = rou_min * math.sin(as_radian(t1))
    x2 = rou_min * math.cos(as_radian(t2))
    y2 = rou_min * math.sin(as_radian(t2))

    beta = (t1 - t2) % 360.0
    if beta > 180.0:
        beta = 360.0 - beta
    h_auto = rou_min * (1.0 - float(h_ratio) * math.cos(as_radian(beta / 2.0)))

    if h is None or float(h) > rou_min:
        h_val = h_auto
    else:
        h_val = float(h)

    h2 = h_val - rou_min * (1.0 - math.cos(as_radian(beta / 2.0)))
    if w < 0:
        h2 = -h2

    dis = 0.5 * math.hypot(x1 - x2, y1 - y2)
    p0 = (-dis, 0.0)
    p2 = (dis, 0.0)
    p1 = (0.0, 0.0) if w == 0 else (0.0, (1.0 + w) / w * h2)
    d = quadratic_bezier(p0, p1, p2, w=w)

    # Match R's exact precedence: ((t1 + t2) / 2) % 360
    alpha = -90.0 + (360.0 - (((t1 + t2) / 2.0) % 360.0))
    if (theta_mid > 270.0 or theta_mid < 90.0) and ((t1 - 180.0) * (t2 - 180.0) < 0.0):
        alpha += 180.0

    arad = as_radian(alpha)
    ca, sa = math.cos(arad), math.sin(arad)
    # R: matrix(c(cos(a), sin(a), -sin(a), cos(a)), nrow=2); d %*% mat
    mat = np.array([[ca, -sa], [sa, ca]], dtype=float)
    d = d @ mat

    x0 = (x1 + x2) / 2.0
    y0 = (y1 + y2) / 2.0
    d[:, 0] += x0
    d[:, 1] += y0

    if ((t1 - t2) % 360.0) < 180.0:
        d = d[::-1].copy()

    if rou1 > rou2:
        p_start = np.array([[rou1 * math.cos(as_radian(t1)), rou1 * math.sin(as_radian(t1))]], dtype=float)
        d = np.vstack([p_start, d])
    elif rou1 < rou2:
        p_end = np.array([[rou2 * math.cos(as_radian(t2)), rou2 * math.sin(as_radian(t2))]], dtype=float)
        d = np.vstack([d, p_end])
    return d


def draw_arrowhead(
    x0: float,
    y0: float,
    angle: float = 0.0,
    arr_length: float = 0.4,
    arr_width: float | None = None,
    arr_adj: float = 1.0,
    arr_type: str = "triangle",
    lcol: Any = "black",
    lty: Any = 1,
    arr_col: Any = None,
    arr_lwd: float = 0.1,
) -> None:
    """Draw an arrowhead at (x0, y0) oriented along `angle` degrees (ported from shape::Arrowhead)."""
    if arr_type == "none":
        return
    if arr_width is None:
        arr_width = arr_length / 2.0
    if arr_col is None:
        arr_col = lcol

    npoint = 5
    if arr_type == "curved":
        rad = 0.7
        ln = 0.25 * math.pi
        xs = np.linspace(1.5 * math.pi + ln, 1.5 * math.pi, npoint)
        r1 = np.column_stack([-rad * np.cos(xs), rad + rad * np.sin(xs)])
        xs_rev = xs[::-1]
        r2 = np.column_stack([-rad * np.cos(xs_rev), -rad - rad * np.sin(xs_rev)])
        mid_x = r2[-1, 0]
        rd = r1[0, 1]
        xs3 = np.linspace(math.pi / 2.0, 1.5 * math.pi, 3 * npoint)
        r3 = np.column_stack([mid_x - rd * 0.25 * np.cos(xs3), -rd * np.sin(xs3)])
        rr = np.vstack([r1, r2, r3])
        rr[:, 0] *= 2.6
        rr[:, 1] *= 3.45
    elif arr_type in ("circle", "ellipse"):
        if arr_type == "circle":
            arr_width = arr_length
        rad = 0.1
        t = np.linspace(0.0, 2.0 * math.pi, 15 * npoint)
        rr = 6.22 * np.column_stack([-rad + rad * np.sin(t), rad * np.cos(t)])
    elif arr_type == "T":
        rr = 6.22 * np.array([[-0.01, -0.1], [0.0, -0.1], [0.0, 0.1], [-0.01, 0.1]], dtype=float)
    else:  # triangle / simple
        rr = 6.22 * np.array([[-0.2, -0.1], [0.0, 0.0], [-0.2, 0.1]], dtype=float)

    if arr_adj == 0.5:
        rr[:, 0] -= float(np.min(rr[:, 0])) / 2.0
    elif arr_adj == 0.0:
        rr[:, 0] -= float(np.min(rr[:, 0]))

    s_cm = float(convert_length(1.0, "cm"))
    scale = _STATE.canvas_overlay_scale if _STATE.canvas_overlay_scale is not None else 1.0
    s_cm *= scale

    dx = rr[:, 0] * float(arr_length)
    dy = rr[:, 1] * float(arr_width)
    ang = as_radian(angle)
    ca, sa = math.cos(ang), math.sin(ang)
    rx = x0 + (ca * dx - sa * dy) * s_cm
    ry = y0 + (sa * dx + ca * dy) * s_cm

    ax = ensure_canvas(initialize=False)
    ax.add_patch(
        MplPolygon(
            np.column_stack([rx, ry]),
            closed=True,
            facecolor=resolve_color(arr_col),
            edgecolor=resolve_color(lcol),
            linewidth=float(arr_lwd),
            linestyle=resolve_lty(lty),
            joinstyle="miter",
        )
    )


def circos_link(
    sector_index1: str,
    point1: float | Sequence[float],
    sector_index2: str,
    point2: float | Sequence[float],
    rou: float | None = None,
    rou1: float | None = None,
    rou2: float | None = None,
    h: float | None = None,
    h_ratio: float = 0.5,
    w: float = 1.0,
    h2: float | None = None,
    w2: float | None = None,
    inverse: bool = False,
    col: Any = "black",
    lwd: float = 1.0,
    lty: Any = 1,
    border: Any = None,
    directional: int = 0,
    arr_length: float | None = None,
    arr_width: float | None = None,
    arr_type: str = "triangle",
    arr_lty: Any = None,
    arr_lwd: float | None = None,
    arr_col: Any = None,
    reduce_to_mid_line: bool = False,
    **kwargs: Any,
) -> None:
    """Draw a line or ribbon link between points/intervals in two sectors."""
    nkw = normalize_kwargs(kwargs)
    sector_index1 = str(nkw.get("sector_index1", sector_index1))
    sector_index2 = str(nkw.get("sector_index2", sector_index2))
    h_ratio = float(nkw.get("h_ratio", h_ratio))
    arr_type = nkw.get("arr_type", arr_type)
    arr_length = nkw.get("arr_length", arr_length)
    arr_width = nkw.get("arr_width", arr_width)
    arr_lty = nkw.get("arr_lty", arr_lty)
    arr_lwd = nkw.get("arr_lwd", arr_lwd)
    arr_col = nkw.get("arr_col", arr_col)
    reduce_to_mid_line = bool(nkw.get("reduce_to_mid_line", reduce_to_mid_line))

    if rou is None:
        rou = get_most_inside_radius()
    if rou1 is None:
        rou1 = rou
    if rou2 is None:
        rou2 = rou
    rou1 = float(rou1)
    rou2 = float(rou2)

    if h2 is None:
        h2 = h
    if w2 is None:
        w2 = w
    if border is None:
        border = col
    if arr_length is None:
        arr_length = 0.02 if arr_type == "big.arrow" else 0.4
    if arr_width is None:
        arr_width = float(arr_length) / 2.0
    if arr_lty is None:
        arr_lty = lty
    if arr_lwd is None:
        arr_lwd = lwd
    if arr_col is None:
        arr_col = col

    sdata1 = get_sector_data(sector_index1)
    sdata2 = get_sector_data(sector_index2)

    def _norm_pt(pt: Any) -> np.ndarray:
        if isinstance(pt, pd.DataFrame):
            arr = pt.to_numpy(dtype=float).ravel()
        else:
            arr = np.atleast_1d(np.asarray(pt, dtype=float)).ravel()
        return arr

    p1 = _norm_pt(point1)
    p2 = _norm_pt(point2)

    if circos_par.get_value("ring"):
        if len(p1) == 2 and p1[0] > p1[1]:
            sd = get_sector_data()
            p1[0] -= sd["max.value"] - sd["min.value"]
        if len(p2) == 2 and p2[0] > p2[1]:
            sd = get_sector_data()
            p2[0] -= sd["max.value"] - sd["min.value"]

    p1 = np.sort(p1)
    p2 = np.sort(p2)

    rg1 = sdata1["max.data"] - sdata1["min.data"]
    if circos_par.get_value("message"):
        if np.min(p1) < sdata1["min.data"] - rg1 * 1e-6 or np.max(p1) > sdata1["max.data"] + rg1 * 1e-6:
            warnings.warn(f"Note: The first link end is drawn out of sector '{sector_index1}'.", UserWarning, stacklevel=2)
        rg2 = sdata2["max.data"] - sdata2["min.data"]
        if np.min(p2) < sdata2["min.data"] - rg2 * 1e-6 or np.max(p2) > sdata2["max.data"] + rg2 * 1e-6:
            warnings.warn(f"Note: The second link end is drawn out of sector '{sector_index2}'.", UserWarning, stacklevel=2)

    if reduce_to_mid_line:
        p1 = np.array([float(np.mean(p1))], dtype=float)
        p2 = np.array([float(np.mean(p2))], dtype=float)

    ax = ensure_canvas(initialize=False)
    h_mid = ((h + h2) / 2.0) if (h is not None and h2 is not None) else None
    w_mid = (float(w) + float(w2)) / 2.0

    def _draw_poly(pts: np.ndarray) -> None:
        pts_s = _scale_xy(pts)
        ec = resolve_color(border)
        ls = resolve_lty(lty)
        if ls == "none":
            ec = "none"
        ax.add_patch(
            MplPolygon(pts_s, closed=True, facecolor=resolve_color(col), edgecolor=ec, linewidth=float(lwd), linestyle=ls)
        )

    def _draw_line(pts: np.ndarray, l_col: Any, l_lwd: float, l_lty: Any) -> None:
        pts_s = _scale_xy(pts)
        c_val = resolve_color(l_col, default="black")
        ls = resolve_lty(l_lty)
        if c_val == "none" or ls == "none":
            return
        ax.add_line(Line2D(pts_s[:, 0], pts_s[:, 1], color=c_val, linewidth=float(l_lwd), linestyle=ls, solid_capstyle="butt"))

    def _draw_center_arrows(dcenter: np.ndarray, adj_end: float = 1.0) -> None:
        nr = len(dcenter)
        if nr <= 1 or arr_type == "big.arrow":
            return
        if directional == 1:
            _draw_line(dcenter[:-1], arr_col, arr_lwd, arr_lty)
        elif directional == -1:
            _draw_line(dcenter[1:], arr_col, arr_lwd, arr_lty)
        elif directional == 2:
            if nr > 2:
                _draw_line(dcenter[1:-1], arr_col, arr_lwd, arr_lty)

        dc_s = _scale_xy(dcenter)
        if directional in (1, 2):
            alpha = line_degree(dc_s[-2, 0], dc_s[-2, 1], dc_s[-1, 0], dc_s[-1, 1])
            draw_arrowhead(dc_s[-1, 0], dc_s[-1, 1], alpha, arr_length=arr_length, arr_width=arr_width, arr_adj=adj_end, arr_type=arr_type, arr_col=arr_col, lcol=arr_col, arr_lwd=0.1)
        if directional in (-1, 2):
            alpha = line_degree(dc_s[1, 0], dc_s[1, 1], dc_s[0, 0], dc_s[0, 1])
            draw_arrowhead(dc_s[0, 0], dc_s[0, 1], alpha, arr_length=arr_length, arr_width=arr_width, arr_adj=1.0, arr_type=arr_type, arr_col=arr_col, lcol=arr_col, arr_lwd=0.1)

    if len(p1) == 1 and len(p2) == 1:
        t1 = float(circlize(p1[0], 0.0, sector_index=sector_index1, track_index=0)[0, "theta"])
        t2 = float(circlize(p2[0], 0.0, sector_index=sector_index2, track_index=0)[0, "theta"])
        d = get_quadratic_points(t1, t2, rou1, rou2, h=h, h_ratio=h_ratio, w=w)
        _draw_line(d, col, lwd, lty)
        if len(d) > 1:
            d_s = _scale_xy(d)
            if directional in (1, 2):
                alpha = line_degree(d_s[-2, 0], d_s[-2, 1], d_s[-1, 0], d_s[-1, 1])
                draw_arrowhead(d_s[-1, 0], d_s[-1, 1], alpha, arr_length=arr_length, arr_width=arr_width, arr_adj=1.0, arr_type=arr_type, arr_col=col, lcol=col, arr_lwd=0.1)
            if directional in (-1, 2):
                alpha = line_degree(d_s[1, 0], d_s[1, 1], d_s[0, 0], d_s[0, 1])
                draw_arrowhead(d_s[0, 0], d_s[0, 1], alpha, arr_length=arr_length, arr_width=arr_width, arr_adj=1.0, arr_type=arr_type, arr_col=col, lcol=col, arr_lwd=0.1)

    elif len(p1) == 1:
        t1 = float(circlize(p1[0], 0.0, sector_index=sector_index1, track_index=0)[0, "theta"])
        t21 = float(circlize(p2[0], 0.0, sector_index=sector_index2, track_index=0)[0, "theta"])
        t22 = float(circlize(p2[1], 0.0, sector_index=sector_index2, track_index=0)[0, "theta"])
        if degree_diff2(t1, t21) <= degree_diff(t22, t21) and degree_diff2(t22, t1) <= degree_diff(t22, t21):
            d = get_quadratic_points(t22, t21, max(rou1, rou2), max(rou1, rou2), h=h, h_ratio=h_ratio, w=w)
            r = arc_points(t22, t21, rou2)
            _draw_poly(np.vstack([d, r[::-1]]))
        else:
            if degree_diff(t1, t21) > degree_diff(t1, t22):
                d1 = get_quadratic_points(t1, t21, rou1, rou2, h=h, h_ratio=h_ratio, w=w)
                d2 = get_quadratic_points(t1, t22, rou1, rou2, h=h2, h_ratio=h_ratio, w=w2)
                d1x = get_quadratic_points(t1, t21, rou1, rou2 - float(arr_length), h=h, h_ratio=h_ratio, w=w)
                d2x = get_quadratic_points(t1, t22, rou1, rou2 - float(arr_length), h=h2, h_ratio=h_ratio, w=w2)
            else:
                d1 = get_quadratic_points(t1, t21, rou1, rou2, h=h2, h_ratio=h_ratio, w=w2)
                d2 = get_quadratic_points(t1, t22, rou1, rou2, h=h, h_ratio=h_ratio, w=w)
                d1x = get_quadratic_points(t1, t21, rou1, rou2 - float(arr_length), h=h2, h_ratio=h_ratio, w=w2)
                d2x = get_quadratic_points(t1, t22, rou1, rou2 - float(arr_length), h=h, h_ratio=h_ratio, w=w)
            dcenter = get_quadratic_points(t1, (t21 + t22) / 2.0, rou1, rou2, h=h_mid, h_ratio=h_ratio, w=w_mid)
            if arr_type == "big.arrow" and directional == 1:
                r2 = arc_midpoint(t22, t21, rou2)
                poly = np.vstack([d1x, r2[::-1], d2x[::-1]])
            else:
                r2 = arc_points(t22, t21, rou2)
                poly = np.vstack([d1, r2[::-1], d2[::-1]])
            _draw_poly(poly)
            _draw_center_arrows(dcenter, adj_end=1.0)

    elif len(p2) == 1:
        t2 = float(circlize(p2[0], 0.0, sector_index=sector_index2, track_index=0)[0, "theta"])
        t11 = float(circlize(p1[0], 0.0, sector_index=sector_index1, track_index=0)[0, "theta"])
        t12 = float(circlize(p1[1], 0.0, sector_index=sector_index1, track_index=0)[0, "theta"])
        if degree_diff2(t2, t11) <= degree_diff2(t12, t11) and degree_diff2(t12, t2) <= degree_diff2(t12, t11):
            d = get_quadratic_points(t12, t11, max(rou1, rou2), max(rou1, rou2), h=h, h_ratio=h_ratio, w=w)
            r = arc_points(t12, t11, rou1)
            _draw_poly(np.vstack([d, r[::-1]]))
        else:
            if degree_diff(t2, t11) > degree_diff(t2, t12):
                d1 = get_quadratic_points(t11, t2, rou1, rou2, h=h, h_ratio=h_ratio, w=w)
                d2 = get_quadratic_points(t12, t2, rou1, rou2, h=h2, h_ratio=h_ratio, w=w2)
                d1x = get_quadratic_points(t11, t2, rou1 - float(arr_length), rou2, h=h, h_ratio=h_ratio, w=w)
                d2x = get_quadratic_points(t12, t2, rou1 - float(arr_length), rou2, h=h2, h_ratio=h_ratio, w=w2)
            else:
                d1 = get_quadratic_points(t11, t2, rou1, rou2, h=h2, h_ratio=h_ratio, w=w2)
                d2 = get_quadratic_points(t12, t2, rou1, rou2, h=h, h_ratio=h_ratio, w=w)
                d1x = get_quadratic_points(t11, t2, rou1 - float(arr_length), rou2, h=h2, h_ratio=h_ratio, w=w2)
                d2x = get_quadratic_points(t12, t2, rou1 - float(arr_length), rou2, h=h, h_ratio=h_ratio, w=w)
            dcenter = get_quadratic_points((t11 + t12) / 2.0, t2, rou1, rou2, h=h_mid, h_ratio=h_ratio, w=w_mid)
            if arr_type == "big.arrow" and directional == -1:
                r1 = arc_midpoint(t12, t11, rou1)
                poly = np.vstack([d1x[::-1], r1[::-1], d2x])
            else:
                r1 = arc_points(t12, t11, rou1)
                poly = np.vstack([d1[::-1], r1[::-1], d2])
            _draw_poly(poly)
            _draw_center_arrows(dcenter, adj_end=1.0)

    else:
        t11 = float(circlize(p1[0], 0.0, sector_index=sector_index1, track_index=0)[0, "theta"])
        t12 = float(circlize(p1[1], 0.0, sector_index=sector_index1, track_index=0)[0, "theta"])
        t21 = float(circlize(p2[0], 0.0, sector_index=sector_index2, track_index=0)[0, "theta"])
        t22 = float(circlize(p2[1], 0.0, sector_index=sector_index2, track_index=0)[0, "theta"])

        if (
            degree_diff2(t12, t21) <= degree_diff2(t22, t21) + degree_diff2(t12, t11)
            and degree_diff2(t12, t21) >= degree_diff2(t22, t11)
            and abs(rou1 - rou2) < 1e-5
        ):
            rm = max(rou1, rou2)
            d = get_quadratic_points(t12, t21, rm, rm, h=h, h_ratio=h_ratio, w=w)
            r = arc_points(t12, t21, rm)
            _draw_poly(np.vstack([d, r[::-1]]))
        elif (
            degree_diff2(t22, t11) <= degree_diff2(t12, t11) + degree_diff2(t22, t21)
            and degree_diff2(t22, t11) >= degree_diff2(t12, t21)
            and abs(rou1 - rou2) < 1e-5
        ):
            rm = max(rou1, rou2)
            d = get_quadratic_points(t22, t11, rm, rm, h=h, h_ratio=h_ratio, w=w)
            r = arc_points(t22, t11, rm)
            _draw_poly(np.vstack([d, r[::-1]]))
        else:
            d1x = d2x = None
            if degree_diff(t11, t22) > degree_diff(t12, t21):
                if inverse:
                    d1 = get_quadratic_points(t11, t21, rou1, rou2, h=h, h_ratio=h_ratio, w=w)
                    d2 = get_quadratic_points(t12, t22, rou1, rou2, h=h, h_ratio=h_ratio, w=w)
                else:
                    d1 = get_quadratic_points(t11, t22, rou1, rou2, h=h, h_ratio=h_ratio, w=w)
                    d2 = get_quadratic_points(t12, t21, rou1, rou2, h=h2, h_ratio=h_ratio, w=w2)
                if directional == 1:
                    if inverse:
                        d1x = get_quadratic_points(t11, t21, rou1, rou2 - float(arr_length), h=h, h_ratio=h_ratio, w=w)
                        d2x = get_quadratic_points(t12, t22, rou1, rou2 - float(arr_length), h=h, h_ratio=h_ratio, w=w)
                    else:
                        d1x = get_quadratic_points(t11, t22, rou1, rou2 - float(arr_length), h=h, h_ratio=h_ratio, w=w)
                        d2x = get_quadratic_points(t12, t21, rou1, rou2 - float(arr_length), h=h2, h_ratio=h_ratio, w=w2)
                elif directional == -1:
                    if inverse:
                        d1x = get_quadratic_points(t11, t21, rou1 - float(arr_length), rou2, h=h, h_ratio=h_ratio, w=w)
                        d2x = get_quadratic_points(t12, t22, rou1 - float(arr_length), rou2, h=h, h_ratio=h_ratio, w=w)
                    else:
                        d1x = get_quadratic_points(t11, t22, rou1 - float(arr_length), rou2, h=h, h_ratio=h_ratio, w=w)
                        d2x = get_quadratic_points(t12, t21, rou1 - float(arr_length), rou2, h=h2, h_ratio=h_ratio, w=w2)
                elif directional == 2:
                    d1x = get_quadratic_points(t11, t21 if inverse else t22, rou1 - float(arr_length), rou2 - float(arr_length), h=h, h_ratio=h_ratio, w=w)
                    d2x = get_quadratic_points(t12, t22 if inverse else t21, rou1 - float(arr_length), rou2 - float(arr_length), h=h if inverse else h2, h_ratio=h_ratio, w=w if inverse else w2)
            else:
                if inverse:
                    d1 = get_quadratic_points(t11, t21, rou1, rou2, h=h, h_ratio=h_ratio, w=w)
                    d2 = get_quadratic_points(t12, t22, rou1, rou2, h=h, h_ratio=h_ratio, w=w)
                else:
                    d1 = get_quadratic_points(t11, t22, rou1, rou2, h=h2, h_ratio=h_ratio, w=w2)
                    d2 = get_quadratic_points(t12, t21, rou1, rou2, h=h, h_ratio=h_ratio, w=w)
                if directional == 1:
                    if inverse:
                        d1x = get_quadratic_points(t11, t21, rou1, rou2 - float(arr_length), h=h, h_ratio=h_ratio, w=w)
                        d2x = get_quadratic_points(t12, t22, rou1, rou2 - float(arr_length), h=h, h_ratio=h_ratio, w=w)
                    else:
                        d1x = get_quadratic_points(t11, t22, rou1, rou2 - float(arr_length), h=h2, h_ratio=h_ratio, w=w2)
                        d2x = get_quadratic_points(t12, t21, rou1, rou2 - float(arr_length), h=h, h_ratio=h_ratio, w=w)
                elif directional == -1:
                    if inverse:
                        d1x = get_quadratic_points(t11, t21, rou1 - float(arr_length), rou2, h=h, h_ratio=h_ratio, w=w)
                        d2x = get_quadratic_points(t12, t22, rou1 - float(arr_length), rou2, h=h, h_ratio=h_ratio, w=w)
                    else:
                        d1x = get_quadratic_points(t11, t22, rou1 - float(arr_length), rou2, h=h2, h_ratio=h_ratio, w=w2)
                        d2x = get_quadratic_points(t12, t21, rou1 - float(arr_length), rou2, h=h, h_ratio=h_ratio, w=w)
                elif directional == 2:
                    d1x = get_quadratic_points(t11, t21 if inverse else t22, rou1 - float(arr_length), rou2 - float(arr_length), h=h if inverse else h2, h_ratio=h_ratio, w=w if inverse else w2)
                    d2x = get_quadratic_points(t12, t22 if inverse else t21, rou1 - float(arr_length), rou2 - float(arr_length), h=h, h_ratio=h_ratio, w=w)

            dcenter = get_quadratic_points((t11 + t12) / 2.0, (t21 + t22) / 2.0, rou1, rou2, h=h_mid, h_ratio=h_ratio, w=w_mid)
            r2 = arc_points(t22, t21, rou2)
            r1 = arc_points(t12, t11, rou1)
            if inverse:
                r2 = r2[::-1]

            if arr_type == "big.arrow" and directional in (1, -1, 2):
                if directional == 1:
                    r2 = arc_midpoint(t22, t21, rou2)
                elif directional == -1:
                    r1 = arc_midpoint(t12, t11, rou1)
                else:
                    r1 = arc_midpoint(t12, t11, rou1)
                    r2 = arc_midpoint(t22, t21, rou2)
                poly = np.vstack([d1x, r2, d2x[::-1], r1])
            else:
                poly = np.vstack([d1, r2, d2[::-1], r1])

            _draw_poly(poly)
            _draw_center_arrows(dcenter, adj_end=1.5)


def arrange_links_evenly(df: pd.DataFrame, directional: int = 0) -> pd.DataFrame:
    """Arrange single-line links evenly along each sector's xlim."""
    if not is_circos_initialized():
        raise RuntimeError("Circular layout needs to be initialized.")

    from ._chord import chord_diagram

    df_in = pd.DataFrame(df).copy()
    sectors = get_all_sector_index()
    c1 = df_in.iloc[:, 0].astype(str)
    c2 = df_in.iloc[:, 1].astype(str)
    if not set(c1).issubset(set(sectors)) or not set(c2).issubset(set(sectors)):
        raise ValueError("The two columns in `df` should only contain sector names.")

    df2 = chord_diagram(df_in, directional=int(directional), plot=False)
    all_ends = list(c1) + list(c2)
    counts: dict[str, int] = {}
    for s in all_ends:
        counts[s] = counts.get(s, 0) + 1

    pos1 = np.empty(len(df2), dtype=float)
    pos2 = np.empty(len(df2), dtype=float)
    for i in range(len(df2)):
        s1 = str(df2.iloc[i]["rn"])
        s2 = str(df2.iloc[i]["cn"])
        xlim1 = get_cell_meta_data("xlim", sector_index=s1)
        xlim2 = get_cell_meta_data("xlim", sector_index=s2)
        pos1[i] = (float(df2.iloc[i]["x1"]) - 0.5) / counts[s1] * (xlim1[1] - xlim1[0]) + xlim1[0]
        pos2[i] = (float(df2.iloc[i]["x2"]) - 0.5) / counts[s2] * (xlim2[1] - xlim2[0]) + xlim2[0]

    return pd.DataFrame(
        {
            "sector1": df2["rn"].tolist(),
            "sector2": df2["cn"].tolist(),
            "pos1": pos1,
            "pos2": pos2,
        }
    )
