"""Nested circular plots combining an outer zoom ring with an inner circular layout."""

from __future__ import annotations

import copy
import warnings
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._low_level import circos_polygon
from ._plot import circos_track_plot_region
from ._state import (
    _STATE,
    circos_clear,
    circos_par,
    get_all_sector_index,
    get_most_inside_radius,
    is_circos_initialized,
    normalize_kwargs,
)
from ._utils import circlize, convert_length, recycle_to_length, reverse_circlize


def _coerce_correspondance(correspondance: Any) -> pd.DataFrame:
    if isinstance(correspondance, pd.DataFrame):
        df = correspondance.copy()
    else:
        df = pd.DataFrame(correspondance)
    if df.shape[1] < 6:
        raise ValueError("`correspondance` must have 6 columns: (s1, start1, end1, s2, start2, end2).")
    df = df.iloc[:, :6].copy()
    df.columns = ["s1", "start1", "end1", "s2", "start2", "end2"]
    df["s1"] = df["s1"].astype(str)
    df["start1"] = pd.to_numeric(df["start1"], errors="raise").astype(float)
    df["end1"] = pd.to_numeric(df["end1"], errors="raise").astype(float)
    df["s2"] = df["s2"].astype(str)
    df["start2"] = pd.to_numeric(df["start2"], errors="raise").astype(float)
    df["end2"] = pd.to_numeric(df["end2"], errors="raise").astype(float)
    return df.reset_index(drop=True)


def _cubic_bezier_2d(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    n: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, n)
    omt = 1.0 - t
    x = (omt**3) * p0[0] + 3.0 * (omt**2) * t * p1[0] + 3.0 * omt * (t**2) * p2[0] + (t**3) * p3[0]
    y = (omt**3) * p0[1] + 3.0 * (omt**2) * t * p1[1] + 3.0 * omt * (t**2) * p2[1] + (t**3) * p3[1]
    return x, y


def _restore_par(saved_par: dict[str, Any]) -> None:
    for k, v in saved_par.items():
        if v is None and k in ("circle.margin",):
            continue
        try:
            circos_par.set_value(k, v, _suppress_warn=True)
        except (KeyError, ValueError, TypeError):
            pass


def circos_nested(
    f1: Callable[[], Any],
    f2: Callable[[], Any],
    correspondance: Any,
    connection_height: float | None = None,
    connection_col: Any = None,
    connection_border: Any = "black",
    connection_lty: Any = 1,
    connection_lwd: Any = 1.0,
    adjust_start_degree: bool = True,
    **kwargs: Any,
) -> None:
    """Connect an outer circular plot `f1` with a nested inner circular plot `f2`."""
    nkw = normalize_kwargs(kwargs)
    connection_height = nkw.pop("connection_height", connection_height)
    connection_col = nkw.pop("connection_col", connection_col)
    connection_border = nkw.pop("connection_border", connection_border)
    connection_lty = nkw.pop("connection_lty", connection_lty)
    connection_lwd = nkw.pop("connection_lwd", connection_lwd)
    adjust_start_degree = bool(nkw.pop("adjust_start_degree", adjust_start_degree))

    df = _coerce_correspondance(correspondance)
    n_rows = len(df)

    cols = [None] * n_rows if connection_col is None else recycle_to_length(connection_col, n_rows)
    borders = [None] * n_rows if connection_border is None else recycle_to_length(connection_border, n_rows)
    ltys = recycle_to_length(connection_lty, n_rows)
    lwds = [float(v) for v in recycle_to_length(connection_lwd, n_rows)]

    # Sort correspondance by (start1, end1, start2, end2) while keeping style vectors aligned
    if n_rows > 0:
        order_idx = sorted(
            range(n_rows),
            key=lambda i: (df.loc[i, "start1"], df.loc[i, "end1"], df.loc[i, "start2"], df.loc[i, "end2"]),
        )
        df = df.iloc[order_idx].reset_index(drop=True)
        cols = [cols[i] for i in order_idx]
        borders = [borders[i] for i in order_idx]
        ltys = [ltys[i] for i in order_idx]
        lwds = [lwds[i] for i in order_idx]

    saved_par = copy.deepcopy(circos_par.as_dict(r_keys=True))
    orig_ax = _STATE.ax
    orig_fignum = plt.gcf().number if plt.get_fignums() else None

    # Step 1: Dry-run f2() in a temporary offscreen figure to inspect sectors & coordinates
    tmp_fig = plt.figure(figsize=(6.0, 6.0))
    c2_theta1_all: dict[int, float] = {}
    c2_theta2_all: dict[int, float] = {}
    f2_sectors: list[str] = []
    try:
        circos_clear()
        _STATE.ax = tmp_fig.add_subplot(111)
        f2()
        if not is_circos_initialized():
            raise RuntimeError("`f2()` did not initialize a circular layout.")
        xlim2 = circos_par.get_value("canvas.xlim")
        ylim2 = circos_par.get_value("canvas.ylim")
        if not (np.allclose(xlim2, [-1.0, 1.0]) and np.allclose(ylim2, [-1.0, 1.0])):
            raise ValueError("`canvas.xlim` and `canvas.ylim` in `f2()` must be [-1, 1].")
        if adjust_start_degree and abs(float(circos_par.get_value("start.degree"))) > 1e-9:
            raise ValueError("`circos.par(start.degree=...)` should not be modified in `f2()` when `adjust_start_degree=True`.")
        f2_sectors = get_all_sector_index()
        f2_set = set(f2_sectors)
        for idx, row in df.iterrows():
            if row["s2"] in f2_set:
                c2_theta1_all[idx] = float(circlize(row["start2"], 1.0, sector_index=row["s2"])[0, 0])
                c2_theta2_all[idx] = float(circlize(row["end2"], 1.0, sector_index=row["s2"])[0, 0])
    finally:
        circos_clear()
        plt.close(tmp_fig)
        if orig_fignum is not None and plt.fignum_exists(orig_fignum):
            plt.figure(orig_fignum)
        _STATE.ax = orig_ax
        _restore_par(saved_par)

    # Step 2: Run outer plot f1() on the target canvas
    f1()
    if not is_circos_initialized():
        raise RuntimeError("`f1()` did not initialize a circular layout.")

    xlim1 = circos_par.get_value("canvas.xlim")
    ylim1 = circos_par.get_value("canvas.ylim")
    if abs(float(xlim1[0]) + float(xlim1[1])) > 1e-9 or abs(float(ylim1[0]) + float(ylim1[1])) > 1e-9:
        raise ValueError("`canvas.xlim` and `canvas.ylim` in `f1()` must be symmetric around 0.")
    if abs(abs(float(xlim1[0])) - abs(float(ylim1[0]))) > 1e-9:
        raise ValueError("Canvas limits in `f1()` must be square (|xlim| == |ylim|).")

    if connection_height is None:
        connection_height_val = float(convert_length(5.0, "mm"))
    else:
        connection_height_val = float(connection_height)

    f1_sectors = get_all_sector_index()
    f1_set = set(f1_sectors)
    f2_set = set(f2_sectors)

    valid_indices = [
        idx
        for idx, row in df.iterrows()
        if (row["s1"] in f1_set) and (row["s2"] in f2_set) and (idx in c2_theta1_all)
    ]

    c1_theta1 = np.array(
        [float(circlize(df.loc[idx, "start1"], 1.0, sector_index=df.loc[idx, "s1"])[0, 0]) for idx in valid_indices],
        dtype=float,
    )
    c1_theta2 = np.array(
        [float(circlize(df.loc[idx, "end1"], 1.0, sector_index=df.loc[idx, "s1"])[0, 0]) for idx in valid_indices],
        dtype=float,
    )
    c2_theta1 = np.array([c2_theta1_all[idx] for idx in valid_indices], dtype=float)
    c2_theta2 = np.array([c2_theta2_all[idx] for idx in valid_indices], dtype=float)

    r1 = float(get_most_inside_radius())
    r2 = r1 - connection_height_val
    if r2 <= 0.0:
        raise ValueError(f"Not enough radial space inside outer plot (r1={r1:.4f}, connection_height={connection_height_val:.4f}).")

    if len(valid_indices) > 1:
        f1_rank = {s: i for i, s in enumerate(f1_sectors)}
        f2_rank = {s: i for i, s in enumerate(f2_sectors)}
        o1 = sorted(
            range(len(valid_indices)),
            key=lambda k: (
                f1_rank[df.loc[valid_indices[k], "s1"]],
                df.loc[valid_indices[k], "start1"],
                df.loc[valid_indices[k], "end1"],
            ),
        )
        o2 = sorted(
            range(len(valid_indices)),
            key=lambda k: (
                f2_rank[df.loc[valid_indices[k], "s2"]],
                df.loc[valid_indices[k], "start2"],
                df.loc[valid_indices[k], "end2"],
            ),
        )
        if o1 != o2:
            warnings.warn(
                "Order of regions in outer plot differs from inner plot; connection ribbons may cross.",
                UserWarning,
                stacklevel=2,
            )

    offset = 0.0
    if adjust_start_degree and len(valid_indices) > 0:
        c1_mid = 0.5 * (c1_theta1 + c1_theta2)
        c2_mid = 0.5 * (c2_theta1 + c2_theta2)
        offsets = np.arange(360, dtype=float)
        # Minimize sum of shortest angular distances between c1_mid and (c2_mid + offset)
        diff = ((c1_mid[:, None] - (c2_mid[:, None] + offsets[None, :])) + 180.0) % 360.0 - 180.0
        total_dist = np.sum(np.abs(diff), axis=0)
        offset = float(offsets[int(np.argmin(total_dist))])
        c2_theta1 = c2_theta1 + offset
        c2_theta2 = c2_theta2 + offset

    # Step 3: Draw connection track between r1 and r2
    circos_track_plot_region(
        sectors=f1_sectors,
        ylim=[0.0, 1.0],
        track_height=connection_height_val,
        track_margin=[0.0, 0.0],
        cell_padding=[0.0, 0.0, 0.0, 0.0],
        bg_border=None,
    )

    op_warn = circos_par.get_value("points.overflow.warning")
    circos_par.set_value("points_overflow_warning", False, _suppress_warn=True)
    try:
        for k, orig_i in enumerate(valid_indices):
            s1 = str(df.loc[orig_i, "s1"])
            c1_t1, c1_t2 = float(c1_theta1[k]), float(c1_theta2[k])
            c2_t1, c2_t2 = float(c2_theta1[k]), float(c2_theta2[k])
            c1_m = 0.5 * (c1_t1 + c1_t2)
            c2_m = 0.5 * (c2_t1 + c2_t2)
            shift = round((c1_m - c2_m) / 360.0) * 360.0
            c2_t1 += shift
            c2_t2 += shift

            coords = reverse_circlize(
                [c1_t1, c1_t2, c2_t1, c2_t2],
                [r1, r1, r2, r2],
                sector_index=s1,
            )
            x11, y11 = float(coords[0, 0]), float(coords[0, 1])
            x12, y12 = float(coords[1, 0]), float(coords[1, 1])
            x21, y21 = float(coords[2, 0]), float(coords[2, 1])
            x22, y22 = float(coords[3, 0]), float(coords[3, 1])

            dy1 = y21 - y11
            dy2 = y22 - y12
            lx, ly = _cubic_bezier_2d(
                (x11, y11),
                (x11, y11 + dy1 / 3.0),
                (x21, y11 + 2.0 * dy1 / 3.0),
                (x21, y21),
            )
            rx, ry = _cubic_bezier_2d(
                (x22, y22),
                (x22, y12 + 2.0 * dy2 / 3.0),
                (x12, y12 + dy2 / 3.0),
                (x12, y12),
            )
            poly_x = np.concatenate([lx, rx, [x11]])
            poly_y = np.concatenate([ly, ry, [y11]])
            circos_polygon(
                poly_x,
                poly_y,
                sector_index=s1,
                col=cols[orig_i],
                border=borders[orig_i],
                lty=ltys[orig_i],
                lwd=lwds[orig_i],
            )
    finally:
        circos_par.set_value("points_overflow_warning", op_warn, _suppress_warn=True)

    # Step 4: Overlay f2() inside radius r2 on the same Matplotlib Axes
    circos_clear(keep_ax=True)
    try:
        _STATE.canvas_overlay_scale = r2
        if adjust_start_degree:
            circos_par.set_value("start_degree", offset, _suppress_warn=True)
        f2()
    finally:
        circos_clear(keep_ax=False)
