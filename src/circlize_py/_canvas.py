"""Matplotlib canvas management, physical unit conversions, and text metrics."""

from __future__ import annotations

import functools
import re
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
import numpy as np

from ._colors import resolve_font
from ._state import _STATE, circos_par

INCH_PER_MM = 0.03937007874015748
MM_PER_INCH = 25.4
BASE_FONTSIZE_PT = 10.0


def ensure_canvas(ax: Any = None, initialize: bool = False) -> Any:
    """Return active Matplotlib Axes, creating or resetting it if initializing a new plot."""
    if ax is not None:
        target_ax = ax
        if initialize and _STATE.canvas_overlay_scale is None and getattr(target_ax, "_circlize_used", False):
            target_ax.cla()
    elif _STATE.canvas_overlay_scale is not None and _STATE.ax is not None:
        return _STATE.ax
    elif initialize:
        if plt.get_fignums():
            target_ax = plt.gca()
            if getattr(target_ax, "_circlize_used", False):
                target_ax.cla()
        else:
            fig = plt.figure(figsize=(6.0, 6.0))
            target_ax = fig.add_subplot(111)
    else:
        if _STATE.ax is not None:
            return _STATE.ax
        if plt.get_fignums():
            target_ax = plt.gca()
        else:
            fig = plt.figure(figsize=(6.0, 6.0))
            target_ax = fig.add_subplot(111)

    if initialize or not getattr(target_ax, "_circlize_used", False):
        xlim = circos_par.get_value("canvas.xlim")
        ylim = circos_par.get_value("canvas.ylim")
        target_ax.set_xlim(xlim[0], xlim[1])
        target_ax.set_ylim(ylim[0], ylim[1])
        target_ax.set_aspect("equal", adjustable="box")
        target_ax.axis("off")
        target_ax._circlize_used = True

    _STATE.ax = target_ax
    return target_ax


def get_canvas_pt_per_inch(ax: Any = None) -> float:
    """Return canvas data units per physical inch on the plot box."""
    if ax is None:
        ax = _STATE.ax
    if ax is None:
        ax = ensure_canvas(initialize=False)

    fig = ax.figure
    fw, fh = fig.get_size_inches()
    pos = ax.get_position()
    pw = max(float(fw * pos.width), 1e-6)
    ph = max(float(fh * pos.height), 1e-6)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    dx = max(float(xmax - xmin), 1e-9)
    dy = max(float(ymax - ymin), 1e-9)

    # With aspect='equal', the smaller scale governs both axes
    scale_inch = min(pw / dx, ph / dy)
    pt_per_inch = 1.0 / scale_inch
    if _STATE.canvas_overlay_scale is not None and _STATE.canvas_overlay_scale > 0:
        pt_per_inch = pt_per_inch / _STATE.canvas_overlay_scale
    return float(pt_per_inch)


@functools.lru_cache(maxsize=4096)
def _measure_text_pt(text: str, size_pt: float, weight: str, style: str, family: str) -> tuple[float, float, float, float]:
    """Measure (width_pt, height_pt, ascent_pt, descent_pt) using Matplotlib TextPath."""
    if not text:
        return (0.0, 0.0, 0.0, 0.0)
    fp = FontProperties(family=family or "sans-serif", weight=weight, style=style)
    if text.strip() == "":
        w_pair = TextPath((0, 0), "||", size=size_pt, prop=fp).get_extents()
        w_wrap = TextPath((0, 0), f"|{text}|", size=size_pt, prop=fp).get_extents()
        w_pt = max(0.0, float(w_wrap.width - w_pair.width))
        h_pt = float(size_pt * 0.72)
        return (w_pt, h_pt, h_pt, 0.0)
    ext = TextPath((0, 0), text, size=size_pt, prop=fp).get_extents()
    w_pt = max(0.0, float(ext.width))
    h_pt = max(0.0, float(ext.height))
    ascent_pt = max(0.0, float(ext.ymax))
    descent_pt = max(0.0, float(-ext.ymin))
    return (w_pt, h_pt, ascent_pt, descent_pt)


def strwidth_inches(text: str, cex: float = 1.0, font: Any = 1, family: str = "sans-serif") -> float:
    weight, style = resolve_font(font)
    w_pt, _, _, _ = _measure_text_pt(str(text), float(cex) * BASE_FONTSIZE_PT, weight, style, str(family or "sans-serif"))
    return w_pt / 72.0


def strheight_inches(text: str, cex: float = 1.0, font: Any = 1, family: str = "sans-serif") -> float:
    weight, style = resolve_font(font)
    _, h_pt, _, _ = _measure_text_pt(str(text), float(cex) * BASE_FONTSIZE_PT, weight, style, str(family or "sans-serif"))
    if h_pt == 0.0 and str(text):
        h_pt = float(cex) * BASE_FONTSIZE_PT * 0.72
    return h_pt / 72.0


def str_ascent_mm(text: str, cex: float = 1.0, font: Any = 1, family: str = "sans-serif") -> float:
    weight, style = resolve_font(font)
    _, _, asc_pt, _ = _measure_text_pt(str(text), float(cex) * BASE_FONTSIZE_PT, weight, style, str(family or "sans-serif"))
    return (asc_pt / 72.0) * MM_PER_INCH


def str_descent_mm(text: str, cex: float = 1.0, font: Any = 1, family: str = "sans-serif") -> float:
    weight, style = resolve_font(font)
    _, _, _, desc_pt = _measure_text_pt(str(text), float(cex) * BASE_FONTSIZE_PT, weight, style, str(family or "sans-serif"))
    return (desc_pt / 72.0) * MM_PER_INCH


def strwidth_canvas(text: str, cex: float = 1.0, font: Any = 1, family: str = "sans-serif", ax: Any = None) -> float:
    return strwidth_inches(text, cex=cex, font=font, family=family) * get_canvas_pt_per_inch(ax)


def strheight_canvas(text: str, cex: float = 1.0, font: Any = 1, family: str = "sans-serif", ax: Any = None) -> float:
    return strheight_inches(text, cex=cex, font=font, family=family) * get_canvas_pt_per_inch(ax)


def parse_unit(spec: str) -> tuple[float, str]:
    """Parse unit string like '2mm', '-1.5 inches', '3cm' -> (value, canonical_unit)."""
    s = str(spec).strip()
    m = re.match(r"^(-?\d+(?:\.\d+)?)\s*(mm|cm|inches|inche|inch|in)$", s, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Format of the unit is incorrect ({spec!r}). Expected e.g. '2mm', '-2.1 inches', '1cm'.")
    val = float(m.group(1))
    u = m.group(2).lower()
    if u in ("inche", "inch", "in", "inches"):
        u = "inches"
    return (val, u)
