"""Coordinate transformations, unit conversions, adjacency helpers, and geometry utilities."""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from ._canvas import (
    BASE_FONTSIZE_PT,
    INCH_PER_MM,
    ensure_canvas,
    get_canvas_pt_per_inch,
    parse_unit,
    str_ascent_mm,
    str_descent_mm,
    strheight_canvas,
    strwidth_canvas,
)
from ._state import (
    _STATE,
    circos_par,
    get_all_sector_index,
    get_cell_data,
    get_cell_meta_data,
    get_current_sector_index,
    get_current_track_index,
    get_sector_data,
    is_circos_initialized,
    normalize_kwargs,
)


class NamedMatrix2D(np.ndarray):
    """2D float64 NumPy array supporting R-like named column indexing (e.g. pos[:, 'theta'])."""

    def __new__(cls, input_array: Any, colnames: tuple[str, str] = ("theta", "rou")) -> "NamedMatrix2D":
        obj = np.asarray(input_array, dtype=float).view(cls)
        obj._colnames = tuple(colnames)
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        if obj is None:
            return
        self._colnames = getattr(obj, "_colnames", ("theta", "rou"))

    def __getitem__(self, key: Any) -> Any:
        colnames = getattr(self, "_colnames", ())
        if isinstance(key, str) and key in colnames:
            idx = colnames.index(key)
            return np.asarray(super().__getitem__((slice(None), idx)))
        if isinstance(key, tuple) and len(key) == 2 and isinstance(key[1], str) and key[1] in colnames:
            idx = colnames.index(key[1])
            res = super().__getitem__((key[0], idx))
            if isinstance(res, NamedMatrix2D):
                return np.asarray(res)
            return res
        return super().__getitem__(key)

    def __getattr__(self, name: str) -> Any:
        colnames = getattr(self, "_colnames", ())
        if name in colnames:
            return self[name]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")


class DegreeValue:
    """Marker class for degree-unit offsets in circos.text(..., adj=...)."""

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def __float__(self) -> float:
        return self.value

    def __repr__(self) -> str:
        return f"degree({self.value})"


def degree(x: float) -> DegreeValue:
    """Mark a numeric value as degrees for use in `adj` of `circos_text`."""
    return DegreeValue(x)


def fontsize(x: float) -> float:
    """Convert font size in points to cex multiplier."""
    return float(x) / BASE_FONTSIZE_PT


def as_radian(deg: Any) -> Any:
    return np.asarray(deg, dtype=float) / 180.0 * math.pi if not np.isscalar(deg) else float(deg) / 180.0 * math.pi


def as_degree(rad: Any) -> Any:
    return np.asarray(rad, dtype=float) / math.pi * 180.0 if not np.isscalar(rad) else float(rad) / math.pi * 180.0


def _extract_xy_pair(x: Any, y: Any = None) -> tuple[np.ndarray, np.ndarray]:
    if y is None:
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, 0].astype(float), arr[:, 1].astype(float)
        raise ValueError("If `y` is omitted, `x` must be a 2-column array/DataFrame.")
    x_arr = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    y_arr = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    if len(x_arr) == 1 and len(y_arr) > 1:
        x_arr = np.repeat(x_arr, len(y_arr))
    elif len(y_arr) == 1 and len(x_arr) > 1:
        y_arr = np.repeat(y_arr, len(x_arr))
    return x_arr, y_arr


def circlize(
    x: Any,
    y: Any = None,
    sector_index: str | None = None,
    track_index: int | None = None,
    **kwargs: Any,
) -> NamedMatrix2D:
    """Transform data coordinates (x, y) in a cell to polar coordinates (theta, rou)."""
    nkw = normalize_kwargs(kwargs)
    if "sector_index" in nkw:
        sector_index = nkw["sector_index"]
    if "track_index" in nkw:
        track_index = nkw["track_index"]

    if sector_index is None:
        sector_index = get_current_sector_index()
    if track_index is None:
        track_index = get_current_track_index()

    x_arr, y_arr = _extract_xy_pair(x, y)
    sdata = get_sector_data(sector_index)

    span = sdata["max.value"] - sdata["min.value"]
    deg_span = abs(sdata["start.degree"] - sdata["end.degree"])
    theta = sdata["start.degree"] - (x_arr - sdata["min.value"]) / span * deg_span

    if not circos_par.get_value("xaxis.clock.wise"):
        theta = sdata["start.degree"] - theta + sdata["end.degree"]

    if track_index == 0:
        rou = np.ones_like(theta, dtype=float)
    else:
        cdata = get_cell_data(sector_index, track_index)
        cell_ylim = get_cell_meta_data("cell.ylim", sector_index, track_index)
        y_range = cell_ylim[1] - cell_ylim[0]
        rou = cdata["track_start"] - (cell_ylim[1] - y_arr) / y_range * cdata["track_height"]

    m = np.column_stack([theta, rou])
    return NamedMatrix2D(m, colnames=("theta", "rou"))


def reverse_circlize(
    x: Any,
    y: Any = None,
    sector_index: str | None = None,
    track_index: int | None = None,
    **kwargs: Any,
) -> NamedMatrix2D:
    """Transform polar coordinates (theta, rou) back to data coordinates (x, y)."""
    nkw = normalize_kwargs(kwargs)
    if "sector_index" in nkw:
        sector_index = nkw["sector_index"]
    if "track_index" in nkw:
        track_index = nkw["track_index"]

    if sector_index is None:
        sector_index = get_current_sector_index()
    if track_index is None:
        track_index = get_current_track_index()

    theta, rou = _extract_xy_pair(x, y)
    sdata = get_sector_data(sector_index)

    deg_span = abs(sdata["end.degree"] - sdata["start.degree"])
    x_out = (sdata["start.degree"] - theta) / deg_span * (sdata["max.value"] - sdata["min.value"]) + sdata["min.value"]

    if not circos_par.get_value("xaxis.clock.wise"):
        x_out = sdata["max.value"] - x_out + sdata["min.value"]

    if track_index > 0:
        cdata = get_cell_data(sector_index, track_index)
        cell_ylim = get_cell_meta_data("cell.ylim", sector_index, track_index)
        y_out = (cdata["track_height"] - (cdata["track_start"] - rou)) / cdata["track_height"] * (cell_ylim[1] - cell_ylim[0]) + cell_ylim[0]
    else:
        y_out = np.ones_like(x_out, dtype=float)

    m = np.column_stack([x_out, y_out])
    return NamedMatrix2D(m, colnames=("x", "y"))


def polar2Cartesian(d: Any) -> np.ndarray:
    """Convert (theta_deg, rou) matrix to Cartesian (x, y), applying overlay scaling if active."""
    arr = np.asarray(d, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    theta = np.deg2rad(arr[:, 0])
    rou = arr[:, 1]
    scale = _STATE.canvas_overlay_scale if _STATE.canvas_overlay_scale is not None else 1.0
    x = rou * np.cos(theta) * scale
    y = rou * np.sin(theta) * scale
    return np.column_stack([x, y])


polar_to_cartesian = polar2Cartesian


def lines_expand(
    x: Sequence[float],
    y: Sequence[float],
    sector_index: str | None = None,
    track_index: int | None = None,
) -> np.ndarray:
    """Subdivide straight segments in data coordinates so they bend smoothly as circular arcs."""
    if sector_index is None:
        sector_index = get_current_sector_index()
    if track_index is None:
        track_index = get_current_track_index()

    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    if len(x_arr) == 0:
        return np.empty((0, 2), dtype=float)

    nx: list[float] = [float(x_arr[0])]
    ny: list[float] = [float(y_arr[0])]
    seg_len = 2.0 * math.pi / float(circos_par.get_value("unit.circle.segments"))

    for i in range(1, len(x_arr)):
        xi, yi = float(x_arr[i]), float(y_arr[i])
        xp, yp = float(x_arr[i - 1]), float(y_arr[i - 1])
        if math.isnan(xi) or math.isnan(yi):
            nx.append(math.nan)
            ny.append(math.nan)
            continue
        if math.isnan(xp) or math.isnan(yp):
            nx.append(xi)
            ny.append(yi)
            continue

        if xp <= xi:
            xs_pair, ys_pair = [xp, xi], [yp, yi]
        else:
            xs_pair, ys_pair = [xi, xp], [yi, yp]

        td2 = circlize(xs_pair, ys_pair, sector_index=sector_index, track_index=track_index)
        a = abs(float(td2[0, 0]) - float(td2[1, 0])) / 180.0 * math.pi
        b = abs(float(td2[0, 1]) - float(td2[1, 1]))
        length = math.hypot(a, b)
        ncut = int(math.floor(length / seg_len))
        if ncut > 0:
            j = np.arange(1, ncut + 1, dtype=float) / (ncut + 1.0)
            nx.extend((xp + (xi - xp) * j).tolist())
            ny.extend((yp + (yi - yp) * j).tolist())
        nx.append(xi)
        ny.append(yi)

    return np.column_stack([nx, ny])


def recycle_with_factors(x: Any, factors: Sequence[str], levels: Sequence[str]) -> list[Any]:
    """Recycle scalar or level-length sequence to match factors length."""
    if isinstance(x, (str, bytes, int, float, np.number, bool, np.bool_)) or x is None:
        return [x] * len(factors)
    if isinstance(x, dict):
        return [x.get(str(f)) for f in factors]
    seq = list(x)
    if len(seq) == 1:
        return seq * len(factors)
    if len(seq) == len(factors):
        return seq
    if len(seq) == len(levels):
        lmap = {str(lv): seq[i] for i, lv in enumerate(levels)}
        return [lmap[str(f)] for f in factors]
    return [seq[i % len(seq)] for i in range(len(factors))]


def recycle_with_levels(x: Any, levels: Sequence[str]) -> list[Any]:
    if isinstance(x, (str, bytes, int, float, np.number, bool, np.bool_)) or x is None:
        return [x] * len(levels)
    if isinstance(x, dict):
        return [x.get(str(lv)) for lv in levels]
    seq = list(x)
    if len(seq) == 1:
        return seq * len(levels)
    if len(seq) == len(levels):
        return seq
    return [seq[i % len(seq)] for i in range(len(levels))]


def recycle_to_length(x: Any, n: int) -> list[Any]:
    if isinstance(x, (str, bytes, int, float, np.number, bool, np.bool_)) or x is None:
        return [x] * n
    if isinstance(x, tuple) and len(x) in (3, 4) and all(isinstance(v, (int, float, np.number)) for v in x):
        return [x] * n
    seq = list(x)
    if len(seq) == n:
        return seq
    if len(seq) == 1:
        return seq * n
    return [seq[i % len(seq)] for i in range(n)]


def check_track_position(track_index: int, track_start: float, track_height: float) -> None:
    track_margin = circos_par.get_value("track.margin")
    if track_start - track_height - track_margin[1] < -1e-12 or track_start - track_height < -1e-12 or track_start < -1e-12:
        raise ValueError(f"not enough space for cells at track index '{track_index}'.")
    if track_start - track_margin[0] - track_height - track_margin[1] < -1e-12:
        raise ValueError(f"not enough space for bottom margin of cells at track index '{track_index}'.")
    if track_index > 1:
        prev_bottom = get_cell_meta_data("cell.bottom.radius", track_index=track_index - 1)
        if track_start > prev_bottom + 1e-12:
            raise ValueError("Track overlaps with previous track.")


def check_points_position(
    x: Sequence[float],
    y: Sequence[float],
    sector_index: str | None = None,
    track_index: int | None = None,
) -> None:
    if sector_index is None:
        sector_index = get_current_sector_index()
    if track_index is None:
        track_index = get_current_track_index()
    if not circos_par.get_value("points.overflow.warning"):
        return
    cx = get_cell_meta_data("cell.xlim", sector_index, track_index)
    cy = get_cell_meta_data("cell.ylim", sector_index, track_index)
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    valid = ~(np.isnan(x_arr) | np.isnan(y_arr))
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]
    eps = 1e-12
    out = (x_arr < cx[0] - eps) | (x_arr > cx[1] + eps) | (y_arr < cy[0] - eps) | (y_arr > cy[1] + eps)
    n_out = int(np.sum(out))
    if n_out > 0 and circos_par.get_value("message"):
        noun = "point is" if n_out == 1 else "points are"
        warnings.warn(
            f"Note: {n_out} {noun} out of plotting region in sector '{sector_index}', track '{track_index}'.",
            UserWarning,
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Physical unit conversion functions
# ---------------------------------------------------------------------------

def _validate_unit(unit: str) -> str:
    u = str(unit).strip().lower()
    if u in ("inche", "inch", "in", "inches"):
        return "inches"
    if u in ("mm", "cm", "canvas"):
        return u
    raise ValueError(f"Unsupported unit {unit!r}. Must be 'mm', 'cm', or 'inches'.")


def convert_length(x: float | Sequence[float], unit: str = "mm") -> float | np.ndarray:
    """Convert absolute physical units (mm/cm/inches) to canvas radius units."""
    u = _validate_unit(unit)
    is_sc = np.isscalar(x)
    arr = np.asarray(x, dtype=float)
    pt_per_inch = get_canvas_pt_per_inch()
    if u == "inches":
        out = arr * pt_per_inch
    elif u == "mm":
        out = arr * pt_per_inch * INCH_PER_MM
    elif u == "cm":
        out = arr * pt_per_inch * INCH_PER_MM * 10.0
    else:
        out = arr
    return float(out) if is_sc else out


convert_height = convert_length
uh = convert_length


def convert_unit_in_data_coordinate(
    x: float | Sequence[float],
    unit: str = "mm",
    direction: str = "x",
    sector_index: str | None = None,
    track_index: int | None = None,
    h: float | Sequence[float] | None = None,
) -> float | np.ndarray:
    """Convert absolute physical length (mm/cm/inches/canvas) to data units inside a cell."""
    if sector_index is None:
        sector_index = get_current_sector_index()
    if track_index is None:
        track_index = get_current_track_index()
    if h is None:
        h = get_cell_meta_data("ycenter", sector_index=sector_index, track_index=track_index)

    u = _validate_unit(unit)
    is_sc = np.isscalar(x)
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))

    pt_per_inch = get_canvas_pt_per_inch()
    if u == "inches":
        len_canvas = x_arr * pt_per_inch
    elif u == "mm":
        len_canvas = x_arr * pt_per_inch * INCH_PER_MM
    elif u == "cm":
        len_canvas = x_arr * pt_per_inch * INCH_PER_MM * 10.0
    else:
        len_canvas = x_arr

    xlim = get_cell_meta_data("xlim", sector_index=sector_index, track_index=track_index)
    ylim = get_cell_meta_data("ylim", sector_index=sector_index, track_index=track_index)

    if direction == "x":
        h_arr = np.atleast_1d(np.asarray(h, dtype=float))
        if len(h_arr) == 1 and len(x_arr) > 1:
            h_arr = np.repeat(h_arr, len(x_arr))
        w_data = float(xlim[1] - xlim[0])
        w_canvas = np.empty(len(x_arr), dtype=float)
        for i in range(len(x_arr)):
            coor = circlize(xlim, [h_arr[i], h_arr[i]], sector_index=sector_index, track_index=track_index)
            w_canvas[i] = abs(float(coor[0, 0]) - float(coor[1, 0])) / 180.0 * math.pi * float(coor[0, 1])
    elif direction == "y":
        coor = circlize([xlim[0], xlim[0]], ylim, sector_index=sector_index, track_index=track_index)
        w_data = float(ylim[1] - ylim[0])
        w_canvas = abs(float(coor[1, 1]) - float(coor[0, 1]))
    else:
        raise ValueError("`direction` must be 'x' or 'y'.")

    out = len_canvas * abs(w_data / w_canvas)
    return float(out[0]) if is_sc else out


def convert_x(
    x: float | Sequence[float],
    unit: str = "mm",
    sector_index: str | None = None,
    track_index: int | None = None,
    h: float | Sequence[float] | None = None,
    **kwargs: Any,
) -> float | np.ndarray:
    nkw = normalize_kwargs(kwargs)
    sector_index = nkw.get("sector_index", sector_index)
    track_index = nkw.get("track_index", track_index)
    return convert_unit_in_data_coordinate(x, unit=unit, direction="x", sector_index=sector_index, track_index=track_index, h=h)


ux = convert_x


def convert_y(
    x: float | Sequence[float],
    unit: str = "mm",
    sector_index: str | None = None,
    track_index: int | None = None,
    **kwargs: Any,
) -> float | np.ndarray:
    nkw = normalize_kwargs(kwargs)
    sector_index = nkw.get("sector_index", sector_index)
    track_index = nkw.get("track_index", track_index)
    return convert_unit_in_data_coordinate(x, unit=unit, direction="y", sector_index=sector_index, track_index=track_index)


uy = convert_y


def convert_unit_in_canvas_coordinate(x: float | Sequence[float], unit: str = "mm") -> float | np.ndarray:
    return convert_length(x, unit=unit)


def cm_x(x: float | Sequence[float], sector_index: str | None = None, track_index: int | None = None, **kwargs: Any) -> float | np.ndarray:
    return convert_x(x, unit="cm", sector_index=sector_index, track_index=track_index, **kwargs)


def cm_y(y: float | Sequence[float], sector_index: str | None = None, track_index: int | None = None, **kwargs: Any) -> float | np.ndarray:
    return convert_y(y, unit="cm", sector_index=sector_index, track_index=track_index, **kwargs)


def cm_h(h: float | Sequence[float]) -> float | np.ndarray:
    return convert_length(h, unit="cm")


def mm_x(x: float | Sequence[float], sector_index: str | None = None, track_index: int | None = None, **kwargs: Any) -> float | np.ndarray:
    return convert_x(x, unit="mm", sector_index=sector_index, track_index=track_index, **kwargs)


def mm_y(y: float | Sequence[float], sector_index: str | None = None, track_index: int | None = None, **kwargs: Any) -> float | np.ndarray:
    return convert_y(y, unit="mm", sector_index=sector_index, track_index=track_index, **kwargs)


def mm_h(h: float | Sequence[float]) -> float | np.ndarray:
    return convert_length(h, unit="mm")


def inches_x(x: float | Sequence[float], sector_index: str | None = None, track_index: int | None = None, **kwargs: Any) -> float | np.ndarray:
    return convert_x(x, unit="inches", sector_index=sector_index, track_index=track_index, **kwargs)


def inches_y(y: float | Sequence[float], sector_index: str | None = None, track_index: int | None = None, **kwargs: Any) -> float | np.ndarray:
    return convert_y(y, unit="inches", sector_index=sector_index, track_index=track_index, **kwargs)


def inches_h(h: float | Sequence[float]) -> float | np.ndarray:
    return convert_length(h, unit="inches")


inch_x = inches_x
inch_y = inches_y
inch_h = inches_h


def str2unit(s: str) -> float:
    val, unit = parse_unit(s)
    return float(convert_length(val, unit))


# ---------------------------------------------------------------------------
# Adjacency conversion helpers
# ---------------------------------------------------------------------------

def validate_data_frame(x: Any) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    try:
        return pd.DataFrame(x)
    except Exception as exc:
        raise TypeError("The input should be a DataFrame or convertible to a DataFrame.") from exc


def adjacency_list2matrix(lt: pd.DataFrame, square: bool = False) -> pd.DataFrame:
    """Convert an adjacency list (from, to, [value]) DataFrame to an adjacency matrix (DataFrame)."""
    df = validate_data_frame(lt)
    if df.shape[1] == 2:
        df = df.copy()
        df["value"] = 1.0
    if df.shape[1] < 3:
        raise ValueError("`lt` should be a data frame with at least three columns.")
    if not pd.api.types.is_numeric_dtype(df.iloc[:, 2]):
        raise ValueError("Third column in `lt` should be numeric.")

    c0 = df.iloc[:, 0].astype(str).tolist()
    c1 = df.iloc[:, 1].astype(str).tolist()
    vals = df.iloc[:, 2].astype(float).tolist()

    rn = list(dict.fromkeys(c0))
    cn = list(dict.fromkeys(c1))
    if square:
        nm = list(dict.fromkeys(rn + cn))
        mat = pd.DataFrame(0.0, index=nm, columns=nm)
    else:
        mat = pd.DataFrame(0.0, index=rn, columns=cn)

    for r_k, c_k, v in zip(c0, c1, vals):
        mat.loc[r_k, c_k] = v
    return mat


adjacencyList2Matrix = adjacency_list2matrix


def adjacency_matrix2list(mat: Any, keep_zero: bool = False, **kwargs: Any) -> pd.DataFrame:
    """Convert an adjacency matrix (DataFrame or 2D ndarray) to a 3-column DataFrame (from, to, value)."""
    nkw = normalize_kwargs(kwargs)
    keep_zero = nkw.get("keep_zero", keep_zero)

    if isinstance(mat, pd.DataFrame):
        rn = [str(r) for r in mat.index]
        cn = [str(c) for c in mat.columns]
        arr = mat.to_numpy(dtype=float)
    else:
        arr = np.asarray(mat, dtype=float)
        rn = [str(i + 1) for i in range(arr.shape[0])]
        cn = [str(j + 1) for j in range(arr.shape[1])]

    nr, nc = arr.shape
    from_col = rn * nc
    to_col = [c for c in cn for _ in range(nr)]
    val_col = arr.T.ravel()  # column-major order matching R's as.vector(mat)
    df = pd.DataFrame({"from": from_col, "to": to_col, "value": val_col})
    if not keep_zero:
        df = df[df["value"] != 0].reset_index(drop=True)
    return df


adjacencyMatrix2List = adjacency_matrix2list


# ---------------------------------------------------------------------------
# Rounded rectangle and stacked text helpers
# ---------------------------------------------------------------------------

def roundrect_pos(
    xleft: float,
    ybottom: float,
    xright: float,
    ytop: float,
    radius: float = 0.1,
    asp: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    if xleft > xright:
        xleft, xright = xright, xleft
    if ybottom > ytop:
        ybottom, ytop = ytop, ybottom

    yb = ybottom / asp
    yt = ytop / asp
    w = abs(xright - xleft)
    h = abs(yt - yb)
    r = radius * min(w, h)

    xs: list[float] = []
    ys: list[float] = []

    def add_arc(deg_start: float, deg_end: float, cx: float, cy: float) -> None:
        degs = np.linspace(deg_start, deg_end, 20)
        pts = np.column_stack([r * np.cos(np.deg2rad(degs)) + cx, r * np.sin(np.deg2rad(degs)) + cy])
        xs.extend(pts[:, 0].tolist())
        ys.extend(pts[:, 1].tolist())

    add_arc(180, 90, xleft + r, yt - r)
    xs.extend([xleft + r, xright - r])
    ys.extend([yt, yt])
    add_arc(90, 0, xright - r, yt - r)
    xs.extend([xright, xright])
    ys.extend([yt - r, yb + r])
    add_arc(0, -90, xright - r, yb + r)
    xs.extend([xright - r, xleft + r])
    ys.extend([yb, yb])
    add_arc(270, 180, xleft + r, yb + r)
    xs.extend([xleft, xleft])
    ys.extend([yb + r, yt - r])

    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float) * asp


def circos_roundrect_pos(
    xleft: float,
    ybottom: float,
    xright: float,
    ytop: float,
    radius: float | str = "4mm",
    sector_index: str | None = None,
    track_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if sector_index is None:
        sector_index = get_current_sector_index()
    if track_index is None:
        track_index = get_current_track_index()

    r_val = str2unit(radius) if isinstance(radius, str) else float(radius)

    xplot = get_cell_meta_data("xplot", sector_index=sector_index, track_index=track_index)
    yplot = get_cell_meta_data("yplot", sector_index=sector_index, track_index=track_index)
    cell_width = abs(float(xplot[1] - xplot[0])) / 360.0 * math.pi * 2.0 * float(np.mean(yplot))
    cell_height = abs(float(yplot[1] - yplot[0]))

    cx = get_cell_meta_data("cell.xlim", sector_index=sector_index, track_index=track_index)
    cy = get_cell_meta_data("cell.ylim", sector_index=sector_index, track_index=track_index)
    cx_range = float(cx[1] - cx[0])
    cy_range = float(cy[1] - cy[0])

    w = abs(xright - xleft) / cx_range * cell_width
    h = abs(ytop - ybottom) / cy_range * cell_height
    r_frac = min(r_val * 2.0 / max(w, 1e-12), r_val * 2.0 / max(h, 1e-12), 0.5)

    xl0 = (xleft - cx[0]) / cx_range * cell_width
    xr0 = (xright - cx[0]) / cx_range * cell_width
    yb0 = (ybottom - cy[0]) / cy_range * cell_height
    yt0 = (ytop - cy[0]) / cy_range * cell_height

    px, py = roundrect_pos(xl0, yb0, xr0, yt0, radius=r_frac, asp=1.0)
    return px / cell_width * cx_range + cx[0], py / cell_height * cy_range + cy[0]


def circos_strwidth(
    text: Sequence[str],
    cex: float | Sequence[float] = 1.0,
    font: Any = 1,
    family: str = "sans-serif",
    sector_index: str | None = None,
    track_index: int | None = None,
    h: float | None = None,
) -> np.ndarray:
    if sector_index is None:
        sector_index = get_current_sector_index()
    if track_index is None:
        track_index = get_current_track_index()
    if h is None:
        h = float(get_cell_meta_data("ylim", sector_index=sector_index, track_index=track_index)[0])

    texts = [str(t) for t in text]
    cexs = recycle_to_length(cex, len(texts))
    fonts = recycle_to_length(font, len(texts))
    out = np.empty(len(texts), dtype=float)
    for i, (t, cx, ft) in enumerate(zip(texts, cexs, fonts)):
        w_canvas = sum(strwidth_canvas(ch, cex=float(cx), font=ft, family=family) for ch in t)
        out[i] = float(convert_unit_in_data_coordinate(w_canvas, unit="canvas", direction="x", sector_index=sector_index, track_index=track_index, h=h))
    return out


def circos_strheight(
    text: Sequence[str],
    cex: float | Sequence[float] = 1.0,
    font: Any = 1,
    family: str = "sans-serif",
    sector_index: str | None = None,
    track_index: int | None = None,
) -> np.ndarray:
    if sector_index is None:
        sector_index = get_current_sector_index()
    if track_index is None:
        track_index = get_current_track_index()
    texts = [str(t) for t in text]
    cexs = recycle_to_length(cex, len(texts))
    fonts = recycle_to_length(font, len(texts))
    out = np.empty(len(texts), dtype=float)
    for i, (t, cx, ft) in enumerate(zip(texts, cexs, fonts)):
        h_canvas = strheight_canvas(t, cex=float(cx), font=ft, family=family)
        out[i] = float(convert_unit_in_data_coordinate(h_canvas, unit="canvas", direction="y", sector_index=sector_index, track_index=track_index))
    return out


def circos_str_ascent(
    text: Sequence[str],
    cex: float | Sequence[float] = 1.0,
    font: Any = 1,
    family: str = "sans-serif",
    sector_index: str | None = None,
    track_index: int | None = None,
) -> np.ndarray:
    texts = [str(t) for t in text]
    cexs = recycle_to_length(cex, len(texts))
    fonts = recycle_to_length(font, len(texts))
    return np.array([mm_y(str_ascent_mm(t, cex=float(cx), font=ft, family=family), sector_index=sector_index, track_index=track_index) for t, cx, ft in zip(texts, cexs, fonts)], dtype=float)


def circos_str_descent(
    text: Sequence[str],
    cex: float | Sequence[float] = 1.0,
    font: Any = 1,
    family: str = "sans-serif",
    sector_index: str | None = None,
    track_index: int | None = None,
) -> np.ndarray:
    texts = [str(t) for t in text]
    cexs = recycle_to_length(cex, len(texts))
    fonts = recycle_to_length(font, len(texts))
    return np.array([mm_y(str_descent_mm(t, cex=float(cx), font=ft, family=family), sector_index=sector_index, track_index=track_index) for t, cx, ft in zip(texts, cexs, fonts)], dtype=float)


def stack_text(
    x: np.ndarray,
    text: Sequence[str],
    margin: float = 0.2,
    cex: Any = 1.0,
    font: Any = 1,
    family: str = "sans-serif",
    sector_index: str | None = None,
    track_index: int | None = None,
) -> list[list[int]]:
    """Greedy interval-packing of text labels along x into non-overlapping radial rows."""
    tw = circos_strwidth(text, cex=cex, font=font, family=family, sector_index=sector_index, track_index=track_index)
    th = circos_strheight(text, cex=cex, font=font, family=family, sector_index=sector_index, track_index=track_index)
    e = float(min(np.min(tw), np.min(th))) * margin if len(tw) else 0.0
    tw = tw + e

    x_arr = np.asarray(x, dtype=float)
    x1 = x_arr - tw / 2.0
    x2 = x_arr + tw / 2.0
    ind = np.arange(len(x_arr))

    od = np.argsort(x1, kind="mergesort")
    x1_s = list(x1[od])
    x2_s = list(x2[od])
    ind_s = list(ind[od])

    rows: list[list[int]] = []
    while x1_s:
        row = [ind_s[0]]
        x_right = x2_s[0]
        used = {0}
        for i in range(1, len(x1_s)):
            if x1_s[i] >= x_right:
                row.append(ind_s[i])
                x_right = x2_s[i]
                used.add(i)
        x1_s = [v for j, v in enumerate(x1_s) if j not in used]
        x2_s = [v for j, v in enumerate(x2_s) if j not in used]
        ind_s = [v for j, v in enumerate(ind_s) if j not in used]
        rows.append(row)
    return rows


def refer_to_one_sector(sectors: Sequence[str], x: Sequence[float]) -> pd.DataFrame:
    """Map x positions across all sectors onto an extended coordinate frame of the first sector."""
    sec_list = [str(s) for s in sectors]
    x_arr = np.asarray(x, dtype=float)
    bed = pd.DataFrame({"sectors": sec_list, "x": x_arr, "order": np.arange(len(x_arr))})
    bed = bed.sort_values(["sectors", "x"], kind="mergesort").reset_index(drop=True)

    flipped_xaxis = not circos_par.get_value("xaxis.clock.wise")
    if flipped_xaxis:
        for i, cr in enumerate(bed["sectors"]):
            sd = get_sector_data(cr)
            bed.loc[i, "x"] = sd["max.value"] - bed.loc[i, "x"] + sd["min.value"]
        circos_par.set_value("xaxis_clock_wise", True, _suppress_warn=True)

    try:
        chr0 = get_all_sector_index()[0]
        sd0 = get_sector_data(chr0)
        all_chr = list(dict.fromkeys(bed["sectors"].tolist()))
        rho: list[float] = []
        for cr in all_chr:
            sub = bed[bed["sectors"] == cr]
            pts = circlize(sub["x"].to_numpy(), np.ones(len(sub)), sector_index=cr)
            rho.extend(pts[:, 0].tolist())
        if len(rho) == 0:
            anchor = ((sd0["start.degree"] + sd0["end.degree"]) / 2.0 + 180.0) % 360.0
        elif len(rho) == 1:
            anchor = (rho[0] + 180.0) % 360.0
        else:
            rho_s = np.sort(np.asarray(rho))
            rho_s = np.append(rho_s, rho_s[0] + 360.0)
            idx = int(np.argmax(np.diff(rho_s)))
            anchor = ((rho_s[idx] + rho_s[idx + 1]) / 2.0) % 360.0

        chr_width = sd0["start.degree"] - sd0["end.degree"]
        s1 = sd0["start.degree"] % 360.0
        s2 = sd0["end.degree"] % 360.0
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
        ext1, ext2 = abs(ext1), abs(ext2)
        new_chr_range = (sd0["start.degree"] + chr_width * ext1, sd0["end.degree"] - chr_width * ext2)

        parts = []
        for cr in all_chr:
            sub = bed[bed["sectors"] == cr].copy()
            if cr != chr0:
                dfx1 = circlize(sub["x"].to_numpy(), np.ones(len(sub)), sector_index=cr)
                mask = (dfx1[:, 0] > new_chr_range[0]) | (dfx1[:, 0] < new_chr_range[1])
                if np.any(mask):
                    dfx1[mask, 0] = (dfx1[mask, 0] - new_chr_range[1]) % 360.0 + new_chr_range[1]
                x1 = reverse_circlize(dfx1, sector_index=chr0)[:, 0]
                sub["x"] = x1
            parts.append(sub)
        bed2 = pd.concat(parts, ignore_index=True) if parts else bed.copy()
        bed2["sectors"] = chr0
        return bed2
    finally:
        if flipped_xaxis:
            circos_par.set_value("xaxis_clock_wise", False, _suppress_warn=True)
