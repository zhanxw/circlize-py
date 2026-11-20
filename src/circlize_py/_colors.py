"""Color interpolation, inverse lookup, random colors, and R graphics parameter mapping."""

from __future__ import annotations

import colorsys
import math
from typing import Any, Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._r_colors import R_NAMED_COLORS

# R 4.0 default palette() (1-based indexing in R)
R_PALETTE = [
    "#000000",
    "#DF536B",
    "#61D04F",
    "#2297E6",
    "#28E2E5",
    "#CD0BBC",
    "#F5C710",
    "#9E9E9E",
]

# R colorspace sRGB -> XYZ (Y=100) matrix and D65 whitepoint
_M_SRGB_TO_XYZ = np.array(
    [
        [41.2453, 35.7580, 18.0423],
        [21.2671, 71.5160, 7.2169],
        [1.9334, 11.9193, 95.0227],
    ],
    dtype=float,
)
_M_XYZ_TO_SRGB = np.linalg.inv(_M_SRGB_TO_XYZ)
# CIE standard illuminant D65 used by colorspace::LAB / LUV
_XN, _YN, _ZN = 95.047, 100.0, 108.883


def is_na_like(val: Any) -> bool:
    if val is None or val is pd.NA:
        return True
    if isinstance(val, (float, np.floating)) and math.isnan(float(val)):
        return True
    if isinstance(val, str) and val.strip().upper() in ("NA", "NAN", ""):
        return True
    return False


def _single_color_to_rgba(c: Any) -> tuple[float, float, float, float] | None:
    """Convert a single R/Python color spec to (r, g, b, a) in [0, 1], or None for NA/none."""
    if is_na_like(c):
        return None
    if isinstance(c, (bool, np.bool_)):
        return None if not c else (0.0, 0.0, 0.0, 1.0)
    if isinstance(c, (int, np.integer)):
        idx = int(c)
        if idx <= 0:
            return None
        hex_c = R_PALETTE[(idx - 1) % len(R_PALETTE)]
        return mcolors.to_rgba(hex_c)
    if isinstance(c, (float, np.floating)):
        if float(c).is_integer():
            idx = int(c)
            if idx <= 0:
                return None
            hex_c = R_PALETTE[(idx - 1) % len(R_PALETTE)]
            return mcolors.to_rgba(hex_c)
        return None
    if isinstance(c, str):
        s = c.strip()
        low = s.lower()
        if low in ("none", "na", "transparent"):
            return (0.0, 0.0, 0.0, 0.0) if low == "transparent" else None
        if s.isdigit():
            idx = int(s)
            if idx <= 0:
                return None
            return mcolors.to_rgba(R_PALETTE[(idx - 1) % len(R_PALETTE)])
        if low in R_NAMED_COLORS:
            return mcolors.to_rgba(R_NAMED_COLORS[low])
        if s.startswith("#") and len(s) == 9:
            r = int(s[1:3], 16) / 255.0
            g = int(s[3:5], 16) / 255.0
            b = int(s[5:7], 16) / 255.0
            a = int(s[7:9], 16) / 255.0
            return (r, g, b, a)
        return mcolors.to_rgba(s)
    if isinstance(c, (tuple, list, np.ndarray)):
        arr = [float(v) for v in c]
        if len(arr) == 3:
            if max(arr) > 1.0:
                arr = [v / 255.0 for v in arr]
            return (arr[0], arr[1], arr[2], 1.0)
        if len(arr) == 4:
            if max(arr) > 1.0:
                arr = [v / 255.0 for v in arr]
            return (arr[0], arr[1], arr[2], arr[3])
    return mcolors.to_rgba(c)


def resolve_color(c: Any, default: str = "none") -> Any:
    """Resolve a color spec into a Matplotlib-compatible color ('none' if NA/None)."""
    rgba = _single_color_to_rgba(c)
    if rgba is None:
        return default
    return rgba


def resolve_colors(cols: Any, n: int | None = None, default: str = "none") -> list[Any]:
    """Normalize a single color or sequence of colors to a list of length n."""
    if isinstance(cols, (str, bytes, int, float, np.integer, np.floating, bool, np.bool_)) or cols is None or cols is pd.NA:
        c = resolve_color(cols, default=default)
        return [c] * (1 if n is None else n)
    if isinstance(cols, tuple) and len(cols) in (3, 4) and all(isinstance(v, (int, float, np.number)) for v in cols):
        c = resolve_color(cols, default=default)
        return [c] * (1 if n is None else n)
    arr = list(cols)
    res = [resolve_color(v, default=default) for v in arr]
    if n is not None and len(res) != n:
        if len(res) == 1:
            return res * n
        return [res[i % len(res)] for i in range(n)]
    return res


def color_to_hex(c: Any, include_alpha: bool = True) -> str | None:
    """Convert any color spec to uppercase #RRGGBB or #RRGGBBAA, or None if NA."""
    rgba = _single_color_to_rgba(c)
    if rgba is None:
        return None
    r = max(0, min(255, int(round(rgba[0] * 255))))
    g = max(0, min(255, int(round(rgba[1] * 255))))
    b = max(0, min(255, int(round(rgba[2] * 255))))
    a = max(0, min(255, int(round(rgba[3] * 255))))
    if include_alpha:
        return f"#{r:02X}{g:02X}{b:02X}{a:02X}"
    return f"#{r:02X}{g:02X}{b:02X}"


def add_transparency(col: Any, transparency: float = 0.0) -> Any:
    """Add transparency in [0, 1] (0 = opaque, 1 = transparent) to color(s)."""
    alpha = 1.0 - max(0.0, min(1.0, float(transparency)))
    a_byte = int(round(alpha * 255))

    def _apply(c: Any) -> str | None:
        rgba = _single_color_to_rgba(c)
        if rgba is None:
            return None
        r = max(0, min(255, int(round(rgba[0] * 255))))
        g = max(0, min(255, int(round(rgba[1] * 255))))
        b = max(0, min(255, int(round(rgba[2] * 255))))
        return f"#{r:02X}{g:02X}{b:02X}{a_byte:02X}"

    if isinstance(col, (str, int, float, np.integer, np.floating)) or col is None:
        return _apply(col)
    if isinstance(col, np.ndarray):
        flat = [_apply(v) for v in col.ravel()]
        return np.array(flat, dtype=object).reshape(col.shape)
    return [_apply(v) for v in col]


# ---------------------------------------------------------------------------
# Color space conversions (matching R's colorspace package)
# ---------------------------------------------------------------------------

def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, np.power(np.maximum((c + 0.055) / 1.055, 0.0), 2.4))


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(np.maximum(c, 0.0), 1.0 / 2.4) - 0.055)


def _srgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    lin = _srgb_to_linear(rgb)
    return lin @ _M_SRGB_TO_XYZ.T


def _xyz_to_srgb(xyz: np.ndarray) -> np.ndarray:
    lin = xyz @ _M_XYZ_TO_SRGB.T
    return _linear_to_srgb(lin)


def _xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    x = xyz[..., 0] / _XN
    y = xyz[..., 1] / _YN
    z = xyz[..., 2] / _ZN
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    def f(t: np.ndarray) -> np.ndarray:
        return np.where(t > eps, np.cbrt(np.maximum(t, 0.0)), (kappa * t + 16.0) / 116.0)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116.0 * fy - 16.0
    A = 500.0 * (fx - fy)
    B = 200.0 * (fy - fz)
    return np.stack([L, A, B], axis=-1)


def _lab_to_xyz(lab: np.ndarray) -> np.ndarray:
    L = lab[..., 0]
    A = lab[..., 1]
    B = lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = A / 500.0 + fy
    fz = fy - B / 200.0
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    xr = np.where(fx ** 3 > eps, fx ** 3, (116.0 * fx - 16.0) / kappa)
    yr = np.where(L > kappa * eps, ((L + 16.0) / 116.0) ** 3, L / kappa)
    zr = np.where(fz ** 3 > eps, fz ** 3, (116.0 * fz - 16.0) / kappa)
    return np.stack([xr * _XN, yr * _YN, zr * _ZN], axis=-1)


def _xyz_to_luv(xyz: np.ndarray) -> np.ndarray:
    X, Y, Z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    denom = X + 15.0 * Y + 3.0 * Z
    safe_denom = np.where(denom == 0, 1.0, denom)
    u_prime = np.where(denom == 0, 0.0, 4.0 * X / safe_denom)
    v_prime = np.where(denom == 0, 0.0, 9.0 * Y / safe_denom)

    d_n = _XN + 15.0 * _YN + 3.0 * _ZN
    un_prime = 4.0 * _XN / d_n
    vn_prime = 9.0 * _YN / d_n

    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    yr = Y / _YN
    L = np.where(yr > eps, 116.0 * np.cbrt(np.maximum(yr, 0.0)) - 16.0, kappa * yr)
    U = 13.0 * L * (u_prime - un_prime)
    V = 13.0 * L * (v_prime - vn_prime)
    return np.stack([L, U, V], axis=-1)


def _luv_to_xyz(luv: np.ndarray) -> np.ndarray:
    L, U, V = luv[..., 0], luv[..., 1], luv[..., 2]
    d_n = _XN + 15.0 * _YN + 3.0 * _ZN
    un_prime = 4.0 * _XN / d_n
    vn_prime = 9.0 * _YN / d_n

    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    Y = np.where(L > kappa * eps, ((L + 16.0) / 116.0) ** 3, L / kappa) * _YN

    safe_L = np.where(L == 0, 1.0, L)
    u_prime = np.where(L == 0, un_prime, U / (13.0 * safe_L) + un_prime)
    v_prime = np.where(L == 0, vn_prime, V / (13.0 * safe_L) + vn_prime)
    safe_vp = np.where(v_prime == 0, 1.0, v_prime)
    X = np.where(L == 0, 0.0, Y * (9.0 * u_prime) / (4.0 * safe_vp))
    Z = np.where(L == 0, 0.0, Y * (12.0 - 3.0 * u_prime - 20.0 * v_prime) / (4.0 * safe_vp))
    return np.stack([X, Y, Z], axis=-1)


def _srgb_to_hls(rgb: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rgb, dtype=float)
    for idx in np.ndindex(rgb.shape[:-1]):
        r, g, b = float(rgb[idx + (0,)]), float(rgb[idx + (1,)]), float(rgb[idx + (2,)])
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        out[idx] = [h * 360.0, l, s]
    return out


def _hls_to_srgb(hls: np.ndarray) -> np.ndarray:
    out = np.zeros_like(hls, dtype=float)
    for idx in np.ndindex(hls.shape[:-1]):
        h, l, s = float(hls[idx + (0,)]), float(hls[idx + (1,)]), float(hls[idx + (2,)])
        r, g, b = colorsys.hls_to_rgb((h % 360.0) / 360.0, max(0.0, min(1.0, l)), max(0.0, min(1.0, s)))
        out[idx] = [r, g, b]
    return out


def _convert_from_srgb(rgb: np.ndarray, space: str) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=float)
    if space == "sRGB":
        return rgb.copy()
    if space == "RGB":
        return _srgb_to_linear(rgb)
    if space == "XYZ":
        return _srgb_to_xyz(rgb)
    if space == "LAB":
        return _xyz_to_lab(_srgb_to_xyz(rgb))
    if space == "LUV":
        return _xyz_to_luv(_srgb_to_xyz(rgb))
    if space == "HLS":
        return _srgb_to_hls(rgb)
    raise ValueError(f"`space` should be one of 'RGB', 'LAB', 'XYZ', 'sRGB', 'LUV', 'HLS'; got {space!r}")


def _convert_to_srgb(coords: np.ndarray, space: str) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if space == "sRGB":
        out = coords.copy()
    elif space == "RGB":
        out = _linear_to_srgb(coords)
    elif space == "XYZ":
        out = _xyz_to_srgb(coords)
    elif space == "LAB":
        out = _xyz_to_srgb(_lab_to_xyz(coords))
    elif space == "LUV":
        out = _xyz_to_srgb(_luv_to_xyz(coords))
    elif space == "HLS":
        out = _hls_to_srgb(coords)
    else:
        raise ValueError(f"`space` should be one of 'RGB', 'LAB', 'XYZ', 'sRGB', 'LUV', 'HLS'; got {space!r}")
    return np.clip(out, 0.0, 1.0)


_HCL_TO_MPL_MAP = {
    "viridis": "viridis",
    "plasma": "plasma",
    "inferno": "inferno",
    "magma": "magma",
    "cividis": "cividis",
    "spectral": "Spectral",
    "blue-red": "RdBu_r",
    "blue-red 2": "RdBu_r",
    "blue-red 3": "RdBu_r",
    "red-blue": "RdBu",
    "red-green": "RdYlGn_r",
    "green-brown": "BrBG_r",
    "purple-green": "PRGn",
    "pink-green": "PiYG",
    "rdylbu": "RdYlBu",
    "rdbu": "RdBu",
    "blues": "Blues",
    "reds": "Reds",
    "greens": "Greens",
    "grays": "Greys",
    "greys": "Greys",
    "heat": "YlOrRd_r",
    "terrain": "terrain",
}


def _resolve_hcl_palette(name: str, n: int, reverse: bool = False) -> list[str]:
    low = name.strip().lower()
    mpl_name = _HCL_TO_MPL_MAP.get(low, name)
    try:
        cmap = plt.get_cmap(mpl_name)
    except ValueError:
        cmap = plt.get_cmap(name.capitalize())
    ts = np.linspace(1.0, 0.0, n) if reverse else np.linspace(0.0, 1.0, n)
    rgba = cmap(ts)
    return [mcolors.to_hex(c, keep_alpha=False).upper() for c in rgba]


class ColorRamp2:
    """Callable color mapping function created by color_ramp2 / colorRamp2."""

    def __init__(
        self,
        breaks: Sequence[float],
        colors: Sequence[Any] | None = None,
        transparency: float = 0.0,
        space: str = "LAB",
        hcl_palette: str | None = None,
        reverse: bool = False,
    ) -> None:
        breaks_arr = np.asarray(breaks, dtype=float).ravel()
        breaks_o = breaks_arr.copy()

        if hcl_palette is not None:
            breaks_sorted = np.sort(breaks_arr)
            if len(breaks_sorted) == 2:
                breaks2 = np.linspace(breaks_sorted[0], breaks_sorted[1], 1000)
            else:
                nb = len(breaks_sorted)
                nl = int(math.ceil(1000.0 / nb))
                parts = [np.linspace(breaks_sorted[0], breaks_sorted[1], nl)]
                for i in range(1, nb - 1):
                    parts.append(np.linspace(breaks_sorted[i], breaks_sorted[i + 1], nl)[1:])
                breaks2 = np.concatenate(parts)
            colors = _resolve_hcl_palette(hcl_palette, len(breaks2), reverse=reverse)
            breaks_arr = breaks2

        if colors is None or len(breaks_arr) != len(colors):
            raise ValueError("Length of `breaks` should be equal to `colors`.")

        order = np.argsort(breaks_arr, kind="mergesort")
        breaks_arr = breaks_arr[order]
        cols_ordered = [colors[i] for i in order]

        # Deduplicate breaks keeping first occurrence
        keep_b: list[float] = []
        keep_c: list[Any] = []
        seen: set[float] = set()
        for b, c in zip(breaks_arr, cols_ordered):
            bf = float(b)
            if bf not in seen:
                seen.add(bf)
                keep_b.append(bf)
                keep_c.append(c)

        if len(keep_b) < 2:
            raise ValueError("You should have at least two distinct break values.")

        if space not in ("RGB", "LAB", "XYZ", "sRGB", "LUV", "HLS"):
            raise ValueError("`space` should be in 'RGB', 'LAB', 'XYZ', 'sRGB', 'LUV', 'HLS'.")

        rgb_list = []
        for c in keep_c:
            rgba = _single_color_to_rgba(c)
            if rgba is None:
                raise ValueError(f"Invalid color in colorRamp2: {c!r}")
            rgb_list.append(rgba[:3])
        rgb_arr = np.asarray(rgb_list, dtype=float)
        if space == "LUV":
            black_mask = np.all(rgb_arr == 0.0, axis=1)
            rgb_arr[black_mask] = 1e-5

        self._breaks = np.asarray(keep_b, dtype=float)
        self._rgb = rgb_arr
        self._space_coords = _convert_from_srgb(rgb_arr, space)
        self.space = space
        t_clamped = max(0.0, min(1.0, float(transparency)))
        self.transparency = t_clamped
        self._alpha = 1.0 - t_clamped
        self._alpha_hex = f"{int(round(self._alpha * 255)):02X}"
        self.breaks = breaks_o
        self.colors = self(breaks_o)

    def __call__(self, x: Any = None, return_rgb: bool = False, max_value: float = 1.0) -> Any:
        if x is None:
            raise ValueError("Please specify `x`.")
        is_scalar = np.isscalar(x)
        is_df = isinstance(x, pd.DataFrame)
        df_index = x.index if is_df else None
        df_cols = x.columns if is_df else None

        arr = np.asarray(x.to_numpy() if is_df else x, dtype=float)
        orig_shape = arr.shape
        flat = arr.ravel()
        na_mask = np.isnan(flat)

        if np.all(na_mask):
            if is_scalar:
                return None
            out = np.full(len(flat), None, dtype=object).reshape(orig_shape)
            return pd.DataFrame(out, index=df_index, columns=df_cols) if is_df else out

        valid = flat[~na_mask]
        valid_clamped = np.clip(valid, self._breaks[0], self._breaks[-1])
        # Match R's .bincode(valid_clamped, self._breaks, right=TRUE, include.lowest=TRUE)
        idx = np.searchsorted(self._breaks, valid_clamped, side="left") - 1
        idx = np.clip(idx, 0, len(self._breaks) - 2)

        b1 = self._breaks[idx]
        b2 = self._breaks[idx + 1]
        c1 = self._space_coords[idx]
        c2 = self._space_coords[idx + 1]
        frac = ((valid_clamped - b2) / (b2 - b1))[:, None]
        interp_coords = frac * (c2 - c1) + c2
        srgb = _convert_to_srgb(interp_coords, self.space)
        bytes_rgb = np.clip(np.rint(srgb * 255.0), 0, 255).astype(int)

        if return_rgb:
            rgba = np.empty((len(flat), 4), dtype=float)
            rgba[:] = np.nan
            rgba[~na_mask, :3] = bytes_rgb / 255.0
            rgba[~na_mask, 3] = int(self._alpha_hex, 16) / 255.0
            if max_value != 1.0:
                rgba = rgba * max_value
            if is_scalar:
                return rgba[0]
            return rgba

        hex_valid = [f"#{r:02X}{g:02X}{b:02X}{self._alpha_hex}" for r, g, b in bytes_rgb]
        if is_scalar:
            return hex_valid[0]

        out = np.full(len(flat), None, dtype=object)
        out[~na_mask] = hex_valid
        out = out.reshape(orig_shape)
        if is_df:
            return pd.DataFrame(out, index=df_index, columns=df_cols)
        return out


def color_ramp2(
    breaks: Sequence[float],
    colors: Sequence[Any] | None = None,
    transparency: float = 0.0,
    space: str = "LAB",
    hcl_palette: str | None = None,
    reverse: bool = False,
    alpha: float | None = None,
) -> ColorRamp2:
    """Create a color mapping function by linear interpolation in a color space."""
    if alpha is not None:
        transparency = 1.0 - float(alpha)
    return ColorRamp2(
        breaks=breaks,
        colors=colors,
        transparency=transparency,
        space=space,
        hcl_palette=hcl_palette,
        reverse=reverse,
    )


colorRamp2 = color_ramp2


def col2value(
    r: Any,
    g: Any = None,
    b: Any = None,
    col_fun: ColorRamp2 | None = None,
) -> np.ndarray | float:
    """Invert colors produced by colorRamp2 back to approximate numeric values."""
    if col_fun is None:
        raise ValueError("`col_fun` must be provided.")
    is_scalar = False
    if g is None and b is None:
        if isinstance(r, str):
            is_scalar = True
            rgba = _single_color_to_rgba(r)
            rgb_arr = np.asarray([rgba[:3]], dtype=float)
        elif isinstance(r, pd.DataFrame) or (isinstance(r, np.ndarray) and r.ndim == 2 and r.shape[1] >= 3 and np.issubdtype(r.dtype, np.number)):
            arr = np.asarray(r, dtype=float)
            rgb_arr = arr[:, :3]
        elif isinstance(r, (list, tuple, np.ndarray, pd.Series)):
            items = list(r)
            if len(items) == 3 and all(isinstance(v, (list, tuple, np.ndarray, pd.Series)) for v in items):
                rgb_arr = np.column_stack([np.asarray(items[0], dtype=float), np.asarray(items[1], dtype=float), np.asarray(items[2], dtype=float)])
            else:
                rgb_arr = np.asarray([_single_color_to_rgba(v)[:3] for v in items], dtype=float)
        else:
            rgba = _single_color_to_rgba(r)
            is_scalar = True
            rgb_arr = np.asarray([rgba[:3]], dtype=float)
    else:
        r_arr = np.asarray(r, dtype=float).ravel()
        g_arr = np.asarray(g, dtype=float).ravel()
        b_arr = np.asarray(b, dtype=float).ravel()
        rgb_arr = np.column_stack([r_arr, g_arr, b_arr])
        if np.isscalar(r) and np.isscalar(g) and np.isscalar(b):
            is_scalar = True

    breaks = np.asarray(col_fun.breaks, dtype=float)
    break_rgbs = np.asarray([_single_color_to_rgba(c)[:3] for c in col_fun.colors], dtype=float)
    space = col_fun.space

    m = _convert_from_srgb(rgb_arr, space)
    breaks_m = _convert_from_srgb(break_rgbs, space)
    n_breaks = len(breaks)

    out = np.empty(len(m), dtype=float)
    for i in range(len(m)):
        conds = []
        for ch in range(3):
            c_lo = breaks_m[:-1, ch]
            c_hi = breaks_m[1:, ch]
            conds.append(((c_lo <= m[i, ch]) & (c_hi >= m[i, ch])) | ((c_lo >= m[i, ch]) & (c_hi <= m[i, ch])))
        k_matches = np.where(conds[0] & conds[1] & conds[2])[0]
        if len(k_matches) == 0:
            # Fallback to closest segment by midpoint Euclidean distance
            dists = np.linalg.norm(((breaks_m[:-1] + breaks_m[1:]) / 2.0) - m[i], axis=1)
            k = int(np.argmin(dists))
        else:
            k = int(k_matches[0])
        vals = []
        for ch in range(3):
            denom = breaks_m[k, ch] - breaks_m[k + 1, ch]
            if abs(denom) > 1e-12:
                v_ch = (breaks[k] - breaks[k + 1]) / denom * (m[i, ch] - breaks_m[k, ch]) + breaks[k]
                vals.append(v_ch)
        out[i] = float(np.mean(vals)) if vals else float(breaks[k])

    return float(out[0]) if is_scalar else out


# ---------------------------------------------------------------------------
# Random colors (ported from R/random_color.R)
# ---------------------------------------------------------------------------

_COLOR_DICTIONARY: dict[str, dict[str, Any]] = {}


def _define_color(name: str, hue_range: tuple[float, float] | None, lower_bounds: list[tuple[float, float]]) -> None:
    s_min = lower_bounds[0][0]
    s_max = lower_bounds[-1][0]
    b_min = lower_bounds[-1][1]
    b_max = lower_bounds[0][1]
    _COLOR_DICTIONARY[name] = {
        "hueRange": hue_range,
        "lowerBounds": lower_bounds,
        "saturationRange": (s_min, s_max),
        "brightnessRange": (b_min, b_max),
    }


def _load_color_bounds() -> None:
    _define_color("monochrome", None, [(0, 0), (100, 0)])
    _define_color("red", (-26, 18), [(20, 100), (30, 92), (40, 89), (50, 85), (60, 78), (70, 70), (80, 60), (90, 55), (100, 50)])
    _define_color("orange", (19, 46), [(20, 100), (30, 93), (40, 88), (50, 86), (60, 85), (70, 70), (100, 70)])
    _define_color("yellow", (47, 62), [(25, 100), (40, 94), (50, 89), (60, 86), (70, 84), (80, 82), (90, 80), (100, 75)])
    _define_color("green", (63, 178), [(30, 100), (40, 90), (50, 85), (60, 81), (70, 74), (80, 64), (90, 50), (100, 40)])
    _define_color("blue", (179, 257), [(20, 100), (30, 86), (40, 80), (50, 74), (60, 60), (70, 52), (80, 44), (90, 39), (100, 35)])
    _define_color("purple", (258, 282), [(20, 100), (30, 87), (40, 79), (50, 70), (60, 65), (70, 59), (80, 52), (90, 45), (100, 42)])
    _define_color("pink", (283, 334), [(20, 100), (30, 90), (40, 86), (60, 84), (80, 80), (90, 75), (100, 73)])


_load_color_bounds()


def _get_color_info(hue: float) -> dict[str, Any]:
    h = round(hue)
    if 334 <= h <= 360:
        h -= 360
    for info in _COLOR_DICTIONARY.values():
        hr = info["hueRange"]
        if hr is not None and hr[0] <= h <= hr[1]:
            return info
    return _COLOR_DICTIONARY["red"]


def _get_hue_range(color_input: Any) -> tuple[float, float]:
    if color_input is None:
        return (0.0, 360.0)
    if isinstance(color_input, (int, float, np.number)) and 0 < float(color_input) < 360:
        return (float(color_input), float(color_input))
    if isinstance(color_input, str):
        if color_input in _COLOR_DICTIONARY:
            hr = _COLOR_DICTIONARY[color_input]["hueRange"]
            if hr is not None:
                return hr
            return (0.0, 360.0)
        rgba = _single_color_to_rgba(color_input)
        if rgba is not None:
            h, _, _ = colorsys.rgb_to_hsv(rgba[0], rgba[1], rgba[2])
            return (h * 360.0, h * 360.0)
    return (0.0, 360.0)


def _pick_hue(hue: Any) -> float:
    if hue == "monochrome":
        return 0.0
    hr = _get_hue_range(hue)
    h = float(np.random.uniform(hr[0], hr[1]))
    if h < 0:
        h += 360.0
    return h


def _pick_saturation(hue: float, hue_name: Any, luminosity: str) -> float:
    if hue_name == "monochrome":
        return 0.0
    if luminosity == "random":
        return float(np.random.uniform(0.0, 100.0))
    s_min, s_max = _get_color_info(hue)["saturationRange"]
    if luminosity == "bright":
        s_min = 55.0
    elif luminosity == "dark":
        s_min = s_max - 10.0
    elif luminosity == "light":
        s_max = 55.0
    return float(np.random.uniform(s_min, s_max))


def _get_minimum_brightness(h: float, s: float) -> float:
    lb = _get_color_info(h)["lowerBounds"]
    for i in range(len(lb) - 1):
        s1, v1 = lb[i]
        s2, v2 = lb[i + 1]
        if s1 <= s <= s2:
            m = (v2 - v1) / (s2 - s1)
            b = v1 - m * s1
            return m * s + b
    return 0.0


def _pick_brightness(h: float, s: float, luminosity: str) -> float:
    b_min = _get_minimum_brightness(h, s)
    b_max = 100.0
    if luminosity == "dark":
        b_max = b_min + 20.0
    elif luminosity == "light":
        b_min = (b_max + b_min) / 2.0
    elif luminosity == "random":
        b_min = 0.0
        b_max = 100.0
    return float(np.random.uniform(b_min, b_max))


def rand_color(
    n: int,
    hue: Any = None,
    luminosity: str | None = None,
    transparency: float = 0.0,
    friendly: bool = False,
    seed: int | None = None,
) -> list[str]:
    """Generate n visually distinct random colors (ported from R/random_color.R)."""
    n = int(n)
    if n <= 0:
        return []
    saved_state = np.random.get_state() if seed is not None else None
    if seed is not None:
        np.random.seed(int(seed))
    try:
        if friendly:
            cols = rand_color(n, luminosity="dark", transparency=transparency, friendly=False)
            for _ in range(50):
                bad_idx = []
                for i, c in enumerate(cols):
                    rgba = _single_color_to_rgba(c)
                    h, _, v = colorsys.rgb_to_hsv(rgba[0], rgba[1], rgba[2])
                    h_deg = h * 360.0
                    if v > 0.85 or (40.0 < h_deg < 65.0):
                        bad_idx.append(i)
                if not bad_idx:
                    break
                repl = rand_color(len(bad_idx), luminosity="dark", transparency=transparency, friendly=False)
                for idx, rc in zip(bad_idx, repl):
                    cols[idx] = rc
            return cols

        all_lum = ["bright", "light", "dark"] if luminosity is None else [luminosity]
        out: list[str] = []
        for _ in range(n):
            for _attempt in range(100):
                h = _pick_hue(hue)
                lum_s = all_lum[int(np.random.randint(0, len(all_lum)))]
                lum_b = all_lum[int(np.random.randint(0, len(all_lum)))]
                s = _pick_saturation(h, hue, lum_s)
                b = _pick_brightness(h, s, lum_b)
                r, g, bl = colorsys.hsv_to_rgb((h % 360.0) / 360.0, max(0.0, min(1.0, s / 100.0)), max(0.0, min(1.0, b / 100.0)))
                ri, gi, bi = int(round(r * 255)), int(round(g * 255)), int(round(bl * 255))
                if not (ri == 255 and gi == 255 and bi == 255):
                    break
            hex_c = f"#{ri:02X}{gi:02X}{bi:02X}"
            out.append(add_transparency(hex_c, transparency))
        return out
    finally:
        if saved_state is not None:
            np.random.set_state(saved_state)


# ---------------------------------------------------------------------------
# R lty, pch, and font translation
# ---------------------------------------------------------------------------

def resolve_lty(lty: Any) -> Any:
    """Convert R lty (int 0..6 or string) to a Matplotlib linestyle."""
    if is_na_like(lty):
        return "none"
    if isinstance(lty, (int, float, np.integer, np.floating)):
        v = int(lty)
        if v <= 0:
            return "none"
        v = ((v - 1) % 6) + 1
        return {
            1: "-",
            2: "--",
            3: ":",
            4: "-.",
            5: (0, (8, 4)),
            6: (0, (6, 2, 2, 2)),
        }[v]
    if isinstance(lty, str):
        low = lty.strip().lower()
        mapping = {
            "0": "none",
            "blank": "none",
            "none": "none",
            "1": "-",
            "solid": "-",
            "-": "-",
            "2": "--",
            "dashed": "--",
            "--": "--",
            "3": ":",
            "dotted": ":",
            ":": ":",
            "4": "-.",
            "dotdash": "-.",
            "-.": "-.",
            "5": (0, (8, 4)),
            "longdash": (0, (8, 4)),
            "6": (0, (6, 2, 2, 2)),
            "twodash": (0, (6, 2, 2, 2)),
        }
        return mapping.get(low, lty)
    return lty


def resolve_pch(pch: Any) -> tuple[str, str]:
    """Convert R pch (0..25 or string) to (matplotlib_marker, style_mode)."""
    if isinstance(pch, str):
        if pch.isdigit():
            pch = int(pch)
        else:
            return (pch, "filled")
    if isinstance(pch, (int, float, np.integer, np.floating)):
        code = int(pch)
        if code in (0, 7, 12, 14):
            return ("s", "open")
        if code in (1, 10, 13):
            return ("o", "open")
        if code == 2:
            return ("^", "open")
        if code == 3:
            return ("+", "line_only")
        if code == 4:
            return ("x", "line_only")
        if code in (5, 9, 11):
            return ("D", "open")
        if code == 6:
            return ("v", "open")
        if code == 8:
            return ("*", "line_only")
        if code == 15:
            return ("s", "filled")
        if code in (16, 19):
            return ("o", "filled")
        if code == 17:
            return ("^", "filled")
        if code == 18:
            return ("D", "filled")
        if code == 20:
            return ("o", "filled_small")
        if code == 21:
            return ("o", "bg_filled")
        if code == 22:
            return ("s", "bg_filled")
        if code == 23:
            return ("D", "bg_filled")
        if code == 24:
            return ("^", "bg_filled")
        if code == 25:
            return ("v", "bg_filled")
    return ("o", "open")


def resolve_font(font: Any) -> tuple[str, str]:
    """Convert R font integer (1..4) or string to (fontweight, fontstyle)."""
    if isinstance(font, (int, float, np.integer, np.floating)):
        code = int(font)
        if code == 2:
            return ("bold", "normal")
        if code == 3:
            return ("normal", "italic")
        if code == 4:
            return ("bold", "italic")
        return ("normal", "normal")
    if isinstance(font, str):
        low = font.strip().lower()
        if low in ("2", "bold"):
            return ("bold", "normal")
        if low in ("3", "italic"):
            return ("normal", "italic")
        if low in ("4", "bold.italic", "bold_italic", "bold italic"):
            return ("bold", "italic")
    return ("normal", "normal")
