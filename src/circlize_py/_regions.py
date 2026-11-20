"""Genomic interval sorting, merging (reducing), and overlap helpers."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd


def _extract_start_end(region: Any) -> tuple[np.ndarray, np.ndarray, pd.DataFrame | None]:
    if isinstance(region, pd.DataFrame):
        if region.shape[1] < 2:
            raise ValueError("`region` must have at least two columns (start, end).")
        starts = pd.to_numeric(region.iloc[:, 0], errors="raise").to_numpy(dtype=float)
        ends = pd.to_numeric(region.iloc[:, 1], errors="raise").to_numpy(dtype=float)
        return starts, ends, region
    arr = np.asarray(region, dtype=float)
    if arr.ndim == 1:
        if len(arr) != 2:
            raise ValueError("1D `region` must have length 2.")
        arr = arr.reshape(1, 2)
    if arr.shape[1] < 2:
        raise ValueError("`region` must have at least two columns (start, end).")
    return arr[:, 0], arr[:, 1], None


def order_region(region: Any) -> np.ndarray:
    """Return 0-based indices sorting `region` by (start, end) ascending."""
    starts, ends, _ = _extract_start_end(region)
    if len(starts) == 0:
        return np.array([], dtype=int)
    return np.lexsort((ends, starts))


def sort_region(region: Any) -> pd.DataFrame:
    """Return a copy of `region` sorted by (start, end) ascending."""
    starts, ends, df = _extract_start_end(region)
    od = np.lexsort((ends, starts)) if len(starts) > 0 else np.array([], dtype=int)
    if df is not None:
        return df.iloc[od].reset_index(drop=True)
    arr = np.column_stack([starts, ends])
    return pd.DataFrame(arr[od], columns=["start", "end"])


def reduce_region(region: Any, gap_bp: float = 0.0) -> pd.DataFrame:
    """Merge overlapping or adjacent (within 1 + gap_bp) 1-based closed intervals."""
    starts, ends, _ = _extract_start_end(region)
    n = len(starts)
    if n == 0:
        return pd.DataFrame({"start": pd.Series(dtype=float), "end": pd.Series(dtype=float)})

    od = np.lexsort((ends, starts))
    s_sorted = starts[od]
    e_sorted = ends[od]

    out_s: list[float] = [float(s_sorted[0])]
    out_e: list[float] = [float(e_sorted[0])]
    tol = 1.0 + float(gap_bp)

    for i in range(1, n):
        cur_s = float(s_sorted[i])
        cur_e = float(e_sorted[i])
        if cur_s <= out_e[-1] + tol:
            if cur_e > out_e[-1]:
                out_e[-1] = cur_e
        else:
            out_s.append(cur_s)
            out_e.append(cur_e)

    return pd.DataFrame({"start": out_s, "end": out_e})


def overlap_interval(s1: float, e1: float, s2: float, e2: float) -> float:
    """Return length of overlap between 1-based closed intervals [s1, e1] and [s2, e2]."""
    ov = min(float(e1), float(e2)) - max(float(s1), float(s2)) + 1.0
    return float(ov) if ov > 0.0 else 0.0


def overlap_region(
    region1: Any,
    region2: Any,
    percent: bool = True,
    count_by: Literal["percent", "number"] = "percent",
) -> np.ndarray:
    """Compute bp overlap, fractional overlap, or hit count of `region1` against `region2`."""
    s1, e1, _ = _extract_start_end(region1)
    n1 = len(s1)
    if n1 == 0:
        return np.array([], dtype=float)

    r2 = reduce_region(region2)
    if len(r2) == 0:
        return np.zeros(n1, dtype=float)

    s2 = r2["start"].to_numpy(dtype=float)
    e2 = r2["end"].to_numpy(dtype=float)
    n2 = len(s2)

    overlaps = np.zeros(n1, dtype=float)
    counts = np.zeros(n1, dtype=float)

    # First index in r2 where e2[j] >= s1[i]
    first_j = np.searchsorted(e2, s1, side="left")
    # First index in r2 where s2[j] > e1[i]
    end_j = np.searchsorted(s2, e1, side="right")

    for i in range(n1):
        j0 = int(first_j[i])
        j1 = min(int(end_j[i]), n2)
        if j0 >= j1:
            continue
        a1 = float(s1[i])
        b1 = float(e1[i])
        ov_sum = 0.0
        cnt = 0
        for j in range(j0, j1):
            ov = min(b1, float(e2[j])) - max(a1, float(s2[j])) + 1.0
            if ov > 0.0:
                ov_sum += ov
                cnt += 1
        overlaps[i] = ov_sum
        counts[i] = float(cnt)

    if count_by == "number":
        return counts
    if percent:
        widths = np.maximum(e1 - s1 + 1.0, 1.0)
        return overlaps / widths
    return overlaps
