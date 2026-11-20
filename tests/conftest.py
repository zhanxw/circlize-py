"""Pytest configuration and Matplotlib/circlize fixtures."""

from __future__ import annotations

import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/circlize_mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pytest

from circlize_py import circos_clear


@pytest.fixture(autouse=True)
def _reset_circos_and_figures():
    """Ensure clean circlize state and closed Matplotlib figures around every test."""
    circos_clear()
    plt.close("all")
    yield
    circos_clear()
    plt.close("all")


def render_rgba(fig=None) -> np.ndarray:
    """Render current or given Matplotlib figure to an RGBA uint8 numpy array."""
    target = fig if fig is not None else plt.gcf()
    target.canvas.draw()
    buf = np.asarray(target.canvas.buffer_rgba())
    return buf.copy()


def assert_non_blank(fig=None) -> np.ndarray:
    """Assert rendered figure contains drawn non-background pixels."""
    rgba = render_rgba(fig)
    assert rgba.ndim == 3 and rgba.shape[2] == 4
    rgb = rgba[:, :, :3]
    assert np.std(rgb) > 0.5 or np.any(rgb < 250), "Rendered canvas appears completely blank."
    return rgba
