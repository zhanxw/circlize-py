"""Core state, unit conversions, colors, error validation, and rendered graphics tests."""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import circlize as c_alias
import circlize_py as cp
from conftest import assert_non_blank


def test_package_alias_and_r_colors():
    assert c_alias.__version__ == cp.__version__
    assert c_alias.circos is cp.circos
    assert "grey50" in cp._r_colors.R_NAMED_COLORS
    assert cp.color_to_hex("green", include_alpha=False) == "#00FF00"
    assert cp.color_to_hex("green") == "#00FF00FF"
    assert cp.color_to_hex("grey50", include_alpha=False) == "#7F7F7F"
    assert cp.color_to_hex(1, include_alpha=False) == "#000000"
    assert cp.add_transparency("#FF0000", 0.5) == "#FF000080"


def test_circos_par_validation_and_reset():
    cp.circos_par(track_height=0.25, start_degree=45)
    assert cp.circos_par("track.height") == 0.25
    assert cp.circos_par["start.degree"] == 45.0
    with pytest.raises(ValueError):
        cp.circos_par(gap_degree=400)
    with pytest.raises(ValueError):
        cp.circos_par(track_margin=(0.01,))
    with pytest.raises(ValueError):
        cp.circos_par(circle_margin=(-0.1,))
    with pytest.raises(KeyError):
        cp.circos_par(not_a_real_option=1)

    cp.circos_par(RESET=True)
    assert cp.circos_par("start.degree") == 0.0
    assert cp.circos_par("track.height") == 0.2


def test_cell_meta_and_named_matrix_2d():
    cp.circos_initialize(["A", "B", "C"], xlim=[0, 10])
    cp.circos_track(ylim=[-2, 2], panel_fun=lambda x, y: None)

    cp.set_current_cell("B", 1)
    assert cp.CELL_META.sector_index == "B"
    assert cp.CELL_META.sector.index == "B"
    assert cp.CELL_META.sector_numeric_index == 2
    assert cp.CELL_META.sector.numeric.index == 2
    assert cp.CELL_META.sector_idx == 1
    assert cp.CELL_META.track_index == 1
    assert np.isclose(cp.CELL_META.xcenter, 5.0)
    assert np.isclose(cp.CELL_META.ycenter, 0.0)
    assert np.allclose(cp.CELL_META.xlim, [0.0, 10.0])
    assert "sector.index" in cp.CELL_META.names()

    coords = cp.circlize([0, 5, 10], [0, 1, 2], sector_index="B", track_index=1)
    assert isinstance(coords, cp.NamedMatrix2D)
    assert np.allclose(coords[:, "theta"], coords.theta)
    assert np.allclose(coords[:, "rou"], coords.rou)
    rev = cp.reverse_circlize(coords.theta, coords.rou, sector_index="B", track_index=1)
    assert np.allclose(rev.x, [0, 5, 10])
    assert np.allclose(rev.y, [0, 1, 2])


def test_physical_unit_conversions_and_fontsize():
    cp.circos_initialize(["A"], xlim=[0, 100])
    cp.circos_track(ylim=[0, 10], track_height=0.2)
    mm_len = cp.convert_length(5, "mm")
    cm_len = cp.convert_length(0.5, "cm")
    assert np.isclose(mm_len, cm_len, rtol=1e-9)
    assert np.isclose(cp.mm_h(5), cp.uh(5, "mm"))
    assert np.isclose(cp.inch_h(1), cp.inches_h(1))
    assert np.isclose(cp.ux(5, "mm"), cp.mm_x(5))
    assert np.isclose(cp.uy(5, "mm"), cp.mm_y(5))
    assert np.isclose(cp.ux(1, "cm"), cp.cm_x(1))
    assert np.isclose(cp.uy(1, "cm"), cp.cm_y(1))
    assert cp.fontsize(12) > 0
    assert isinstance(cp.degree(10), cp.DegreeValue)


def test_low_level_and_high_level_rendering_non_blank():
    rng = np.random.default_rng(7)
    fig = plt.figure(figsize=(6, 6))
    cp.circos_initialize(["S1", "S2", "S3"], xlim=[0, 10])

    def _p1(x, y):
        cp.circos_points([2, 5, 8], [0.2, 0.5, 0.8], col="red", pch=16)
        cp.circos_lines([1, 5, 9], [0.1, 0.9, 0.2], col="blue", area=True)
        cp.circos_polygon([3, 4, 4, 3], [0.2, 0.2, 0.6, 0.6], col="#00FF0040", border="green")
        cp.circos_rect(6, 0.2, 8, 0.7, col="#FF00FF40", border="purple")
        cp.circos_segments(1, 0.5, 9, 0.5, col="black", lty=2)
        cp.circos_text(5, 0.9, cp.CELL_META.sector_index, facing="inside", nice_facing=True)
        cp.circos_axis(h="top", labels_cex=0.5)

    cp.circos_track(ylim=[0, 1], panel_fun=_p1)

    cp.circos_track(
        ylim=[-2, 2],
        panel_fun=lambda x, y: cp.circos_boxplot(rng.normal(size=(20, 3)), pos=[2, 5, 8], col="orange"),
    )
    cp.circos_link("S1", [2, 4], "S3", [5, 7], col="#1F77B460", directional=1)
    assert len(plt.gca().patches) + len(plt.gca().lines) + len(plt.gca().collections) > 0
    assert_non_blank(fig)


def test_circos_nested_and_overlay_cleanup():
    wb = cp.load_extdata("tagments_WGBS_DMR")
    dmr1 = wb["DMR1"]
    tagments = wb["tagments"]
    corr = wb["correspondance"]

    def f1():
        cp.circos_par(gap_degree=1.5)
        cp.circos_genomic_initialize(tagments, plot_type=None)
        cp.circos_genomic_track(
            dmr1,
            ylim=[-1, 1],
            track_height=0.12,
            panel_fun=lambda r, v: cp.circos_genomic_points(r, v, pch=16, cex=0.5, col="red"),
        )

    def f2():
        cp.circos_initialize_with_ideogram(
            chromosome_index=corr.iloc[:, 0].unique().tolist(),
            plot_type=("ideogram", "labels"),
            track_height=0.06,
        )

    fig = plt.figure(figsize=(6, 6))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cp.circos_nested(f1, f2, corr.iloc[:, [3, 4, 5, 0, 1, 2]], connection_col="#1B9E7740")
    assert not cp.is_circos_initialized()
    assert cp._state._STATE.canvas_overlay_scale is None
    assert_non_blank(fig)


def test_circos_heatmap_and_genomic_species_fallback():
    mat = pd.DataFrame(
        np.random.default_rng(11).normal(size=(24, 5)),
        index=[f"R{i}" for i in range(24)],
        columns=[f"C{i}" for i in range(5)],
    )
    split = ["G1"] * 12 + ["G2"] * 12
    fig1 = plt.figure(figsize=(6, 6))
    cp.circos.heatmap(
        mat,
        split=split,
        col=cp.color_ramp2([-2, 0, 2], ["blue", "white", "red"]),
        dend_side="inside",
        rownames_side="outside",
    )
    pos = cp.circos.heatmap.get.x([0, 15, "R20"])
    assert len(pos) == 3
    cp.circos.heatmap.link(0, 15, col="black")
    assert_non_blank(fig1)

    cp.circos_clear()
    fig2 = plt.figure(figsize=(6, 6))
    # anoGam1 is in chrom_info_list (no cytoband), tests automatic chromInfo fallback
    cp.circos_initialize_with_ideogram(species="anoGam1")
    secs = cp.get_all_sector_index()
    assert len(secs) > 0
    assert_non_blank(fig2)


def test_circos_heatmap_categorical_split_order():
    rng = np.random.default_rng(123)
    mat1 = pd.DataFrame(
        rng.normal(size=(100, 10)),
        index=[f"G{i}" for i in range(100)],
        columns=[f"S{j}" for j in range(10)],
    )
    mat2 = rng.normal(size=(100, 4))
    split = pd.Categorical(rng.choice(list("abcde"), 100), categories=list("edcba"), ordered=True)

    fig = plt.figure(figsize=(6, 6))
    cp.circos.clear()
    cp.circos.heatmap(
        mat1,
        split=split,
        col=cp.color_ramp2([-2, 0, 2], ["blue", "white", "red"]),
        dend_side="inside",
    )
    cp.circos.heatmap(mat2, col=cp.color_ramp2([-2, 0, 2], ["green", "white", "orange"]))
    assert cp.get_all_sector_index() == ["e", "d", "c", "b", "a"]
    cp.circos.heatmap.link(0, 50, col="black")
    cp.circos.clear()
    assert_non_blank(fig)

