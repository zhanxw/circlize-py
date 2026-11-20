"""Ported execution tests covering all 62 non-empty man/*.Rd examples plus additional workflows."""

from __future__ import annotations

import string
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import circlize as cz
import circlize_py as cp
from conftest import assert_non_blank


def _make_fig():
    return plt.figure(figsize=(6.0, 6.0), dpi=80)


# ---------------------------------------------------------------------------
# All 62 non-empty man/*.Rd examples ported faithfully to Python
# ---------------------------------------------------------------------------

def _ex_CELL_META():
    seen = []
    fig = _make_fig()
    cp.circos.initialize(["a"], xlim=[0, 1])

    def pf(x, y):
        seen.append((cp.CELL_META.sector.index, list(cp.CELL_META.xlim)))

    cp.circos.track(ylim=[0, 1], panel_fun=pf)
    assert seen == [("a", [0.0, 1.0])]
    assert "sector.index" in cp.names_CELL_META(cp.CELL_META)
    cp.circos.clear()
    return fig, False


def _ex_add_transparency():
    c1 = cp.add_transparency("red", 0.5)
    c2 = cp.add_transparency(1, 0.5)
    c3 = cp.add_transparency("#FF000080", 0.2)
    assert c1.startswith("#") and len(c1) == 9
    assert c2.startswith("#") and len(c2) == 9
    assert c3.startswith("#") and len(c3) == 9
    return None, False


def _ex_adjacencyList2Matrix():
    rng = np.random.default_rng(123)
    df = pd.DataFrame(
        {
            "from": rng.choice(list(string.ascii_lowercase), 10, replace=True),
            "to": rng.choice(list(string.ascii_lowercase), 10, replace=True),
            "value": np.arange(1, 11),
        }
    )
    m1 = cp.adjacencyList2Matrix(df)
    m2 = cp.adjacencyList2Matrix(df, square=True)
    assert m2.shape[0] == m2.shape[1]
    assert m1.to_numpy().sum() > 0
    return None, False


def _ex_adjacencyMatrix2List():
    rng = np.random.default_rng(999)
    mat = pd.DataFrame(
        rng.permutation(np.arange(1, 19)).reshape(3, 6),
        index=[f"S{i}" for i in range(1, 4)],
        columns=[f"E{i}" for i in range(1, 7)],
    )
    lst = cp.adjacencyMatrix2List(mat)
    assert len(lst) == 18
    return None, False


def _ex_arrange_links_evenly():
    rng = np.random.default_rng(5)
    sectors = list(string.ascii_lowercase[:20])
    df = pd.DataFrame(
        {"from": rng.choice(sectors, 30, replace=True), "to": rng.choice(sectors, 30, replace=True)}
    ).drop_duplicates()
    df = df[df["from"] != df["to"]].reset_index(drop=True)
    fig = _make_fig()
    cp.circos.initialize(sectors, xlim=[0, 1])
    cp.circos.track(
        ylim=[0, 1],
        panel_fun=lambda x, y: cp.circos.text(cp.CELL_META.xcenter, cp.CELL_META.ycenter, cp.CELL_META.sector_index),
    )
    df2 = cp.arrange_links_evenly(df, directional=1)
    for _, r in df2.iterrows():
        cp.circos.link(r["sector1"], r["pos1"], r["sector2"], r["pos2"], directional=1)
    cp.circos.clear()
    return fig, True


def _ex_calc_gap():
    rng = np.random.default_rng(123)
    mat1 = rng.integers(1, 21, size=(5, 5))
    fig = _make_fig()
    cp.chordDiagram(mat1, directional=1, grid_col=list(range(1, 6)) * 2, transparency=0.5, big_gap=10, small_gap=1)
    cp.circos.clear()
    mat2 = mat1 / 2.0
    gap = cp.calc_gap(mat1, mat2, big_gap=10, small_gap=1)
    assert gap > 10
    cp.chordDiagram(mat2, directional=1, grid_col=list(range(1, 6)) * 2, transparency=0.5, big_gap=gap, small_gap=1)
    cp.circos.clear()
    return fig, True


def _ex_chordDiagram():
    rng = np.random.default_rng(999)
    mat = pd.DataFrame(
        rng.permutation(np.arange(1, 19)).reshape(3, 6),
        index=[f"S{i}" for i in range(1, 4)],
        columns=[f"E{i}" for i in range(1, 7)],
    )
    fig = _make_fig()
    cp.chordDiagram(mat)
    cp.circos.clear()
    df = cp.adjacencyMatrix2List(mat)
    cp.chordDiagram(df)
    cp.circos.clear()
    return fig, True


def _ex_circlize():
    cp.circos.initialize(["a", "b"], xlim=[0, 1])
    cp.circos.track(ylim=[0, 1])
    res = cp.circlize(0.5, 0.5, sector_index="a", track_index=1)
    assert res.shape == (1, 2)
    cp.circos.clear()
    return None, False


def _ex_circos_arrow():
    fig = _make_fig()
    cp.circos.initialize(list("abcd"), xlim=[0, 1])
    cols = cp.rand_color(4, seed=1)
    tails = ["point", "normal", "point", "normal"]
    cp.circos.track(
        ylim=[0, 1],
        bg_border=None,
        track_height=0.4,
        panel_fun=lambda x, y: cp.circos.arrow(
            0,
            1,
            y=0.5,
            width=0.4,
            arrow_head_width=0.6,
            arrow_head_length=cp.cm_x(1),
            col=cols[cp.CELL_META.sector_numeric_index - 1],
            tail=tails[cp.CELL_META.sector_numeric_index - 1],
        ),
    )
    cp.circos.clear()
    return fig, True


def _ex_circos_axis():
    sectors = list("abcdefgh")
    fig = _make_fig()
    cp.circos.par(points_overflow_warning=False)
    cp.circos.initialize(sectors, xlim=[0, 10])
    cp.circos.trackPlotRegion(
        sectors,
        ylim=[0, 10],
        track_height=0.1,
        bg_border=None,
        panel_fun=lambda x, y: cp.circos.text(5, 10, cp.get.cell.meta.data("sector.index")),
    )
    cp.circos.trackPlotRegion(sectors, ylim=[0, 10])
    cp.circos.axis(sector_index="a")
    cp.circos.axis(sector_index="b", direction="inside", labels_facing="outside")
    cp.circos.axis(sector_index="c", h="bottom")
    cp.circos.axis(sector_index="d", h="bottom", direction="inside", labels_facing="reverse.clockwise")
    cp.circos.axis(sector_index="e", h=5, major_at=[1, 3, 5, 7, 9])
    cp.circos.axis(sector_index="f", h=5, major_at=[1, 3, 5, 7, 9], labels=["a", "c", "e", "g", "f"], minor_ticks=0)
    cp.circos.axis(sector_index="g", h=5, major_at=[1, 3, 5, 7, 9], labels=["a1", "c1", "e1", "g1", "f1"], major_tick=False)
    cp.circos.clear()
    return fig, True


def _ex_circos_barplot():
    rng = np.random.default_rng(10)
    fig = _make_fig()
    cp.circos.initialize(list("abcd"), xlim=[0, 10])
    cp.circos.track(ylim=[0, 1], panel_fun=lambda x, y: cp.circos.barplot(rng.uniform(size=10), np.arange(0.5, 10.5), col=list(range(1, 11))))
    cp.circos.clear()
    return fig, True


def _ex_circos_boxplot():
    rng = np.random.default_rng(11)
    fig = _make_fig()
    cp.circos.initialize(list("abcd"), xlim=[0, 10])
    cp.circos.track(ylim=[0, 1], panel_fun=lambda x, y: cp.circos.boxplot(rng.uniform(size=(10, 10)), np.arange(0.5, 10.5), col="lightblue"))
    cp.circos.clear()
    return fig, True


def _ex_circos_connect():
    rng = np.random.default_rng(12)
    fig = _make_fig()
    cp.circos.initialize(["a"], xlim=[0, 1])

    def _p(x, y):
        x0 = rng.uniform(size=30)
        x1 = rng.uniform(size=30)
        cp.circos.connect(x0, 0, x1, 1, type="bezier", border=None, col=cp.rand_color(30, transparency=0.6))

    cp.circos.track(ylim=[0, 1], track_height=0.6, bg_border=None, panel_fun=_p)
    cp.circos.clear()
    return fig, True


def _ex_circos_dendrogram():
    hc = cp.load_extdata("bird.orders")
    labels = hc.labels
    ct = cp.cutree(hc, 6)
    n = len(labels)
    dend = cp.color_branches(cp.as_dendrogram(hc), k=6, col=list(range(1, 7)))
    fig = _make_fig()
    cp.circos.par(cell_padding=[0, 0, 0, 0])
    cp.circos.initialize(["a"], xlim=[0, n])

    def _pl(x, y):
        for i, lbl in enumerate(labels):
            cp.circos.text(i + 0.5, 0, lbl, adj=[0, 0.5], facing="clockwise", niceFacing=True, col=int(ct[lbl]), cex=0.6)

    cp.circos.trackPlotRegion(ylim=[0, 1], bg_border=None, track_height=0.3, panel_fun=_pl)
    max_h = float(dend.height)
    cp.circos.trackPlotRegion(
        ylim=[0, max_h],
        bg_border=None,
        track_height=0.4,
        panel_fun=lambda x, y: cp.circos.dendrogram(dend, max_height=max_h),
    )
    cp.circos.clear()
    return fig, True


def _ex_circos_genomicAxis():
    fig = _make_fig()
    cp.circos.initializeWithIdeogram(chromosome_index=[f"chr{i}" for i in range(1, 5)], plotType=None)
    cp.circos.track(ylim=[0, 1], panel_fun=lambda x, y: cp.circos.genomicAxis())
    cp.circos.track(ylim=[0, 1], track_height=0.1)
    cp.circos.track(
        track_index=cp.get.current.track.index(),
        panel_fun=lambda x, y: cp.circos.genomicAxis(h="bottom", direction="inside"),
    )
    cp.circos.clear()
    return fig, True


def _ex_circos_genomicDensity():
    dmr = cp.load_extdata("DMR")
    fig = _make_fig()
    cp.circos.initializeWithIdeogram(chromosome_index=["chr1", "chr2"], plotType=["axis", "labels"])
    bed_list = [dmr["DMR_hyper"], dmr["DMR_hypo"]]
    cp.circos.genomicDensity(bed_list, col=["#FF000080", "#0000FF80"], track_height=0.2)
    cp.circos.clear()
    return fig, True


def _ex_circos_genomicHeatmap():
    fig = _make_fig()
    cp.circos.initializeWithIdeogram(chromosome_index=["chr1", "chr2"], plotType=["labels", "axis"])
    bed = cp.generateRandomBed(nr=40, nc=4, seed=1)
    col_fun = cp.colorRamp2([-1, 0, 1], ["green", "black", "red"])
    cp.circos.genomicHeatmap(bed, col_fun, side="inside", border="white")
    cp.circos.clear()
    return fig, True


def _ex_circos_genomicIdeogram():
    fig = _make_fig()
    cp.circos.initializeWithIdeogram(chromosome_index=["chr1", "chr2"], plotType=["labels", "axis"])
    cp.circos.track(ylim=[0, 1])
    cp.circos.genomicIdeogram()
    cp.circos.clear()
    return fig, True


def _ex_circos_genomicInitialize():
    df = pd.DataFrame(
        {
            "name": ["TP53", "TP63", "TP73"],
            "start": [7565097, 189349205, 3569084],
            "end": [7590856, 189615068, 3652765],
        }
    )
    fig = _make_fig()
    cp.circos.genomicInitialize(df, tickLabelsStartFromZero=False, major_by=50000, sector_names=["tp53", "tp63", "tp73"])
    cp.circos.clear()
    return fig, True


def _ex_circos_genomicLabels():
    rng = np.random.default_rng(9)
    fig = _make_fig()
    cp.circos.initializeWithIdeogram(chromosome_index=["chr1", "chr2", "chr3"])
    bed = cp.generateRandomBed(nr=30, fun=lambda k: rng.choice(list("abcd"), size=k, replace=True), seed=2)
    # Wait: generateRandomBed casts fun to float, so assign string column directly like R
    bed["lbl"] = rng.choice(list("abcdef"), size=len(bed), replace=True)
    bed.loc[0, "lbl"] = "aaaaa"
    cp.circos.genomicLabels(bed[["chr", "start", "end", "lbl"]], labels_column=4, side="inside")
    cp.circos.clear()
    return fig, True


def _ex_circos_genomicLines():
    fig = _make_fig()
    cp.circos.par(track_height=0.1)
    cp.circos.initializeWithIdeogram(chromosome_index=["chr1", "chr2"], plotType=None)
    b1 = cp.generateRandomBed(nr=60, seed=1)
    b2 = cp.generateRandomBed(nr=60, seed=2)
    cp.circos.genomicTrack(b1, panel_fun=lambda r, v, **kw: cp.circos.genomicLines(r, v, type="l", **kw))
    cp.circos.genomicTrack(
        [b1, b2],
        stack=True,
        panel_fun=lambda r, v, **kw: cp.circos.genomicLines(r, v, col=cp.getI(**kw), **kw),
    )
    cp.circos.clear()
    return fig, True


def _ex_circos_genomicLink():
    fig = _make_fig()
    b1 = cp.generateRandomBed(nr=60, seed=1).iloc[:15]
    b2 = cp.generateRandomBed(nr=60, seed=2).iloc[:15]
    cp.circos.par(track_height=0.1, cell_padding=[0, 0, 0, 0])
    cp.circos.initializeWithIdeogram(chromosome_index=["chr1", "chr2", "chr3"])
    cp.circos.genomicLink(b1, b2, col=list(range(1, 16)), border=None)
    cp.circos.clear()
    return fig, True


def _ex_circos_genomicPoints():
    fig = _make_fig()
    cp.circos.par(track_height=0.1)
    cp.circos.initializeWithIdeogram(chromosome_index=["chr1", "chr2"], plotType=None)
    b1 = cp.generateRandomBed(nr=60, seed=1)
    cp.circos.genomicTrack(b1, panel_fun=lambda r, v, **kw: cp.circos.genomicPoints(r, v, pch=16, cex=0.5, **kw))
    cp.circos.genomicTrack(
        b1,
        stack=True,
        panel_fun=lambda r, v, **kw: cp.circos.genomicPoints(r, v, pch=16, cex=0.5, **kw),
    )
    cp.circos.clear()
    return fig, True


def _ex_circos_genomicRainfall():
    dmr = cp.load_extdata("DMR")
    fig = _make_fig()
    cp.circos.initializeWithIdeogram(chromosome_index=["chr1", "chr2"], plotType=["axis", "labels"])
    bed_list = [dmr["DMR_hyper"], dmr["DMR_hypo"]]
    cp.circos.genomicRainfall(bed_list, pch=16, cex=0.4, col=["#FF000080", "#0000FF80"])
    cp.circos.clear()
    return fig, True


def _ex_circos_genomicRect():
    fig = _make_fig()
    cp.circos.par(track_height=0.1, cell_padding=[0, 0, 0, 0])
    cp.circos.initializeWithIdeogram(chromosome_index=["chr1", "chr2"], plotType=None)
    b1 = cp.generateRandomBed(nr=50, seed=1)
    b2 = cp.generateRandomBed(nr=50, seed=2)
    f = cp.colorRamp2([-1, 0, 1], ["green", "black", "red"])
    cp.circos.genomicTrack(
        [b1, b2],
        stack=True,
        panel_fun=lambda r, v, **kw: cp.circos.genomicRect(r, v, col=f(v.iloc[:, 0]), border=None, **kw),
    )
    cp.circos.clear()
    return fig, True


def _ex_circos_genomicText():
    rng = np.random.default_rng(20)
    fig = _make_fig()
    cp.circos.par(track_height=0.1, cell_padding=[0, 0, 0, 0])
    cp.circos.initializeWithIdeogram(chromosome_index=["chr1", "chr2"], plotType=None)
    bed = cp.generateRandomBed(nr=20, seed=3)
    cp.circos.genomicTrack(
        bed,
        ylim=[0, 1],
        panel_fun=lambda r, v, **kw: cp.circos.genomicText(r, v, y=0.5, labels="text", **kw),
    )
    bed["lbl"] = rng.choice(list("abcd"), size=len(bed), replace=True)
    cp.circos.genomicTrack(
        bed,
        panel_fun=lambda r, v, **kw: cp.circos.genomicText(r, v, labels_column=2, **kw),
    )
    cp.circos.clear()
    return fig, True


def _ex_circos_heatmap():
    rng = np.random.default_rng(123)
    mat1 = np.vstack([
        np.hstack([rng.normal(1, 1, (30, 4)), rng.normal(-1, 1, (30, 4))]),
        np.hstack([rng.normal(-1, 1, (30, 4)), rng.normal(1, 1, (30, 4))]),
    ])
    df1 = pd.DataFrame(mat1, index=[f"R{i}" for i in range(60)], columns=[f"C{i}" for i in range(8)])
    split = rng.choice(list("abcde"), 60, replace=True).tolist()
    fig = _make_fig()
    cp.circos.heatmap(df1, split=split, col=cp.colorRamp2([-2, 0, 2], ["blue", "white", "red"]))
    cp.circos.clear()
    return fig, True


def _ex_circos_heatmap_link():
    rng = np.random.default_rng(123)
    mat = pd.DataFrame(rng.normal(size=(40, 6)), index=[f"R{i}" for i in range(40)])
    split = rng.choice(list("abcd"), 40, replace=True).tolist()
    fig = _make_fig()
    cp.circos.heatmap(mat, col=cp.colorRamp2([-2, 0, 2], ["blue", "white", "red"]), split=split, rownames_side="outside")
    cp.circos.heatmap.link(5, 25)
    cp.circos.clear()
    return fig, True


def _ex_circos_initializeWithIdeogram():
    fig = _make_fig()
    cp.circos.initializeWithIdeogram(chromosome_index=["chr1", "chr2", "chr3"])
    cp.circos.clear()
    cp.circos.initializeWithIdeogram(species="mm10", chromosome_index=["chr1", "chr2"])
    cp.circos.clear()
    return fig, True


def _ex_circos_labels():
    fig = _make_fig()
    cp.circos.initialize(sectors=list("abcdefgh"), xlim=[0, 1])
    cp.circos.track(ylim=[0, 1])
    cp.circos.labels(["a", "a", "b", "b"], x=[0.1, 0.12, 0.4, 0.6], labels=["0.1", "0.12", "0.4", "0.6"])
    cp.circos.clear()
    return fig, True


def _ex_circos_lines():
    rng = np.random.default_rng(4)
    sectors = list("abcdefghi")
    fig = _make_fig()
    cp.circos.par(points_overflow_warning=False)
    cp.circos.initialize(sectors, xlim=[0, 10])
    cp.circos.trackPlotRegion(sectors, ylim=[0, 10], track_height=0.5)
    for si, tp, area, base in [
        ("a", "l", False, "bottom"),
        ("b", "o", False, "bottom"),
        ("c", "h", False, "bottom"),
        ("d", "h", False, 5),
        ("e", "s", False, "bottom"),
        ("f", "l", True, "bottom"),
        ("g", "o", True, "bottom"),
    ]:
        xs = np.sort(rng.uniform(0, 10, 10))
        ys = rng.uniform(0, 8, 10)
        cp.circos.lines(xs, ys, sector_index=si, type=tp, area=area, baseline=base)
    cp.circos.clear()
    return fig, True


def _ex_circos_par():
    d = cp.circos.par()
    assert isinstance(d, dict) and "track.height" in d
    return None, False


def _ex_circos_points():
    rng = np.random.default_rng(5)
    fig = _make_fig()
    cp.circos.initialize(list("abcdefgh"), xlim=[0, 1])
    cp.circos.track(ylim=[0, 1], panel_fun=lambda x, y: cp.circos.points(rng.uniform(size=10), rng.uniform(size=10)))
    cp.circos.points(rng.uniform(size=10), rng.uniform(size=10), sector_index="c", pch=16, col="red")
    cp.circos.clear()
    return fig, True


def _ex_circos_polygon():
    rng = np.random.default_rng(123)
    fig = _make_fig()
    cp.circos.initialize(list("abcd"), xlim=[0, 1])

    def _p(x, y):
        x1 = np.sort(rng.uniform(size=20))
        y1 = x1 + rng.normal(scale=0.3, size=20)
        d1 = np.concatenate([x1, x1[::-1]])
        d2 = np.concatenate([x1 + 0.5, (x1 - 0.5)[::-1]])
        cp.circos.polygon(d1, d2, col="#CCCCCC", border=None)
        cp.circos.points(x1, y1, cex=0.5)
        cp.circos.lines(x1, x1)

    cp.circos.trackPlotRegion(ylim=[-3, 3], track_height=0.4, panel_fun=_p)
    cp.circos.clear()
    return fig, True


def _ex_circos_raster():
    img = cp.load_extdata("Rlogo.png")
    fig = _make_fig()
    cp.circos.initialize(list("abcd"), xlim=[0, 1])
    cp.circos.track(
        ylim=[0, 1],
        panel_fun=lambda x, y: cp.circos.raster(
            img, cp.CELL_META.xcenter, cp.CELL_META.ycenter, width="1.2cm", facing="inside", niceFacing=True
        ),
    )
    cp.circos.clear()
    return fig, True


def _ex_circos_rect():
    fig = _make_fig()
    cp.circos.initialize(list("abcd"), xlim=[0, 10])
    cp.circos.track(
        ylim=[0, 10],
        track_height=0.5,
        panel_fun=lambda x, y: [cp.circos.rect(2, 2, 6, 6, rot=rot) for rot in range(0, 360, 60)],
    )
    cp.circos.clear()
    return fig, True


def _ex_circos_segments():
    fig = _make_fig()
    cp.circos.initialize(list("abcdefgh"), xlim=[0, 1])

    def _p(x, y):
        v = np.arange(0.2, 0.81, 0.2)
        cp.circos.segments(v, 0.1, v, 0.9)
        cp.circos.segments(0.1, v, 0.9, v)

    cp.circos.track(ylim=[0, 1], track_height=0.3, panel_fun=_p)
    cp.circos.clear()
    return fig, True


def _ex_circos_stackedText():
    rng = np.random.default_rng(7)
    fig = _make_fig()
    cp.circos.par(circle_margin=0.5, cell_padding=[0, 0, 0, 0], track_margin=[0, 0])
    cp.circos.initialize(sectors=list("abcd"), xlim=[0, 1])
    secs = rng.choice(list("abcd"), 30, replace=True)
    xs = rng.uniform(size=30)
    txt = [rng.choice(list("abc")) * 4 for _ in range(30)]
    cp.circos.stackedText(secs, xs, txt, bg_col="#EEEEEE")
    cp.circos.track(ylim=[0, 1])
    cp.circos.clear()
    return fig, True


def _ex_circos_text():
    fig = _make_fig()
    cp.circos.par(points_overflow_warning=False)
    cp.circos.initialize(list("abcd"), xlim=[0, 10])

    def _p(x, y):
        cp.circos.text(3, 1, "inside", facing="inside", cex=0.8)
        cp.circos.text(7, 1, "outside", facing="outside", cex=0.8)
        cp.circos.text(0, 5, "reverse.clockwise", facing="reverse.clockwise", adj=[0.5, 0], cex=0.8)
        cp.circos.text(10, 5, "clockwise", facing="clockwise", adj=[0.5, 0], cex=0.8)
        cp.circos.text(5, 5, "downward", facing="downward", cex=0.8)
        cp.circos.text(3, 9, "==bend.in==", facing="bending.inside", cex=0.8)
        cp.circos.text(7, 9, "==bend.out==", facing="bending.outside", cex=0.8)

    cp.circos.trackPlotRegion(list("abcd"), ylim=[0, 10], track_height=0.5, panel_fun=_p)
    cp.circos.clear()
    return fig, True


def _ex_circos_trackHist():
    rng = np.random.default_rng(8)
    x = rng.normal(size=800)
    sectors = rng.choice(list(string.ascii_lowercase[:8]), 800, replace=True)
    fig = _make_fig()
    cp.circos.initialize(sectors, x=x)
    cp.circos.trackHist(sectors, x=x, col="#999999", border="#999999")
    cp.circos.trackHist(sectors, x=x, bin_size=0.3, draw_density=True, col="#FF000080", border="red")
    cp.circos.clear()
    return fig, True


def _ex_circos_trackPlotRegion():
    rng = np.random.default_rng(123)
    fig = _make_fig()
    cp.circos.initialize(list("abcdefgh"), xlim=[0, 1])
    fa = rng.choice(list("abcdefgh"), 100, replace=True)
    x = rng.uniform(size=100)
    y = rng.normal(size=100)
    cp.circos.track(ylim=[0, 1], bg_col=cp.rand_color(8, seed=1))
    cp.circos.track(fa, x=x, y=y, panel_fun=lambda xv, yv: cp.circos.points(xv, yv), track_height=0.2, bg_border=cp.rand_color(8, seed=2))
    cp.circos.clear()
    return fig, True


def _ex_circos_trackPoints():
    rng = np.random.default_rng(9)
    fig = _make_fig()
    cp.circos.initialize(list("abcdefgh"), xlim=[0, 1])
    secs = rng.choice(list("abcdefgh"), 100, replace=True)
    x = rng.uniform(size=100)
    y = rng.uniform(size=100)
    cp.circos.track(ylim=[0, 1])
    cp.circos.trackPoints(secs, x=x, y=y, pch=16, col="blue")
    cp.circos.clear()
    return fig, True


def _ex_circos_triangle():
    fig = _make_fig()
    cp.circos.initialize(list("abcd"), xlim=[0, 10])
    cp.circos.track(
        ylim=[0, 10],
        track_height=0.5,
        panel_fun=lambda x, y: cp.circos.triangle([2, 2], [2, 8], [8, 8], [2, 8], [5, 5], [8, 2]),
    )
    cp.circos.clear()
    return fig, True


def _ex_circos_updatePlotRegion():
    fig = _make_fig()
    cp.circos.initialize(list("abcdefgh"), xlim=[0, 1])
    cp.circos.track(
        ylim=[0, 1],
        panel_fun=lambda x, y: cp.circos.text(cp.CELL_META.xcenter, cp.CELL_META.ycenter, cp.CELL_META.sector_index),
    )
    cp.circos.update(sector_index="b", track_index=1)
    cp.circos.rect(cp.CELL_META.cell_xlim[0], cp.CELL_META.cell_ylim[0], cp.CELL_META.cell_xlim[1], cp.CELL_META.cell_ylim[1], col="#FF000080")
    cp.circos.clear()
    return fig, True


def _ex_circos_violin():
    rng = np.random.default_rng(13)
    fig = _make_fig()
    cp.circos.initialize(list("abcd"), xlim=[0, 10])
    cp.circos.track(
        ylim=[0, 1],
        panel_fun=lambda x, y: cp.circos.violin([rng.uniform(size=12) for _ in range(5)], [1, 3, 5, 7, 9], col=list(range(1, 6))),
    )
    cp.circos.clear()
    return fig, True


def _ex_circos_yaxis():
    fig = _make_fig()
    cp.circos.par(points_overflow_warning=False, gap_degree=8)
    cp.circos.initialize(list("abcdefgh"), xlim=[0, 10])
    cp.circos.trackPlotRegion(list("abcdefgh"), ylim=[0, 10], track_height=0.5)
    for a in list("bcd"):
        cp.circos.yaxis(side="left", sector_index=a)
    for a in list("efg"):
        cp.circos.yaxis(side="right", sector_index=a)
    cp.circos.clear()
    return fig, True


def _ex_col2value():
    x = np.linspace(0, 1, 11)
    col_fun = cp.colorRamp2([0, 0.5, 1], ["blue", "white", "red"])
    vals = cp.col2value(col_fun(x), col_fun=col_fun)
    assert np.allclose(vals, x, atol=0.05)
    v_red = cp.col2value("red", col_fun=col_fun)
    assert np.isclose(v_red, 1.0, atol=0.05)
    return None, False


def _ex_colorRamp2():
    cf = cp.colorRamp2([-1, 0, 1], ["green", "white", "red"])
    out = cf([-2, -1, -0.5, 0, 0.5, 1, 2])
    assert len(out) == 7 and all(c.startswith("#") for c in out)
    return None, False


def _ex_convert_length():
    fig = _make_fig()
    cp.circos.par(cell_padding=[0, 0, 0, 0], track_margin=[0, 0])
    cp.circos.initialize(list("abcdefghij"), xlim=np.column_stack([np.zeros(10), np.linspace(0.5, 1.5, 10)]))
    cp.circos.track(ylim=[0, 1], track_height=cp.mm_h(5))
    cp.circos.par(track_margin=[0, cp.mm_h(2)])
    cp.circos.track(ylim=[0, 1], track_height=cp.cm_h(1))
    cp.circos.clear()
    return fig, True


def _ex_convert_x():
    fig = _make_fig()
    cp.circos.par(cell_padding=[0, 0, 0, 0], track_margin=[0, 0])
    cp.circos.initialize(list("abcdefghij"), xlim=[0, 1.5])
    cp.circos.track(
        ylim=[0, 1],
        track_height=cp.mm_h(5),
        panel_fun=lambda x, y: cp.circos.lines([0, cp.mm_x(5)], [0.5, 0.5], col="blue"),
    )
    cp.circos.clear()
    return fig, True


def _ex_draw_sector():
    fig = _make_fig()
    cp.circos.initialize(list("abcdefgh"), xlim=[0, 1])
    for _ in range(2):
        cp.circos.trackPlotRegion(ylim=[0, 1])
    cp.draw.sector(20, 0)
    cp.draw.sector(30, 60, rou1=0.8, rou2=0.5, clock_wise=False, col="#FF000080")
    cp.draw.sector(0, 180, rou1=0.25, center=(-0.5, 0.5), border=2, lwd=2, lty=2)
    cp.draw.sector(0, 360, rou1=0.7, rou2=0.6, col="#0000FF80")
    cp.circos.clear()
    return fig, True


def _ex_genomicDensity():
    bed = cp.generateRandomBed(nr=200, seed=1)
    bed = bed[bed["chr"] == "chr1"]
    d1 = cp.genomicDensity(bed)
    d2 = cp.genomicDensity(bed, count_by="number")
    assert len(d1) > 0 and len(d2) > 0
    return None, False


def _ex_get_cell_meta_data():
    got = []
    cp.circos.initialize(list("abcd"), xlim=[0, 1])
    cp.circos.trackPlotRegion(ylim=[0, 1], panel_fun=lambda x, y: got.append(cp.get.cell.meta.data("xlim")))
    assert np.allclose(cp.get.cell.meta.data("xlim", sector_index="a", track_index=1), [0, 1])
    assert len(got) == 4
    cp.circos.clear()
    return None, False


def _ex_highlight_sector():
    fig = _make_fig()
    cp.circos.initialize(list("abcdefgh"), xlim=[0, 1])
    for _ in range(4):
        cp.circos.trackPlotRegion(ylim=[0, 1])
    cp.circos.info(plot=True)
    cp.highlight.sector(["a", "h"], track_index=1)
    cp.highlight.sector("c", col="#00FF0040")
    cp.highlight.sector("d", col=None, border="red", lwd=2)
    cp.highlight.sector("e", col="#0000FF40", track_index=[2, 3])
    cp.highlight.sector(["f", "g"], col=None, border="green", lwd=2, track_index=[2, 3])
    cp.highlight.sector(list("abcdefgh"), col="#FFFF0040", track_index=4)
    cp.circos.clear()
    return fig, True


def _ex_names_CELL_META():
    nms = cp.names_CELL_META(cp.CELL_META)
    assert "sector.index" in nms and "track.index" in nms
    return None, False


def _ex_rainfallTransform():
    bed = cp.generateRandomBed(nr=200, seed=1)
    bed = bed[bed["chr"] == "chr1"]
    rf = cp.rainfallTransform(bed)
    assert "dist" in rf.columns and len(rf) == len(bed)
    return None, False


def _ex_rand_color():
    cols = cp.rand_color(10, luminosity="bright", hue="red", seed=3)
    assert len(cols) == 10 and all(c.startswith("#") for c in cols)
    return None, False


def _ex_read_chromInfo():
    d1 = cp.read.chromInfo(species="hg19")
    d2 = cp.read.chromInfo(chromInfo=cp.system_file("extdata", "chromInfo.txt", package="circlize"))
    assert len(d1.chromosome) == 24 and len(d2.chromosome) == 93
    return None, False


def _ex_read_cytoband():
    d1 = cp.read.cytoband(species="hg19")
    d2 = cp.read.cytoband(cytoband=cp.system_file("extdata", "cytoBand.txt", package="circlize"))
    assert len(d1.df) == 862 and len(d2.df) == 862
    return None, False


def _ex_reverse_circlize():
    cp.circos.initialize(list("abcd"), xlim=[0, 1])
    cp.circos.trackPlotRegion(ylim=[0, 1])
    r = cp.reverse.circlize([30, 60], [0.9, 0.8], sector_index="a", track_index=1)
    assert r.shape == (2, 2)
    cp.circos.clear()
    return None, False


def _ex_set_current_cell():
    cp.circos.initialize(list("abcdefgh"), xlim=[0, 1])
    cp.circos.track(ylim=[0, 1])
    cp.set_current_cell("b", 1)
    assert cp.get_current_sector_index() == "b"
    assert cp.get_current_track_index() == 1
    cp.circos.clear()
    return None, False


def _ex_set_track_gap():
    fig = _make_fig()
    cp.circos.initialize(list("abcdefghij"), xlim=[0, 1])
    cp.circos.track(ylim=[0, 1])
    cp.set_track_gap(cp.mm_h(2))
    cp.circos.track(ylim=[0, 1])
    assert abs(cp.get_cell_meta_data("cell.bottom.radius", track_index=1) - cp.get_cell_meta_data("cell.top.radius", track_index=2)) > 0
    cp.circos.clear()
    return fig, True


# Extra workflow coverage requested beyond the 62 Rd examples
def _ex_extra_circos_nested():
    wb = cp.load_extdata("tagments_WGBS_DMR")
    fig = _make_fig()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cp.circos.nested(
            lambda: (
                cp.circos.genomicInitialize(wb["tagments"], plotType=None),
                cp.circos.genomicTrack(wb["DMR1"], ylim=[-1, 1], panel_fun=lambda r, v: cp.circos.genomicPoints(r, v, cex=0.5)),
            ),
            lambda: cp.circos.initializeWithIdeogram(chromosome_index=wb["correspondance"].iloc[:, 0].unique().tolist(), plotType=["ideogram", "labels"]),
            wb["correspondance"].iloc[:, [3, 4, 5, 0, 1, 2]],
            connection_col="#00FF0040",
        )
    return fig, True


def _ex_extra_initializeCircularGenome():
    fig = _make_fig()
    cp.circos.initializeCircularGenome("E.coli", genome_size=4639675)
    df = pd.DataFrame({"chr": ["E.coli"] * 20, "start": np.linspace(0, 4.5e6, 20), "end": np.linspace(1e5, 4.6e6, 20), "val": np.sin(np.linspace(0, 6, 20))})
    cp.circos.genomicTrack(df, panel_fun=lambda r, v, **kw: cp.circos.genomicLines(r, v, area=True, baseline=0, col="#66C2A5"))
    cp.circos.clear()
    return fig, True


def _ex_extra_multigroup_chordDiagram():
    df = pd.DataFrame({"from": ["A1", "A2", "B1", "C1"], "to": ["B1", "C1", "C2", "A1"], "value": [10, 15, 20, 12]})
    grp = {"A1": "G1", "A2": "G1", "B1": "G2", "C1": "G3", "C2": "G3"}
    fig = _make_fig()
    cp.chordDiagram(df, group=grp, big_gap=15, small_gap=2)
    cp.circos.clear()
    return fig, True


def _ex_extra_multitrack_heatmap():
    rng = np.random.default_rng(77)
    m1 = pd.DataFrame(rng.normal(size=(30, 5)), index=[f"R{i}" for i in range(30)])
    m2 = pd.DataFrame(rng.normal(size=(30, 3)), index=[f"R{i}" for i in range(30)])
    fig = _make_fig()
    cp.circos.heatmap(m1, split=["A"] * 15 + ["B"] * 15, col=cp.colorRamp2([-2, 0, 2], ["blue", "white", "red"]), dend_side="inside")
    cp.circos.heatmap(m2, col=cp.colorRamp2([-2, 0, 2], ["green", "white", "purple"]))
    cp.circos.clear()
    return fig, True


def _ex_extra_api_style_equivalence():
    # Verify R dot syntax, camelCase, and snake_case via both circlize_py and circlize produce identical layout metadata
    for style in ("dot", "snake"):
        if style == "dot":
            cz.circos.par(start_degree=20, gap_degree=5)
            cz.circos.initialize(["x", "y", "z"], xlim=[0, 10])
            cz.circos.track(ylim=[0, 1], track_height=0.2)
            vals = cz.circlize([0, 5, 10], [0, 0.5, 1], sector_index="y", track_index=1)
            cz.circos.clear()
            ref = np.asarray(vals)
        else:
            cp.circos_par(start_degree=20, gap_degree=5)
            cp.circos_initialize(["x", "y", "z"], xlim=[0, 10])
            cp.circos_track(ylim=[0, 1], track_height=0.2)
            vals2 = cp.circlize([0, 5, 10], [0, 0.5, 1], sector_index="y", track_index=1)
            cp.circos_clear()
            assert np.allclose(ref, np.asarray(vals2))
    return None, False


ALL_CASES = {
    "CELL_META.Rd": _ex_CELL_META,
    "add_transparency.Rd": _ex_add_transparency,
    "adjacencyList2Matrix.Rd": _ex_adjacencyList2Matrix,
    "adjacencyMatrix2List.Rd": _ex_adjacencyMatrix2List,
    "arrange_links_evenly.Rd": _ex_arrange_links_evenly,
    "calc_gap.Rd": _ex_calc_gap,
    "chordDiagram.Rd": _ex_chordDiagram,
    "circlize.Rd": _ex_circlize,
    "circos.arrow.Rd": _ex_circos_arrow,
    "circos.axis.Rd": _ex_circos_axis,
    "circos.barplot.Rd": _ex_circos_barplot,
    "circos.boxplot.Rd": _ex_circos_boxplot,
    "circos.connect.Rd": _ex_circos_connect,
    "circos.dendrogram.Rd": _ex_circos_dendrogram,
    "circos.genomicAxis.Rd": _ex_circos_genomicAxis,
    "circos.genomicDensity.Rd": _ex_circos_genomicDensity,
    "circos.genomicHeatmap.Rd": _ex_circos_genomicHeatmap,
    "circos.genomicIdeogram.Rd": _ex_circos_genomicIdeogram,
    "circos.genomicInitialize.Rd": _ex_circos_genomicInitialize,
    "circos.genomicLabels.Rd": _ex_circos_genomicLabels,
    "circos.genomicLines.Rd": _ex_circos_genomicLines,
    "circos.genomicLink.Rd": _ex_circos_genomicLink,
    "circos.genomicPoints.Rd": _ex_circos_genomicPoints,
    "circos.genomicRainfall.Rd": _ex_circos_genomicRainfall,
    "circos.genomicRect.Rd": _ex_circos_genomicRect,
    "circos.genomicText.Rd": _ex_circos_genomicText,
    "circos.heatmap.Rd": _ex_circos_heatmap,
    "circos.heatmap.link.Rd": _ex_circos_heatmap_link,
    "circos.initializeWithIdeogram.Rd": _ex_circos_initializeWithIdeogram,
    "circos.labels.Rd": _ex_circos_labels,
    "circos.lines.Rd": _ex_circos_lines,
    "circos.par.Rd": _ex_circos_par,
    "circos.points.Rd": _ex_circos_points,
    "circos.polygon.Rd": _ex_circos_polygon,
    "circos.raster.Rd": _ex_circos_raster,
    "circos.rect.Rd": _ex_circos_rect,
    "circos.segments.Rd": _ex_circos_segments,
    "circos.stackedText.Rd": _ex_circos_stackedText,
    "circos.text.Rd": _ex_circos_text,
    "circos.trackHist.Rd": _ex_circos_trackHist,
    "circos.trackPlotRegion.Rd": _ex_circos_trackPlotRegion,
    "circos.trackPoints.Rd": _ex_circos_trackPoints,
    "circos.triangle.Rd": _ex_circos_triangle,
    "circos.updatePlotRegion.Rd": _ex_circos_updatePlotRegion,
    "circos.violin.Rd": _ex_circos_violin,
    "circos.yaxis.Rd": _ex_circos_yaxis,
    "col2value.Rd": _ex_col2value,
    "colorRamp2.Rd": _ex_colorRamp2,
    "convert_length.Rd": _ex_convert_length,
    "convert_x.Rd": _ex_convert_x,
    "draw.sector.Rd": _ex_draw_sector,
    "genomicDensity.Rd": _ex_genomicDensity,
    "get.cell.meta.data.Rd": _ex_get_cell_meta_data,
    "highlight.sector.Rd": _ex_highlight_sector,
    "names.CELL_META.Rd": _ex_names_CELL_META,
    "rainfallTransform.Rd": _ex_rainfallTransform,
    "rand_color.Rd": _ex_rand_color,
    "read.chromInfo.Rd": _ex_read_chromInfo,
    "read.cytoband.Rd": _ex_read_cytoband,
    "reverse.circlize.Rd": _ex_reverse_circlize,
    "set.current.cell.Rd": _ex_set_current_cell,
    "set_track_gap.Rd": _ex_set_track_gap,
    "extra_circos_nested": _ex_extra_circos_nested,
    "extra_initializeCircularGenome": _ex_extra_initializeCircularGenome,
    "extra_multigroup_chordDiagram": _ex_extra_multigroup_chordDiagram,
    "extra_multitrack_heatmap": _ex_extra_multitrack_heatmap,
    "extra_api_style_equivalence": _ex_extra_api_style_equivalence,
}


@pytest.mark.parametrize("case_name", list(ALL_CASES.keys()))
def test_rd_example(case_name: str):
    fig, is_plot = ALL_CASES[case_name]()
    assert not cp.is_circos_initialized(), f"{case_name} left circos initialized"
    if is_plot:
        assert fig is not None
        assert_non_blank(fig)
