"""Examples 02: High-level statistical tracks (histograms, barplots, boxplots, violins, dendrograms)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from circlize_py import (
    CELL_META,
    as_dendrogram,
    circos,
    color_branches,
    load_extdata,
    rand_color,
)


def run_examples(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    rng = np.random.default_rng(42)

    # 1. Histograms, barplots, boxplots, and violin plots in a single layout
    sectors = list("abcd")
    fa = np.repeat(sectors, 150)
    x_hist = rng.normal(loc=0.0, scale=1.0, size=600)

    fig = plt.figure(figsize=(7.5, 7.5), dpi=120)
    circos.clear()
    circos.par(gap_after=[4, 4, 4, 12], track_height=0.16)
    circos.initialize(sectors=sectors, xlim=(0.0, 10.0))

    # Track 1: Barplots
    def panel_bar(xv, yv):
        vals = rng.uniform(0.5, 4.5, size=8)
        circos.barplot(vals, pos=np.arange(1, 9) + 0.5, col=rand_color(8, hue="blue"))
        if CELL_META.sector_numeric_index == 1:
            circos.yaxis(side="left", labels_cex=0.55)
        circos.text(5.0, 5.2, f"Sector {CELL_META.sector_index}", cex=0.8)

    circos.track(ylim=(0.0, 5.8), panel_fun=panel_bar)

    # Track 2: Boxplots
    def panel_box(xv, yv):
        mat = rng.normal(loc=0.0, scale=1.0, size=(30, 5)) + np.linspace(-1.0, 1.0, 5)
        circos.boxplot(mat, pos=[1.5, 3.2, 5.0, 6.8, 8.5], col=["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854"])
        if CELL_META.sector_numeric_index == 1:
            circos.yaxis(side="left", labels_cex=0.55)

    circos.track(ylim=(-3.8, 3.8), panel_fun=panel_box)

    # Track 3: Violins
    def panel_violin(xv, yv):
        mat = rng.normal(loc=0.0, scale=0.8, size=(40, 4)) + np.linspace(-0.8, 0.8, 4)
        circos.violin(mat, pos=[2.0, 4.0, 6.0, 8.0], col="#E5C494")
        if CELL_META.sector_numeric_index == 1:
            circos.yaxis(side="left", labels_cex=0.55)

    circos.track(ylim=(-3.5, 3.5), panel_fun=panel_violin)
    circos.clear()

    out1 = output_dir / "02a_bar_box_violin_tracks.png"
    fig.savefig(out1, bbox_inches="tight")
    plt.close(fig)
    saved.append(out1)

    # 2. Track histogram + bird.orders phylogenetic dendrogram
    hc = load_extdata("bird.orders")
    dend = color_branches(as_dendrogram(hc), k=5, col=["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00"])
    labels = dend.labels()
    n_leaves = len(labels)
    dend_height = float(dend.height)

    fig2 = plt.figure(figsize=(8.0, 8.0), dpi=120)
    circos.clear()
    circos.par(cell_padding=(0.0, 0.0, 0.0, 0.0))
    circos.initialize(["birds"], xlim=(0.0, float(n_leaves)))

    def panel_labels(xv, yv):
        for i, lbl in enumerate(labels):
            circos.text(i + 0.5, 0.0, lbl, facing="clockwise", niceFacing=True, adj=(0.0, 0.5), cex=0.6)

    circos.track(ylim=(0.0, 1.0), bg_border=None, track_height=0.28, panel_fun=panel_labels)

    def panel_dend(xv, yv):
        circos.dendrogram(dend, max_height=dend_height, facing="inside")

    circos.track(ylim=(0.0, dend_height), bg_border=None, track_height=0.45, panel_fun=panel_dend)
    circos.clear()

    out2 = output_dir / "02b_bird_orders_dendrogram.png"
    fig2.savefig(out2, bbox_inches="tight")
    plt.close(fig2)
    saved.append(out2)

    # 3. circos.trackHist standalone example
    fig3 = plt.figure(figsize=(6.8, 6.8), dpi=120)
    circos.clear()
    circos.initialize(sectors=fa, x=x_hist)
    circos.trackHist(sectors=fa, x=x_hist, col="#999999", bg_col="#F8F8F8")
    circos.trackHist(sectors=fa, x=x_hist, bin_size=0.3, col="#E41A1C", force_ylim=False, draw_density=True)
    circos.clear()

    out3 = output_dir / "02c_track_hist.png"
    fig3.savefig(out3, bbox_inches="tight")
    plt.close(fig3)
    saved.append(out3)

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/output"))
    args = parser.parse_args()
    paths = run_examples(args.output_dir)
    for p in paths:
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
