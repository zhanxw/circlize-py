"""Examples 01: Basic circular layout, low-level primitives, raster images, arrows, and labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from circlize_py import (
    CELL_META,
    circos,
    convert_length,
    draw,
    highlight,
    load_extdata,
    rand_color,
)


def run_examples(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    rng = np.random.default_rng(123)

    # 1. Classic README multi-track circular plot using R-style circos.* dot syntax
    sectors = list("abcdefgh")
    x = rng.normal(size=800)
    y = rng.normal(size=800)
    fa = np.repeat(sectors, 100)

    fig = plt.figure(figsize=(7.5, 7.5), dpi=120)
    circos.clear()
    circos.par(track_height=0.14, cell_padding=(0.02, 0.0, 0.02, 0.0))
    circos.initialize(sectors=fa, x=x)

    def panel_t1(xv, yv):
        si = CELL_META.sector_index
        xlim = CELL_META.xlim
        ylim = CELL_META.ylim
        circos.text(np.mean(xlim), np.mean(ylim), si, cex=0.9, facing="inside", niceFacing=True)
        circos.axis(h="top", labels_cex=0.55)

    circos.trackPlotRegion(sectors=fa, y=y, bg_col=rand_color(8, luminosity="light"), panel_fun=panel_t1)

    def panel_t2(xv, yv):
        circos.points(xv, yv, pch=16, cex=0.5, col=rand_color(1, luminosity="dark"))
        si = CELL_META.sector_index
        if si in ("a", "c"):
            circos.lines(np.sort(xv), np.sin(np.sort(xv)), col="#D95F02", lwd=1.5)

    circos.trackPlotRegion(sectors=fa, x=x, y=y, panel_fun=panel_t2)

    def panel_t3(xv, yv):
        xlim = CELL_META.xlim
        m = np.mean(xlim)
        circos.rect(xlim[0], 0.1, m, 0.5, col="#1B9E7780", border="#1B9E77")
        circos.triangle(m, 0.1, m + (xlim[1] - m) * 0.4, 0.9, xlim[1], 0.1, col="#7570B380", border="#7570B3")
        circos.segments(xlim[0], 0.8, xlim[1], 0.8, col="navy", lty=2)

    circos.track(ylim=(0.0, 1.0), panel_fun=panel_t3)

    # Highlight two sectors across tracks 2 and 3, and draw a center badge
    highlight.sector(["a", "b"], track_index=[2, 3], col="#FF000025", border="red", text="Group 1", facing="bending.inside")
    draw.sector(0, 360, rou1=0.25, col="#F5F5F5", border="#CCCCCC")
    circos.clear()

    out1 = output_dir / "01a_basic_tracks_and_primitives.png"
    fig.savefig(out1, bbox_inches="tight")
    plt.close(fig)
    saved.append(out1)

    # 2. Arrows, raster images, curved stacked labels, and updatePlotRegion
    doodle = load_extdata("doodle")
    rlogo = load_extdata("Rlogo.png")
    fig2 = plt.figure(figsize=(7.5, 7.5), dpi=120)
    circos.clear()
    circos.par(cell_padding=(0.0, 0.0, 0.0, 0.0))
    circos.initialize(["S1", "S2", "S3", "S4"], xlim=(0, 10))

    def panel_arrow(xv, yv):
        if CELL_META.sector_numeric_index % 2 == 1:
            circos.arrow(1, 9, y=0.5, width=0.5, col="#66C2A5", border="black")
        else:
            circos.arrow(1, 9, y=0.5, width=0.5, col="#FC8D62", arrow_position="start", tail="point")
        circos.text(5, 0.5, CELL_META.sector_index, facing="bending.inside", niceFacing=True)

    circos.track(ylim=(0, 1), track_height=0.12, bg_border=None, panel_fun=panel_arrow)

    def panel_raster(xv, yv):
        img = rlogo if CELL_META.sector_index == "S1" else doodle[CELL_META.sector_numeric_index]
        circos.raster(img, 5, 0.5, width="1cm", facing="inside", niceFacing=True)

    circos.track(ylim=(0, 1), track_height=0.2, panel_fun=panel_raster)

    circos.updatePlotRegion(sector_index="S2", track_index=2, bg_col="#FFF7BC")
    circos.raster(rlogo, 5, 0.5, width="1.2cm", facing="inside", niceFacing=True)
    circos.text(5, 0.88, "Updated Cell", facing="inside", col="#D95F02", cex=0.75)

    circos.stackedText(
        ["S1", "S1", "S3", "S4"],
        [2.0, 2.3, 5.0, 6.0],
        ["Tag1", "Tag2", "Peak", "Site"],
        bg_col="#EEEEEE",
        cex=0.65,
    )

    # Connect track labels with anti-overlapping layout
    circos.labels(
        sectors=["S1", "S1", "S3", "S4"],
        x=[2.0, 7.5, 5.0, 6.0],
        labels=["GeneA", "GeneB", "Peak1", "LocusX"],
        side="inside",
        connection_height=convert_length(5, "mm"),
    )
    circos.clear()

    out2 = output_dir / "01b_arrows_raster_and_labels.png"
    fig2.savefig(out2, bbox_inches="tight")
    plt.close(fig2)
    saved.append(out2)
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
