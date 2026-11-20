"""Examples 04: Circular split heatmaps with dendrograms/links and nested zoom plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from circlize_py import (
    CELL_META,
    circos,
    colorRamp2,
    load_extdata,
    rand_color,
)


def run_examples(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    rng = np.random.default_rng(101)

    # 1. Multi-track circular heatmap with split, dendrogram, row labels, and link
    mat1 = rng.normal(loc=0.0, scale=1.0, size=(60, 10))
    mat1[:20, :5] += 1.5
    mat1[20:40, 5:] += 1.5
    mat2 = rng.uniform(-1.0, 1.0, size=(60, 4))
    rownames = [f"G{i:02d}" for i in range(60)]
    split = ["Cluster1"] * 20 + ["Cluster2"] * 20 + ["Cluster3"] * 20
    df1 = pd.DataFrame(mat1, index=rownames, columns=[f"S{i}" for i in range(10)])
    df2 = pd.DataFrame(mat2, index=rownames, columns=[f"T{i}" for i in range(4)])

    col_fun1 = colorRamp2([-2.0, 0.0, 2.0], ["#2166AC", "#F7F7F7", "#B2182B"])
    col_fun2 = colorRamp2([-1.0, 0.0, 1.0], ["#1B7837", "#F7F7F7", "#762A83"])

    fig = plt.figure(figsize=(8.0, 8.0), dpi=120)
    circos.clear()
    circos.par(gap_after=[4, 4, 14])
    circos.heatmap(
        df1,
        split=split,
        col=col_fun1,
        dend_side="inside",
        rownames_side="outside",
        rownames_cex=0.55,
        track_height=0.28,
    )
    circos.heatmap(df2, col=col_fun2, track_height=0.14)
    # Connect two rows by 0-based row indices or row names
    circos.heatmap.link(0, 35, col="#D95F02", lwd=1.8, directional=1)
    circos.heatmap.link("G05", "G48", col="#7570B3", lwd=1.8)
    circos.clear()

    out1 = output_dir / "04a_circular_heatmap_split.png"
    fig.savefig(out1, bbox_inches="tight")
    plt.close(fig)
    saved.append(out1)

    # 2. Nested zoom plot using bundled tagments_WGBS_DMR dataset
    wb = load_extdata("tagments_WGBS_DMR")
    dmr1 = wb["DMR1"]
    tagments = wb["tagments"]
    corr = wb["correspondance"]
    col_meth = colorRamp2([-0.8, 0.0, 0.8], ["blue", "white", "red"])

    def f1():
        circos.par(gap_degree=1.5, track_margin=(0.0, 0.005))
        circos.genomicInitialize(tagments, plot_type=None)
        circos.genomicTrack(
            dmr1,
            ylim=(-1.0, 1.0),
            track_height=0.15,
            bg_col="#EFEFEF",
            panel_fun=lambda region, value: circos.genomicPoints(
                region,
                value,
                pch=16,
                cex=0.6,
                col=[col_meth(v) for v in value.iloc[:, 0]],
            ),
        )

    def f2():
        circos.initializeWithIdeogram(
            chromosome_index=corr.iloc[:, 0].unique().tolist(),
            plot_type=("ideogram", "labels"),
            track_height=0.06,
        )
        circos.genomicTrack(
            corr.iloc[:, 0:3],
            ylim=(0.0, 1.0),
            track_height=0.1,
            panel_fun=lambda region, value: circos.genomicRect(region, ybottom=0.1, ytop=0.9, col="#377EB880"),
        )

    fig2 = plt.figure(figsize=(8.0, 8.0), dpi=120)
    circos.clear()
    circos.nested(
        f1,
        f2,
        corr.iloc[:, [3, 4, 5, 0, 1, 2]],
        connection_col=rand_color(len(corr), transparency=0.75),
        connection_border="#555555",
    )

    out2 = output_dir / "04b_nested_zoom_wgbs.png"
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
