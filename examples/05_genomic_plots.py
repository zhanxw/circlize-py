"""Examples 05: Genomic plots — ideograms, genomic tracks/links, DMR density/rainfall, heatmaps, labels, circular genomes, and transcript isoforms."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from circlize_py import (
    circos,
    colorRamp2,
    generateRandomBed,
    getI,
    highlight,
    load_extdata,
)


def run_examples(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # 1. Human ideogram + stacked multi-layer genomic points, area lines, stack rect, and links
    bed1 = generateRandomBed(nr=250, nc=2, seed=1)
    bed2 = generateRandomBed(nr=250, nc=2, seed=2)
    link1 = generateRandomBed(nr=40, nc=0, seed=3)
    link2 = generateRandomBed(nr=40, nc=0, seed=4)
    n_links = min(len(link1), len(link2))
    link1 = link1.iloc[:n_links].reset_index(drop=True)
    link2 = link2.iloc[:n_links].reset_index(drop=True)

    fig1 = plt.figure(figsize=(8.0, 8.0), dpi=120)
    circos.clear()
    circos.initializeWithIdeogram()

    def panel_pts(region, value, **kwargs):
        i = getI(**kwargs)
        col = "#E41A1C" if i == 1 else "#377EB8"
        circos.genomicPoints(region, value, numeric_column=0, pch=16, cex=0.5, col=col, **kwargs)

    circos.genomicTrack([bed1, bed2], panel_fun=panel_pts)

    circos.genomicTrack(
        bed1,
        numeric_column="value1",
        panel_fun=lambda region, value, **kw: circos.genomicLines(
            region, value, numeric_column=0, area=True, col="#4DAF4A80", border="#4DAF4A", **kw
        ),
    )

    circos.genomicTrack(
        [bed1.iloc[::4, :3], bed2.iloc[::4, :3]],
        stack=True,
        panel_fun=lambda region, value, **kw: circos.genomicRect(
            region,
            value,
            ytop=getI(**kw) + 0.35,
            ybottom=getI(**kw) - 0.35,
            col="#E41A1C" if getI(**kw) == 1 else "#377EB8",
            border=None,
            **kw,
        ),
    )

    circos.genomicLink(link1, link2, col="#7570B355", border=None)
    highlight.chromosome("chr1", col="#FF000018", border="red")
    circos.clear()

    out1 = output_dir / "05a_ideogram_tracks_and_links.png"
    fig1.savefig(out1, bbox_inches="tight")
    plt.close(fig1)
    saved.append(out1)

    # 2. DMR rainfall, genomicDensity, genomicHeatmap, and genomicLabels
    dmr = load_extdata("DMR")
    dmr_hyper = dmr["DMR_hyper"]
    dmr_hypo = dmr["DMR_hypo"]

    hm_bed = generateRandomBed(nr=80, nc=5, seed=11)
    lbl_bed = generateRandomBed(nr=35, nc=0, seed=12)
    lbl_bed["gene"] = [f"Locus_{i+1}" for i in range(len(lbl_bed))]

    fig2 = plt.figure(figsize=(8.0, 8.0), dpi=120)
    circos.clear()
    circos.initializeWithIdeogram(plot_type=("ideogram", "labels"))
    circos.genomicLabels(lbl_bed, labels_column="gene", side="outside", cex=0.55)
    circos.genomicRainfall([dmr_hyper, dmr_hypo], pch=16, cex=0.45, col=["#E41A1C80", "#377EB880"])
    circos.genomicDensity(dmr_hyper, col="#E41A1C80", track_height=0.1)
    circos.genomicDensity(dmr_hypo, col="#377EB880", track_height=0.1)
    circos.genomicHeatmap(
        hm_bed,
        col=colorRamp2([-1.5, 0.0, 1.5], ["blue", "white", "red"]),
        side="inside",
        heatmap_height=0.12,
    )
    circos.clear()

    out2 = output_dir / "05b_dmr_density_rainfall_heatmap_labels.png"
    fig2.savefig(out2, bbox_inches="tight")
    plt.close(fig2)
    saved.append(out2)

    # 3. Circular bacterial genome + TP53/TP63/TP73 transcript isoforms
    rng = np.random.default_rng(55)
    fig3 = plt.figure(figsize=(7.5, 7.5), dpi=120)
    circos.clear()
    circos.initializeCircularGenome("E.coli", genome_size=4639675)
    gc_bed = pd.DataFrame(
        {
            "chr": ["E.coli"] * 120,
            "start": np.linspace(0, 4600000, 120),
            "end": np.linspace(30000, 4630000, 120),
            "val": rng.uniform(0.35, 0.65, size=120),
        }
    )
    circos.genomicTrack(
        gc_bed,
        panel_fun=lambda r, v, **kw: circos.genomicLines(r, v, area=True, baseline=0.5, col="#66C2A580", border="#1B9E77"),
    )
    r1 = pd.DataFrame({"chr": ["E.coli"] * 8, "start": [1e5, 6e5, 1.1e6, 1.6e6, 2.2e6, 2.8e6, 3.3e6, 3.9e6], "end": [2e5, 7e5, 1.2e6, 1.7e6, 2.3e6, 2.9e6, 3.4e6, 4.0e6]})
    r2 = pd.DataFrame({"chr": ["E.coli"] * 8, "start": [2.4e6, 3.1e6, 4.1e6, 0.4e6, 0.9e6, 1.5e6, 2.0e6, 4.3e6], "end": [2.5e6, 3.2e6, 4.2e6, 0.5e6, 1.0e6, 1.6e6, 2.1e6, 4.4e6]})
    circos.genomicLink(r1, r2, col="#FC8D6280")
    circos.clear()

    out3 = output_dir / "05c_circular_genome_and_links.png"
    fig3.savefig(out3, bbox_inches="tight")
    plt.close(fig3)
    saved.append(out3)

    # 4. Transcript exon visualization from bundled tp_family
    tp_df = load_extdata("tp_family_df")
    fig4 = plt.figure(figsize=(7.2, 7.2), dpi=120)
    circos.clear()
    circos.par(gap_degree=8, track_margin=(0.0, 0.02))
    circos.genomicInitialize(tp_df[["gene", "start", "end"]])
    # Count transcripts per gene for stacked exon plotting
    tx_counts = tp_df.groupby("gene")["transcript"].nunique().to_dict()
    max_tx = max(tx_counts.values())

    def panel_exons(region, value, **kw):
        gene = region  # region is 2-col start/end; value holds transcript/exon
        # Draw gene spans per transcript row
        tx_names = list(dict.fromkeys(value["transcript"].tolist()))
        for y_idx, tx in enumerate(tx_names, start=1):
            sub = region[value["transcript"].to_numpy() == tx]
            circos.segments(float(sub["start"].min()), y_idx, float(sub["end"].max()), y_idx, col="#888888", lwd=0.8)
            circos.genomicRect(sub, ybottom=y_idx - 0.3, ytop=y_idx + 0.3, col="#2B8CBE", border="black", lwd=0.5)

    circos.genomicTrack(
        tp_df[["gene", "start", "end", "transcript", "exon"]],
        ylim=(0.5, max_tx + 0.5),
        bg_col="#F9F9F9",
        panel_fun=panel_exons,
    )
    circos.clear()

    out4 = output_dir / "05d_tp_family_transcripts.png"
    fig4.savefig(out4, bbox_inches="tight")
    plt.close(fig4)
    saved.append(out4)

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
