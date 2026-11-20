"""Examples 03: Links, evenly-spaced ribbons, and Chord diagrams (matrix, DataFrame, multi-group)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from circlize_py import (
    CELL_META,
    arrange_links_evenly,
    calc_gap,
    chordDiagram,
    circos,
    highlight,
    rand_color,
)


def run_examples(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    rng = np.random.default_rng(7)

    # 1. Links with ribbons, arrowheads, and arrange_links_evenly
    sectors = list("abcdefgh")
    fig = plt.figure(figsize=(7.0, 7.0), dpi=120)
    circos.clear()
    circos.initialize(sectors, xlim=(0, 10))
    circos.track(
        ylim=(0, 1),
        track_height=0.08,
        bg_col=rand_color(8, luminosity="light"),
        panel_fun=lambda x, y: circos.text(5, 0.5, CELL_META.sector_index, cex=0.9),
    )

    circos.link("a", 5, "c", 5, col="#E41A1C", lwd=1.8, directional=1)
    circos.link("b", [2, 5], "f", [4, 7], col="#377EB880", border="#377EB8", directional=1)
    links = pd.DataFrame(
        {
            "from": ["d", "d", "e"],
            "to": ["g", "h", "a"],
        }
    )
    arranged = arrange_links_evenly(links, directional=0)
    for _, row in arranged.iterrows():
        circos.link(row["sector1"], row["pos1"], row["sector2"], row["pos2"], col="#4DAF4A80", lwd=1.5)
    circos.clear()

    out1 = output_dir / "03a_links_and_ribbons.png"
    fig.savefig(out1, bbox_inches="tight")
    plt.close(fig)
    saved.append(out1)

    # 2. Directional Chord diagram from named matrix with custom preAllocateTracks
    mat = pd.DataFrame(
        rng.integers(5, 30, size=(4, 5)),
        index=[f"S{i}" for i in range(1, 5)],
        columns=[f"E{i}" for i in range(1, 6)],
    )
    grid_col = {
        "S1": "#E41A1C",
        "S2": "#377EB8",
        "S3": "#4DAF4A",
        "S4": "#984EA3",
        "E1": "grey70",
        "E2": "grey70",
        "E3": "grey70",
        "E4": "grey70",
        "E5": "grey70",
    }

    fig2 = plt.figure(figsize=(7.5, 7.5), dpi=120)
    circos.clear()
    chordDiagram(
        mat,
        grid_col=grid_col,
        directional=1,
        direction_type=["diffHeight", "arrows"],
        link_arr_type="big.arrow",
        diffHeight=-0.04,
        annotationTrack=["grid"],
        preAllocateTracks={"track.height": 0.14},
    )
    circos.track(
        track_index=1,
        bg_border=None,
        panel_fun=lambda x, y: circos.text(
            CELL_META.xcenter,
            CELL_META.ylim[0],
            CELL_META.sector_index,
            facing="clockwise",
            niceFacing=True,
            adj=(0.0, 0.5),
            cex=0.85,
        ),
    )
    circos.clear()

    out2 = output_dir / "03b_directional_chord_diagram.png"
    fig2.savefig(out2, bbox_inches="tight")
    plt.close(fig2)
    saved.append(out2)

    # 3. Chord diagram from DataFrame with multi-group layout and calc_gap
    df = pd.DataFrame(
        {
            "from": ["A1", "A1", "A2", "B1", "B2", "C1", "C2", "A3"],
            "to": ["B1", "C1", "B2", "C2", "A1", "A2", "B1", "C2"],
            "value": [15, 22, 18, 25, 12, 19, 14, 10],
        }
    )
    group = {"A1": "GrpA", "A2": "GrpA", "A3": "GrpA", "B1": "GrpB", "B2": "GrpB", "C1": "GrpC", "C2": "GrpC"}
    df_half = df.assign(value=df["value"] * 0.5)
    _ = calc_gap(df, df_half, big_gap=18.0, small_gap=2.0)

    fig3 = plt.figure(figsize=(7.2, 7.2), dpi=120)
    circos.clear()
    chordDiagram(df, group=group, big_gap=18.0, small_gap=2.0, reduce=0.0)
    highlight.sector(["A1", "A2", "A3"], track_index=1, col="#E41A1C20", border="#E41A1C", text="GrpA", facing="bending.inside")
    highlight.sector(["B1", "B2"], track_index=1, col="#377EB820", border="#377EB8", text="GrpB", facing="bending.inside")
    highlight.sector(["C1", "C2"], track_index=1, col="#4DAF4A20", border="#4DAF4A", text="GrpC", facing="bending.inside")
    circos.clear()

    out3 = output_dir / "03c_multigroup_chord_dataframe.png"
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
