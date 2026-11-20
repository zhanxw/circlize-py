# circlize-py

Python port of the R **`circlize`** circular visualization library (v0.4.16), built on `numpy`, `pandas`, `matplotlib`, `scipy`, and `Pillow`.

All core circular layouts, low-level primitives, high-level statistical tracks, hierarchical dendrograms, chord diagrams, split circular heatmaps, nested zoom plots, genomic ideograms, and all bundled datasets are available in pure Python with full 3-way API parity so R examples translate almost verbatim.

**Companion Book & Full Guide:** See [**circlize-book-py**](https://github.com/zhanxw/circlize-book-py) for the Python edition of *Circular Visualization in R*.

---

## Installation

Install from PyPI (Python ≥ 3.10):

```bash
pip install circlize-py
```

Or install editable from a local clone for development & testing:

```bash
pip install -e ".[dev]"
```

You can import either package name interchangeably:

```python
import circlize_py
# or
import circlize
```

---

## 3-Way API Parity

Every exported function supports three interchangeable call styles plus automatic normalization of R dotted / camelCase keyword arguments (e.g. `track.height`, `niceFacing`, `bg.col`, `preAllocateTracks`):

| Style | Example | Notes |
| --- | --- | --- |
| **Idiomatic Python (`snake_case`)** | `circos_initialize(...)`, `circos_track(...)`, `chord_diagram(...)` | Recommended for new Python scripts |
| **R camelCase (`NAMESPACE` aliases)** | `chordDiagram(...)`, `colorRamp2(...)`, `generateRandomBed(...)` | Exact R top-level camelCase names |
| **R dot-syntax namespaces** | `circos.initialize(...)`, `circos.par(...)`, `circos.track(...)`, `circos.heatmap.link(...)`, `draw.sector(...)`, `highlight.sector(...)`, `read.cytoband(...)`, `reverse.circlize(...)`, `posTransform.text(...)`, `cytoband.col(...)` | Translates R examples line-for-line |

> [!NOTE]
> Top-level `set` and `print` namespaces are intentionally **not** shadowed to avoid overriding Python built-in `set` and `print`. Use `set_current_cell(...)`, `set_current_sector_index(...)`, `set_current_track_index(...)`, or `circos.info()`.

---

## Quick Start

### R Dot-Syntax Style

```python
import matplotlib.pyplot as plt
import numpy as np
from circlize import CELL_META, circos, rand_color

rng = np.random.default_rng(1)
sectors = list("abcdefgh")
fa = np.repeat(sectors, 100)
x = rng.normal(size=800)
y = rng.normal(size=800)

fig = plt.figure(figsize=(7, 7), dpi=120)
circos.par(track_height=0.15)
circos.initialize(sectors=fa, x=x)

def panel_fn(xv, yv):
    circos.points(xv, yv, pch=16, cex=0.5, col="#1B9E77")
    circos.axis(labels_cex=0.5)
    circos.text(
        CELL_META.xcenter,
        CELL_META.ylim[1],
        CELL_META.sector_index,
        facing="inside",
        niceFacing=True,
    )

circos.track(sectors=fa, x=x, y=y, bg_col=rand_color(8, luminosity="light", seed=1), panel_fun=panel_fn)
circos.link("a", 0.0, "e", 0.0, col="red", directional=1)
circos.clear()
fig.savefig("quick_start.png", bbox_inches="tight")
plt.close(fig)
```

<p align="center">
  <img src="https://raw.githubusercontent.com/zhanxw/circlize-py/main/examples/figures/quick_start.png" alt="Quick Start Output" width="460"/>
</p>

### Chord Diagram

```python
import matplotlib.pyplot as plt
import pandas as pd
from circlize_py import chord_diagram, circos

fig = plt.figure(figsize=(7, 7), dpi=120)
mat = pd.DataFrame(
    [[10, 4, 2], [3, 12, 5], [7, 1, 9]],
    index=["S1", "S2", "S3"],
    columns=["E1", "E2", "E3"],
)
chord_diagram(mat, directional=1, diff_height=-0.04)
circos.clear()
fig.savefig("quick_start_chord.png", bbox_inches="tight")
plt.close(fig)
```

<p align="center">
  <img src="https://raw.githubusercontent.com/zhanxw/circlize-py/main/examples/figures/quick_start_chord.png" alt="Quick Start Chord Diagram" width="460"/>
</p>

---

## Example Gallery

All gallery scripts live in [`examples/`](examples/) and save PNG figures directly:

### [01 — Basic Layout & Low-Level Primitives](examples/01_basic_layout_and_low_level.py)

| Basic Tracks & Primitives | Arrows, Raster Images & Stacked Labels |
| :---: | :---: |
| ![Basic Tracks and Primitives](https://raw.githubusercontent.com/zhanxw/circlize-py/main/examples/figures/01a_basic_tracks_and_primitives.png) | ![Arrows, Raster Images, and Stacked Labels](https://raw.githubusercontent.com/zhanxw/circlize-py/main/examples/figures/01b_arrows_raster_and_labels.png) |

### [02 — High-Level Statistical Tracks & Dendrograms](examples/02_high_level_tracks.py)

| Barplot, Boxplot & Violin Tracks | Bird Orders Phylogenetic Dendrogram | Track Histograms & Density |
| :---: | :---: | :---: |
| ![Barplot, Boxplot, and Violin Tracks](https://raw.githubusercontent.com/zhanxw/circlize-py/main/examples/figures/02a_bar_box_violin_tracks.png) | ![Bird Orders Phylogenetic Dendrogram](https://raw.githubusercontent.com/zhanxw/circlize-py/main/examples/figures/02b_bird_orders_dendrogram.png) | ![Track Histograms and Density Curves](https://raw.githubusercontent.com/zhanxw/circlize-py/main/examples/figures/02c_track_hist.png) |

### [03 — Links & Chord Diagrams](examples/03_links_and_chord_diagram.py)

| Links, Ribbons & Arrangement | Directional Matrix Chord Diagram | Multi-Group DataFrame Chord Diagram |
| :---: | :---: | :---: |
| ![Links, Ribbons, and Even Arrangement](https://raw.githubusercontent.com/zhanxw/circlize-py/main/examples/figures/03a_links_and_ribbons.png) | ![Directional Matrix Chord Diagram](https://raw.githubusercontent.com/zhanxw/circlize-py/main/examples/figures/03b_directional_chord_diagram.png) | ![Multi-Group DataFrame Chord Diagram](https://raw.githubusercontent.com/zhanxw/circlize-py/main/examples/figures/03c_multigroup_chord_dataframe.png) |

### [04 — Split Circular Heatmaps & Nested Zoom Plots](examples/04_circular_heatmap_and_nested.py)

| Multi-Track Split Circular Heatmap | Nested WGBS Tagment Zoom Plot |
| :---: | :---: |
| ![Multi-Track Split Circular Heatmap](https://raw.githubusercontent.com/zhanxw/circlize-py/main/examples/figures/04a_circular_heatmap_split.png) | ![Nested WGBS Tagment Zoom Plot](https://raw.githubusercontent.com/zhanxw/circlize-py/main/examples/figures/04b_nested_zoom_wgbs.png) |

### [05 — Genomic Visualization](examples/05_genomic_plots.py)

| Human Ideogram, Tracks & Links | DMR Rainfall, Density, Heatmap & Labels |
| :---: | :---: |
| ![Human Ideogram, Genomic Tracks, and Links](https://raw.githubusercontent.com/zhanxw/circlize-py/main/examples/figures/05a_ideogram_tracks_and_links.png) | ![DMR Rainfall, Density, Heatmap, and Labels](https://raw.githubusercontent.com/zhanxw/circlize-py/main/examples/figures/05b_dmr_density_rainfall_heatmap_labels.png) |
| **Circular *E. coli* Genome & Links** | **TP53 / TP63 / TP73 Transcript Exons** |
| ![Circular E. coli Genome and Links](https://raw.githubusercontent.com/zhanxw/circlize-py/main/examples/figures/05c_circular_genome_and_links.png) | ![TP53 / TP63 / TP73 Transcript Exons](https://raw.githubusercontent.com/zhanxw/circlize-py/main/examples/figures/05d_tp_family_transcripts.png) |

---

## Indexing & Signature Conventions

To preserve natural Python idioms while keeping R examples ergonomic:

1. **`track_index` & `CELL_META.sector.numeric.index` (1-based)**:
   - `track_index` runs `1, 2, ..., n_tracks` (just like R `circlize`, `track.index = 1`).
   - `CELL_META.sector.numeric.index` / `CELL_META.sector_numeric_index` is 1-based (`1..n_sectors`), matching R.
   - `CELL_META.sector_idx` gives the 0-based Python index (`0..n_sectors-1`).
2. **Heatmap row indices (0-based Python indexing)**:
   - `CELL_META.subset` and `CELL_META.row_order` use 0-based row indices (`0..n_rows-1`).
   - `circos.heatmap.link(row_ind1, row_ind2)` and `circos.heatmap.get.x(row_ind)` accept 0-based integer indices or row-name strings.
3. **Genomic value column selectors**:
   - Prefer column name strings (e.g. `numeric_column="value1"`, `labels_column="gene"`).
   - 0-based column indices work cleanly (`numeric_column=0` inside `panel_fun(region, value)` or `3` on a full BED table).
   - Automatic 1-based fallback applies when `idx == df.shape[1]` (so standard R 4-column BED `labels.column=4` / `numeric.column=4` and 1-column `value` `labels.column=1` work without edits).
4. **R `I()` AsIs objects**:
   - Use `I([...])` from `circlize_py` (`from circlize_py import I`) for raw data coordinates in `circos.rect(..., padding=I(...))` or `highlight.sector(..., padding=I(...))`.
5. **Colors & NamedMatrix2D**:
   - All 657 R named colors (e.g. `"grey50"`, `"red3"`, `"cornflowerblue"`), R numeric palette `1..8`, hex `#RRGGBB` / `#RRGGBBAA`, and Matplotlib color strings are supported.
   - `circlize()` and `reverse_circlize()` return a `NamedMatrix2D` supporting numpy indexing `[0, 0]`, column selection `[:, "theta"]`, and attribute access `.theta`, `.rou`, `.x`, `.y`.

---

## Bundled Datasets (`extdata`)

All R `inst/extdata/` files are bundled natively inside `circlize_py/extdata/`:

```python
from circlize_py import load_extdata, read, system_file

cb = read.cytoband()                       # 24 chromosomes, 862 cytobands (hg19)
cb_mm10 = read.cytoband(species="mm10")    # 21 chromosomes, 403 cytobands (27 assemblies bundled)
ci_ano = read.chromInfo(species="anoGam1") # 7 chromosomes (86 chromInfo assemblies bundled)

dmr = load_extdata("DMR")                  # {"DMR_hyper": (3150, 3), "DMR_hypo": (1188, 3)}
wb = load_extdata("tagments_WGBS_DMR")     # {"DMR1": (1102, 4), "tagments": (54, 4), "correspondance": (54, 6)}
tp_df = load_extdata("tp_family_df")       # (425, 5) DataFrame of TP53/TP63/TP73 exons
tp = load_extdata("tp_family")             # Nested dict for ["TP73", "TP63", "TP53"]
hc = load_extdata("bird.orders")           # HClust phylogenetic tree (23 bird orders)
doodles = load_extdata("doodle")           # List of 32 RGB uint8 image arrays
logo = load_extdata("Rlogo.png")           # (700, 800, 4) RGBA uint8 numpy array

path_txt = system_file("extdata", "cytoBand.txt", package="circlize")
```

---

## Running Examples & Tests

Run all standalone gallery scripts:

```bash
python examples/01_basic_layout_and_low_level.py
python examples/02_high_level_tracks.py
python examples/03_links_and_chord_diagram.py
python examples/04_circular_heatmap_and_nested.py
python examples/05_genomic_plots.py
```

Run the test suite (unit tests, full `.Rd` example ports, and numerical R parity checks via `Rscript`):

```bash
MPLBACKEND=Agg pytest -v
```

---

## Package Architecture

| Module | Contents |
| --- | --- |
| `_state.py`, `_canvas.py`, `_utils.py` | Layout state, `circos.par`, `CELL_META`, Matplotlib canvas sizing, `circlize`/`reverse_circlize`, physical unit converters, adjacency helpers |
| `_colors.py`, `_r_colors.py` | 657 R colors, `colorRamp2` (`LAB`, `RGB`, `sRGB`, `HLS`, `XYZ`, `LUV`), `col2value`, `add_transparency`, `rand_color` |
| `_plot.py`, `_low_level.py` | `circos_initialize`, `circos_track*`, `draw_sector`, `highlight_sector`, `points`, `lines`, `polygon`, `rect`, `segments`, `text`, axes, barplot/boxplot/violin/hist, arrows, raster, labels |
| `_dendrogram.py`, `_link.py`, `_chord.py` | `HClust`, `as_dendrogram`, `color_branches`, `circos_dendrogram`, Bezier links & arrowheads, `chord_diagram*`, `calc_gap` |
| `_heatmap.py`, `_nested.py` | Split circular heatmaps + clustering + links, `circos_nested` zoom overlay |
| `_regions.py`, `_genomic_utils.py`, `_genomic.py` | Interval operations, `read_cytoband`/`read_chromInfo`, `generate_random_bed`, ideograms, `circos_genomic_*`, `genomic_density`, `rainfall_transform`, `pos_transform_*`, `circos_genomic_heatmap`, `circos_genomic_labels` |
