"""Numerical parity tests comparing Python circlize_py outputs against installed R circlize."""

from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

import circlize_py as cp

HAS_RSCRIPT = shutil.which("Rscript") is not None


def _run_r_json(r_body: str):
    if not HAS_RSCRIPT:
        pytest.skip("Rscript not installed")
    script = (
        "suppressPackageStartupMessages(library(circlize));\n"
        "suppressPackageStartupMessages(library(jsonlite));\n"
        "pdf(NULL);\n"
        + r_body
        + "\ndev.off() > /dev/null;\n"
    ).replace("\ndev.off() > /dev/null;", "\ninvisible(dev.off());")
    res = subprocess.run(
        ["Rscript", "--vanilla", "-e", script],
        capture_output=True,
        text=True,
        check=True,
    )
    # Grab last non-empty JSON line
    lines = [ln for ln in res.stdout.splitlines() if ln.strip()]
    return json.loads(lines[-1])


def test_r_parity_circlize_reverse_and_polar():
    r_out = _run_r_json(
        """
        circos.par(start.degree = 30, gap.degree = c(5, 10, 15))
        circos.initialize(sectors = c("A", "B", "C"), xlim = matrix(c(0, 10, -5, 5, 100, 200), ncol = 2, byrow = TRUE))
        circos.track(ylim = c(0, 10), track.height = 0.25)
        c1 <- circlize(c(0, 5, 10), c(0, 5, 10), sector.index = "A", track.index = 1)
        c2 <- circlize(c(-5, 0, 5), c(2, 5, 8), sector.index = "B", track.index = 1)
        r1 <- reverse.circlize(c1[, "theta"], c1[, "rou"], sector.index = "A", track.index = 1)
        pc <- circlize:::polar2Cartesian(c1)
        m <- get.cell.meta.data("cell.top.radius", sector.index = "A", track.index = 1)
        mb <- get.cell.meta.data("cell.bottom.radius", sector.index = "A", track.index = 1)
        circos.clear()
        cat(toJSON(list(c1 = unname(c1), c2 = unname(c2), r1 = unname(r1), pc = unname(pc), top = m, bot = mb), auto_unbox = TRUE, digits = 12))
        """
    )

    cp.circos_par(start_degree=30, gap_degree=[5, 10, 15])
    cp.circos_initialize(sectors=["A", "B", "C"], xlim=np.array([[0, 10], [-5, 5], [100, 200]]))
    cp.circos_track(ylim=[0, 10], track_height=0.25)

    c1_py = cp.circlize([0, 5, 10], [0, 5, 10], sector_index="A", track_index=1)
    c2_py = cp.circlize([-5, 0, 5], [2, 5, 8], sector_index="B", track_index=1)
    r1_py = cp.reverse_circlize(c1_py[:, "theta"], c1_py[:, "rou"], sector_index="A", track_index=1)
    pc_py = cp.polar2Cartesian(c1_py)

    assert np.isclose(cp.get_cell_meta_data("cell.top.radius", "A", 1), r_out["top"], atol=1e-9)
    assert np.isclose(cp.get_cell_meta_data("cell.bottom.radius", "A", 1), r_out["bot"], atol=1e-9)
    assert np.allclose(np.asarray(c1_py), np.asarray(r_out["c1"]), atol=1e-9)
    assert np.allclose(np.asarray(c2_py), np.asarray(r_out["c2"]), atol=1e-9)
    assert np.allclose(np.asarray(r1_py), np.asarray(r_out["r1"]), atol=1e-9)
    assert np.allclose(np.asarray(pc_py), np.asarray(r_out["pc"]), atol=1e-9)


def test_r_parity_xaxis_counter_clockwise():
    r_out = _run_r_json(
        """
        circos.par(xaxis.clock.wise = FALSE)
        circos.initialize(sectors = c("A", "B"), xlim = c(0, 10))
        circos.track(ylim = c(0, 1))
        c1 <- circlize(c(0, 5, 10), c(0, 0.5, 1), sector.index = "A", track.index = 1)
        r1 <- reverse.circlize(c1[, "theta"], c1[, "rou"], sector.index = "A", track.index = 1)
        circos.clear()
        cat(toJSON(list(c1 = unname(c1), r1 = unname(r1)), auto_unbox = TRUE, digits = 12))
        """
    )

    cp.circos_par(xaxis_clock_wise=False)
    cp.circos_initialize(sectors=["A", "B"], xlim=[0, 10])
    cp.circos_track(ylim=[0, 1])
    c1_py = cp.circlize([0, 5, 10], [0, 0.5, 1], sector_index="A", track_index=1)
    r1_py = cp.reverse_circlize(c1_py[:, 0], c1_py[:, 1], sector_index="A", track_index=1)
    assert np.allclose(c1_py, np.asarray(r_out["c1"]), atol=1e-9)
    assert np.allclose(r1_py, np.asarray(r_out["r1"]), atol=1e-9)


@pytest.mark.parametrize("space", ["LAB", "RGB", "sRGB", "HLS", "XYZ", "LUV"])
def test_r_parity_color_ramp_and_col2value(space: str):
    r_out = _run_r_json(
        f"""
        f <- colorRamp2(c(-2, 0, 2), c("blue", "white", "red"), space = "{space}")
        xs <- c(-3, -2, -1, 0, 0.5, 2, 4)
        cols <- f(xs)
        vals <- col2value(cols, col_fun = f)
        f_tr <- colorRamp2(c(-2, 0, 2), c("blue", "white", "red"), space = "{space}", transparency = 0.4)
        cols_tr <- f_tr(xs)
        cat(toJSON(list(cols = cols, vals = vals, cols_tr = cols_tr), auto_unbox = TRUE, digits = 12))
        """
    )

    f_py = cp.colorRamp2([-2, 0, 2], ["blue", "white", "red"], space=space)
    xs = [-3, -2, -1, 0, 0.5, 2, 4]
    cols_py = f_py(xs)
    vals_py = cp.col2value(cols_py, col_fun=f_py)

    for c_py, c_r in zip(cols_py, r_out["cols"]):
        rgb_py = np.array([int(c_py[1:3], 16), int(c_py[3:5], 16), int(c_py[5:7], 16), int(c_py[7:9], 16)])
        rgb_r = np.array([int(c_r[1:3], 16), int(c_r[3:5], 16), int(c_r[5:7], 16), int(c_r[7:9], 16)])
        assert np.all(np.abs(rgb_py - rgb_r) <= 2), f"{space}: {c_py} vs {c_r}"

    # Compare Python col2value directly against R col2value across all 6 color spaces
    assert np.allclose(vals_py, np.asarray(r_out["vals"], dtype=float), atol=0.02)

    f_tr_py = cp.colorRamp2([-2, 0, 2], ["blue", "white", "red"], space=space, transparency=0.4)
    cols_tr_py = f_tr_py(xs)
    for c_py, c_r in zip(cols_tr_py, r_out["cols_tr"]):
        assert abs(int(c_py[7:9], 16) - int(c_r[7:9], 16)) <= 1


def test_r_parity_chord_diagram_and_calc_gap_and_adjacency():
    r_out = _run_r_json(
        """
        m <- matrix(c(10, 4, 2, 3, 12, 5, 7, 1, 9), nrow = 3, byrow = TRUE)
        rownames(m) <- c("S1", "S2", "S3")
        colnames(m) <- c("E1", "E2", "E3")
        df <- adjacencyMatrix2List(m)
        m2 <- adjacencyList2Matrix(df)
        gap <- calc_gap(m, m / 2, big.gap = 12, small.gap = 2)
        res <- chordDiagram(m, plot = FALSE)
        circos.clear()
        cat(toJSON(list(df = df, m2 = unname(m2), gap = gap, res = res), auto_unbox = TRUE, digits = 12))
        """
    )

    m = pd.DataFrame(
        [[10, 4, 2], [3, 12, 5], [7, 1, 9]],
        index=["S1", "S2", "S3"],
        columns=["E1", "E2", "E3"],
    )
    df_py = cp.adjacencyMatrix2List(m)
    m2_py = cp.adjacencyList2Matrix(df_py)
    assert np.allclose(m2_py.to_numpy(), np.asarray(r_out["m2"]), atol=1e-9)

    gap_py = cp.calc_gap(m, m / 2.0, big_gap=12, small_gap=2)
    assert np.isclose(gap_py, r_out["gap"], atol=1e-9)

    res_py = cp.chordDiagram(m, plot=False)
    r_df = pd.DataFrame(r_out["res"])
    for col in ("x1", "x2", "o1", "o2", "value1", "value2"):
        assert np.allclose(res_py[col].to_numpy(dtype=float), r_df[col].to_numpy(dtype=float), atol=1e-8)
    assert res_py["rn"].tolist() == r_df["rn"].tolist()
    assert res_py["cn"].tolist() == r_df["cn"].tolist()


def test_r_parity_genomic_density_rainfall_and_smart_align():
    r_out = _run_r_json(
        """
        bed <- data.frame(chr=c("chr1","chr1","chr1","chr2","chr2"), start=c(10,50,200,20,100), end=c(30,80,220,40,150))
        gd_p_t <- genomicDensity(bed, window.size=100, overlap=TRUE, count_by="percent")
        gd_p_f <- genomicDensity(bed, window.size=100, overlap=FALSE, count_by="percent")
        gd_n_t <- genomicDensity(bed, window.size=100, overlap=TRUE, count_by="number")
        rf_min <- rainfallTransform(bed, mode="min")
        rf_mean <- rainfallTransform(bed, mode="mean")
        rf_left <- rainfallTransform(bed, mode="left")
        rf_right <- rainfallTransform(bed, mode="right")
        sa <- smartAlign(c(1, 2, 10, 11), c(4, 5, 13, 14), c(0, 15))
        cat(toJSON(list(gd_p_t=gd_p_t, gd_p_f=gd_p_f, gd_n_t=gd_n_t, rf_min=rf_min, rf_mean=rf_mean, rf_left=rf_left, rf_right=rf_right, sa=sa), auto_unbox=TRUE, digits=12))
        """
    )

    bed = pd.DataFrame(
        {
            "chr": ["chr1", "chr1", "chr1", "chr2", "chr2"],
            "start": [10, 50, 200, 20, 100],
            "end": [30, 80, 220, 40, 150],
        }
    )

    for key, ov, cb in [
        ("gd_p_t", True, "percent"),
        ("gd_p_f", False, "percent"),
        ("gd_n_t", True, "number"),
    ]:
        py_gd = cp.genomic_density(bed, window_size=100, overlap=ov, count_by=cb)
        r_gd = pd.DataFrame(r_out[key])
        assert py_gd["chr"].tolist() == r_gd["chr"].tolist()
        assert np.allclose(py_gd["start"], r_gd["start"])
        assert np.allclose(py_gd["end"], r_gd["end"])
        assert np.allclose(py_gd["value"], r_gd["value"], atol=1e-9)

    for key, mode in [("rf_min", "min"), ("rf_mean", "mean"), ("rf_left", "left"), ("rf_right", "right")]:
        py_rf = cp.rainfall_transform(bed, mode=mode)
        r_rf = pd.DataFrame(r_out[key])
        r_vals = pd.to_numeric(r_rf["dist"], errors="coerce").to_numpy(dtype=float)
        assert np.allclose(py_rf["dist"].to_numpy(dtype=float), r_vals, equal_nan=True)

    sa_py = cp.smart_align([1, 2, 10, 11], [4, 5, 13, 14], [0, 15])
    r_sa = pd.DataFrame(r_out["sa"])
    assert np.allclose(sa_py["start"], r_sa["start"], atol=1e-9)
    assert np.allclose(sa_py["end"], r_sa["end"], atol=1e-9)


def test_r_parity_cytoband_chrominfo_sort_chr_and_bird_dendrogram():
    r_out = _run_r_json(
        """
        cb <- read.cytoband(species="hg19")
        ci <- read.chromInfo(species="hg19")
        cols <- cytoband.col(c("gneg", "gpos25", "gpos50", "gpos75", "gpos100", "gvar", "stalk", "acen", "unknown"))
        sc <- circlize:::sort_chr(c("chr10", "chr2", "chr1", "chrX", "chrY", "chrM", "chr11_random", "chrUn_gl"))
        e <- new.env()
        load(system.file("extdata", "bird.orders.RData", package="circlize"), envir=e)
        d <- as.dendrogram(e$hc)
        ct <- unname(cutree(e$hc, k=4))
        cat(toJSON(list(
          cb_nrow=nrow(cb$df), cb_chrs=cb$chromosome, cb_lens=unname(cb$chr.len),
          ci_nrow=nrow(ci$df), ci_chrs=ci$chromosome, ci_lens=unname(ci$chr.len),
          cols=cols, sc=sc, labels=labels(d), height=attr(d, "height"), ct=ct
        ), auto_unbox=TRUE, digits=12))
        """
    )

    cb_py = cp.read_cytoband(species="hg19")
    assert len(cb_py.df) == r_out["cb_nrow"]
    assert cb_py.chromosome == r_out["cb_chrs"]
    assert np.allclose([cb_py.chr_len[c] for c in cb_py.chromosome], r_out["cb_lens"])

    ci_py = cp.read_chrom_info(species="hg19")
    assert len(ci_py.df) == r_out["ci_nrow"]
    assert ci_py.chromosome == r_out["ci_chrs"]
    assert np.allclose([ci_py.chr_len[c] for c in ci_py.chromosome], r_out["ci_lens"])

    cols_py = cp.cytoband_col(["gneg", "gpos25", "gpos50", "gpos75", "gpos100", "gvar", "stalk", "acen", "unknown"])
    assert cols_py == r_out["cols"]

    sc_py = cp.sort_chr(["chr10", "chr2", "chr1", "chrX", "chrY", "chrM", "chr11_random", "chrUn_gl"])
    assert sc_py == r_out["sc"]

    hc_py = cp.load_extdata("bird.orders")
    dend_py = cp.as_dendrogram(hc_py)
    assert dend_py.labels() == r_out["labels"]
    assert np.isclose(dend_py.height, r_out["height"], atol=1e-9)
    ct_py = cp.cutree(hc_py, k=4)
    assert [ct_py[lbl] for lbl in hc_py.labels] == r_out["ct"]
