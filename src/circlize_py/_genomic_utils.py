"""Genomic data loading, cytoband/chromInfo parsers, chromosome sorting, and random BED generation."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from ._dendrogram import HClust
from ._state import circos_par, normalize_kwargs

_EXTDATA_DIR = Path(__file__).resolve().parent / "extdata"
_UNSET = object()

_CYTOBAND_PALETTE: dict[str, str] = {
    "gpos100": "#000000",
    "gpos": "#000000",
    "gpos75": "#828282",
    "gpos66": "#A0A0A0",
    "gpos50": "#C8C8C8",
    "gpos33": "#D2D2D2",
    "gpos25": "#C8C8C8",
    "gvar": "#DCDCDC",
    "gneg": "#FFFFFF",
    "acen": "#D92F27",
    "stalk": "#647FA4",
}

_R_ASSET_MAP: dict[str, str] = {
    "cytoband.txt": "cytoBand.txt",
    "chrominfo.txt": "chromInfo.txt",
    "rlogo.png": "Rlogo.png",
    "dmr.rdata": "DMR.json.gz",
    "dmr": "DMR.json.gz",
    "tagments_wgbs_dmr.rdata": "tagments_WGBS_DMR.json.gz",
    "tagments_wgbs_dmr": "tagments_WGBS_DMR.json.gz",
    "tp_family.rdata": "tp_family.json.gz",
    "tp_family": "tp_family.json.gz",
    "tp_family_df.rds": "tp_family_df.json.gz",
    "tp_family_df": "tp_family_df.json.gz",
    "bird.orders.rdata": "bird_orders.json.gz",
    "bird.orders": "bird_orders.json.gz",
    "bird_orders": "bird_orders.json.gz",
    "doodle.rdata": "doodle.npz",
    "doodle": "doodle.npz",
    "cytoband_list.rds": "cytoband_list.json.gz",
    "cytoband_list": "cytoband_list.json.gz",
    "chrom_info_list.rds": "chrom_info_list.json.gz",
    "chrom_info_list": "chrom_info_list.json.gz",
}


class CytobandData(dict):
    """Dictionary holding cytoband/chromInfo DataFrame, chromosome list, and chr.len map."""

    @property
    def df(self) -> pd.DataFrame:
        return self["df"]

    @property
    def chromosome(self) -> list[str]:
        return self["chromosome"]

    @property
    def chr_len(self) -> dict[str, float]:
        return self["chr.len"]


def extdata_path(name: str) -> Path:
    """Return Path to a bundled asset in `circlize_py/extdata`."""
    direct = _EXTDATA_DIR / name
    if direct.exists():
        return direct
    mapped = _R_ASSET_MAP.get(name.lower())
    if mapped is not None:
        cand = _EXTDATA_DIR / mapped
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Extdata asset not found: {name!r} in {_EXTDATA_DIR}")


def system_file(*parts: str, package: str = "circlize") -> str:
    """R `system.file(..., package='circlize')` compatible path resolver."""
    cleaned = [p for p in parts if p and p != "extdata"]
    if not cleaned:
        return str(_EXTDATA_DIR)
    rel = Path(*cleaned)
    direct = _EXTDATA_DIR / rel
    if direct.exists():
        return str(direct)
    base = cleaned[-1]
    return str(extdata_path(base))


def _load_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _normalize_bed_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("start", "end", "methDiff"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_extdata(name: str) -> Any:
    """Load a bundled circlize dataset by logical name, R filename, or `system_file()` path."""
    p = Path(name)
    if p.exists():
        fname = p.name
    else:
        mapped = _R_ASSET_MAP.get(name.lower())
        if mapped is not None:
            p = _EXTDATA_DIR / mapped
            fname = p.name
        else:
            p = extdata_path(name)
            fname = p.name

    low = fname.lower()
    if low == "dmr.json.gz":
        raw = _load_json_gz(p)
        return {
            "DMR_hyper": _normalize_bed_types(pd.DataFrame(raw["DMR_hyper"])),
            "DMR_hypo": _normalize_bed_types(pd.DataFrame(raw["DMR_hypo"])),
        }
    if low == "tagments_wgbs_dmr.json.gz":
        raw = _load_json_gz(p)
        return {
            "DMR1": _normalize_bed_types(pd.DataFrame(raw["DMR1"])),
            "tagments": _normalize_bed_types(pd.DataFrame(raw["tagments"])),
            "correspondance": _normalize_bed_types(pd.DataFrame(raw["correspondance"])),
        }
    if low == "tp_family_df.json.gz":
        return _normalize_bed_types(pd.DataFrame(_load_json_gz(p)))
    if low == "tp_family.json.gz":
        raw = _load_json_gz(p)
        out: dict[str, dict[str, pd.DataFrame]] = {}
        for gene, tx_map in raw.items():
            out[gene] = {}
            for tx, rows in tx_map.items():
                df = pd.DataFrame(rows)
                if "_row" in df.columns:
                    df = df.set_index("_row")
                    df.index.name = None
                out[gene][tx] = _normalize_bed_types(df)
        return out
    if low == "bird_orders.json.gz":
        raw = _load_json_gz(p)
        return HClust(
            merge=np.asarray(raw["merge"], dtype=int),
            height=np.asarray(raw["height"], dtype=float),
            order=np.asarray(raw["order"], dtype=int),
            labels=[str(x) for x in raw["labels"]],
            method=str(raw.get("method", "complete")),
        )
    if low == "doodle.npz":
        with np.load(p) as z:
            return [np.asarray(z[k]) for k in z.files]
    if low in ("cytoband_list.json.gz", "chrom_info_list.json.gz"):
        raw = _load_json_gz(p)
        return {sp: pd.DataFrame(rows) for sp, rows in raw.items()}
    if low == "cytoband.txt":
        return pd.read_csv(
            p,
            sep="\t",
            header=None,
            names=["V1", "V2", "V3", "V4", "V5"],
            dtype={"V1": str, "V4": str, "V5": str},
            keep_default_na=False,
        )
    if low == "chrominfo.txt":
        return pd.read_csv(
            p,
            sep="\t",
            header=None,
            usecols=[0, 1],
            names=["V1", "V2"],
            dtype={"V1": str},
            keep_default_na=False,
        )
    if low == "rlogo.png":
        return np.asarray(Image.open(p))
    raise ValueError(f"Unsupported extdata file format: {p}")


def usable_chromosomes(species: str | None) -> list[str] | None:
    """Return canonical chromosome list for standard model organism genome assemblies."""
    if species is None:
        return None
    prefix = re.sub(r"\d+$", "", str(species))
    if prefix == "hg":
        return [f"chr{c}" for c in list(range(1, 23)) + ["X", "Y"]]
    if prefix == "mm":
        return [f"chr{c}" for c in list(range(1, 20)) + ["X", "Y"]]
    if prefix == "rn":
        return [f"chr{c}" for c in list(range(1, 21)) + ["X", "Y"]]
    if prefix == "dm":
        return ["chr2L", "chr2R", "chr3L", "chr3R", "chr4", "chrX"]
    if prefix == "ce":
        return ["chrI", "chrII", "chrIII", "chrIV", "chrV", "chrX"]
    if prefix == "sacCer":
        romans = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV"]
        return [f"chr{r}" for r in romans]
    return None


def sort_chr(chromosome: Sequence[str]) -> list[str]:
    """Sort chromosome names naturally (numeric chromosomes first, then sex/other), matching R `sort_chr`."""
    chroms = sorted(str(c) for c in chromosome)
    if not chroms:
        return chroms
    if not re.match(r"^(chr|\d)", chroms[0]):
        return chroms

    inds = [re.sub(r"_.*$", "", c.replace("chr", "", 1) if "chr" in c else c) for c in chroms]
    # Note: R uses gsub("chr", "", c)
    inds = [re.sub(r"_.*$", "", c.replace("chr", "")) for c in chroms]

    num_items: list[tuple[int, int, str]] = []
    letter_items: list[tuple[str, int, str]] = []
    for idx, (c, ind) in enumerate(zip(chroms, inds)):
        if re.match(r"^\d+$", ind):
            num_items.append((int(ind), idx, c))
        else:
            letter_items.append((ind, idx, c))

    num_items.sort(key=lambda t: (t[0], t[1]))
    letter_items.sort(key=lambda t: (t[0], t[1]))
    return [t[2] for t in num_items] + [t[2] for t in letter_items]


def cytoband_col(x: Any) -> str | list[str]:
    """Map cytoband stain names (gneg, gpos25, acen, ...) to hex colors."""
    if isinstance(x, str):
        return _CYTOBAND_PALETTE.get(x, "#FFFFFF")
    if isinstance(x, (pd.Series, pd.Index, np.ndarray, list, tuple)):
        return [_CYTOBAND_PALETTE.get(str(v), "#FFFFFF") for v in x]
    return _CYTOBAND_PALETTE.get(str(x), "#FFFFFF")


def _read_tabular_genomic(source: Any, expected_cols: int) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    path_str = str(source)
    if os.path.exists(path_str):
        comp = "gzip" if path_str.endswith(".gz") else "infer"
        df = pd.read_csv(path_str, sep="\t", header=None, compression=comp, keep_default_na=False)
        if df.shape[1] < expected_cols:
            raise ValueError(f"Expected at least {expected_cols} columns in {path_str}, got {df.shape[1]}.")
        return df.iloc[:, :expected_cols]
    raise FileNotFoundError(f"Cannot read genomic annotation source: {source!r}")


def read_cytoband(
    cytoband: Any = None,
    species: str | None = None,
    chromosome_index: Any = _UNSET,
    sort_chr_flag: bool = True,
    **kwargs: Any,
) -> CytobandData:
    """Read UCSC cytoBand data or bundled species cytoband table."""
    nkw = normalize_kwargs(kwargs)
    chromosome_index = nkw.pop("chromosome_index", chromosome_index)
    if "sort_chr" in nkw:
        sort_chr_flag = bool(nkw.pop("sort_chr"))

    if chromosome_index is _UNSET:
        chromosome_index = usable_chromosomes(species)

    if chromosome_index is not None:
        sort_chr_flag = False

    if species is not None:
        tmpdir = Path(str(circos_par.get_value("__tempdir__")))
        cached = tmpdir / f"{species}_cytoBand.txt.gz"
        if cached.exists():
            d = _read_tabular_genomic(cached, 5)
        else:
            cb_map = _load_json_gz(_EXTDATA_DIR / "cytoband_list.json.gz")
            if species in cb_map:
                d = pd.DataFrame(cb_map[species])
            else:
                raise ValueError(
                    f"Species {species!r} is not in bundled cytoband_list ({len(cb_map)} assemblies). "
                    "Pass a local cytoBand file path or DataFrame to `read_cytoband(file)`."
                )
    elif cytoband is None:
        d = load_extdata("cytoBand.txt")
    else:
        d = _read_tabular_genomic(cytoband, 5)

    d = d.copy()
    if d.shape[1] < 5:
        raise ValueError("`cytoband` data must have 5 columns (chr, start, end, band, stain).")
    d = d.iloc[:, :5].copy()
    d.columns = ["V1", "V2", "V3", "V4", "V5"]
    d["V1"] = d["V1"].astype(str)
    d["V2"] = pd.to_numeric(d["V2"], errors="raise").astype(float)
    d["V3"] = pd.to_numeric(d["V3"], errors="raise").astype(float)
    d["V4"] = d["V4"].astype(str)
    d["V5"] = d["V5"].astype(str)

    if chromosome_index is not None:
        req = [chromosome_index] if isinstance(chromosome_index, str) else [str(c) for c in chromosome_index]
        present = set(d["V1"].tolist())
        filtered_idx = [c for c in req if c in present]
        d = d[d["V1"].isin(filtered_idx)].copy()
        chromosome = filtered_idx
    else:
        chromosome = list(dict.fromkeys(d["V1"].tolist()))

    if len(d) == 0 or len(chromosome) == 0:
        raise ValueError("Cannot find any chromosome in cytoband data (check 'chr' prefix).")

    if sort_chr_flag:
        chromosome = sort_chr(chromosome)

    chr_len: dict[str, float] = {}
    frames: list[pd.DataFrame] = []
    for ch in chromosome:
        sub = d[d["V1"] == ch]
        chr_len[ch] = float(sub["V3"].max())
        frames.append(sub)

    dnew = pd.concat(frames, ignore_index=True)
    return CytobandData({"df": dnew, "chromosome": chromosome, "chr.len": chr_len})


def read_chrom_info(
    chrom_info: Any = None,
    species: str | None = None,
    chromosome_index: Any = _UNSET,
    sort_chr_flag: bool = True,
    **kwargs: Any,
) -> CytobandData:
    """Read UCSC chromInfo data or bundled species chromosome sizes."""
    nkw = normalize_kwargs(kwargs)
    chromosome_index = nkw.pop("chromosome_index", chromosome_index)
    if "sort_chr" in nkw:
        sort_chr_flag = bool(nkw.pop("sort_chr"))

    if chromosome_index is _UNSET:
        chromosome_index = usable_chromosomes(species)

    if chromosome_index is not None:
        sort_chr_flag = False

    if species is not None:
        tmpdir = Path(str(circos_par.get_value("__tempdir__")))
        cached = tmpdir / f"{species}_chromInfo.txt.gz"
        if cached.exists():
            d = _read_tabular_genomic(cached, 2)
        else:
            ci_map = _load_json_gz(_EXTDATA_DIR / "chrom_info_list.json.gz")
            if species in ci_map:
                d = pd.DataFrame(ci_map[species])
            else:
                raise ValueError(
                    f"Species {species!r} is not in bundled chrom_info_list ({len(ci_map)} assemblies). "
                    "Pass a local chromInfo file path or DataFrame to `read_chrom_info(file)`."
                )
    elif chrom_info is None:
        d = load_extdata("chromInfo.txt")
    else:
        d = _read_tabular_genomic(chrom_info, 2)

    d = d.iloc[:, :2].copy()
    d.columns = ["chr", "len"]
    d["chr"] = d["chr"].astype(str)
    d["len"] = pd.to_numeric(d["len"], errors="raise").astype(float)
    d = d.drop_duplicates(subset=["chr"], keep="first")

    if chromosome_index is not None:
        req = [chromosome_index] if isinstance(chromosome_index, str) else [str(c) for c in chromosome_index]
        present = set(d["chr"].tolist())
        filtered_idx = [c for c in req if c in present]
        d = d[d["chr"].isin(filtered_idx)].copy()
        chromosome = filtered_idx
    else:
        chromosome = list(dict.fromkeys(d["chr"].tolist()))

    if len(d) == 0 or len(chromosome) == 0:
        raise ValueError("Cannot find any chromosome in chromInfo data (check 'chr' prefix).")

    if sort_chr_flag:
        chromosome = sort_chr(chromosome)

    len_map = dict(zip(d["chr"].tolist(), d["len"].tolist()))
    chr_len = {ch: float(len_map[ch]) for ch in chromosome}
    dnew = pd.DataFrame(
        {
            "chr": chromosome,
            "start": [0.0] * len(chromosome),
            "end": [chr_len[ch] for ch in chromosome],
        }
    )
    return CytobandData({"df": dnew, "chromosome": chromosome, "chr.len": chr_len})


read_chromInfo = read_chrom_info


def validate_region(
    region: Any,
    start_col: int = 1,
    end_col: int = 2,
    check_chr: bool = True,
) -> None:
    """Validate genomic region DataFrame/array (numeric and end >= start)."""
    if isinstance(region, pd.DataFrame):
        df = region
    else:
        df = pd.DataFrame(region)

    if len(df) == 0:
        return
    if df.shape[1] <= max(start_col, end_col):
        raise ValueError("`region` does not have enough columns for start/end.")
    if check_chr and start_col >= 1:
        # Ensure chromosome column is non-null
        if df.iloc[:, 0].isna().any():
            raise ValueError("Chromosome column contains missing values.")

    starts = pd.to_numeric(df.iloc[:, start_col], errors="coerce").to_numpy(dtype=float)
    ends = pd.to_numeric(df.iloc[:, end_col], errors="coerce").to_numpy(dtype=float)
    if np.any(np.isnan(starts)) or np.any(np.isnan(ends)):
        raise ValueError("Region start/end coordinates must be numeric and non-NA.")
    if np.any(ends < starts):
        raise ValueError("Region `end` must be >= `start`.")


def generate_random_bed(
    nr: int = 10000,
    nc: int = 1,
    fun: Callable[[int], Any] | None = None,
    species: str | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate random genomic intervals across chromosomes with `nc` numeric value columns."""
    rng = np.random.default_rng(seed)
    if fun is None:
        fun = lambda k: rng.normal(0.0, 0.5, size=k)

    cyto = read_cytoband(species=species)
    chr_len = cyto["chr.len"]
    chromosome = cyto["chromosome"]
    total_len = sum(chr_len.values())

    frames: list[pd.DataFrame] = []
    for ch in chromosome:
        clen = int(round(chr_len[ch]))
        if clen < 2:
            continue
        k = int(round(nr * 2.0 * clen / total_len))
        if k % 2 == 1:
            k += 1
        if k == 0:
            continue
        k = min(k, clen if clen % 2 == 0 else clen - 1)
        if k < 2:
            continue
        breaks = np.sort(rng.choice(np.arange(1, clen + 1), size=k, replace=False))
        starts = breaks[0::2].astype(float)
        ends = breaks[1::2].astype(float)
        n_int = len(starts)
        data: dict[str, Any] = {
            "chr": [ch] * n_int,
            "start": starts,
            "end": ends,
        }
        for c in range(1, int(nc) + 1):
            colname = f"value{c}"
            vals = np.asarray(fun(n_int))
            if vals.shape == ():
                vals = np.repeat(vals.item(), n_int)
            data[colname] = vals
        frames.append(pd.DataFrame(data))

    if not frames:
        cols = ["chr", "start", "end"] + [f"value{c}" for c in range(1, int(nc) + 1)]
        return pd.DataFrame(columns=cols)

    df = pd.concat(frames, ignore_index=True)
    if nc == 0:
        return df[["chr", "start", "end"]]
    return df


generateRandomBed = generate_random_bed
