"""Global state, circos.par options, cell metadata, and CELL_META proxy."""

from __future__ import annotations

import copy
import re
import tempfile
import warnings
from typing import Any, Callable, Iterable, Sequence

import numpy as np


def r_key_to_snake(name: str) -> str:
    """Convert R dotted/camelCase parameter names to canonical snake_case."""
    if name.startswith("__") and name.endswith("__"):
        return name
    s = name.replace(".", "_")
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


def normalize_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Normalize R-style kwargs (dotted or camelCase) to snake_case."""
    out: dict[str, Any] = {}
    for k, v in kwargs.items():
        nk = r_key_to_snake(k)
        out[nk] = v
    return out


def snake_to_r_meta_name(name: str) -> str:
    """Normalize metadata key between snake_case and R dotted names."""
    if name in ("sector_idx",):
        return name
    return name.replace("_", ".")


class _CircosState:
    def __init__(self) -> None:
        self.sector_data: dict[str, dict[str, float]] | None = None
        self.sector_order: list[str] = []
        self.cell_data: dict[str, dict[int, dict[str, Any]]] | None = None
        self.current_track_index: int = 0
        self.current_sector_index: str | None = None
        self.ax: Any = None
        self.canvas_overlay_scale: float | None = None
        self.current_genomic_param: dict[str, Any] | None = None

    def reset_layout(self, keep_ax: bool = False) -> None:
        self.sector_data = None
        self.sector_order = []
        self.cell_data = None
        self.current_track_index = 0
        self.current_sector_index = None
        self.current_genomic_param = None
        if not keep_ax:
            self.ax = None
            self.canvas_overlay_scale = None


_STATE = _CircosState()


def is_circos_initialized() -> bool:
    return _STATE.sector_data is not None


class CircosPar:
    """Global layout options matching R's GlobalOptions-backed circos.par."""

    _SYNONYMS = {
        "gap_after": "gap_degree",
        "help": "message",
    }
    _VISIBLE_KEYS = [
        "start.degree",
        "gap.degree",
        "gap.after",
        "track.margin",
        "unit.circle.segments",
        "cell.padding",
        "track.height",
        "points.overflow.warning",
        "circle.margin",
        "canvas.xlim",
        "canvas.ylim",
        "major.by.degree",
        "clock.wise",
        "xaxis.clock.wise",
        "message",
        "help",
    ]

    def __init__(self) -> None:
        self._tmpdir = tempfile.mkdtemp(prefix="circlize_py_")
        self._init_defaults()

    def _init_defaults(self) -> None:
        self._values: dict[str, Any] = {
            "start_degree": 0.0,
            "gap_degree": 1.0,
            "track_margin": [0.01, 0.01],
            "unit_circle_segments": 500,
            "cell_padding": [0.02, 1.0, 0.02, 1.0],
            "track_height": 0.2,
            "points_overflow_warning": True,
            "circle_margin": [0.0, 0.0, 0.0, 0.0],
            "canvas_xlim": None,  # computed dynamically from circle_margin when None
            "canvas_ylim": None,
            "major_by_degree": 10.0,
            "clock_wise": True,
            "xaxis_clock_wise": True,
            "message": True,
            "ring": False,
            "__tempdir__": self._tmpdir,
            "__omar__": False,
            "__tempenv__": {},
        }

    def _canonical_key(self, key: str) -> str:
        sk = r_key_to_snake(key)
        return self._SYNONYMS.get(sk, sk)

    def get_value(self, key: str) -> Any:
        ck = self._canonical_key(key)
        if ck == "canvas_xlim" and self._values["canvas_xlim"] is None:
            cm = self._values["circle_margin"]
            return [-(1.0 + cm[0]), 1.0 + cm[1]]
        if ck == "canvas_ylim" and self._values["canvas_ylim"] is None:
            cm = self._values["circle_margin"]
            return [-(1.0 + cm[2]), 1.0 + cm[3]]
        if ck == "default_track_height":
            return self._values["track_height"]
        if ck not in self._values:
            raise KeyError(f"Unknown circos.par option: {key!r}")
        val = self._values[ck]
        if isinstance(val, list):
            return list(val)
        return val

    def set_value(self, key: str, value: Any, _suppress_warn: bool = False) -> None:
        ck = self._canonical_key(key)
        if ck == "default_track_height":
            warnings.warn(
                "`default.track.height` is replaced by `track.height`, ignoring this setting.",
                UserWarning,
                stacklevel=3,
            )
            return

        if ck not in self._values:
            raise KeyError(f"Unknown circos.par option: {key!r}")

        if ck == "start_degree":
            if is_circos_initialized() and not _suppress_warn:
                warnings.warn(
                    "'start.degree' can only be modified before `circos.initialize`, or maybe you forgot to call `circos.clear` in your last plot.",
                    UserWarning,
                    stacklevel=3,
                )
            self._values["start_degree"] = float(value)
        elif ck == "gap_degree":
            if is_circos_initialized() and not _suppress_warn:
                warnings.warn(
                    "'gap.degree' can only be modified before `circos.initialize`, or maybe you forgot to call `circos.clear` in your last plot.",
                    UserWarning,
                    stacklevel=3,
                )
            if isinstance(value, dict):
                vals = {str(k): float(v) for k, v in value.items()}
                if any(v < 0 or v >= 360 for v in vals.values()):
                    raise ValueError("`gap.degree` values must be in [0, 360).")
                self._values["gap_degree"] = vals
            elif isinstance(value, (list, tuple, np.ndarray)):
                vals_list = [float(v) for v in value]
                if any(v < 0 or v >= 360 for v in vals_list):
                    raise ValueError("`gap.degree` values must be in [0, 360).")
                self._values["gap_degree"] = vals_list[0] if len(vals_list) == 1 else vals_list
            else:
                fv = float(value)
                if fv < 0 or fv >= 360:
                    raise ValueError("`gap.degree` value must be in [0, 360).")
                self._values["gap_degree"] = fv
        elif ck == "track_margin":
            vals = [float(v) for v in value]
            if len(vals) != 2:
                raise ValueError("`track.margin` must have length 2.")
            self._values["track_margin"] = vals
        elif ck == "unit_circle_segments":
            self._values["unit_circle_segments"] = int(value)
        elif ck == "cell_padding":
            vals = [float(v) for v in value]
            if len(vals) == 2:
                vals = vals + vals
            elif len(vals) != 4:
                raise ValueError("`cell.padding` must have length 2 or 4.")
            if is_circos_initialized() and not _suppress_warn:
                old = self._values["cell_padding"]
                vals = [vals[0], old[1], vals[2], old[3]]
            self._values["cell_padding"] = vals
        elif ck == "track_height":
            self._values["track_height"] = float(value)
        elif ck == "points_overflow_warning":
            self._values["points_overflow_warning"] = bool(value)
        elif ck == "circle_margin":
            if is_circos_initialized() and not _suppress_warn:
                warnings.warn(
                    "'circle.margin' can only be modified before `circos.initialize`, or maybe you forgot to call `circos.clear` in your last plot.",
                    UserWarning,
                    stacklevel=3,
                )
            vals = [float(value)] if isinstance(value, (int, float, np.number)) else [float(v) for v in value]
            if any(v <= 0 for v in vals):
                raise ValueError("The value of `circle.margin` should be positive.")
            if len(vals) == 1:
                vals = vals * 4
            elif len(vals) == 2:
                vals = [vals[0], vals[0], vals[1], vals[1]]
            elif len(vals) != 4:
                raise ValueError("Length of `circle.margin` can only be 1, 2, or 4.")
            self._values["circle_margin"] = vals
        elif ck == "canvas_xlim":
            if is_circos_initialized() and not _suppress_warn:
                warnings.warn(
                    "'canvas.xlim' can only be modified before `circos.initialize`, or maybe you forgot to call `circos.clear` in your last plot.",
                    UserWarning,
                    stacklevel=3,
                )
            if value is None:
                self._values["canvas_xlim"] = None
            else:
                vals = [float(v) for v in value]
                if len(vals) != 2:
                    raise ValueError("`canvas.xlim` must have length 2.")
                self._values["canvas_xlim"] = vals
        elif ck == "canvas_ylim":
            if is_circos_initialized() and not _suppress_warn:
                warnings.warn(
                    "'canvas.ylim' can only be modified before `circos.initialize`, or maybe you forgot to call `circos.clear` in your last plot.",
                    UserWarning,
                    stacklevel=3,
                )
            if value is None:
                self._values["canvas_ylim"] = None
            else:
                vals = [float(v) for v in value]
                if len(vals) != 2:
                    raise ValueError("`canvas.ylim` must have length 2.")
                self._values["canvas_ylim"] = vals
        elif ck == "major_by_degree":
            self._values["major_by_degree"] = float(value)
        elif ck == "clock_wise":
            if is_circos_initialized() and not _suppress_warn:
                warnings.warn(
                    "'clock.wise' can only be modified before `circos.initialize`, or maybe you forgot to call `circos.clear` in your last plot.",
                    UserWarning,
                    stacklevel=3,
                )
            self._values["clock_wise"] = bool(value)
        elif ck == "xaxis_clock_wise":
            self._values["xaxis_clock_wise"] = bool(value)
        elif ck == "message":
            self._values["message"] = bool(value)
        elif ck in ("ring", "__omar__"):
            self._values[ck] = bool(value)
        elif ck == "__tempdir__":
            self._values[ck] = str(value)
        elif ck == "__tempenv__":
            self._values[ck] = value

    def reset(self) -> None:
        tmpdir = self._values.get("__tempdir__", self._tmpdir)
        self._init_defaults()
        self._values["__tempdir__"] = tmpdir

    def as_dict(self, r_keys: bool = True) -> dict[str, Any]:
        if r_keys:
            return {k: self.get_value(k) for k in self._VISIBLE_KEYS if k not in ("gap.after", "help")}
        return {r_key_to_snake(k): self.get_value(k) for k in self._VISIBLE_KEYS if k not in ("gap.after", "help")}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Handle RESET / READ.ONLY flags
        nkw = normalize_kwargs(kwargs)
        reset_flag = nkw.pop("reset", False)
        nkw.pop("read_only", None)
        nkw.pop("local", None)
        nkw.pop("add", None)
        suppress_warn = nkw.pop("_suppress_warn", False)

        if reset_flag:
            self.reset()
            if not args and not nkw:
                return None

        if not args and not nkw:
            return self.as_dict(r_keys=True)

        if args and not nkw:
            if len(args) == 1:
                arg = args[0]
                if isinstance(arg, dict):
                    old = {k: self.get_value(k) for k in arg}
                    for k, v in arg.items():
                        self.set_value(k, v, _suppress_warn=suppress_warn)
                    return old
                if isinstance(arg, (list, tuple)):
                    return {str(k): self.get_value(str(k)) for k in arg}
                return self.get_value(str(arg))
            return {str(k): self.get_value(str(k)) for k in args}

        if nkw:
            old = {}
            for k in nkw:
                try:
                    old[k] = self.get_value(k)
                except KeyError:
                    old[k] = None
            for k, v in nkw.items():
                self.set_value(k, v, _suppress_warn=suppress_warn)
            return old
        return None

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return self.get_value(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
            return
        self.set_value(name, value)

    def __getitem__(self, key: str) -> Any:
        return self.get_value(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.set_value(key, value)

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in self.as_dict().items())
        return f"CircosPar({items})"


circos_par = CircosPar()


def get_all_sector_index() -> list[str]:
    if _STATE.sector_data is None:
        return []
    return list(_STATE.sector_order)


def get_all_track_index() -> list[int]:
    if _STATE.cell_data is None:
        return []
    max_t = 0
    for tracks in _STATE.cell_data.values():
        if tracks:
            max_t = max(max_t, max(tracks.keys()))
    if max_t == 0:
        return []
    return list(range(1, max_t + 1))


def get_sector_data(sector_index: str | None = None) -> dict[str, float]:
    if sector_index is None:
        sector_index = get_current_sector_index()
    if _STATE.sector_data is None:
        raise RuntimeError("Circular layout has not been initialized yet.")
    sid = str(sector_index)
    if sid not in _STATE.sector_data:
        raise ValueError(f"Cannot find sector: {sid!r}.")
    return _STATE.sector_data[sid]


def get_current_track_index() -> int:
    return _STATE.current_track_index


def set_current_track_index(x: int) -> None:
    _STATE.current_track_index = int(x)


def get_current_sector_index() -> str | None:
    return _STATE.current_sector_index


def set_current_sector_index(x: str) -> None:
    sx = str(x)
    if sx not in get_all_sector_index():
        raise ValueError(f"Cannot find {sx!r} in all available sector names.")
    _STATE.current_sector_index = sx


def set_current_cell(sector_index: str, track_index: int) -> None:
    set_current_sector_index(str(sector_index))
    set_current_track_index(int(track_index))


def get_current_chromosome() -> str | None:
    return get_current_sector_index()


def get_cell_data(sector_index: str | None = None, track_index: int | None = None) -> dict[str, Any]:
    if sector_index is None:
        sector_index = get_current_sector_index()
    if track_index is None:
        track_index = get_current_track_index()
    if _STATE.cell_data is None or sector_index is None:
        raise RuntimeError("Cell data is not initialized.")
    return _STATE.cell_data[str(sector_index)][int(track_index)]


def set_cell_data(sector_index: str, track_index: int, **kwargs: Any) -> None:
    if _STATE.cell_data is None:
        raise RuntimeError("Cell data is not initialized.")
    _STATE.cell_data[str(sector_index)][int(track_index)] = dict(kwargs)


def has_cell(sector_index: str | None, track_index: int | None) -> bool:
    if _STATE.cell_data is None or sector_index is None or track_index is None:
        return False
    sid = str(sector_index)
    tid = int(track_index)
    return sid in _STATE.cell_data and tid in _STATE.cell_data[sid]


def add_track_meta_data(name: str, value: Any, track_index: int | None = None) -> None:
    if track_index is None:
        track_index = get_current_track_index()
    env = circos_par.get_value("__tempenv__")
    tmeta = env.setdefault("track.meta.data", {})
    tkey = str(track_index)
    tmeta.setdefault(tkey, {})[name] = value


def add_sector_meta_data(name: str, value: Any, sector_index: str | None = None) -> None:
    if sector_index is None:
        sector_index = get_current_sector_index()
    env = circos_par.get_value("__tempenv__")
    smeta = env.setdefault("sector.meta.data", {})
    skey = str(sector_index)
    smeta.setdefault(skey, {})[name] = value


def get_cell_meta_data(
    name: str,
    sector_index: str | None = None,
    track_index: int | None = None,
) -> Any:
    if sector_index is None:
        sector_index = get_current_sector_index()
    if track_index is None:
        track_index = get_current_track_index()

    if sector_index is None or len(_STATE.sector_order) == 0:
        raise RuntimeError("It seems the circular plot has not been initialized.")
    if track_index is None or track_index == 0:
        raise RuntimeError("It seems the track has not been created.")

    sid = str(sector_index)
    tid = int(track_index)
    if sid not in get_all_sector_index():
        raise ValueError(f"Cannot find sector: {sid!r}.")
    if tid not in get_all_track_index():
        raise ValueError(f"Cannot find track: {tid}.")

    if name == "sector_idx":
        return _STATE.sector_order.index(sid)

    r_name = snake_to_r_meta_name(name)
    sdata = get_sector_data(sid)
    cdata = get_cell_data(sid, tid)

    if r_name == "xlim":
        return np.array(cdata["xlim"], dtype=float)
    if r_name == "ylim":
        return np.array(cdata["ylim"], dtype=float)
    if r_name == "xrange":
        xlim = cdata["xlim"]
        return float(xlim[1] - xlim[0])
    if r_name == "yrange":
        ylim = cdata["ylim"]
        return float(ylim[1] - ylim[0])
    if r_name == "xcenter":
        xlim = cdata["xlim"]
        return float((xlim[1] + xlim[0]) / 2.0)
    if r_name == "ycenter":
        ylim = cdata["ylim"]
        return float((ylim[1] + ylim[0]) / 2.0)
    if r_name == "cell.xlim":
        return np.array(cdata["cell_xlim"], dtype=float)
    if r_name == "cell.ylim":
        return np.array(cdata["cell_ylim"], dtype=float)
    if r_name == "sector.numeric.index":
        return _STATE.sector_order.index(sid) + 1
    if r_name == "sector.index":
        return sid
    if r_name == "track.index":
        return tid
    if r_name == "xplot":
        return np.array([sdata["start.degree"], sdata["end.degree"]], dtype=float)
    if r_name == "yplot":
        return np.array([cdata["track_start"] - cdata["track_height"], cdata["track_start"]], dtype=float)
    if r_name == "cell.width":
        return float(sdata["start.degree"] - sdata["end.degree"])
    if r_name == "cell.height":
        return float(cdata["track_height"])
    if r_name == "track.margin":
        return list(cdata["track_margin"])
    if r_name == "cell.padding":
        return list(cdata["cell_padding"])
    if r_name == "cell.start.degree":
        return float(sdata["start.degree"])
    if r_name == "cell.end.degree":
        return float(sdata["end.degree"])
    if r_name == "cell.bottom.radius":
        return float(cdata["track_start"] - cdata["track_height"])
    if r_name == "cell.top.radius":
        return float(cdata["track_start"])
    if r_name == "bg.col":
        return cdata["bg_col"]
    if r_name == "bg.border":
        return cdata["bg_border"]
    if r_name == "bg.lty":
        return cdata["bg_lty"]
    if r_name == "bg.lwd":
        return cdata["bg_lwd"]
    if r_name == "track.height":
        return float(cdata["track_height"])

    env = circos_par.get_value("__tempenv__")
    tmeta = env.get("track.meta.data", {})
    if str(tid) in tmeta:
        if name in tmeta[str(tid)]:
            return tmeta[str(tid)][name]
        if r_name in tmeta[str(tid)]:
            return tmeta[str(tid)][r_name]
    smeta = env.get("sector.meta.data", {})
    if sid in smeta:
        if name in smeta[sid]:
            return smeta[sid][name]
        if r_name in smeta[sid]:
            return smeta[sid][r_name]
    return None


class _SubMetaProxy:
    """Enables R-like chained access such as CELL_META.cell.xlim or CELL_META.sector.numeric.index."""

    def __init__(self, prefix: str) -> None:
        self._prefix = prefix

    def __getattr__(self, name: str) -> Any:
        full = f"{self._prefix}.{name}"
        if full in ("sector.numeric",):
            return _SubMetaProxy(full)
        return get_cell_meta_data(full)

    def __getitem__(self, name: str) -> Any:
        return get_cell_meta_data(f"{self._prefix}.{name}")


class CellMetaProxy:
    """Easy access to metadata of the current cell (CELL_META.xlim, CELL_META.cell.xlim, CELL_META['xlim'])."""

    _BASE_NAMES = [
        "xlim",
        "ylim",
        "xrange",
        "yrange",
        "xcenter",
        "ycenter",
        "cell.xlim",
        "cell.ylim",
        "sector.numeric.index",
        "sector.index",
        "track.index",
        "xplot",
        "yplot",
        "cell.width",
        "cell.height",
        "track.margin",
        "cell.padding",
        "cell.start.degree",
        "cell.end.degree",
        "cell.bottom.radius",
        "cell.top.radius",
        "bg.col",
        "bg.border",
        "bg.lty",
        "bg.lwd",
        "track.height",
    ]

    def names(self) -> list[str]:
        nm = list(self._BASE_NAMES)
        sid = get_current_sector_index()
        tid = get_current_track_index()
        env = circos_par.get_value("__tempenv__")
        if tid > 0:
            tmeta = env.get("track.meta.data", {}).get(str(tid), {})
            nm.extend(tmeta.keys())
        if sid is not None:
            smeta = env.get("sector.meta.data", {}).get(sid, {})
            nm.extend(smeta.keys())
        return nm

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in ("cell", "sector", "track", "bg"):
            return _SubMetaProxy(name)
        return get_cell_meta_data(name)

    def __getitem__(self, name: str) -> Any:
        return get_cell_meta_data(name)

    def __dir__(self) -> list[str]:
        return [n.replace(".", "_") for n in self.names()]

    def __repr__(self) -> str:
        return (
            "Please use in a form of `CELL_META.name` or `CELL_META['name']` "
            "where `name` is supported in `get_cell_meta_data()`. "
            "Call `CELL_META.names()` for supported names."
        )


CELL_META = CellMetaProxy()


def get_most_inside_radius() -> float:
    tracks = get_all_track_index()
    if not tracks:
        return 1.0
    last = tracks[-1]
    return float(
        get_cell_meta_data("cell.bottom.radius", track_index=last)
        - get_cell_meta_data("track.margin", track_index=last)[0]
        - circos_par.get_value("track.margin")[1]
    )


def circos_clear(keep_ax: bool = False) -> None:
    """Reset the circular layout state and circos.par parameters."""
    _STATE.reset_layout(keep_ax=keep_ax)
    circos_par.reset()


def circos_info(
    sector_index: str | Sequence[str] | None = None,
    track_index: int | Sequence[int] | None = None,
    plot: bool = False,
    **kwargs: Any,
) -> None:
    """Print or plot sector/track/cell metadata."""
    nkw = normalize_kwargs(kwargs)
    if "sector_index" in nkw:
        sector_index = nkw["sector_index"]
    if "track_index" in nkw:
        track_index = nkw["track_index"]

    sectors = get_all_sector_index()
    tracks = get_all_track_index()

    if plot:
        from ._low_level import circos_text

        for s in sectors:
            for t in tracks:
                cx = get_cell_meta_data("cell.xlim", sector_index=s, track_index=t)
                cy = get_cell_meta_data("cell.ylim", sector_index=s, track_index=t)
                circos_text(
                    float(np.mean(cx)),
                    float(np.mean(cy)),
                    labels=f"{s}:{t}",
                    sector_index=s,
                    track_index=t,
                    facing="downward",
                )
        return None

    if sector_index is None and track_index is None:
        if sectors:
            print("All your sectors:")
            print(sectors)
        else:
            print("No sector has been created")
        print()
        if tracks:
            print("All your tracks:")
            print(tracks)
        else:
            print("No track has been created")
        print()
    else:
        if track_index is None:
            t_list = tracks
        elif isinstance(track_index, (int, np.integer)):
            t_list = [int(track_index)]
        else:
            t_list = [int(t) for t in track_index]

        if sector_index is None:
            s_list = sectors
        elif isinstance(sector_index, str):
            s_list = [sector_index]
        else:
            s_list = [str(s) for s in sector_index]

        for s in s_list:
            for t in t_list:
                print(f"sector index: '{s}'")
                print(f"track index: {t}")
                xlim = get_cell_meta_data("xlim", s, t)
                ylim = get_cell_meta_data("ylim", s, t)
                cx = get_cell_meta_data("cell.xlim", s, t)
                cy = get_cell_meta_data("cell.ylim", s, t)
                xp = get_cell_meta_data("xplot", s, t)
                yp = get_cell_meta_data("yplot", s, t)
                cw = get_cell_meta_data("cell.width", s, t)
                ch = get_cell_meta_data("cell.height", s, t)
                tm = get_cell_meta_data("track.margin", s, t)
                cp = get_cell_meta_data("cell.padding", s, t)
                print(f"xlim: [{xlim[0]}, {xlim[1]}]")
                print(f"ylim: [{ylim[0]}, {ylim[1]}]")
                print(f"cell.xlim: [{cx[0]}, {cx[1]}]")
                print(f"cell.ylim: [{cy[0]}, {cy[1]}]")
                print(f"xplot (degree): [{xp[0]}, {xp[1]}]")
                print(f"yplot (radius): [{yp[0]}, {yp[1]}]")
                print(f"cell.width (degree): {cw}")
                print(f"cell.height (radius): {ch}")
                print(f"track.margin: [{tm[0]}, {tm[1]}]")
                print(f"cell.padding: [{cp[0]}, {cp[1]}, {cp[2]}, {cp[3]}]")
                print()

    cur_s = get_current_sector_index()
    cur_t = get_current_track_index()
    if cur_s is not None:
        print(f"Your current sector.index is {cur_s}")
    if cur_t > 0:
        print(f"Your current track.index is {cur_t}")
    return None


def show_index() -> None:
    warnings.warn("`show.index` is deprecated, please use `circos.info` instead.", DeprecationWarning, stacklevel=2)
    circos_info(plot=True)
