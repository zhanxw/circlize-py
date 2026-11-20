"""Hierarchical clustering, dendrogram construction, branch coloring, and circular dendrogram rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.spatial.distance import pdist

from ._low_level import circos_lines, circos_points
from ._state import (
    get_current_sector_index,
    get_current_track_index,
    normalize_kwargs,
    set_current_cell,
)
from ._utils import recycle_to_length


@dataclass
class HClust:
    """R-compatible hclust structure (merge, height, order, labels)."""

    merge: np.ndarray  # shape (n-1, 2), 1-based R encoding: <0 is leaf -(idx), >0 is cluster row
    height: np.ndarray  # shape (n-1,)
    order: np.ndarray  # shape (n,), 1-based leaf indices
    labels: list[str]
    method: str = "complete"


@dataclass
class DendrogramNode:
    """R-compatible dendrogram tree node."""

    height: float
    members: int
    midpoint: float | None = None
    leaf: bool = False
    label: str | None = None
    leaf_index: int | None = None  # 0-based index into original observations
    children: list["DendrogramNode"] = field(default_factory=list)
    edge_par: dict[str, Any] = field(default_factory=dict)
    node_par: dict[str, Any] | None = None
    x: float | None = None

    def labels(self) -> list[str]:
        if self.leaf:
            return [self.label if self.label is not None else str(self.leaf_index)]
        out: list[str] = []
        for ch in self.children:
            out.extend(ch.labels())
        return out

    def leaf_indices(self) -> list[int]:
        """Return 0-based leaf indices in left-to-right dendrogram order."""
        if self.leaf:
            return [int(self.leaf_index if self.leaf_index is not None else 0)]
        out: list[int] = []
        for ch in self.children:
            out.extend(ch.leaf_indices())
        return out

    def order_dendrogram(self) -> list[int]:
        """Return 1-based leaf indices in left-to-right order (matching R order.dendrogram)."""
        return [i + 1 for i in self.leaf_indices()]

    def copy(self) -> "DendrogramNode":
        return DendrogramNode(
            height=self.height,
            members=self.members,
            midpoint=self.midpoint,
            leaf=self.leaf,
            label=self.label,
            leaf_index=self.leaf_index,
            children=[c.copy() for c in self.children],
            edge_par=dict(self.edge_par),
            node_par=dict(self.node_par) if self.node_par is not None else None,
            x=self.x,
        )

    def _recompute_stats(self) -> None:
        if self.leaf:
            self.members = 1
            self.midpoint = 0.0
            return
        for c in self.children:
            c._recompute_stats()
        self.members = sum(c.members for c in self.children)
        if len(self.children) == 1:
            c0 = self.children[0]
            self.midpoint = 0.0 if c0.leaf else float(c0.midpoint or 0.0)
        elif len(self.children) >= 2:
            # First and last child centers relative to leftmost leaf of self
            offsets = []
            acc = 0
            for c in self.children:
                center_rel = acc + (0.0 if c.leaf else float(c.midpoint or 0.0))
                offsets.append(center_rel)
                acc += c.members
            self.midpoint = (offsets[0] + offsets[-1]) / 2.0


def hclust_from_data(
    mat: np.ndarray,
    distance_method: str = "euclidean",
    clustering_method: str = "complete",
    labels: Sequence[str] | None = None,
) -> HClust:
    """Cluster rows of `mat` using scipy and wrap in an R-compatible HClust."""
    arr = np.asarray(mat, dtype=float)
    n = arr.shape[0]
    lbls = [str(l) for l in labels] if labels is not None else [str(i + 1) for i in range(n)]
    if n == 1:
        return HClust(
            merge=np.empty((0, 2), dtype=int),
            height=np.empty(0, dtype=float),
            order=np.array([1], dtype=int),
            labels=lbls,
            method=clustering_method,
        )

    dist_map = {
        "euclidean": "euclidean",
        "maximum": "chebyshev",
        "manhattan": "cityblock",
        "canberra": "canberra",
        "binary": "hamming",
        "minkowski": "minkowski",
    }
    link_map = {
        "complete": "complete",
        "single": "single",
        "average": "average",
        "upgma": "average",
        "ward.d": "ward",
        "ward.d2": "ward",
        "ward": "ward",
        "centroid": "centroid",
        "median": "median",
        "mcquitty": "weighted",
    }
    dm = dist_map.get(distance_method.lower(), distance_method)
    lm = link_map.get(clustering_method.lower(), clustering_method)
    d = pdist(arr, metric=dm)
    Z = scipy_linkage(d, method=lm)
    return hclust_from_linkage(Z, labels=lbls, method=clustering_method)


def hclust_from_linkage(Z: np.ndarray, labels: Sequence[str] | None = None, method: str = "complete") -> HClust:
    """Convert a scipy (n-1, 4) linkage matrix to R-style HClust."""
    Z = np.asarray(Z, dtype=float)
    n = Z.shape[0] + 1
    lbls = [str(l) for l in labels] if labels is not None else [str(i + 1) for i in range(n)]
    merge = np.empty((n - 1, 2), dtype=int)
    heights = Z[:, 2].astype(float)
    for k in range(n - 1):
        for col in (0, 1):
            idx = int(Z[k, col])
            merge[k, col] = -(idx + 1) if idx < n else (idx - n + 1)

    def _collect_order(node_id: int) -> list[int]:
        if node_id < 0:
            return [-node_id]
        row = node_id - 1
        return _collect_order(int(merge[row, 0])) + _collect_order(int(merge[row, 1]))

    order = np.array(_collect_order(n - 1), dtype=int)
    return HClust(merge=merge, height=heights, order=order, labels=lbls, method=method)


def as_dendrogram(obj: Any, labels: Sequence[str] | None = None) -> DendrogramNode:
    """Convert HClust, scipy linkage matrix, or dict into a DendrogramNode."""
    if isinstance(obj, DendrogramNode):
        return obj
    if isinstance(obj, dict) and "merge" in obj and "height" in obj:
        obj = HClust(
            merge=np.asarray(obj["merge"], dtype=int),
            height=np.asarray(obj["height"], dtype=float),
            order=np.asarray(obj.get("order", []), dtype=int),
            labels=[str(v) for v in obj.get("labels", [])],
            method=str(obj.get("method", "unknown")),
        )
    elif isinstance(obj, np.ndarray) and obj.ndim == 2 and obj.shape[1] == 4:
        obj = hclust_from_linkage(obj, labels=labels)

    if not isinstance(obj, HClust):
        raise TypeError("Expected DendrogramNode, HClust, or scipy linkage matrix.")

    merge = obj.merge
    heights = obj.height
    lbls = obj.labels
    n_merges = len(heights)
    if n_merges == 0:
        return DendrogramNode(
            height=0.0,
            members=1,
            midpoint=0.0,
            leaf=True,
            label=lbls[0] if lbls else "1",
            leaf_index=0,
        )

    def _build(node_id: int) -> DendrogramNode:
        if node_id < 0:
            idx0 = -node_id - 1
            lbl = lbls[idx0] if idx0 < len(lbls) else str(idx0 + 1)
            return DendrogramNode(height=0.0, members=1, midpoint=0.0, leaf=True, label=lbl, leaf_index=idx0)
        row = node_id - 1
        left = _build(int(merge[row, 0]))
        right = _build(int(merge[row, 1]))
        return DendrogramNode(height=float(heights[row]), members=left.members + right.members, children=[left, right])

    root = _build(n_merges)
    root._recompute_stats()
    return root


def cutree(obj: HClust | DendrogramNode, k: int) -> dict[str, int]:
    """Cut a hierarchical clustering tree into k clusters, returning {label: cluster_id (1..k)}."""
    k = int(k)
    if isinstance(obj, DendrogramNode):
        # Cut by taking top k-1 highest internal nodes
        subs = cut_dendrogram(obj, k)
        out: dict[str, int] = {}
        for cid, sub in enumerate(subs, start=1):
            for lbl in sub.labels():
                out[lbl] = cid
        return out

    hc = obj
    n = len(hc.labels)
    if k <= 1:
        return {lbl: 1 for lbl in hc.labels}
    if k >= n:
        return {lbl: i + 1 for i, lbl in enumerate(hc.labels)}

    parent = list(range(n + len(hc.height)))
    # Only merge the first n - k steps
    active: list[int] = [-(i + 1) for i in range(n)]
    groups: dict[int, list[int]] = {-(i + 1): [i] for i in range(n)}
    for step in range(n - k):
        c1, c2 = int(hc.merge[step, 0]), int(hc.merge[step, 1])
        merged = groups.pop(c1) + groups.pop(c2)
        groups[step + 1] = merged

    # Order clusters by first leaf appearance (matching R cutree)
    ordered_groups = sorted(groups.values(), key=lambda g: min(g))
    res: dict[str, int] = {}
    for cid, idxs in enumerate(ordered_groups, start=1):
        for idx0 in idxs:
            res[hc.labels[idx0]] = cid
    return res


def cut_dendrogram(dend: DendrogramNode, k: int) -> list[DendrogramNode]:
    """Split a dendrogram into k subtrees by cutting the k-1 highest merges."""
    k = int(k)
    if k <= 1 or dend.leaf:
        return [dend.copy()]

    # Priority expansion of the highest internal node in active list
    nodes = [dend.copy()]
    while len(nodes) < k:
        best_i = -1
        best_h = -1.0
        for i, nd in enumerate(nodes):
            if not nd.leaf and nd.height > best_h:
                best_h = nd.height
                best_i = i
        if best_i < 0:
            break
        target = nodes.pop(best_i)
        # Preserve left-to-right order
        nodes = nodes[:best_i] + target.children + nodes[best_i:]
    return nodes


def color_branches(dend: DendrogramNode, k: int = 1, col: Sequence[Any] | None = None) -> DendrogramNode:
    """Color the k subtrees of a dendrogram (similar to R's dendextend::color_branches)."""
    root = dend.copy()
    k = int(k)
    if col is None:
        cols = list(range(1, k + 1))
    else:
        cols = recycle_to_length(col, k)

    if k <= 1 or root.leaf:
        _paint_subtree(root, cols[0])
        return root

    # Find the k-1 highest internal nodes to identify the k subtree roots
    active: list[DendrogramNode] = [root]
    while len(active) < k:
        best_i = -1
        best_h = -1.0
        for i, nd in enumerate(active):
            if not nd.leaf and nd.height > best_h:
                best_h = nd.height
                best_i = i
        if best_i < 0:
            break
        target = active.pop(best_i)
        active = active[:best_i] + target.children + active[best_i:]

    for sub, c_val in zip(active, cols):
        _paint_subtree(sub, c_val)
    return root


def _paint_subtree(node: DendrogramNode, col: Any) -> None:
    node.edge_par["col"] = col
    for c in node.children:
        _paint_subtree(c, col)


def reorder_dendrogram(dend: DendrogramNode, wts: Sequence[float]) -> DendrogramNode:
    """Reorder dendrogram branches so leaf weighted means increase from left to right (R's stats::reorder)."""
    root = dend.copy()
    w_arr = np.asarray(wts, dtype=float).ravel()

    def _reorder(node: DendrogramNode) -> tuple[float, int]:
        if node.leaf:
            idx = int(node.leaf_index or 0)
            return float(w_arr[idx]), 1
        vals = []
        tot_w = 0.0
        tot_n = 0
        for c in node.children:
            mean_c, n_c = _reorder(c)
            vals.append((mean_c, c))
            tot_w += mean_c * n_c
            tot_n += n_c
        vals.sort(key=lambda pair: pair[0])
        node.children = [p[1] for p in vals]
        return (tot_w / max(tot_n, 1)), tot_n

    _reorder(root)
    root._recompute_stats()
    return root


def adjust_dend_by_x(dend: DendrogramNode, x_pos: Sequence[float]) -> DendrogramNode:
    """Assign explicit x coordinates to dendrogram leaves and propagate midpoint x upwards."""
    root = dend.copy()
    x_arr = np.asarray(x_pos, dtype=float).ravel()

    # Note: x_pos corresponds to ordered leaves (in dendrogram left-to-right order)
    leaf_counter = 0

    def _assign(node: DendrogramNode) -> float:
        nonlocal leaf_counter
        if node.leaf:
            node.x = float(x_arr[leaf_counter])
            leaf_counter += 1
            return node.x
        xs = [_assign(c) for c in node.children]
        node.x = float((xs[0] + xs[-1]) / 2.0)
        return node.x

    _assign(root)
    return root


def circos_dendrogram(
    dend: DendrogramNode | HClust | np.ndarray,
    facing: str = "outside",
    max_height: float | None = None,
    use_x_attr: bool = False,
    sector_index: str | None = None,
    track_index: int | None = None,
    **kwargs: Any,
) -> None:
    """Draw a circular dendrogram inside a cell."""
    nkw = normalize_kwargs(kwargs)
    max_height = nkw.get("max_height", max_height)
    use_x_attr = bool(nkw.get("use_x_attr", use_x_attr))
    sector_index = nkw.get("sector_index", sector_index)
    track_index = nkw.get("track_index", track_index)
    if sector_index is None:
        sector_index = get_current_sector_index()
    if track_index is None:
        track_index = get_current_track_index()

    if not isinstance(dend, DendrogramNode):
        dend = as_dendrogram(dend)

    if facing not in ("outside", "inside"):
        raise ValueError("`facing` must be 'outside' or 'inside'.")

    os_idx = get_current_sector_index()
    ot_idx = get_current_track_index()
    set_current_cell(sector_index, track_index)
    try:
        if max_height is None:
            max_height = float(dend.height)

        lbls = dend.labels()
        x_map = {lbl: idx + 0.5 for idx, lbl in enumerate(lbls)}

        def _get_node_x(nd: DendrogramNode) -> float:
            if use_x_attr and nd.x is not None:
                return float(nd.x)
            if nd.leaf:
                return float(x_map[nd.label if nd.label is not None else str(nd.leaf_index)])
            first_lbl = nd.labels()[0]
            return float(nd.midpoint or 0.0) + float(x_map[first_lbl])

        def _draw_node(nd: DendrogramNode) -> None:
            height = float(nd.height)
            n = len(nd.children)
            if n == 0:
                return
            xl = [_get_node_x(c) for c in nd.children]
            yl = [float(c.height) for c in nd.children]
            ep_list = [
                {
                    "col": c.edge_par.get("col", "black"),
                    "lty": c.edge_par.get("lty", 1),
                    "lwd": c.edge_par.get("lwd", 1.0),
                }
                for c in nd.children
            ]

            if facing == "outside":
                if n == 1:
                    circos_lines([xl[0], xl[0]], [max_height - yl[0], max_height - height], col=ep_list[0]["col"], lty=ep_list[0]["lty"], lwd=ep_list[0]["lwd"], straight=True)
                else:
                    circos_lines([xl[0], xl[0]], [max_height - yl[0], max_height - height], col=ep_list[0]["col"], lty=ep_list[0]["lty"], lwd=ep_list[0]["lwd"], straight=True)
                    circos_lines([xl[0], (xl[0] + xl[1]) / 2.0], [max_height - height, max_height - height], col=ep_list[0]["col"], lty=ep_list[0]["lty"], lwd=ep_list[0]["lwd"])
                    for i in range(1, n - 1):
                        circos_lines([xl[i], xl[i]], [max_height - yl[i], max_height - height], col=ep_list[i]["col"], lty=ep_list[i]["lty"], lwd=ep_list[i]["lwd"], straight=True)
                        circos_lines([(xl[i - 1] + xl[i]) / 2.0, (xl[i] + xl[i + 1]) / 2.0], [max_height - height, max_height - height], col=ep_list[i]["col"], lty=ep_list[i]["lty"], lwd=ep_list[i]["lwd"])
                    circos_lines([xl[-1], xl[-1]], [max_height - yl[-1], max_height - height], col=ep_list[-1]["col"], lty=ep_list[-1]["lty"], lwd=ep_list[-1]["lwd"], straight=True)
                    circos_lines([xl[-1], (xl[-1] + xl[-2]) / 2.0], [max_height - height, max_height - height], col=ep_list[-1]["col"], lty=ep_list[-1]["lty"], lwd=ep_list[-1]["lwd"])
                if nd.node_par is not None:
                    circos_points([float(np.mean(xl)) / 2.0], [max_height - height], col=nd.node_par.get("col", "black"), pch=nd.node_par.get("pch", 1), cex=nd.node_par.get("cex", 1.0))
            else:
                if n == 1:
                    circos_lines([xl[0], xl[0]], [yl[0], height], col=ep_list[0]["col"], lty=ep_list[0]["lty"], lwd=ep_list[0]["lwd"], straight=True)
                else:
                    circos_lines([xl[0], xl[0]], [yl[0], height], col=ep_list[0]["col"], lty=ep_list[0]["lty"], lwd=ep_list[0]["lwd"], straight=True)
                    circos_lines([xl[0], (xl[0] + xl[1]) / 2.0], [height, height], col=ep_list[0]["col"], lty=ep_list[0]["lty"], lwd=ep_list[0]["lwd"])
                    for i in range(1, n - 1):
                        circos_lines([xl[i], xl[i]], [yl[i], height], col=ep_list[i]["col"], lty=ep_list[i]["lty"], lwd=ep_list[i]["lwd"], straight=True)
                        circos_lines([(xl[i - 1] + xl[i]) / 2.0, (xl[i] + xl[i + 1]) / 2.0], [height, height], col=ep_list[i]["col"], lty=ep_list[i]["lty"], lwd=ep_list[i]["lwd"])
                    circos_lines([xl[-1], xl[-1]], [yl[-1], height], col=ep_list[-1]["col"], lty=ep_list[-1]["lty"], lwd=ep_list[-1]["lwd"], straight=True)
                    circos_lines([xl[-1], (xl[-1] + xl[-2]) / 2.0], [height, height], col=ep_list[-1]["col"], lty=ep_list[-1]["lty"], lwd=ep_list[-1]["lwd"])
                if nd.node_par is not None:
                    circos_points([float(np.mean(xl)) / 2.0], [height], col=nd.node_par.get("col", "black"), pch=nd.node_par.get("pch", 1), cex=nd.node_par.get("cex", 1.0))

            for idx_c, c in enumerate(nd.children):
                if c.leaf:
                    if c.node_par is not None:
                        y_pt = max_height if facing == "outside" else 0.0
                        circos_points([xl[idx_c]], [y_pt], col=c.node_par.get("col", "black"), pch=c.node_par.get("pch", 1), cex=c.node_par.get("cex", 1.0))
                else:
                    _draw_node(c)

        if not dend.leaf:
            _draw_node(dend)
    finally:
        if os_idx is not None:
            set_current_cell(os_idx, ot_idx)
