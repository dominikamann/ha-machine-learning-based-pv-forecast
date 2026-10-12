"""Histogram gradient boosting in numpy (quantile and squared-error loss).

scikit-learn publishes no wheels for Home Assistant OS (musl on aarch64), so the integration ships
this small implementation of the same algorithm: features are binned into at most 255 bins, trees
grow best-first on gradient histograms, and quantile leaves take the weighted quantile of the
residuals (as scikit-learn's HistGradientBoostingRegressor does).
"""

from __future__ import annotations

import numpy as np


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if len(values) == 0 or weights.sum() <= 0:
        return 0.0
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    i = int(np.searchsorted(cumulative, q * cumulative[-1]))
    return float(values[order][min(i, len(values) - 1)])


class _Binner:
    """Maps each feature to small integer bins; NaN gets the last bin."""

    def __init__(self, max_bins: int) -> None:
        self.max_bins = max_bins
        self.edges: list[np.ndarray] = []

    def fit(self, X: np.ndarray) -> _Binner:
        probs = np.linspace(0, 1, self.max_bins + 1)[1:-1]
        self.edges = []
        for column in X.T:
            finite = column[np.isfinite(column)]
            self.edges.append(
                np.unique(np.quantile(finite, probs)) if finite.size else np.array([])
            )
        return self

    @property
    def missing_bin(self) -> int:
        return self.max_bins

    def transform(self, X: np.ndarray) -> np.ndarray:
        out = np.empty(X.shape, dtype=np.uint8)
        for j, edges in enumerate(self.edges):
            column = X[:, j]
            binned = np.searchsorted(edges, column, side="right")
            out[:, j] = np.where(np.isfinite(column), binned, self.missing_bin)
        return out


class _Tree:
    """Flat arrays; a node is a leaf when feature == -1. Rows with bin <= threshold go left."""

    def __init__(self) -> None:
        self.feature: list[int] = []
        self.threshold: list[int] = []
        self.left: list[int] = []
        self.right: list[int] = []
        self.value: list[float] = []

    def add(self) -> int:
        self.feature.append(-1)
        self.threshold.append(0)
        self.left.append(-1)
        self.right.append(-1)
        self.value.append(0.0)
        return len(self.value) - 1

    def freeze(self) -> tuple[np.ndarray, ...]:
        return (
            np.array(self.feature, dtype=np.int32),
            np.array(self.threshold, dtype=np.int32),
            np.array(self.left, dtype=np.int32),
            np.array(self.right, dtype=np.int32),
            np.array(self.value, dtype=float),
        )


def _predict_tree(tree: tuple[np.ndarray, ...], Xb: np.ndarray) -> np.ndarray:
    feature, threshold, left, right, value = tree
    rows = np.arange(len(Xb))
    node = np.zeros(len(Xb), dtype=np.int32)
    while True:
        f = feature[node]
        active = f >= 0
        if not active.any():
            return value[node]
        n = node[active]
        go_left = Xb[rows[active], f[active]] <= threshold[n]
        node[active] = np.where(go_left, left[n], right[n])


class GradientBoosting:
    """Gradient boosted regression trees with quantile or squared-error loss."""

    def __init__(
        self,
        loss: str = "quantile",
        quantile: float = 0.5,
        learning_rate: float = 0.05,
        max_iter: int = 300,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.1,
        max_bins: int = 255,
    ) -> None:
        if loss not in ("quantile", "squared_error"):
            raise ValueError(f"unknown loss {loss!r}")
        self.loss = loss
        self.quantile = quantile
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2 = l2_regularization
        self.max_bins = max_bins
        self.baseline = 0.0
        self.trees: list[tuple[np.ndarray, ...]] = []
        self.binner = _Binner(max_bins)

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> GradientBoosting:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        keep = w > 0
        X, y, w = X[keep], y[keep], w[keep]
        Xb = self.binner.fit(X).transform(X)
        self.baseline = (
            _weighted_quantile(y, w, self.quantile)
            if self.loss == "quantile"
            else float(np.average(y, weights=w))
        )
        pred = np.full(len(y), self.baseline)
        offsets = np.arange(X.shape[1]) * (self.max_bins + 1)
        codes = (Xb.astype(np.int64) + offsets).ravel()
        self.trees = []
        for _ in range(self.max_iter):
            residual = y - pred
            if self.loss == "quantile":
                gradient = w * np.where(residual > 0, self.quantile, self.quantile - 1)
            else:
                gradient = w * residual
            tree, leaves = self._grow(Xb, codes, gradient, w, X.shape[1])
            for node, rows in leaves:
                value = self._leaf_value(residual[rows], w[rows], gradient[rows])
                tree.value[node] = self.learning_rate * value
                pred[rows] += self.learning_rate * value
            self.trees.append(tree.freeze())
        return self

    def _leaf_value(self, residual: np.ndarray, w: np.ndarray, gradient: np.ndarray) -> float:
        if self.loss == "quantile":
            return _weighted_quantile(residual, w, self.quantile)
        return float(gradient.sum() / (w.sum() + self.l2))

    def _histograms(
        self,
        rows: np.ndarray,
        codes: np.ndarray,
        gradient: np.ndarray,
        w: np.ndarray,
        n_features: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        size = n_features * (self.max_bins + 1)
        flat = codes.reshape(-1, n_features)[rows].ravel()
        g = np.bincount(flat, np.repeat(gradient[rows], n_features), size)
        h = np.bincount(flat, np.repeat(w[rows], n_features), size)
        c = np.bincount(flat, minlength=size).astype(float)
        shape = (n_features, self.max_bins + 1)
        return g.reshape(shape), h.reshape(shape), c.reshape(shape)

    def _best_split(
        self,
        rows: np.ndarray,
        codes: np.ndarray,
        gradient: np.ndarray,
        w: np.ndarray,
        n_features: int,
    ) -> tuple[float, int, int]:
        if len(rows) < 2 * self.min_samples_leaf:
            return 0.0, -1, -1
        g, h, c = self._histograms(rows, codes, gradient, w, n_features)
        gl, hl, cl = (np.cumsum(a, axis=1)[:, :-1] for a in (g, h, c))
        gt, ht, ct = g.sum(axis=1)[:, None], h.sum(axis=1)[:, None], c.sum(axis=1)[:, None]
        gr, hr, cr = gt - gl, ht - hl, ct - cl
        with np.errstate(divide="ignore", invalid="ignore"):
            gain = gl**2 / (hl + self.l2) + gr**2 / (hr + self.l2) - gt**2 / (ht + self.l2)
        valid = (cl >= self.min_samples_leaf) & (cr >= self.min_samples_leaf) & (hl > 0) & (hr > 0)
        gain = np.where(valid, gain, -np.inf)
        best = int(np.argmax(gain))
        feature, threshold = divmod(best, gain.shape[1])
        best_gain = float(gain[feature, threshold])
        return (best_gain, feature, threshold) if best_gain > 1e-12 else (0.0, -1, -1)

    def _grow(
        self,
        Xb: np.ndarray,
        codes: np.ndarray,
        gradient: np.ndarray,
        w: np.ndarray,
        n_features: int,
    ) -> tuple[_Tree, list[tuple[int, np.ndarray]]]:
        tree = _Tree()
        root = tree.add()
        all_rows = np.arange(len(Xb))
        candidates = [(root, all_rows, *self._best_split(all_rows, codes, gradient, w, n_features))]
        leaves: list[tuple[int, np.ndarray]] = []
        n_leaves = 1
        while candidates:
            candidates.sort(key=lambda c: c[2])
            node, rows, gain, feature, threshold = candidates.pop()
            if feature < 0 or n_leaves >= self.max_leaf_nodes:
                leaves.append((node, rows))
                continue
            go_left = Xb[rows, feature] <= threshold
            left, right = tree.add(), tree.add()
            tree.feature[node], tree.threshold[node] = feature, threshold
            tree.left[node], tree.right[node] = left, right
            n_leaves += 1
            for child, child_rows in ((left, rows[go_left]), (right, rows[~go_left])):
                split = self._best_split(child_rows, codes, gradient, w, n_features)
                candidates.append((child, child_rows, *split))
        return tree, leaves

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xb = self.binner.transform(np.asarray(X, dtype=float))
        out = np.full(len(Xb), self.baseline)
        for tree in self.trees:
            out += _predict_tree(tree, Xb)
        return out
