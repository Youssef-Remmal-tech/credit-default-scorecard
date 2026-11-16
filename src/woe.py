"""Weight-of-evidence binning and information value.

WoE convention: ln(share of goods / share of bads). Positive WoE = lower risk.
Continuous features get quantile bins merged until the bad rate is monotonic.
Low-cardinality features (status codes, month counts) keep their natural order
without a monotonic constraint, because codes like -2 (no use) and -1 (paid in
full) are categories, not points on a scale.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SMOOTH = 0.5  # avoids log(0) for bins with no goods or no bads


@dataclass
class FeatureBins:
    feature: str
    edges: list            # right-closed upper edges; last is +inf
    table: pd.DataFrame = field(repr=False)

    @property
    def iv(self) -> float:
        return float(self.table["iv"].sum())

    def bin_index(self, x: pd.Series) -> np.ndarray:
        return np.searchsorted(np.asarray(self.edges[:-1]), x.to_numpy(), side="left")

    def transform(self, x: pd.Series) -> np.ndarray:
        return self.table["woe"].to_numpy()[self.bin_index(x)]


def _stats(x: np.ndarray, y: np.ndarray, edges: list) -> pd.DataFrame:
    idx = np.searchsorted(np.asarray(edges[:-1]), x, side="left")
    k = len(edges)
    n = np.bincount(idx, minlength=k)
    bad = np.bincount(idx, weights=y, minlength=k)
    return pd.DataFrame({"n": n, "bad": bad, "good": n - bad})


def _woe_table(stats: pd.DataFrame, edges: list) -> pd.DataFrame:
    t = stats.copy()
    tot_good, tot_bad = t["good"].sum(), t["bad"].sum()
    pg = (t["good"] + SMOOTH) / (tot_good + SMOOTH * len(t))
    pb = (t["bad"] + SMOOTH) / (tot_bad + SMOOTH * len(t))
    t["bad_rate"] = t["bad"] / t["n"]
    t["woe"] = np.log(pg / pb)
    t["iv"] = (pg - pb) * t["woe"]
    lows = [-np.inf] + list(edges[:-1])
    t["bin"] = [_label(lo, hi) for lo, hi in zip(lows, edges)]
    t["share"] = t["n"] / t["n"].sum()
    return t[["bin", "n", "share", "bad", "bad_rate", "woe", "iv"]]


def _label(lo, hi) -> str:
    fmt = lambda v: f"{v:.4g}"
    if np.isinf(lo):
        return f"<= {fmt(hi)}"
    if np.isinf(hi):
        return f"> {fmt(lo)}"
    return f"({fmt(lo)}, {fmt(hi)}]"


def _merge(edges: list, i: int) -> list:
    """Merge bin i with bin i+1 by removing the edge between them."""
    return edges[:i] + edges[i + 1:]


def fit_feature(x: pd.Series, y: pd.Series, max_bins: int = 10,
                min_share: float = 0.05) -> FeatureBins:
    xv, yv = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
    uniq = np.unique(xv)
    categorical = len(uniq) <= 12

    if categorical:
        edges = list(uniq[:-1]) + [np.inf]
    else:
        qs = np.unique(np.quantile(xv, np.linspace(0, 1, max_bins + 1)[1:-1]))
        edges = list(qs) + [np.inf]

    # 1) enforce monotonic bad rate for continuous features
    if not categorical:
        direction = np.sign(pd.Series(xv).corr(pd.Series(yv), method="spearman")) or 1.0
        while len(edges) > 1:
            s = _stats(xv, yv, edges)
            br = (s["bad"] / s["n"]).to_numpy()
            viol = [i for i in range(len(br) - 1) if (br[i + 1] - br[i]) * direction < 0]
            if not viol:
                break
            edges = _merge(edges, viol[0])

    # 2) merge bins smaller than min_share into the neighbour with the closest bad rate
    while len(edges) > 1:
        s = _stats(xv, yv, edges)
        share = (s["n"] / s["n"].sum()).to_numpy()
        small = np.where(share < min_share)[0]
        if len(small) == 0:
            break
        i = int(small[np.argmin(share[small])])
        br = (s["bad"] / s["n"].clip(lower=1)).to_numpy()
        if i == 0:
            j = 0
        elif i == len(edges) - 1:
            j = i - 1
        else:
            j = i - 1 if abs(br[i] - br[i - 1]) <= abs(br[i] - br[i + 1]) else i
        edges = _merge(edges, j)

    table = _woe_table(_stats(xv, yv, edges), edges)
    table.insert(0, "feature", x.name)
    return FeatureBins(feature=x.name, edges=edges, table=table)


def fit_all(df: pd.DataFrame, features: list, target: str, **kw) -> dict:
    return {f: fit_feature(df[f], df[target], **kw) for f in features}


def transform_all(df: pd.DataFrame, bins: dict) -> pd.DataFrame:
    return pd.DataFrame({f: b.transform(df[f]) for f, b in bins.items()}, index=df.index)
