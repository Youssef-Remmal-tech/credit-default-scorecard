"""Validation metrics: discrimination, calibration, stability, cut-off strategy, fairness."""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from . import config


def ks_stat(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def summary(y, p) -> dict:
    auc = roc_auc_score(y, p)
    return {"auc": auc, "gini": 2 * auc - 1, "ks": ks_stat(y, p), "brier": brier_score_loss(y, p)}


def bootstrap_gini_diff(y, p_a, p_b, n=1000, seed=42) -> dict:
    """Paired bootstrap of Gini(b) - Gini(a) on the same resampled customers."""
    rng = np.random.default_rng(seed)
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    diffs, ga, gb = [], [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        a = 2 * roc_auc_score(y[i], p_a[i]) - 1
        b = 2 * roc_auc_score(y[i], p_b[i]) - 1
        ga.append(a); gb.append(b); diffs.append(b - a)
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"gini_a_ci": q(ga), "gini_b_ci": q(gb),
            "diff_mean": float(np.mean(diffs)), "diff_ci": q(diffs), "n": len(diffs)}


def calibration_table(y, p, bins=10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["decile"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False) + 1
    return df.groupby("decile").agg(n=("y", "size"), predicted_pd=("p", "mean"),
                                    observed_default_rate=("y", "mean")).reset_index()


def psi(expected, actual, bins=10) -> tuple[float, pd.DataFrame]:
    """Population stability index of a score, using deciles of the expected (training) distribution."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    contrib = (a - e) * np.log(a / e)
    return float(contrib.sum()), pd.DataFrame({"expected_share": e, "actual_share": a, "psi": contrib})


def cutoff_table(y, score) -> pd.DataFrame:
    """Approve the highest scores first. For each approval rate, report the cut-off and outcomes."""
    order = np.argsort(-np.asarray(score))
    y_sorted = np.asarray(y)[order]
    s_sorted = np.asarray(score)[order]
    n, total_bad = len(y_sorted), y_sorted.sum()
    rows = []
    for rate in np.round(np.arange(0.30, 1.0001, 0.01), 2):
        k = int(round(rate * n))
        approved = y_sorted[:k]
        rows.append({
            "approval_rate": rate,
            "cutoff_score": float(s_sorted[k - 1]),
            "bad_rate_approved": float(approved.mean()),
            "defaults_declined_share": float(1 - approved.sum() / total_bad),
            "goods_declined_share": float(((1 - y_sorted[k:]).sum()) / (n - total_bad)),
        })
    return pd.DataFrame(rows)


def age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, [0, 29, 39, 49, 200], labels=["21-29", "30-39", "40-49", "50+"])


def fairness_table(test: pd.DataFrame, score: np.ndarray, cutoff: float) -> pd.DataFrame:
    df = test.assign(score=score, approved=(score >= cutoff).astype(int),
                     age_band=age_band(test["age"]))
    out = []
    for attr in ["sex", "age_band"]:
        g = df.groupby(attr, observed=True)
        t = pd.DataFrame({
            "n": g.size(),
            "approval_rate": g["approved"].mean(),
            "observed_default_rate": g[config.TARGET].mean(),
            "default_rate_if_approved": g.apply(lambda d: d.loc[d.approved == 1, config.TARGET].mean(),
                                                include_groups=False),
            "auc_within_group": g.apply(lambda d: roc_auc_score(d[config.TARGET], -d["score"]),
                                        include_groups=False),
        }).reset_index().rename(columns={attr: "group"})
        t.insert(0, "attribute", attr)
        t["approval_ratio_vs_highest"] = t["approval_rate"] / t["approval_rate"].max()
        out.append(t)
    return pd.concat(out, ignore_index=True)


def risk_grades(y, score, cuts=(470, 500, 530, 545)) -> pd.DataFrame:
    """Group scores into five grades (A safest to E riskiest) and report volume and default rate."""
    labels = ["E", "D", "C", "B", "A"]
    edges = [-np.inf, *cuts, np.inf]
    g = pd.cut(np.asarray(score), edges, labels=labels, right=False)
    df = pd.DataFrame({"grade": g, "y": np.asarray(y), "score": np.asarray(score)})
    t = df.groupby("grade", observed=False).agg(customers=("y", "size"), defaults=("y", "sum"),
                                                default_rate=("y", "mean"),
                                                min_score=("score", "min"), max_score=("score", "max"))
    t["share"] = t["customers"] / t["customers"].sum()
    return t.iloc[::-1].reset_index()
