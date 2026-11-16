"""Static charts for the README. Two-series palette: blue and orange (colour-blind-safe pair)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f2a37", "#6b7480", "#e3e6ea"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "text.color": INK,
})


def iv_chart(selection_log, path):
    d = selection_log.sort_values("iv")
    fig, ax = plt.subplots(figsize=(7, 4.6))
    colors = [BLUE if dec == "kept" else "#b9c3cf" for dec in d["decision"]]
    ax.barh(d["feature"], d["iv"], color=colors, height=0.62)
    ax.axvline(0.02, color=MUTED, lw=1, ls="--")
    ax.text(0.02, len(d) - 0.4, " IV 0.02 threshold", color=MUTED, fontsize=8, va="center")
    ax.set_title("Information value by feature")
    ax.set_xlabel("Information value (blue = kept in scorecard, grey = dropped)")
    ax.grid(axis="y", visible=False)
    fig.savefig(path); plt.close(fig)


def roc_chart(y, p_champ, p_chall, labels, path):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for p, c, lab in [(p_champ, BLUE, labels[0]), (p_chall, ORANGE, labels[1])]:
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=c, lw=2, label=lab)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    ax.set_xlabel("False positive rate (goods flagged)")
    ax.set_ylabel("True positive rate (defaults caught)")
    ax.set_title("ROC curve, hold-out sample")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(path); plt.close(fig)


def score_distribution(score, y, path):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bins = np.linspace(np.percentile(score, 0.5), np.percentile(score, 99.5), 40)
    ax.hist(score[y == 0], bins=bins, color=BLUE, alpha=0.35, label="Did not default", density=True)
    ax.hist(score[y == 0], bins=bins, color=BLUE, histtype="step", lw=1.8, density=True)
    ax.hist(score[y == 1], bins=bins, color=ORANGE, histtype="step", lw=2.2, label="Defaulted", density=True)
    ax.set_xlabel("Scorecard points (higher = lower risk)")
    ax.set_ylabel("Density")
    ax.set_title("Score separates defaulters from non-defaulters")
    ax.legend(frameon=False)
    fig.savefig(path); plt.close(fig)


def calibration_chart(cal_champ, cal_chall, path):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    lim = max(cal_champ[["predicted_pd", "observed_default_rate"]].max().max(),
              cal_chall[["predicted_pd", "observed_default_rate"]].max().max()) * 1.05
    ax.plot([0, lim], [0, lim], color=MUTED, lw=1, ls="--", label="Perfect calibration")
    ax.plot(cal_champ["predicted_pd"], cal_champ["observed_default_rate"], "o-", color=BLUE, lw=2, ms=6, label="Scorecard")
    ax.plot(cal_chall["predicted_pd"], cal_chall["observed_default_rate"], "s-", color=ORANGE, lw=2, ms=6, label="Gradient boosting")
    ax.set_xlabel("Predicted probability of default (decile mean)")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration by decile, hold-out")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(path); plt.close(fig)


def cutoff_chart(cut, base_rate, target_rate, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cut["approval_rate"] * 100, cut["bad_rate_approved"] * 100, color=BLUE, lw=2)
    ax.axhline(base_rate * 100, color=MUTED, lw=1, ls="--")
    ax.text(31, base_rate * 100 - 0.9, f"Approve everyone: {base_rate:.1%} default", color=MUTED, fontsize=8)
    row = cut.loc[(cut["approval_rate"] - target_rate).abs().idxmin()]
    ax.plot(row["approval_rate"] * 100, row["bad_rate_approved"] * 100, "o", color=BLUE, ms=8,
            markeredgecolor="white", markeredgewidth=2)
    ax.annotate(f"Approve {row['approval_rate']:.0%}: {row['bad_rate_approved']:.1%} default\n"
                f"cut-off {row['cutoff_score']:.0f} points",
                (row["approval_rate"] * 100, row["bad_rate_approved"] * 100),
                xytext=(-150, 25), textcoords="offset points", fontsize=9,
                arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.set_xlabel("Approval rate (%)")
    ax.set_ylabel("Default rate among approved (%)")
    ax.set_title("Cut-off strategy: approval rate vs risk accepted")
    fig.savefig(path); plt.close(fig)
