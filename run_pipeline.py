"""End-to-end pipeline: SQL features -> scorecard + challenger -> validation -> outputs.

Usage:  python run_pipeline.py
"""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config, plots
from src.build_features import build_feature_mart
from src.evaluate import (bootstrap_gini_diff, calibration_table, cutoff_table,
                          fairness_table, psi, risk_grades, summary)
from src.scorecard import build_models, reason_codes


def main():
    for d in (config.FIGURES, config.TABLES, config.DOCS):
        d.mkdir(parents=True, exist_ok=True)

    print("1/5  Building features in DuckDB")
    mart = build_feature_mart()

    train, test = train_test_split(mart, test_size=config.TEST_SIZE, stratify=mart[config.TARGET],
                                   random_state=config.SEED)
    y_tr, y_te = train[config.TARGET].to_numpy(), test[config.TARGET].to_numpy()

    print("2/5  Fitting scorecard and challenger")
    bins, selection_log, card, challenger, ch_info = build_models(train)

    print("3/5  Validating on the hold-out sample")
    s_tr, s_te = card.score(train), card.score(test)
    pd_card_tr, pd_card_te = card.pd_from_score(s_tr), card.pd_from_score(s_te)
    pd_gbm_te = challenger.predict_proba(test[config.MODEL_FEATURES])[:, 1]
    pd_gbm_tr = challenger.predict_proba(train[config.MODEL_FEATURES])[:, 1]

    m_card_tr, m_card_te = summary(y_tr, pd_card_tr), summary(y_te, pd_card_te)
    m_gbm_tr, m_gbm_te = summary(y_tr, pd_gbm_tr), summary(y_te, pd_gbm_te)
    boot = bootstrap_gini_diff(y_te, pd_card_te, pd_gbm_te, n=config.N_BOOTSTRAP, seed=config.SEED)
    cal_card, cal_gbm = calibration_table(y_te, pd_card_te), calibration_table(y_te, pd_gbm_te)
    psi_val, psi_tab = psi(s_tr, s_te)
    cut = cutoff_table(y_te, s_te)
    target = cut.loc[(cut["approval_rate"] - config.TARGET_APPROVAL_RATE).abs().idxmin()]
    fair = fairness_table(test, s_te, target["cutoff_score"])
    # Same approval rate, challenger model: what does its extra Gini buy?
    cut_gbm = cutoff_table(y_te, -pd_gbm_te)
    target_gbm = cut_gbm.loc[(cut_gbm["approval_rate"] - config.TARGET_APPROVAL_RATE).abs().idxmin()]

    grades_tr = risk_grades(y_tr, s_tr, config.GRADE_CUTS)
    grades_te = risk_grades(y_te, s_te, config.GRADE_CUTS)
    # Example decline reasons: the lowest-scoring, median and a borderline hold-out customer
    order = np.argsort(s_te)
    examples = []
    for label, pos in [("lowest score", 0), ("just below 80% cut-off", int(np.searchsorted(np.sort(s_te), target["cutoff_score"])) - 1)]:
        row = test.iloc[order[pos]]
        reasons = reason_codes(card, row)
        examples.append({"case": label, "score": int(s_te[order[pos]]),
                         "reasons": "; ".join(f"{f} (-{p} pts)" for f, p in reasons)})
    examples = pd.DataFrame(examples)

    print("4/5  Writing tables and charts")
    grades_tr.to_csv(config.TABLES / "risk_grades_train.csv", index=False)
    grades_te.to_csv(config.TABLES / "risk_grades_holdout.csv", index=False)
    examples.to_csv(config.TABLES / "reason_code_examples.csv", index=False)
    pd.concat([b.table for b in bins.values()]).to_csv(config.TABLES / "woe_bins_all_features.csv", index=False)
    selection_log.to_csv(config.TABLES / "feature_selection.csv", index=False)
    card.points_table.to_csv(config.TABLES / "scorecard_points.csv", index=False)
    pd.concat([cal_card.assign(model="scorecard"), cal_gbm.assign(model="gradient_boosting")]) \
        .to_csv(config.TABLES / "calibration.csv", index=False)
    psi_tab.to_csv(config.TABLES / "psi_score.csv", index=False)
    cut.to_csv(config.TABLES / "cutoff_strategy.csv", index=False)
    fair.to_csv(config.TABLES / "fairness_monitoring.csv", index=False)

    plots.iv_chart(selection_log, config.FIGURES / "01_information_value.png")
    plots.score_distribution(s_te, y_te, config.FIGURES / "02_score_distribution.png")
    plots.roc_chart(y_te, pd_card_te, pd_gbm_te,
                    [f"Scorecard (AUC {m_card_te['auc']:.3f})", f"Gradient boosting (AUC {m_gbm_te['auc']:.3f})"],
                    config.FIGURES / "03_roc.png")
    plots.calibration_chart(cal_card, cal_gbm, config.FIGURES / "04_calibration.png")
    plots.cutoff_chart(cut, y_te.mean(), config.TARGET_APPROVAL_RATE, config.FIGURES / "05_cutoff_strategy.png")

    metrics = {
        "data": {"customers": int(len(mart)), "default_rate": float(mart[config.TARGET].mean()),
                 "train_n": int(len(train)), "test_n": int(len(test))},
        "scorecard": {"features": card.features, "n_features": len(card.features),
                      "train": m_card_tr, "test": m_card_te,
                      "score_min_test": float(s_te.min()), "score_max_test": float(s_te.max())},
        "challenger": {"train": m_gbm_tr, "test": m_gbm_te, **ch_info},
        "bootstrap_gini_challenger_minus_scorecard": boot,
        "psi_score_train_vs_test": psi_val,
        "risk_grades_holdout": grades_te.to_dict(orient="records"),
        "cutoff_at_target_approval": target.to_dict(),
        "challenger_at_target_approval": {k: float(target_gbm[k]) for k in
                                          ["approval_rate", "bad_rate_approved", "defaults_declined_share"]},
    }
    (config.ROOT / "outputs" / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    # data for the interactive cut-off explorer
    explorer = cut[["approval_rate", "cutoff_score", "bad_rate_approved",
                    "defaults_declined_share", "goods_declined_share"]].round(4)
    (config.DOCS / "cutoff_data.js").write_text(
        "window.CUTOFF_DATA = " + explorer.to_json(orient="records") + ";\n"
        f"window.BASE_RATE = {y_te.mean():.4f};\n")

    print("5/5  Done")
    print(json.dumps({"scorecard_test": m_card_te, "challenger_test": m_gbm_te,
                      "gini_diff": boot["diff_mean"], "gini_diff_ci": boot["diff_ci"],
                      "psi": psi_val, "features": card.features,
                      "at_80pct": target.to_dict(), "gbm_at_80pct": target_gbm.to_dict()}, indent=2, default=float))


if __name__ == "__main__":
    main()
