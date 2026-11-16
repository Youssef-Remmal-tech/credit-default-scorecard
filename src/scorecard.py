"""Champion scorecard (WoE logistic regression) and challenger (gradient boosting)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from . import config
from .woe import fit_all, transform_all


def select_features(bins: dict, woe_train: pd.DataFrame) -> tuple[list, pd.DataFrame]:
    """Keep features with IV >= MIN_IV, then drop the weaker of any highly correlated pair."""
    iv = pd.Series({f: b.iv for f, b in bins.items()}).sort_values(ascending=False)
    log = []
    kept = []
    for f in iv.index:
        if iv[f] < config.MIN_IV:
            log.append((f, iv[f], "dropped: IV below threshold"))
            continue
        clash = [k for k in kept if abs(np.corrcoef(woe_train[f], woe_train[k])[0, 1]) > config.MAX_WOE_CORR]
        if clash:
            log.append((f, iv[f], f"dropped: WoE correlation > {config.MAX_WOE_CORR} with {clash[0]}"))
            continue
        kept.append(f)
        log.append((f, iv[f], "kept"))
    return kept, pd.DataFrame(log, columns=["feature", "iv", "decision"])


def fit_champion(woe_train: pd.DataFrame, y: pd.Series, features: list) -> tuple[LogisticRegression, list, list]:
    """Logistic regression on WoE.

    Higher WoE means lower risk, so every coefficient for P(default) should be negative.
    A positive coefficient means the feature is only adjusting for overlap with others;
    it would give customers points for looking riskier, so it is removed and the model refitted.
    """
    feats, dropped = list(features), []
    while True:
        model = LogisticRegression(C=1.0, max_iter=1000)
        model.fit(woe_train[feats], y)
        coefs = pd.Series(model.coef_[0], index=feats)
        wrong = coefs[coefs > 0]
        if wrong.empty:
            return model, feats, dropped
        worst = wrong.idxmax()
        feats.remove(worst)
        dropped.append(worst)


class Scorecard:
    """Turns the WoE logistic model into an additive points table (integer points, as lenders use)."""

    def __init__(self, bins: dict, model: LogisticRegression, features: list):
        self.bins = {f: bins[f] for f in features}
        self.model = model
        self.features = features
        self.factor = config.PDO / np.log(2)
        self.offset = config.BASE_SCORE - self.factor * np.log(config.BASE_ODDS)
        n = len(features)
        b0 = model.intercept_[0]
        rows = []
        for f, beta in zip(features, model.coef_[0]):
            t = self.bins[f].table
            pts = -(beta * t["woe"] + b0 / n) * self.factor + self.offset / n
            for _, r in t.assign(points=pts).iterrows():
                rows.append({"feature": f, "bin": r["bin"], "share": r["share"],
                             "bad_rate": r["bad_rate"], "woe": r["woe"], "points": int(round(r["points"]))})
        self.points_table = pd.DataFrame(rows)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        total = np.zeros(len(df))
        for f in self.features:
            pts = self.points_table.loc[self.points_table["feature"] == f, "points"].to_numpy()
            total += pts[self.bins[f].bin_index(df[f])]
        return total

    def pd_from_score(self, score: np.ndarray) -> np.ndarray:
        log_odds_good = (score - self.offset) / self.factor
        return 1 / (1 + np.exp(log_odds_good))

    def predict_pd(self, df: pd.DataFrame) -> np.ndarray:
        woe = transform_all(df, self.bins)[self.features]
        return self.model.predict_proba(woe)[:, 1]


def fit_challenger(X: pd.DataFrame, y: pd.Series) -> tuple[HistGradientBoostingClassifier, dict]:
    grid = {
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "max_iter": [200, 400],
        "min_samples_leaf": [50],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    search = GridSearchCV(
        HistGradientBoostingClassifier(random_state=config.SEED, early_stopping=False),
        grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(X, y)
    return search.best_estimator_, {"best_params": search.best_params_, "cv_auc": search.best_score_}


def build_models(train: pd.DataFrame):
    y = train[config.TARGET]
    bins = fit_all(train, config.MODEL_FEATURES, config.TARGET,
                   max_bins=config.MAX_BINS, min_share=config.MIN_BIN_SHARE)
    woe_train = transform_all(train, bins)
    selected, selection_log = select_features(bins, woe_train)
    model, final_feats, sign_dropped = fit_champion(woe_train, y, selected)
    selection_log.loc[selection_log["feature"].isin(sign_dropped), "decision"] = \
        "dropped: counter-intuitive coefficient sign in the regression"
    card = Scorecard(bins, model, final_feats)
    challenger, ch_info = fit_challenger(train[config.MODEL_FEATURES], y)
    return bins, selection_log, card, challenger, ch_info


def reason_codes(card: "Scorecard", row: pd.Series, top: int = 3) -> list[tuple[str, int]]:
    """Adverse-action style reasons: the features where this customer lost the most points
    compared with the best bin of that feature."""
    lost = []
    for f in card.features:
        t = card.points_table[card.points_table["feature"] == f].reset_index(drop=True)
        i = int(card.bins[f].bin_index(pd.Series([row[f]]))[0])
        lost.append((f, int(t["points"].max() - t.loc[i, "points"])))
    lost.sort(key=lambda x: -x[1])
    return [r for r in lost[:top] if r[1] > 0]
