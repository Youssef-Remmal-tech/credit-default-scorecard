# Model Validation Report: Credit Card Default Scorecard

| | |
|---|---|
| **Model** | Behavioural application scorecard, WoE logistic regression, 7 characteristics |
| **Purpose** | Rank existing card customers by probability of default next month; support an approval cut-off |
| **Challenger** | Histogram gradient boosting, 14 characteristics |
| **Data** | UCI Default of Credit Card Clients, 30,000 customers, Apr–Sep 2005 |
| **Validation sample** | Stratified 30% hold-out, 9,000 customers, 1,991 defaults, untouched until final scoring |
| **Overall opinion** | **Fit for purpose as a ranking and cut-off tool**, with three findings to address before production use |

---

## 1. Conceptual soundness

**Target.** Default on the next monthly payment, a binary flag supplied in the source data. All characteristics come from the six months before the target month, so there is no look-ahead.

**Characteristics.** 14 behavioural candidates were engineered in SQL. Selection followed three written rules, applied on the training sample only:

| Rule | Removed |
|---|---|
| Information value below 0.02 | none |
| WoE correlation above 0.75 with a stronger characteristic | worst status (3m, 6m), current late streak, max and latest utilisation |
| Positive coefficient (would reward riskier behaviour) | months paid in full, months inactive |

Every retained characteristic has a negative coefficient on WoE, so each bin's points move in the expected direction.

**Binning.** Continuous characteristics are monotonic in default rate by construction, with every bin holding at least 5% of customers. Repayment status is treated as categorical. That treatment is justified: "paid in full" (−1) shows a 17.0% default rate against 12.6% for "revolving" (0), so imposing an ordinal scale would misstate risk.

**Protected characteristics.** Sex, age and marital status are excluded from the model and used only for outcome monitoring.

## 2. Discrimination

| Metric | Train | Hold-out | Challenger hold-out |
|---|---|---|---|
| AUC | 0.766 | **0.745** | 0.776 |
| Gini | 0.533 | **0.490** (95% CI 0.465–0.516) | 0.552 (95% CI 0.528–0.576) |
| KS | 0.417 | **0.401** | 0.425 |

The train-to-hold-out drop of 0.043 Gini is modest for a 7-characteristic model. It suggests limited overfitting.

**Risk grades** (cut points fixed in configuration, verified on training data first):

| Grade | Score | Hold-out customers | Hold-out default rate | Training default rate |
|---|---|---|---|---|
| A | 545+ | 4,269 (47.4%) | 10.6% | 9.7% |
| B | 530–544 | 1,936 (21.5%) | 16.2% | 15.9% |
| C | 510–529 | 1,112 (12.4%) | 25.7% | 25.8% |
| D | 480–509 | 830 (9.2%) | 41.7% | 43.6% |
| E | below 480 | 853 (9.5%) | 69.5% | 69.9% |

Default rates rise strictly from A to E in both samples, and each grade's hold-out rate is within 2 percentage points of its training rate.

## 3. Calibration

Mean predicted PD and observed default rate by hold-out decile (full table: `outputs/tables/calibration.csv`):

- The top decile predicts 68.9% and observes 68.6%.
- Seven of the ten deciles (1–3 and 5–8) under-predict, by 1.1 to 2.0 percentage points.
- Decile 9 over-predicts by 3.1 points (42.4% predicted, 39.2% observed).

The Brier score is 0.140, against 0.136 for the challenger.

## 4. Stability

The PSI of the score between training and hold-out is **0.001** (threshold for review: 0.10). Both samples come from the same period, so this confirms sampling stability only, not stability over time.

## 5. Benchmarking against the challenger

| | Scorecard | Challenger |
|---|---|---|
| Gini gain (paired bootstrap, 1,000 resamples) | | **+0.062**, 95% CI [0.047, 0.078] |
| Default rate among approved, 80% approval | 14.2% | 13.7% |
| Defaults among 7,200 approved (per 9,000 applicants) | 1,023 | 986 |
| Explainability | Integer points, reason codes | Requires post-hoc explanation |

The challenger is statistically stronger. At the operating point the gain is 37 defaults per 9,000 applicants. The trade-off between that gain and explainability is a business decision. This report records it and takes no position on it.

## 6. Fairness monitoring at the 80% approval cut-off (511 points)

| Group | Approval rate | Ratio to highest group | Default rate if approved | AUC within group |
|---|---|---|---|---|
| Female | 81.6% | 1.00 | 13.3% | 0.744 |
| Male | 79.3% | 0.97 | 15.8% | 0.746 |
| 21–29 | 77.0% | 0.92 | 13.5% | 0.762 |
| 30–39 | 83.6% | 1.00 | 13.2% | 0.734 |
| 40–49 | 81.7% | 0.98 | 16.0% | 0.746 |
| 50+ | 79.2% | 0.95 | 17.2% | 0.725 |

All approval-rate ratios are above 0.80, and ranking power is similar across groups. Approved customers aged 50+ default at 17.2% against 13.2% for ages 30–39, which indicates the score understates risk for older customers at a given score.

## 7. Findings

| # | Finding | Severity | Recommendation |
|---|---|---|---|
| F1 | No out-of-time validation. The data covers one six-month window. | **High** | Validate on a later vintage before use. Recompute PSI monthly. |
| F2 | The score under-predicts for customers aged 50+ when approved (17.2% vs 13.2% for 30–39). | **Medium** | Monitor by age band each month. Investigate missing characteristics such as income or tenure before any recalibration. Do not add age as an input without a documented legal justification. |
| F3 | Under-prediction of 1.1 to 2.0 points in seven of the ten deciles, and over-prediction of 3.1 points in decile 9. | **Medium** | Apply a calibration overlay if the PD is used for pricing or provisioning. Ranking use is unaffected. |
| F4 | Repayment status codes are documented inconsistently in the source. | **Low** | Confirm the code definitions with the data owner. The categorical treatment already limits the impact. |
| F5 | Next-month horizon only. | **Low** (for this use) | A 12-month or lifetime PD is needed for IFRS 9 or capital models. |

## 8. Reproducibility

- `python run_pipeline.py` regenerates every figure and table in this report from the raw file, whose SHA-256 checksum is verified on load.
- `pytest` runs 8 tests covering SQL integrity, binning monotonicity, points arithmetic, PSI and cut-off logic.
- A clean re-run produced byte-identical output tables.
