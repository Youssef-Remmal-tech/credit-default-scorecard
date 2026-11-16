-- 02_features.sql
-- Behavioural features built from the 6-month statement history.
-- Only account behaviour and the credit limit are used as model inputs.
-- Sex, age and marital status are kept for fairness monitoring only (see README).

CREATE OR REPLACE TABLE int_statement_enriched AS
SELECT
    s.*,
    c.credit_limit,
    -- The payment made in month m settles the bill issued one month earlier (month_idx + 1).
    LEAD(s.bill_amt) OVER w                         AS prior_bill_amt,
    s.bill_amt / NULLIF(c.credit_limit, 0)          AS utilisation,
    CASE WHEN s.repay_status >= 1 THEN 1 ELSE 0 END AS is_late
FROM stg_statements s
JOIN stg_customers c USING (customer_id)
WINDOW w AS (PARTITION BY s.customer_id ORDER BY s.month_idx);

-- Current delinquency streak: consecutive late months counting back from September.
-- Gaps-and-islands: the streak ends at the first on-time month.
CREATE OR REPLACE TABLE int_late_streak AS
WITH flagged AS (
    SELECT
        customer_id, month_idx, is_late,
        SUM(1 - is_late) OVER (PARTITION BY customer_id ORDER BY month_idx
                               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS on_time_seen
    FROM int_statement_enriched
)
SELECT customer_id, SUM(CASE WHEN on_time_seen = 0 THEN 1 ELSE 0 END) AS current_late_streak
FROM flagged
GROUP BY customer_id;

CREATE OR REPLACE TABLE feature_mart AS
WITH agg AS (
    SELECT
        customer_id,
        MAX(CASE WHEN month_idx = 1 THEN repay_status END)              AS status_latest,
        MAX(repay_status)                                               AS worst_status_6m,
        MAX(CASE WHEN month_idx <= 3 THEN repay_status END)             AS worst_status_3m,
        SUM(is_late)                                                    AS months_late_6m,
        SUM(CASE WHEN repay_status = -1 THEN 1 ELSE 0 END)              AS months_paid_in_full_6m,
        SUM(CASE WHEN repay_status = -2 THEN 1 ELSE 0 END)              AS months_inactive_6m,
        MAX(CASE WHEN month_idx = 1 THEN utilisation END)               AS utilisation_latest,
        AVG(utilisation)                                                AS utilisation_avg_6m,
        MAX(utilisation)                                                AS utilisation_max_6m,
        -- Balance trend: slope of bill against time (positive = balance growing), scaled by limit.
        REGR_SLOPE(bill_amt, -month_idx) / NULLIF(MAX(credit_limit), 0) AS balance_trend,
        -- Share of the prior bill repaid, averaged over months that had a positive prior bill.
        AVG(CASE WHEN prior_bill_amt > 0
                 THEN LEAST(paid_amt / prior_bill_amt, 1.5) END)        AS repayment_ratio_avg,
        SUM(CASE WHEN prior_bill_amt > 0 AND paid_amt = 0 THEN 1 ELSE 0 END) AS months_no_payment
    FROM int_statement_enriched
    GROUP BY customer_id
)
SELECT
    c.customer_id,
    LN(c.credit_limit)                  AS log_credit_limit,
    a.status_latest,
    a.worst_status_6m,
    a.worst_status_3m,
    a.months_late_6m,
    s.current_late_streak,
    a.months_paid_in_full_6m,
    a.months_inactive_6m,
    a.utilisation_latest,
    a.utilisation_avg_6m,
    a.utilisation_max_6m,
    COALESCE(a.balance_trend, 0)        AS balance_trend,
    COALESCE(a.repayment_ratio_avg, 1)  AS repayment_ratio_avg,  -- no positive prior bills -> nothing owed
    a.months_no_payment,
    -- monitoring-only attributes
    c.sex,
    c.age,
    c.marriage_code,
    c.education_code,
    c.defaulted
FROM stg_customers c
JOIN agg a USING (customer_id)
JOIN int_late_streak s USING (customer_id)
ORDER BY c.customer_id;
