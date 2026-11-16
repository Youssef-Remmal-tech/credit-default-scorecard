-- 03_checks.sql
-- Data-quality checks. Each query returns 0 when the check passes.
SELECT 'duplicate_customers' AS check_name, COUNT(*) - COUNT(DISTINCT customer_id) AS failures FROM stg_customers
UNION ALL
SELECT 'statements_not_6_per_customer', COUNT(*) FROM (
    SELECT customer_id FROM stg_statements GROUP BY customer_id HAVING COUNT(*) <> 6)
UNION ALL
SELECT 'mart_row_mismatch', ABS((SELECT COUNT(*) FROM feature_mart) - (SELECT COUNT(*) FROM stg_customers))
UNION ALL
SELECT 'null_model_features', COUNT(*) FROM feature_mart
WHERE log_credit_limit IS NULL OR status_latest IS NULL OR utilisation_latest IS NULL
   OR repayment_ratio_avg IS NULL OR balance_trend IS NULL
UNION ALL
SELECT 'target_not_binary', COUNT(*) FROM stg_customers WHERE defaulted NOT IN (0, 1);
