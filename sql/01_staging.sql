-- 01_staging.sql
-- Load the raw UCI file and split it into two tidy tables:
--   stg_customers   one row per customer (static attributes + target)
--   stg_statements  one row per customer per month (6 months, long format)
-- Undocumented category codes are folded into "other" rather than dropped.

CREATE OR REPLACE TABLE raw_credit AS
SELECT * FROM read_csv_auto(getvariable('raw_path'), header = true);

CREATE OR REPLACE TABLE stg_customers AS
SELECT
    "ID"                                        AS customer_id,
    CAST(LIMIT_BAL AS DOUBLE)                   AS credit_limit,
    CASE SEX WHEN 1 THEN 'male' WHEN 2 THEN 'female' END AS sex,
    CASE WHEN EDUCATION IN (1, 2, 3) THEN EDUCATION ELSE 4 END AS education_code,  -- 0,5,6 undocumented -> 4 (other)
    CASE WHEN MARRIAGE IN (1, 2) THEN MARRIAGE ELSE 3 END AS marriage_code,         -- 0 undocumented -> 3 (other)
    AGE                                         AS age,
    "default.payment.next.month"                AS defaulted
FROM raw_credit;

-- Wide -> long. month_idx 1 = September 2005 (most recent) ... 6 = April 2005.
-- repay_status: -2 no consumption, -1 paid in full, 0 revolving credit, k = k months late.
CREATE OR REPLACE TABLE stg_statements AS
WITH long AS (
    UNPIVOT raw_credit
    ON (PAY_0, BILL_AMT1, PAY_AMT1) AS "1",
       (PAY_2, BILL_AMT2, PAY_AMT2) AS "2",
       (PAY_3, BILL_AMT3, PAY_AMT3) AS "3",
       (PAY_4, BILL_AMT4, PAY_AMT4) AS "4",
       (PAY_5, BILL_AMT5, PAY_AMT5) AS "5",
       (PAY_6, BILL_AMT6, PAY_AMT6) AS "6"
    INTO NAME month_idx VALUE repay_status, bill_amt, paid_amt
)
SELECT
    "ID"                            AS customer_id,
    CAST(month_idx AS INTEGER)      AS month_idx,
    CAST(repay_status AS INTEGER)   AS repay_status,
    CAST(bill_amt AS DOUBLE)        AS bill_amt,
    CAST(paid_amt AS DOUBLE)        AS paid_amt
FROM long;
