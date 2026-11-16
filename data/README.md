# Data

`raw/UCI_Credit_Card.csv` is the UCI **Default of Credit Card Clients** dataset (Yeh & Lien, 2009), in the CSV layout Kaggle distributes as [uciml/default-of-credit-card-clients-dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset).

- 30,000 rows, 25 columns, licence CC BY 4.0
- SHA-256: `a0f0ab49d6326671d6cd83be5c88dcf18007025fe9a53ecd699119c871176ca1` (checked on every run)

| Column | Meaning |
|---|---|
| `LIMIT_BAL` | Credit limit (NT dollars) |
| `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` | Demographics. Sex, age and marriage are used only for fairness monitoring |
| `PAY_0`, `PAY_2` … `PAY_6` | Repayment status, Sep → Apr 2005. −2 no use, −1 paid in full, 0 revolving, k = k months late |
| `BILL_AMT1` … `BILL_AMT6` | Statement balance, Sep → Apr |
| `PAY_AMT1` … `PAY_AMT6` | Amount paid, Sep → Apr |
| `default.payment.next.month` | Target: 1 = defaulted in October 2005 |

`processed/` is created by the pipeline and is not committed.
