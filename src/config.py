"""Project-wide settings. Change values here, not inside the modules."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "UCI_Credit_Card.csv"
RAW_SHA256 = "a0f0ab49d6326671d6cd83be5c88dcf18007025fe9a53ecd699119c871176ca1"
PROCESSED = ROOT / "data" / "processed"
SQL_DIR = ROOT / "sql"
FIGURES = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"
DOCS = ROOT / "docs"

SEED = 42
TEST_SIZE = 0.30
TARGET = "defaulted"

# Inputs the model may use: account behaviour and credit limit only.
MODEL_FEATURES = [
    "log_credit_limit",
    "status_latest",
    "worst_status_6m",
    "worst_status_3m",
    "months_late_6m",
    "current_late_streak",
    "months_paid_in_full_6m",
    "months_inactive_6m",
    "utilisation_latest",
    "utilisation_avg_6m",
    "utilisation_max_6m",
    "balance_trend",
    "repayment_ratio_avg",
    "months_no_payment",
]
# Protected characteristics (Equality Act 2010) are excluded from the model
# and used only to monitor outcomes.
MONITORING_ONLY = ["sex", "age", "marriage_code"]

# Scorecard binning and selection rules
MAX_BINS = 10
MIN_BIN_SHARE = 0.05
MIN_IV = 0.02
MAX_WOE_CORR = 0.75

# Score scaling: 600 points at good:bad odds of 50:1, odds double every 20 points
BASE_SCORE = 600
BASE_ODDS = 50
PDO = 20

# Risk grades (score lower bounds for D, C, B, A); E is everything below 480
GRADE_CUTS = (480, 510, 530, 545)

# Business scenario used for the cut-off and fairness tables
TARGET_APPROVAL_RATE = 0.80
N_BOOTSTRAP = 1000
