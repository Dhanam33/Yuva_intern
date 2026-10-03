"""Shared dataset schema and project paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"

RAW_DATA_PATH = DATA_DIR / "customer_churn_raw.csv"

ID_COLUMN = "customer_id"
TARGET_COLUMN = "churn"
TARGET_MAP = {"No": 0, "Yes": 1}

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_tickets",
    "senior_citizen",
]
CATEGORICAL_FEATURES = [
    "contract",
    "internet_service",
    "payment_method",
    "paperless_billing",
    "partner",
    "dependents",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Monthly charges are predicted without using total_charges, which is calculated
# from monthly charges and tenure and would therefore leak information.
REGRESSION_FEATURES = [
    "tenure_months",
    "support_tickets",
    "senior_citizen",
    "contract",
    "internet_service",
    "payment_method",
    "paperless_billing",
    "partner",
    "dependents",
]
