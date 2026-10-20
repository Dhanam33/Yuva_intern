"""Shared paths and constants for the Week 4 capstone.

Importing this module must never train a model or touch the network, so the
API, the tests and the training script can all depend on it safely.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Filesystem layout
# --------------------------------------------------------------------------
WEEK_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = WEEK_DIR / "data" / "customer_churn_raw.csv"
OUTPUT_DIR = WEEK_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "model"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = WEEK_DIR / "reports"
STATIC_DIR = WEEK_DIR / "static"
DOCS_DIR = WEEK_DIR / "docs"

MODEL_PATH = MODEL_DIR / "customer_churn_pipeline.joblib"
MODEL_PICKLE_PATH = MODEL_DIR / "customer_churn_pipeline.pkl"
MODEL_CARD_PATH = OUTPUT_DIR / "model_card.json"
MANIFEST_PATH = OUTPUT_DIR / "model_manifest.json"

# --------------------------------------------------------------------------
# Problem definition
# --------------------------------------------------------------------------
ID_COLUMN = "customer_id"
TARGET_COLUMN = "churn"
TARGET_MAP = {"No": 0, "Yes": 1}
POSITIVE_LABEL = "Yes"
NEGATIVE_LABEL = "No"

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

CATEGORY_LEVELS = {
    "contract": ["Month-to-month", "One year", "Two year"],
    "internet_service": ["DSL", "Fiber optic", "No"],
    "payment_method": ["Electronic check", "Credit card", "Bank transfer", "Mailed check"],
    "paperless_billing": ["Yes", "No"],
    "partner": ["Yes", "No"],
    "dependents": ["Yes", "No"],
}

# Observed ranges in the training data. They are documentation, not validation:
# the API accepts anything within the wider, physically sensible bounds below.
OBSERVED_RANGES = {
    "tenure_months": (1.0, 71.0),
    "monthly_charges": (18.0, 106.24),
    "total_charges": (0.0, 6881.94),
    "support_tickets": (0.0, 6.0),
    "senior_citizen": (0.0, 1.0),
}

# --------------------------------------------------------------------------
# Model identity
# --------------------------------------------------------------------------
MODEL_NAME = "customer-churn-classifier"
MODEL_VERSION = "4.0.0"
MODEL_DESCRIPTION = (
    "Logistic Regression classifier predicting 90-day customer churn from "
    "account, contract and service features."
)

# Hyperparameters selected by GridSearchCV in Week 3 (5-fold CV, 0.806 ROC-AUC).
BEST_PARAMS = {"C": 0.1, "penalty": "l2", "solver": "lbfgs", "class_weight": "balanced"}
DEFAULT_PARAMS = {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "class_weight": "balanced"}

# Operating threshold. 0.5 is the neutral default; the cost analysis in
# Week 2 showed 0.40 maximises F1 on this problem.
DEFAULT_THRESHOLD = 0.5
F1_OPTIMAL_THRESHOLD = 0.40

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
