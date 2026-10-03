"""Create the reproducible, intentionally imperfect teaching dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .schema import RAW_DATA_PATH


def generate_customer_churn_data(n_rows: int = 1800, seed: int = 2026) -> pd.DataFrame:
    """Return a deterministic synthetic customer-churn dataset.

    The data is explicitly synthetic, not a record of real customers. Churn is
    generated from plausible contract, tenure, service, and support signals;
    feature missingness is then injected to make preprocessing exercises real.
    """
    rng = np.random.default_rng(seed)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_rows,
        p=[0.56, 0.24, 0.20],
    )
    internet_service = rng.choice(
        ["Fiber optic", "DSL", "No"], size=n_rows, p=[0.46, 0.36, 0.18]
    )
    payment_method = rng.choice(
        ["Electronic check", "Credit card", "Bank transfer", "Mailed check"],
        size=n_rows,
        p=[0.31, 0.23, 0.27, 0.19],
    )
    paperless_billing = rng.choice(["Yes", "No"], size=n_rows, p=[0.72, 0.28])
    partner = rng.choice(["Yes", "No"], size=n_rows, p=[0.50, 0.50])
    dependents = rng.choice(["Yes", "No"], size=n_rows, p=[0.31, 0.69])
    senior_citizen = rng.binomial(1, 0.17, size=n_rows)
    tenure_months = np.rint(72 * rng.beta(1.65, 1.55, size=n_rows)).astype(int)

    service_addon = np.select(
        [internet_service == "Fiber optic", internet_service == "DSL"],
        [58.0, 34.0],
        default=0.0,
    )
    contract_discount = np.select(
        [contract == "One year", contract == "Two year"], [2.0, 5.0], default=0.0
    )
    monthly_charges = np.clip(
        22.0 + service_addon + rng.normal(0.0, 9.5, n_rows) - contract_discount,
        18.0,
        125.0,
    ).round(2)

    support_rate = 0.9 + 0.35 * (internet_service == "Fiber optic")
    support_tickets = np.minimum(rng.poisson(support_rate), 9).astype(int)
    total_charges = np.maximum(
        0.0,
        tenure_months * monthly_charges
        - np.where(contract == "Two year", 0.02 * tenure_months * monthly_charges, 0.0)
        + rng.normal(0.0, 90.0, n_rows),
    ).round(2)

    contract_risk = np.select(
        [contract == "Month-to-month", contract == "One year"],
        [0.90, -0.45],
        default=-1.15,
    )
    log_odds = (
        -1.80
        + contract_risk
        + 0.035 * (monthly_charges - 70.0)
        - 0.025 * (tenure_months - 36.0)
        + 0.24 * support_tickets
        + 0.34 * (internet_service == "Fiber optic")
        + 0.26 * (payment_method == "Electronic check")
        + 0.25 * senior_citizen
        - 0.25 * (dependents == "Yes")
        + 0.15 * (paperless_billing == "Yes")
    )
    churn_probability = 1.0 / (1.0 + np.exp(-log_odds))
    churn = np.where(rng.random(n_rows) < churn_probability, "Yes", "No")

    data = pd.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(1, n_rows + 1)],
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "support_tickets": support_tickets,
            "senior_citizen": senior_citizen,
            "contract": contract,
            "internet_service": internet_service,
            "payment_method": payment_method,
            "paperless_billing": paperless_billing,
            "partner": partner,
            "dependents": dependents,
            "churn": churn,
        }
    )

    # Missingness is seeded and deliberately limited to input features; labels
    # and identifiers remain intact so rows can be evaluated consistently.
    missing_rates = {
        "tenure_months": 0.035,
        "monthly_charges": 0.045,
        "total_charges": 0.050,
        "support_tickets": 0.040,
        "contract": 0.025,
        "internet_service": 0.020,
        "payment_method": 0.030,
        "paperless_billing": 0.020,
        "partner": 0.015,
        "dependents": 0.015,
    }
    for column, rate in missing_rates.items():
        data.loc[rng.random(n_rows) < rate, column] = np.nan

    return data


def write_raw_dataset(path=RAW_DATA_PATH, n_rows: int = 1800, seed: int = 2026) -> pd.DataFrame:
    """Generate and write the raw CSV, creating its parent directory as needed."""
    frame = generate_customer_churn_data(n_rows=n_rows, seed=seed)
    path = path if hasattr(path, "parent") else __import__("pathlib").Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame
