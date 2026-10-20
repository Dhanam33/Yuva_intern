"""Per-prediction explanations for a linear model.

For logistic regression the log-odds decompose exactly:

    logit(p) = intercept + sum_i(coef_i * x_i)

so each transformed feature contributes a number of log-odds that can be
reported without approximation. This is the main practical advantage of a
linear model in production: the explanation is the model, not a post-hoc
approximation of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import CATEGORICAL_FEATURES, NUMERIC_FEATURES

TYPE_LABELS = [
    (0.15, "low"),
    (0.40, "medium"),
    (1.01, "high"),
]


def risk_band(probability: float) -> str:
    """Bucket a probability into a label a human can act on."""
    for ceiling, label in TYPE_LABELS:
        if probability < ceiling:
            return label
    return "high"


def _display(transformed_name: str, record: dict) -> tuple[str, object]:
    """Map a transformed column back to a readable feature and its value.

    ``contract_Month-to-month`` becomes ``("contract = Month-to-month", "Month-to-month")``
    so the explanation reads like a sentence rather than like a matrix index.
    """
    if transformed_name in NUMERIC_FEATURES:
        return transformed_name, record.get(transformed_name)

    for column in CATEGORICAL_FEATURES:
        if transformed_name.startswith(column + "_"):
            level = transformed_name[len(column) + 1:]
            return f"{column} = {level}", record.get(column)

    return transformed_name, record.get(transformed_name)


def explain(pipeline, record: dict, top_k: int = 5) -> list[dict]:
    """Return the features that moved this prediction the most.

    Contributions are in log-odds and sum, with the intercept, to the model's
    log-odds for the positive class.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    frame = pd.DataFrame([record])
    transformed = np.asarray(preprocessor.transform(frame))[0]
    names = preprocessor.get_feature_names_out()
    coefficients = model.coef_[0]

    contributions = transformed * coefficients
    order = np.argsort(-np.abs(contributions))[:top_k]

    results = []
    for index in order:
        name = str(names[index])
        label, value = _display(name, record)
        amount = float(contributions[index])
        if abs(amount) < 1e-12:
            continue
        results.append(
            {
                "feature": label,
                "value": value if not isinstance(value, float) else round(value, 2),
                "contribution": round(amount, 4),
                "direction": "increases risk" if amount > 0 else "decreases risk",
            }
        )
    return results


def log_odds_breakdown(pipeline, record: dict) -> dict:
    """Expose the arithmetic so a reader can verify the explanation."""
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    transformed = np.asarray(preprocessor.transform(pd.DataFrame([record])))[0]
    contributions = transformed * model.coef_[0]
    intercept = float(model.intercept_[0])
    return {
        "intercept": round(intercept, 4),
        "sum_of_contributions": round(float(contributions.sum()), 4),
        "log_odds": round(intercept + float(contributions.sum()), 4),
    }
