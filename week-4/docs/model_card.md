# Model card — customer churn classifier

| Field | Value |
|---|---|
| Model name | `customer-churn-classifier` |
| Version | `4.0.0` |
| Algorithm | LogisticRegression (scikit-learn Pipeline) |
| Trained at | 2026-10-20T13:39:26+00:00 |
| Training rows | 1,440 |
| Threshold | 0.50 (caller-overridable) |
| Artifact | `customer_churn_pipeline.joblib`, 5.7 KiB |

## Intended use

Ranking existing customers by churn risk so a retention team can prioritise outreach. Decision support only.

## Features

`tenure_months`, `monthly_charges`, `total_charges`, `support_tickets`, `senior_citizen`, `contract`, `internet_service`, `payment_method`, `paperless_billing`, `partner`, `dependents`

## Hyperparameters

```json
{
  "C": 0.1,
  "penalty": "l2",
  "solver": "lbfgs",
  "class_weight": "balanced"
}
```

Selected by GridSearchCV, 5-fold CV, selected in Week 3.

## Metrics

Measured on 360 held-out rows, scored once.

| Metric | Value |
|---|---|
| accuracy | 0.678 |
| balanced accuracy | 0.671 |
| precision | 0.432 |
| recall | 0.656 |
| f1 | 0.521 |
| roc auc | 0.768 |
| pr auc | 0.517 |
| mcc | 0.308 |
| log loss | 0.571 |
| brier | 0.196 |
| threshold | 0.500 |

Confusion matrix: TN 181, FP 83, FN 33, TP 63.

## Limitations

- Trained on synthetic teaching data; not validated on real customers.
- Do not use for credit, pricing, hiring or any consequential decision.
- No subgroup fairness audit has been performed.
- Predictions are only as current as the features supplied.
- Performance drifts if customer behaviour changes.

## Reproducing this card

```bash
python src/train.py
```

The card is written by that command. Nothing in it is typed by hand.
