# Yuva Internship — Four-Week Machine-Learning Project

This repository contains completed, runnable deliverables for the four-week Python and machine-learning task: data preprocessing, supervised learning, clustering/model evaluation, and an end-to-end prediction API.

**GitHub project URL:** https://github.com/Dhanam33/Yuva_intern
**Working branch for this submission:** `arena/01a1004b-yuva-intern`

> **Dataset note:** the bundled 1,800-row customer-churn CSV is reproducible synthetic teaching data, not real customer information. It intentionally contains missing values and categorical fields so all requested preprocessing steps can be demonstrated. Model scores must not be interpreted as evidence of real-world business performance.

## Deliverables

| Week | Completed work | Report / artifacts |
|---|---|---|
| 1 — Python, NumPy, Pandas, preprocessing | Data loading, EDA, duplicate/ID audit, missing-value imputation, one-hot encoding, scaling, mutual-information feature selection | `reports/Week_1_Data_Preprocessing_Report.docx`; cleaned/encoded exports and charts in `outputs/` |
| 2 — supervised learning | Linear Regression for numeric monthly charges; Logistic Regression, Decision Tree, Random Forest, and KNN for churn; held-out comparison with metrics | `reports/Week_2_Supervised_Learning_Report.docx`; classification and regression metric files in `outputs/` |
| 3 — unsupervised learning and evaluation | K-Means and Agglomerative clustering; silhouette comparisons; PCA visualization; 5-fold stratified CV; regularization tuning; confusion matrix, precision, recall, F1, ROC-AUC | `reports/Week_3_Unsupervised_Learning_and_Evaluation_Report.docx`; evaluation artifacts in `outputs/` |
| 4 — deployment and capstone | Full preprocessing/model pipeline serialized with Joblib; FastAPI health and prediction endpoints; example request; project documentation and slide deck | `reports/Week_4_AI_Deployment_Capstone_Report.docx`; `reports/Capstone_Presentation.pptx`; `artifacts/churn_model.joblib` |

Each Word report includes its own **200+ word submission description**. The descriptions are also provided together in `reports/SUBMISSION_DESCRIPTIONS.md` for copying into the submission form. `docs/python_ml_fundamentals.md` summarizes the Python, NumPy, Pandas, and ML concepts used.

## Run locally

Requires Python 3.11 or newer.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

# Reproduce the deliverables in order
python scripts/generate_dataset.py
python -m yuva_ml.week1_preprocessing
python -m yuva_ml.week2_supervised
python -m yuva_ml.week3_unsupervised
python -m yuva_ml.train
python scripts/build_reports.py
pytest
```

The generation seed, split seed, and preprocessing details are fixed for reproducibility. The Week 1 all-row transformed CSV and mutual-information ranking are learning artifacts only; Week 2/3 model evaluation fits preprocessing inside training data and CV folds to prevent information leakage. Feature selection, if used for modeling, should also be fitted within those folds.

## Run the API

After installing dependencies and training/including the artifact:

```bash
uvicorn yuva_ml.api:app --app-dir src --host 0.0.0.0 --port 8000
```

- Health check: `GET http://127.0.0.1:8000/health`
- Interactive API documentation: `http://127.0.0.1:8000/docs`
- Prediction: `POST /predict` with one or more customer records (up to 100 per request). Missing fields are accepted and imputed; unknown categories are tolerated.

Example:

```json
{
  "instances": [
    {
      "customer_id": "demo-001",
      "tenure_months": 4,
      "monthly_charges": 105.0,
      "support_tickets": 4,
      "contract": "Month-to-month",
      "internet_service": "Fiber optic",
      "payment_method": "Electronic check"
    }
  ]
}
```

Example response:

```json
{
  "predictions": [
    {
      "customer_id": "demo-001",
      "prediction": "Yes",
      "churn_probability": 0.964383
    }
  ]
}
```

The API is a local educational deployment, not a public production service. Before real use, replace the synthetic data, validate performance on representative data, choose a decision threshold with business costs in mind, add authentication/rate limiting/monitoring, and manage model-version compatibility. Only load Joblib artifacts from trusted sources.

## Repository layout

```text
src/yuva_ml/       Shared preprocessing, weekly exercises, training, and FastAPI app
scripts/           Dataset/report generation scripts
data/              Reproducible raw synthetic CSV and data notes
outputs/            Cleaned data, metrics, CV/clustering results, and plots
artifacts/          Serialized fitted pipeline and model metadata
reports/            Four .docx reports, descriptions, and capstone .pptx
tests/               Automated preprocessing and API tests
```
