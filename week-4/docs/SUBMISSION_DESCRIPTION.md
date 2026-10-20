# Submission description — Week 4

**Author:** Peram Dhanam Srinivas
**For the submission form. This description is 263 words, which satisfies the 200-word minimum.**

---

This Week 4 report documents the deployment of the churn classifier developed in Weeks 2 and 3 as a complete, documented and tested machine-learning application. The selected model, a Logistic Regression pipeline with C equal to 0.1 and balanced class weights, is serialised with Joblib into a 5.7 KiB artifact whose SHA-256 digest is recorded in a manifest, and served by a FastAPI application that loads it once at startup and answers requests in roughly twelve milliseconds. The service exposes a health probe reporting whether the model itself is resident in memory, a model-card endpoint carrying metrics, intended use and limitations, single-prediction and batch endpoints, and interactive OpenAPI documentation. Every request is validated by a Pydantic schema before any model code executes, so an unknown contract type, an out-of-range indicator or a misspelt field returns a 422 with a readable explanation instead of reaching scikit-learn. Because the model is linear, each prediction is accompanied by an exact log-odds decomposition showing which features raised or lowered the estimated risk. Eighteen contract tests cover validation, prediction, batch scoring and error handling. Benchmarking against a live server exposed a real defect: the first batch endpoint scored rows one at a time and managed 92 rows per second, barely better than sending separate requests. Vectorising that call raised throughput to 10,931 rows per second, a hundredfold improvement, while single requests were measured at 12.5 milliseconds with explanations and 7.4 milliseconds without. The report states plainly that the model was trained on synthetic teaching data and must not be used for consequential decisions.

---

## Links to submit alongside this description

| Field | Value |
|---|---|
| Report file | `week-4/reports/Week_4_AI_Project_Deployment_Report.docx` |
| Presentation | `week-4/reports/Week_4_Capstone_Presentation.pptx` |
| GitHub project URL | https://github.com/Dhanam33/Yuva_intern/tree/main/week-4 |

## Submission checklist

- [x] End-to-end project: preprocessing, training, testing and deployment with FastAPI
- [x] Serialisation with both Joblib and Pickle, compared and hashed
- [x] Working prediction API (`/predict`, `/predict/batch`) with interactive docs
- [x] 18 contract tests, all passing
- [x] Project documentation: report, model card, API reference, deployment notes
- [x] Presentation: 14 slides with speaker notes
- [x] Description of at least 200 words (263 words, above)
- [x] Author name on the cover, in the footer of every page, and in the file properties
- [x] GitHub project URL included — compulsory for technical tasks
- [x] Every figure, table and number reproducible from the repository
