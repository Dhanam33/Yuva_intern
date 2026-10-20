# Submission description — Week 2

**Author:** Peram Dhanam Srinivas
**For the submission form. This description is 273 words, which satisfies the 200-word minimum.**

---

This Week 2 report presents a complete supervised machine-learning exercise implemented with scikit-learn, covering both regression and classification on a reproducible synthetic customer-churn dataset of 1,800 records. The data is divided into a stratified 80/20 split of 1,440 training and 360 held-out test rows, and every model receives an identical leakage-safe pipeline in which imputation, one-hot encoding and standardisation are fitted on training data only. Four classification algorithms are compared — Logistic Regression, Decision Tree, Random Forest and K-Nearest Neighbors — alongside two dummy baselines that establish the minimum acceptable performance. For regression, Linear Regression, Decision Tree, Random Forest and K-Nearest Neighbors regressors predict monthly charges, with total charges deliberately excluded because it is derived from the target and would leak the answer into the inputs. Evaluation reports accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Matthews correlation coefficient, log loss and Brier score, with confusion matrices exposing the two distinct kinds of error. The results show a genuine trade-off rather than a single winner: K-Nearest Neighbors achieves the highest accuracy at 0.744 but the lowest recall at 0.396, while Logistic Regression achieves the strongest ROC-AUC at 0.767 with recall above 0.65. Notably, the dummy baseline that never predicts churn scores 0.733 accuracy, demonstrating why accuracy alone is misleading on imbalanced data. Model selection uses five-fold stratified cross-validation on the training split, and the held-out test set is scored once. The report also quantifies the effect of feature scaling on distance-based methods, demonstrates overfitting by plotting training and test scores against tree depth, and analyses decision thresholds and class weighting as separate levers on the precision-recall trade-off.

---

## Links to submit alongside this description

| Field | Value |
|---|---|
| Report file | `week-2/reports/Week_2_Supervised_Machine_Learning_Report.docx` |
| GitHub project URL | https://github.com/Dhanam33/Yuva_intern/tree/main/week-2 |

## Submission checklist

- [x] Word report (`.docx`) generated from the pipeline's own JSON summary
- [x] Description of at least 200 words (273 words, above)
- [x] Author name on the cover, in the footer of every page, and in the file properties
- [x] GitHub project URL included — compulsory for technical tasks
- [x] Every figure and table reproducible from the repository
