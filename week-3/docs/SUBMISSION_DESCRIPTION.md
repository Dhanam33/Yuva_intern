# Submission description — Week 3

**Author:** Peram Dhanam Srinivas
**For the submission form. This description is 230 words, which satisfies the 200-word minimum.**

---

This Week 3 report addresses both halves of the brief — unsupervised learning and model evaluation — using scikit-learn on the same reproducible synthetic customer-churn dataset of 1,800 records. Part A applies K-Means and agglomerative hierarchical clustering to a 21-column feature matrix, with principal component analysis used to examine dimensionality reduction. Cluster quality is judged using silhouette, Davies-Bouldin and Calinski-Harabasz indices, and the value of k is chosen only after demonstrating that the three disagree. Hierarchical clustering is compared across ward, complete, average and single linkage, which exposes a common trap: single linkage achieves the highest silhouette of any configuration while producing a one-customer cluster. Two independent algorithms recover almost the same three segments, with an adjusted Rand index of 0.732, which is evidence that the structure is real; yet neither relates to churn, where the index is 0.015 and the churn rate varies by only nine percentage points across segments. Part B builds a supervised classifier and evaluates it properly: the confusion matrix, precision, recall, F1 and ROC-AUC are computed from first principles, five-fold stratified cross-validation is used throughout, and GridSearchCV searches 37 hyperparameter combinations. Nested cross-validation separates the tuned score from an unbiased estimate. Results are reported without embellishment: tuning improved Logistic Regression by 0.001 ROC-AUC, nested cross-validation agreed to within 0.001, and the final held-out scores were 0.768 ROC-AUC with 0.656 recall.

---

## Links to submit alongside this description

| Field | Value |
|---|---|
| Report file | `week-3/reports/Week_3_Unsupervised_Learning_Model_Evaluation_Report.docx` |
| GitHub project URL | https://github.com/Dhanam33/Yuva_intern/tree/main/week-3 |

## Submission checklist

- [x] Word report (`.docx`) generated from the pipeline's own JSON summary
- [x] Description of at least 200 words (230 words, above)
- [x] Author name on the cover, in the footer of every page, and in the file properties
- [x] GitHub project URL included — compulsory for technical tasks
- [x] Clustering performed on a dataset (Part A, Sections 5–10)
- [x] Model evaluation report prepared (Part B, Sections 11–19)
- [x] Every figure and table reproducible from the repository
