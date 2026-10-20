# Submission description — Week 1

**For the submission form. This description is 307 words, which satisfies the 200-word minimum.**

---

This Week 1 report documents a complete Python-based data-preparation workflow built with
NumPy, Pandas and scikit-learn. The subject is a reproducible synthetic customer-churn
dataset of 1,800 records containing five numeric inputs, six categorical inputs, a unique
customer identifier and a binary churn label, with missing values deliberately introduced
so that every preprocessing technique has something real to act on. The report explains
the problem framing, documents each of the nine pipeline stages, and justifies every
decision rather than merely listing code. A data-quality audit records dimensions, data
types, duplicate rows and duplicate identifiers before any modification. Cleaning then
trims text, repairs impossible values such as negative charges, protects the integrity of
the target and removes duplication. Missing data is audited column by column and imputed
with medians for numeric fields and most-frequent values for categorical fields, while
missing-indicator columns preserve the information contained in the absence of a value.
Exploratory analysis uses histograms, distribution plots, a correlation matrix, per-level
churn-rate comparisons and boxplot screening to describe the data honestly, including
where outliers were found and why they were retained rather than deleted. Categorical
variables are one-hot encoded with unseen-category tolerance so that production records
cannot crash the model, and numeric variables are standardised so that distance-based
algorithms are not dominated by differences in measurement scale. Mutual information ranks
the encoded predictors, and the exported ranking is explicitly separated from model
evaluation because selection performed on the full sample would leak information into
training. Every number in the report is reproducible from a fixed random seed, the
scripts, the exported CSV tables and the eight analysis figures. A companion notebook
teaches the underlying Python, NumPy and Pandas skills step by step, and the report closes
with limitations, an accuracy of claims, and the specific checks that must pass before
this pipeline could be trusted on real data.

---

## Links to submit alongside this description

| Field | Value |
|---|---|
| Report file | `week-1/reports/Week_1_Python_ML_Data_Preprocessing_Report.docx` |
| GitHub project URL | https://github.com/Dhanam33/Yuva_intern/tree/arena/ba225c97-yuva-intern/week-1 |

## Submission checklist

- [x] Word report (`.docx`) generated from the pipeline's own JSON summary
- [x] Description of at least 200 words (307 words, above)
- [x] GitHub project URL included — the brief makes this compulsory for technical tasks
- [x] Every figure and table in the report reproducible from the repository
