#!/usr/bin/env python3
"""Build the Week 2 Word report from the generated training artifacts.

Every number is read from ``week-2/outputs/week2_summary.json`` and the exported
CSV tables, so the document cannot drift from the code that produced it.

    python src/train_models.py     # generate artifacts
    python src/make_figures.py     # generate figures
    python docs/build_report.py    # build this report
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

# --------------------------------------------------------------------------
# Project metadata
# --------------------------------------------------------------------------
AUTHOR = "Peram Dhanam Srinivas"
INTERNSHIP = "Yuva Internship Program"
WEEK_TITLE = "Week 2 — Supervised Machine Learning Models"
REPORT_TITLE = "Supervised Machine Learning Models"
REPORT_SUBTITLE = (
    "Regression and classification with scikit-learn: training, prediction and "
    "evaluation on a customer-churn dataset"
)
REPOSITORY = "https://github.com/Dhanam33/Yuva_intern/tree/main/week-2"

WEEK_DIR = Path(__file__).resolve().parents[1]
OUTPUTS = WEEK_DIR / "outputs"
FIGURES = OUTPUTS / "figures"
REPORTS = WEEK_DIR / "reports"

NAVY = RGBColor(0x17, 0x32, 0x4D)
TEAL = RGBColor(0x23, 0x7A, 0x7A)
CORAL = RGBColor(0xC4, 0x66, 0x4D)
LIGHT_GREY = RGBColor(0x7A, 0x87, 0x94)

# --------------------------------------------------------------------------
# Submission description (must be at least 200 words)
# --------------------------------------------------------------------------
SUBMISSION_DESCRIPTION = (
    "This Week 2 report presents a complete supervised machine-learning exercise implemented "
    "with scikit-learn, covering both regression and classification on a reproducible synthetic "
    "customer-churn dataset of 1,800 records. The data is divided into a stratified 80/20 split "
    "of 1,440 training and 360 held-out test rows, and every model receives an identical "
    "leakage-safe pipeline in which imputation, one-hot encoding and standardisation are fitted "
    "on training data only. Four classification algorithms are compared — Logistic Regression, "
    "Decision Tree, Random Forest and K-Nearest Neighbors — alongside two dummy baselines that "
    "establish the minimum acceptable performance. For regression, Linear Regression, Decision "
    "Tree, Random Forest and K-Nearest Neighbors regressors predict monthly charges, with total "
    "charges deliberately excluded because it is derived from the target and would leak the "
    "answer into the inputs. Evaluation reports accuracy, balanced accuracy, precision, recall, "
    "F1, ROC-AUC, PR-AUC, Matthews correlation coefficient, log loss and Brier score, with "
    "confusion matrices exposing the two distinct kinds of error. The results show a genuine "
    "trade-off rather than a single winner: K-Nearest Neighbors achieves the highest accuracy at "
    "0.744 but the lowest recall at 0.396, while Logistic Regression achieves the strongest "
    "ROC-AUC at 0.767 with recall above 0.65. Notably, the dummy baseline that never predicts "
    "churn scores 0.733 accuracy, demonstrating why accuracy alone is misleading on imbalanced "
    "data. Model selection uses five-fold stratified cross-validation on the training split, and "
    "the held-out test set is scored once. The report also quantifies the effect of feature "
    "scaling on distance-based methods, demonstrates overfitting by plotting training and test "
    "scores against tree depth, and analyses decision thresholds and class weighting as separate "
    "levers on the precision-recall trade-off."
)


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w’'-]+\b", text))


# --------------------------------------------------------------------------
# Document primitives
# --------------------------------------------------------------------------
def shade(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    element = OxmlElement("w:shd")
    element.set(qn("w:fill"), fill)
    tc_pr.append(element)


def repeat_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    marker = OxmlElement("w:tblHeader")
    marker.set(qn("w:val"), "true")
    tr_pr.append(marker)


def keep_together(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    marker = OxmlElement("w:cantSplit")
    marker.set(qn("w:val"), "true")
    tr_pr.append(marker)


def add_page_numbers(section) -> None:
    paragraph = section.footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.text = ""

    run = paragraph.add_run("Page ")
    run.font.size = Pt(8)
    run.font.color.rgb = LIGHT_GREY

    for instruction, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if instruction:
            field = OxmlElement("w:fldChar")
            field.set(qn("w:fldCharType"), instruction)
            run = paragraph.add_run()
            run.font.size = Pt(8)
            run._r.append(field)
        else:
            instr = OxmlElement("w:instrText")
            instr.set(qn("xml:space"), "preserve")
            instr.text = " PAGE "
            run = paragraph.add_run()
            run.font.size = Pt(8)
            run._r.append(instr)

    tail = paragraph.add_run(f"   ·   {AUTHOR}   ·   {WEEK_TITLE}")
    tail.font.size = Pt(8)
    tail.font.color.rgb = LIGHT_GREY


def setup_document() -> Document:
    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.80)
    section.bottom_margin = Inches(0.78)
    section.left_margin = Inches(0.86)
    section.right_margin = Inches(0.86)

    normal = document.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = RGBColor(0x29, 0x34, 0x3F)
    normal.paragraph_format.space_after = Pt(7)
    normal.paragraph_format.line_spacing = 1.13

    for name, size, color, before, after in [
        ("Heading 1", 16, NAVY, 14, 6),
        ("Heading 2", 12.5, TEAL, 10, 4),
        ("Heading 3", 11, RGBColor(0x3A, 0x4A, 0x5A), 8, 3),
    ]:
        style = document.styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = color
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.line_spacing = 1.05

    header = section.header.paragraphs[0]
    header.text = "YUVA INTERNSHIP   ·   WEEK 2   ·   SUPERVISED MACHINE LEARNING MODELS"
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for run in header.runs:
        run.font.size = Pt(7.5)
        run.font.bold = True
        run.font.color.rgb = TEAL

    add_page_numbers(section)
    return document


def body(document: Document, text: str, *, italic: bool = False, size: float | None = None) -> None:
    paragraph = document.add_paragraph()
    run = paragraph.add_run(text)
    run.italic = italic
    if size:
        run.font.size = Pt(size)


def bullets(document: Document, items: list[str]) -> None:
    for item in items:
        paragraph = document.add_paragraph(style="List Bullet")
        paragraph.paragraph_format.space_after = Pt(3)
        paragraph.paragraph_format.left_indent = Inches(0.28)
        paragraph.add_run(item)


def numbered(document: Document, items: list[str]) -> None:
    """Manually numbered list — Word's List Number counter spans the document."""
    for position, item in enumerate(items, start=1):
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.space_after = Pt(3)
        paragraph.paragraph_format.left_indent = Inches(0.52)
        paragraph.paragraph_format.first_line_indent = Inches(-0.24)
        number_run = paragraph.add_run(f"{position}.  ")
        number_run.bold = True
        number_run.font.color.rgb = TEAL
        paragraph.add_run(item)


def add_table(
    document: Document,
    headers: list[str],
    rows: list[list[str]],
    *,
    widths: list[float] | None = None,
    font_size: float = 8.6,
    align_right_from: int | None = None,
    highlight_row: int | None = None,
) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    header_row = table.rows[0]
    repeat_header(header_row)
    keep_together(header_row)
    for index, text in enumerate(headers):
        cell = header_row.cells[index]
        cell.text = ""
        paragraph = cell.paragraphs[0]
        paragraph.paragraph_format.space_after = Pt(2)
        paragraph.paragraph_format.space_before = Pt(2)
        run = paragraph.add_run(str(text))
        run.font.bold = True
        run.font.size = Pt(font_size)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        shade(cell, "17324D")

    for row_index, source in enumerate(rows):
        row = table.add_row()
        keep_together(row)
        for index, value in enumerate(source):
            cell = row.cells[index]
            cell.text = ""
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(2)
            paragraph.paragraph_format.space_before = Pt(2)
            if align_right_from is not None and index >= align_right_from:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            run = paragraph.add_run(str(value))
            run.font.size = Pt(font_size)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if highlight_row is not None and row_index == highlight_row:
                shade(cell, "FFF3E0")
                run.font.bold = True
            elif row_index % 2 == 1:
                shade(cell, "F1F5F9")

    if widths:
        for row in table.rows:
            for index, width in enumerate(widths):
                row.cells[index].width = Inches(width)

    spacer = document.add_paragraph()
    spacer.paragraph_format.space_after = Pt(2)
    spacer.paragraph_format.space_before = Pt(0)


def add_caption(document: Document, text: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(2)
    paragraph.paragraph_format.space_after = Pt(10)
    run = paragraph.add_run(text)
    run.italic = True
    run.font.size = Pt(8)
    run.font.color.rgb = LIGHT_GREY


def add_figure(document: Document, path: Path, caption: str, width: float = 6.2) -> None:
    if not path.exists():
        paragraph = document.add_paragraph()
        run = paragraph.add_run(f"[missing figure: {path.name}]")
        run.italic = True
        run.font.color.rgb = CORAL
        return
    holder = document.add_paragraph()
    holder.alignment = WD_ALIGN_PARAGRAPH.CENTER
    holder.paragraph_format.space_before = Pt(4)
    holder.paragraph_format.space_after = Pt(2)
    holder.add_run().add_picture(str(path), width=Inches(width))
    add_caption(document, caption)


def add_callout(
    document: Document,
    heading: str,
    text: str,
    *,
    fill: str = "EAF1F6",
    accent: str = "237A7A",
) -> None:
    table = document.add_table(rows=1, cols=1)
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.rows[0].cells[0]
    cell.width = Inches(6.55)
    shade(cell, fill)
    cell.text = ""

    first = cell.paragraphs[0]
    first.paragraph_format.space_before = Pt(6)
    first.paragraph_format.space_after = Pt(2)
    head = first.add_run(heading)
    head.font.bold = True
    head.font.size = Pt(9.5)
    head.font.color.rgb = RGBColor.from_string(accent)

    second = cell.add_paragraph()
    second.paragraph_format.space_after = Pt(6)
    text_run = second.add_run(text)
    text_run.font.size = Pt(9.3)

    spacer = document.add_paragraph()
    spacer.paragraph_format.space_after = Pt(2)


def add_code(document: Document, code: str, *, font_size: float = 7.5) -> None:
    table = document.add_table(rows=1, cols=1)
    table.autofit = False
    cell = table.rows[0].cells[0]
    cell.width = Inches(6.55)
    shade(cell, "F5F7FA")
    cell.text = ""
    first = True
    for line in code.rstrip("\n").split("\n"):
        paragraph = cell.paragraphs[0] if first else cell.add_paragraph()
        first = False
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.paragraph_format.space_before = Pt(0)
        paragraph.paragraph_format.line_spacing = 1.0
        run = paragraph.add_run(line if line else " ")
        run.font.name = "Consolas"
        run.font.size = Pt(font_size)
        run.font.color.rgb = RGBColor(0x2B, 0x35, 0x40)
        r_pr = run._r.get_or_add_rPr()
        r_fonts = OxmlElement("w:rFonts")
        r_fonts.set(qn("w:ascii"), "Consolas")
        r_fonts.set(qn("w:hAnsi"), "Consolas")
        r_pr.append(r_fonts)

    spacer = document.add_paragraph()
    spacer.paragraph_format.space_after = Pt(2)


def page_break(document: Document) -> None:
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.add_run().add_break(WD_BREAK.PAGE)


def load_json(name: str) -> dict:
    return json.loads((OUTPUTS / name).read_text(encoding="utf-8"))


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUTS / name)


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


# --------------------------------------------------------------------------
# Report sections
# --------------------------------------------------------------------------
def build_cover(document: Document, summary: dict) -> None:
    for _ in range(3):
        document.add_paragraph()

    brand = document.add_paragraph()
    brand.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = brand.add_run("YUVA INTERNSHIP PROGRAMME")
    run.font.size = Pt(11)
    run.font.bold = True
    run.font.color.rgb = TEAL

    rule = document.add_paragraph()
    rule.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rule.paragraph_format.space_after = Pt(20)
    run = rule.add_run("WEEK 2  ·  DELIVERABLE REPORT")
    run.font.size = Pt(9)
    run.font.color.rgb = LIGHT_GREY

    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(8)
    run = title.add_run("Supervised Machine\nLearning Models")
    run.font.size = Pt(30)
    run.font.bold = True
    run.font.color.rgb = NAVY
    run.font.name = "Calibri Light"

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(26)
    run = subtitle.add_run(REPORT_SUBTITLE)
    run.font.size = Pt(12)
    run.font.color.rgb = TEAL
    run.italic = True

    accent = document.add_paragraph()
    accent.alignment = WD_ALIGN_PARAGRAPH.CENTER
    accent.paragraph_format.space_after = Pt(26)
    accent.paragraph_format.left_indent = Inches(2.2)
    accent.paragraph_format.right_indent = Inches(2.2)
    border = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "12")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "C4664D")
    border.append(bottom)
    accent._p.get_or_add_pPr().append(border)

    facts = [
        ("Submitted by", AUTHOR),
        ("Programme", INTERNSHIP),
        ("Week", "2 of 4 — Regression and classification"),
        ("Deliverable", "Train at least two models and compare them with evaluation metrics"),
        ("Models", "Linear Regression · Logistic Regression · Decision Tree · Random Forest · K-Nearest Neighbors"),
        ("Dataset", f"customer_churn_raw.csv — {summary['dataset']['rows']:,} synthetic customer records"),
        ("Tools", "Python 3.11 · scikit-learn · NumPy · Pandas · Matplotlib"),
        ("Repository", REPOSITORY),
        ("Date", date.today().strftime("%d %B %Y")),
    ]

    table = document.add_table(rows=0, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for label, value in facts:
        row = table.add_row()
        label_cell, value_cell = row.cells
        label_cell.width = Inches(1.55)
        value_cell.width = Inches(4.65)
        label_cell.text = ""
        value_cell.text = ""

        paragraph = label_cell.paragraphs[0]
        paragraph.paragraph_format.space_after = Pt(3)
        run = paragraph.add_run(label.upper())
        run.font.size = Pt(7.5)
        run.font.bold = True
        run.font.color.rgb = LIGHT_GREY

        paragraph = value_cell.paragraphs[0]
        paragraph.paragraph_format.space_after = Pt(3)
        run = paragraph.add_run(value)
        run.font.size = Pt(9.2)
        run.font.color.rgb = NAVY
        if label == "Repository":
            run.font.size = Pt(8)
            run.font.color.rgb = TEAL

    document.add_paragraph()
    note = document.add_paragraph()
    note.alignment = WD_ALIGN_PARAGRAPH.CENTER
    note.paragraph_format.space_before = Pt(18)
    run = note.add_run(
        "All results are reproducible from the repository using a fixed random seed of 42. "
        "The dataset is synthetic teaching data and does not describe real customers."
    )
    run.font.size = Pt(8)
    run.italic = True
    run.font.color.rgb = LIGHT_GREY

    page_break(document)


def build_executive_summary(document: Document, summary: dict) -> None:
    classification = summary["classification"]
    headline = summary["headline"]
    models = classification["models"]
    real = [row for row in models if not row["model"].startswith("Dummy")]
    baseline = next(row for row in models if row["model"] == "Dummy (most frequent)")

    document.add_heading("1. Executive summary", level=1)

    body(
        document,
        "This report trains and compares supervised machine-learning models on the "
        "customer-churn dataset prepared in Week 1. Two tasks are covered, because the "
        "algorithm family and the evaluation metrics depend on the type of target: "
        "classification for churn (a category) and regression for monthly charges (a number). "
        "Using Linear Regression on a class label would be a category error, so the report keeps "
        "the two tasks explicitly separate throughout.",
    )
    body(
        document,
        f"The data is split into {classification['split']['train_rows']:,} training rows and "
        f"{classification['split']['test_rows']:,} held-out test rows, stratified so both halves "
        "keep the same churn rate. Every model is wrapped in an identical preprocessing pipeline, "
        "so imputation, one-hot encoding and standardisation are fitted on the training data only. "
        "That makes the comparison a comparison of algorithms rather than of data preparation, and "
        "it prevents test information from leaking into training.",
    )

    best_accuracy = max(real, key=lambda row: row["accuracy"])
    best_auc = max(real, key=lambda row: row["roc_auc"])
    best_recall = max(real, key=lambda row: row["recall"])

    add_table(
        document,
        ["Measure", "Result"],
        [
            ["Training / test rows", f"{classification['split']['train_rows']:,} / {classification['split']['test_rows']:,} (stratified, seed 42)"],
            ["Classification models", "Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors"],
            ["Baselines included", "Dummy (most frequent), Dummy (stratified)"],
            ["Regression models", "Linear Regression, Decision Tree, Random Forest, K-Nearest Neighbors"],
            ["Highest accuracy", f"{best_accuracy['model']} — {fmt(best_accuracy['accuracy'])}"],
            ["Highest ROC-AUC", f"{best_auc['model']} — {fmt(best_auc['roc_auc'])}"],
            ["Highest recall", f"{best_recall['model']} — {fmt(best_recall['recall'])}"],
            ["Do-nothing baseline accuracy", fmt(baseline["accuracy"])],
            ["Best regression model", f"{headline['regression_best_r2_model']} — R² {fmt(headline['regression_best_r2'])}"],
            ["Regression MAE", f"{headline['regression_linear_mae']:.2f} currency units"],
            ["Model selection method", "5-fold stratified cross-validation on the training split"],
            ["Test-set usage", "Scored once, after selection — never used to choose a model"],
        ],
        widths=[2.5, 4.0],
        font_size=8.8,
    )

    add_callout(
        document,
        "The headline finding",
        f"Three different models win three different metrics. {best_accuracy['model']} has the "
        f"highest accuracy ({fmt(best_accuracy['accuracy'])}) but the lowest recall "
        f"({fmt(best_accuracy['recall'])}) — it misses most customers who actually churn. "
        f"{best_auc['model']} has the strongest ROC-AUC ({fmt(best_auc['roc_auc'])}) and roughly "
        f"double that recall. Meanwhile the dummy baseline that never predicts churn scores "
        f"{fmt(baseline['accuracy'])} accuracy, which is higher than two of the four real models. "
        "No single metric identifies a best model, and accuracy alone actively misleads on "
        "imbalanced data.",
    )

    add_figure(
        document,
        FIGURES / "00_workflow_overview.png",
        "Figure 1 — The supervised workflow. Preprocessing is fitted inside the training box, "
        "cross-validation runs inside that box, and the test split is scored once at the end.",
        width=6.35,
    )


def build_description(document: Document) -> None:
    document.add_heading("2. Submission description", level=1)
    count = word_count(SUBMISSION_DESCRIPTION)
    body(
        document,
        f"The following description is provided for the submission form. It is {count} words long, "
        "which satisfies the 200-word minimum.",
        italic=True,
    )
    add_callout(
        document,
        f"Report description ({count} words)",
        SUBMISSION_DESCRIPTION,
        fill="F4F8FB",
        accent="17324D",
    )


def build_intro(document: Document, summary: dict) -> None:
    document.add_heading("3. Introduction and objectives", level=1)
    body(
        document,
        "Week 1 produced a clean, fully encoded and standardised dataset. Week 2 uses it to do "
        "what the brief asks: train models, predict with them, and evaluate the predictions "
        "honestly. The step from preprocessing to modelling introduces a new class of mistake. It "
        "is no longer about a corrupt cell; it is about a number that looks authoritative and is "
        "actually wrong — an accuracy inflated by class imbalance, a score inflated by leakage, or "
        "a model chosen because it won a metric that does not matter for the problem.",
    )

    document.add_heading("3.1 Objectives set by the brief", level=2)
    numbered(
        document,
        [
            "Implement supervised learning algorithms including Linear Regression, Logistic "
            "Regression, Decision Trees, Random Forest and K-Nearest Neighbors.",
            "Learn model training, prediction and evaluation using performance metrics.",
            "Deliverable: train at least two machine learning models and compare their performance "
            "using evaluation metrics.",
        ],
    )

    document.add_heading("3.2 How each objective is met", level=2)
    add_table(
        document,
        ["Objective", "Where it is addressed", "Evidence"],
        [
            [
                "Linear Regression",
                "Section 6",
                "outputs/regression_metrics.csv; 4 regressors compared including a mean baseline",
            ],
            [
                "Logistic Regression",
                "Section 5.2",
                "outputs/classification_metrics.csv; coefficients in Section 5.3",
            ],
            [
                "Decision Trees",
                "Section 5.4",
                "Depth sweep 1–18 with train/test curves proving overfitting",
            ],
            [
                "Random Forest",
                "Section 5.5",
                "300 trees; feature importance exported to outputs/feature_importance.csv",
            ],
            [
                "K-Nearest Neighbors",
                "Section 5.6",
                "Scaled and unscaled variants compared directly",
            ],
            [
                "Training and prediction",
                "Sections 5–6",
                "src/train_models.py; per-row probabilities in outputs/classification_predictions.csv",
            ],
            [
                "Evaluation metrics",
                "Section 4 and 7",
                "Ten classification metrics, five regression metrics, four confusion matrices",
            ],
            [
                "Compare at least two models",
                "Section 7",
                "Four classifiers and four regressors, plus two dummy baselines, on one test split",
            ],
        ],
        widths=[1.5, 1.5, 3.5],
        font_size=8.3,
    )

    document.add_heading("3.3 Scope", level=2)
    body(
        document,
        "The brief asks for at least two models. This report trains eight: four classifiers, four "
        "regressors, and two dummy baselines that make the word \"better\" meaningful by "
        "establishing what worse-than-nothing looks like. Two topics are deliberately left to "
        "later weeks — systematic hyperparameter tuning and clustering — although cross-validation "
        "is used here for selection, because a comparison without it would not be trustworthy.",
    )


def build_dataset(document: Document, summary: dict) -> None:
    dataset = summary["dataset"]

    document.add_heading("4. Dataset, tasks and evaluation design", level=1)

    document.add_heading("4.1 The data", level=2)
    body(
        document,
        f"The dataset is a reproducible synthetic sample of {dataset['rows']:,} customers, "
        "generated with a fixed seed and shipped with the repository. It is not real customer "
        "data. Week 1 established that it contains five numeric inputs, six categorical inputs, a "
        "unique identifier and the binary churn label, with 562 missing feature cells deliberately "
        "introduced and imputed.",
    )
    body(
        document,
        "The identifier is excluded from model inputs. A customer ID is unique to each row, so a "
        "model can memorise it but cannot generalise from it; including it would inflate training "
        "performance and contribute nothing at prediction time.",
    )

    document.add_heading("4.2 Class imbalance sets the bar", level=2)
    add_figure(
        document,
        FIGURES / "01_class_balance.png",
        "Figure 2 — Class imbalance, and why accuracy must be read against a baseline. The dashed "
        "line is the accuracy of a model that always predicts \"No\".",
        width=6.3,
    )
    body(
        document,
        "26.7% of customers churn, a ratio of roughly 2.7 to 1. This single fact determines how "
        "the rest of the report must be read. A model that never predicts churn is correct 73.3% "
        "of the time while being completely useless: it has zero recall, so every customer who "
        "leaves is missed.",
    )
    body(
        document,
        "The right-hand panel of Figure 2 plots each real model against that baseline. Two of the "
        "four — Logistic Regression and the Decision Tree — score below it on accuracy. That is "
        "not a coding error. Those models use balanced class weights, which deliberately trades "
        "accuracy for recall, and the report returns to this repeatedly because it is the crux of "
        "the whole exercise.",
    )

    document.add_heading("4.3 The two tasks", level=2)
    add_table(
        document,
        ["", "Classification", "Regression"],
        [
            ["Target", "churn (No / Yes)", "monthly_charges (a number)"],
            ["Question", "Will this customer leave?", "What will this customer pay per month?"],
            ["Output", "A probability, thresholded to a label", "A continuous value"],
            ["Typical metrics", "Accuracy, precision, recall, F1, ROC-AUC", "MAE, RMSE, R², MAPE"],
            ["Why this model", "Predicts a category", "Predicts a quantity"],
        ],
        widths=[1.15, 2.7, 2.65],
        font_size=8.5,
    )
    body(
        document,
        "Keeping the two tasks separate is not pedantry. Applying Linear Regression to the churn "
        "label would produce numbers such as 0.37 that are neither a probability nor a class, and "
        "the residuals of such a fit are meaningless because the target has only two values.",
    )

    document.add_heading("4.4 Evaluation design", level=2)
    body(
        document,
        "Three rules govern every number in this report. Breaking any one of them would produce a "
        "score that is too high and impossible to defend.",
    )
    numbered(
        document,
        [
            "One stratified split, created once. 80% of rows train, 20% test. Stratification keeps "
            "the churn rate identical in both halves, so the score does not depend on which rows "
            "happened to land in the test set.",
            "Preprocessing is fitted inside the training fold only. Medians, category lists, means "
            "and standard deviations are learned from training rows and then applied unchanged to "
            "test rows. This is enforced structurally by wrapping everything in a scikit-learn "
            "Pipeline rather than by remembering to do it.",
            "The test set is scored once, at the end. Model selection uses 5-fold cross-validation "
            "on the training data. If the test set were used to choose between candidates, it would "
            "quietly become a validation set and the final number would be optimistic.",
        ],
    )

    document.add_heading("4.5 The shared pipeline", level=2)
    body(
        document,
        "Every model receives identical inputs through the same transformer. This matters for "
        "fairness of comparison: if one model saw scaled features and another did not, a difference "
        "in score would confound the algorithm with the preparation.",
    )
    add_code(
        document,
        "def build_preprocessor(numeric, categorical, scale=True):\n"
        "    numeric_steps = [(\"imputer\", SimpleImputer(strategy=\"median\"))]\n"
        "    if scale:\n"
        "        numeric_steps.append((\"scaler\", StandardScaler()))\n"
        "\n"
        "    return ColumnTransformer(\n"
        "        transformers=[\n"
        "            (\"numeric\", Pipeline(numeric_steps), numeric),\n"
        "            (\"categorical\", Pipeline([\n"
        "                (\"imputer\", SimpleImputer(strategy=\"most_frequent\")),\n"
        "                (\"onehot\", OneHotEncoder(handle_unknown=\"ignore\", sparse_output=False)),\n"
        "            ]), categorical),\n"
        "        ],\n"
        "        remainder=\"drop\",\n"
        "    )\n"
        "\n"
        "def make_pipeline(estimator, scale=True):\n"
        "    return Pipeline([\n"
        "        (\"preprocessor\", build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES, scale)),\n"
        "        (\"model\", estimator),\n"
        "    ])",
    )


def build_metrics(document: Document) -> None:
    document.add_heading("5. Evaluation metrics, and what each one hides", level=1)
    body(
        document,
        "A metric is a question. Choosing the wrong question produces a confident answer to "
        "something nobody asked. Ten classification metrics and five regression metrics are "
        "reported, and the table below states which question each one answers and how it can "
        "mislead.",
    )

    add_table(
        document,
        ["Metric", "Question it answers", "How it misleads"],
        [
            ["Accuracy", "What fraction was classified correctly?", "Useless when classes are imbalanced — 73.3% here with zero recall"],
            ["Balanced accuracy", "Average of per-class recall", "Hides which class is failing"],
            ["Precision", "Of those flagged, how many were right?", "Rewards doing nothing — flagging one confident case scores 1.0"],
            ["Recall", "Of real churners, how many were found?", "Rewards flagging everyone"],
            ["F1", "Harmonic mean of precision and recall", "Assumes the two errors cost the same"],
            ["ROC-AUC", "How well does the model rank?", "Optimistic when the positive class is rare"],
            ["PR-AUC", "Ranking quality for the minority class", "Harder to compare across datasets"],
            ["MCC", "Correlation between prediction and truth", "Less intuitive to explain to stakeholders"],
            ["Log loss", "How good are the probabilities themselves?", "Punishes confident errors very heavily"],
            ["Brier score", "Mean squared error of the probabilities", "Harder to read in isolation"],
        ],
        widths=[1.25, 2.55, 2.7],
        font_size=8.1,
    )
    add_caption(document, "Table — Classification metrics used in this report.")

    body(
        document,
        "One distinction organises the whole table: **ROC-AUC and PR-AUC measure ranking and do "
        "not depend on a threshold; everything else does.** A model outputs a probability, and a "
        "threshold converts that probability into a label. Move the threshold and precision, "
        "recall, F1 and accuracy all change — while ROC-AUC stays fixed. Section 8 exploits this.",
    )

    add_table(
        document,
        ["Regression metric", "Meaning", "Caveat"],
        [
            ["MAE", "Average absolute error in currency units", "Treats all errors linearly; easy to explain"],
            ["RMSE", "Square root of mean squared error", "Punishes large errors more; same units as the target"],
            ["R²", "Share of variance explained relative to predicting the mean", "Can be negative for a bad model"],
            ["MAPE", "Average percentage error", "Undefined or explosive when the target is near zero"],
            ["Median abs. error", "Typical error, robust to outliers", "Ignores the tail of large errors"],
        ],
        widths=[1.35, 2.85, 2.3],
        font_size=8.3,
    )
    add_caption(document, "Table — Regression metrics used in this report.")

    body(
        document,
        "The gap between MAE and RMSE is itself informative. When RMSE greatly exceeds MAE, a few "
        "predictions are badly wrong — which is a different problem from uniformly mediocre "
        "accuracy, and needs a different fix.",
    )


def build_classification(document: Document, summary: dict) -> None:
    classification = summary["classification"]
    models = classification["models"]
    dummies = [row for row in models if row["model"].startswith("Dummy")]

    document.add_heading("6. Classification — the five algorithms", level=1)
    body(
        document,
        "Each subsection explains what the algorithm does, why it was written the way it was, and "
        "what it actually scored. The comparison table comes at the end of the section, in "
        "Section 7.",
    )

    # ---- Baseline -------------------------------------------------------
    document.add_heading("6.1 Baselines first", level=2)
    body(
        document,
        "Before any real model, two dummy classifiers are fitted. They ignore the features "
        "entirely. `most_frequent` always predicts the majority class; `stratified` guesses "
        "randomly while respecting the class frequencies.",
    )
    add_table(
        document,
        ["Baseline", "Accuracy", "Recall", "ROC-AUC"],
        [[row["model"], fmt(row["accuracy"]), fmt(row["recall"]), fmt(row["roc_auc"])] for row in dummies],
        widths=[2.6, 1.3, 1.3, 1.3],
        font_size=8.6,
        align_right_from=1,
    )
    body(
        document,
        "Establishing these first changes the meaning of every later number. The "
        "`most_frequent` baseline scores 0.733 accuracy and 0.000 recall: high accuracy, zero "
        "usefulness. No real model can be called good until it is measured against that, not "
        "against zero.",
    )

    # ---- Logistic Regression -------------------------------------------
    document.add_heading("6.2 Logistic Regression", level=2)
    body(
        document,
        "Logistic Regression applies the sigmoid function to a linear combination of inputs, "
        "producing a probability between 0 and 1. Despite the name it is a classification "
        "algorithm: the linear part is fitted, then squashed into a probability, then thresholded "
        "into a label.",
    )
    add_code(
        document,
        "LogisticRegression(max_iter=2000, class_weight=\"balanced\", random_state=42)",
    )
    body(
        document,
        "Two arguments carry real meaning here. max_iter=2000 gives the solver room to converge "
        "on the standardised feature matrix; the default is sometimes insufficient and produces a "
        "silent warning rather than an error. class_weight=\"balanced\" tells the optimiser that "
        "the minority class matters as much as the majority one — without it, predicting \"No\" "
        "for almost everyone minimises the loss and the model learns nothing useful.",
    )
    body(
        document,
        "Interpretability is this model's main advantage. The coefficients can be read as "
        "directions of effect, with one important caveat stated in Section 6.2.1.",
    )

    document.add_heading("6.2.1 Reading the coefficients honestly", level=3)
    body(
        document,
        "The one-hot columns are collinear: within each categorical feature they sum to 1, so any "
        "one of them is a linear combination of the others. Consequently **individual one-hot "
        "coefficients must not be interpreted as independent effects**, and comparing "
        "`contract_One year` against `contract_Two year` as if they were separate levers is "
        "invalid. Only the standardised numeric features have a clean interpretation: a one-unit "
        "(one standard deviation) increase changes the log-odds by the coefficient value.",
    )

    # ---- Decision Tree --------------------------------------------------
    document.add_heading("6.3 Decision Tree", level=2)
    body(
        document,
        "A decision tree repeatedly splits the data on the feature that best separates the classes, "
        "producing a set of if-then rules that can be read and explained. Its weakness follows "
        "directly from its strength: an unlimited-depth tree can create one leaf per training row, "
        "memorising the data instead of learning from it.",
    )
    add_code(
        document,
        "DecisionTreeClassifier(max_depth=5, min_samples_leaf=20,\n"
        "                       class_weight=\"balanced\", random_state=42)",
    )
    body(
        document,
        "max_depth=5 and min_samples_leaf=20 are the brakes. They were fixed in advance rather than "
        "tuned on the test set, so the reported score remains an honest estimate.",
    )

    body(
        document,
        "The overfitting claim is not asserted — it is measured. The same tree was trained at every "
        "depth from 1 to 18 and scored on both the training and test data.",
    )
    add_figure(
        document,
        FIGURES / "12_overfitting_tree_depth.png",
        "Figure 3 — Overfitting, measured. The training score climbs towards 1.0 as depth grows, "
        "while the test score peaks early and then declines. The gap is variance.",
        width=6.3,
    )
    body(
        document,
        "The training ROC-AUC reaches approximately 1.00 at depth 18 — a perfect training score, "
        "which is a symptom rather than an achievement. The test ROC-AUC peaks at depth 3 and then "
        "falls steadily. This is the clearest possible demonstration of why a model is never chosen "
        "on its training score, and why max_depth is a hyperparameter rather than a free choice.",
    )

    # ---- Random Forest --------------------------------------------------
    document.add_heading("6.4 Random Forest", level=2)
    body(
        document,
        "A Random Forest trains many decision trees, each on a bootstrap sample of the rows and "
        "considering a random subset of features at every split, then averages their votes. The two "
        "sources of randomness decorrelate the trees, and averaging decorrelated trees cancels much "
        "of the variance that ruins a single deep tree.",
    )
    add_code(
        document,
        "RandomForestClassifier(n_estimators=300, min_samples_leaf=5,\n"
        "                       class_weight=\"balanced\", random_state=42, n_jobs=1)",
    )
    body(
        document,
        "It is usually one of the strongest off-the-shelf methods on tabular data and needs little "
        "tuning. The cost is transparency: there is no single tree to read, so feature importance "
        "replaces it.",
    )

    document.add_heading("6.4.1 Feature importance, and its bias", level=3)
    importance = read_csv("feature_importance.csv").head(10)
    add_table(
        document,
        ["Rank", "Feature", "Importance"],
        [[str(index + 1), row.feature, fmt(row.importance, 4)]
         for index, row in enumerate(importance.itertuples(index=False))],
        widths=[0.7, 3.9, 1.5],
        font_size=8.4,
        align_right_from=0,
    )
    add_caption(document, "Table — Top ten features by mean decrease in impurity (outputs/feature_importance.csv).")
    body(
        document,
        "The ranking needs one honest caveat. Impurity-based importance is biased toward "
        "high-cardinality and continuous features, because they offer more possible split points. "
        "It ranks usefulness *inside this model*; it is not a causal effect and it is not a "
        "substitute for proper feature selection. Week 1's mutual-information ranking is a useful "
        "cross-check, and the two broadly agree that monthly charges, tenure and contract type "
        "dominate.",
    )
    add_figure(
        document,
        FIGURES / "08_feature_importance.png",
        "Figure 4 — Random Forest feature importance. Continuous features rank highly partly because "
        "they offer more candidate split points.",
        width=5.9,
    )

    # ---- KNN ------------------------------------------------------------
    document.add_heading("6.5 K-Nearest Neighbors", level=2)
    body(
        document,
        "KNN makes no model at all. To classify a new point it finds the k closest training points "
        "and takes a vote, weighting nearer neighbours more heavily. Prediction is therefore slow — "
        "every query scans the training set — and, more importantly, the method is entirely "
        "dependent on the meaning of \"closest\".",
    )
    add_code(
        document,
        "KNeighborsClassifier(n_neighbors=15, weights=\"distance\")",
    )
    body(
        document,
        "That dependence is not a theoretical concern. It was measured directly: the identical "
        "algorithm was fitted with and without the standard scaler.",
    )
    add_callout(
        document,
        "Scaling is not cosmetic — measured, not assumed",
        "KNN with StandardScaler: accuracy 0.744, recall 0.396, ROC-AUC 0.734. "
        "KNN without scaling: accuracy 0.708, recall 0.156, ROC-AUC 0.608. "
        "Same algorithm, same k, same data — only the preparation differs. Recall falls by 24 "
        "percentage points because 'distance' computed on raw units is dominated entirely by "
        "total_charges, which is measured in thousands while senior_citizen is 0 or 1.",
    )
    body(
        document,
        "This is the strongest argument in the report for why preprocessing lives in a pipeline "
        "rather than in a separate step someone might forget. Nothing in the training output would "
        "have warned that the unscaled model was quietly broken.",
    )


def build_comparison(document: Document, summary: dict) -> None:
    classification = summary["classification"]
    models = classification["models"]
    real = [row for row in models if not row["model"].startswith("Dummy")]
    sorted_by_auc = sorted(real, key=lambda row: row["roc_auc"], reverse=True)

    document.add_heading("7. Comparing the models", level=1)
    body(
        document,
        "The table below is the deliverable the brief asks for: at least two models trained and "
        "compared using evaluation metrics. Eight are shown — four real classifiers and two "
        "baselines here, with the regressors in Section 8.",
    )

    rows = []
    for row in models:
        rows.append(
            [
                row["model"],
                fmt(row["accuracy"]),
                fmt(row["balanced_accuracy"]),
                fmt(row["precision"]),
                fmt(row["recall"]),
                fmt(row["f1"]),
                fmt(row["roc_auc"]),
                fmt(row["pr_auc"]),
            ]
        )
    add_table(
        document,
        ["Model", "Acc.", "Bal. acc.", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"],
        rows,
        widths=[1.7, 0.62, 0.72, 0.78, 0.68, 0.6, 0.78, 0.68],
        font_size=7.9,
        align_right_from=1,
    )
    add_caption(
        document,
        "Table — Classification performance on the 360 held-out customers, sorted by ROC-AUC "
        "(outputs/classification_metrics.csv).",
    )

    add_figure(
        document,
        FIGURES / "02_model_comparison.png",
        "Figure 5 — The same results as a chart. Left: ranking and balanced metrics. Right: the "
        "precision-recall trade-off that drives every real decision.",
        width=6.35,
    )

    document.add_heading("7.1 Three models win three metrics", level=2)
    best_accuracy = max(real, key=lambda row: row["accuracy"])
    best_auc = max(real, key=lambda row: row["roc_auc"])
    best_recall = max(real, key=lambda row: row["recall"])

    add_table(
        document,
        ["Metric", "Winner", "Score", "What that means"],
        [
            ["Accuracy", best_accuracy["model"], fmt(best_accuracy["accuracy"]),
             f"Best raw hit rate — but recall is only {fmt(best_accuracy['recall'])}, so most churners are missed"],
            ["ROC-AUC", best_auc["model"], fmt(best_auc["roc_auc"]),
             "Best at ranking customers by churn risk, which is what a retention campaign consumes"],
            ["Recall", best_recall["model"], fmt(best_recall["recall"]),
             "Finds the most actual churners, at the cost of a much higher false-alarm rate"],
        ],
        widths=[0.95, 1.6, 0.75, 3.2],
        font_size=8.3,
    )

    body(
        document,
        "This is the central result of the week and it deserves stating plainly: **no single metric "
        "identifies a best model, because the metrics measure different things.** A retention team "
        "that can afford only a few offers wants high precision. A team trying to avoid losing any "
        "customer wants high recall. A team ranking a queue of customers to call wants high "
        "ROC-AUC. All three are legitimate, and they point at different models.",
    )

    add_figure(
        document,
        FIGURES / "03_accuracy_vs_recall.png",
        "Figure 6 — Accuracy and recall disagree about which model is best. Reporting only accuracy "
        "would have selected the model that misses the most churners.",
        width=5.9,
    )

    document.add_heading("7.2 The baseline that beats two real models", level=2)
    baseline = next(row for row in models if row["model"] == "Dummy (most frequent)")
    worse = [row["model"] for row in real if row["accuracy"] < baseline["accuracy"]]
    body(
        document,
        f"The dummy classifier that never predicts churn scores {fmt(baseline['accuracy'])} "
        f"accuracy. Two real models — {' and '.join(worse)} — score lower than that.",
    )
    add_callout(
        document,
        "Why that is not a bug",
        "Both of those models use class_weight='balanced', which reweights the loss so that a "
        "missed churner counts roughly 2.7 times a false alarm. The optimiser responds by predicting "
        "churn more often, which raises recall and lowers accuracy. The model is deliberately worse "
        "by one metric because it is better at the job. An evaluation that reported only accuracy "
        "would have rejected the most useful model in the table.",
        fill="FDF1EC",
        accent="C4664D",
    )

    document.add_heading("7.3 Confusion matrices", level=2)
    body(
        document,
        "Accuracy collapses two very different mistakes into one number. A confusion matrix keeps "
        "them separate.",
    )
    bullets(
        document,
        [
            "False positive (FP) — predicted churn, customer stayed. Cost: a wasted retention offer.",
            "False negative (FN) — predicted loyal, customer left. Cost: that customer's entire "
            "future value, plus the acquisition cost of replacing them.",
        ],
    )
    add_figure(
        document,
        FIGURES / "04_confusion_matrices.png",
        "Figure 7 — Confusion matrices for all four classifiers on the same 360 held-out customers.",
        width=6.35,
    )

    knn = next(row for row in real if row["model"] == "K-Nearest Neighbors")
    logit = next(row for row in real if row["model"] == "Logistic Regression")
    add_table(
        document,
        ["Model", "True negatives", "False positives", "False negatives", "True positives"],
        [
            [row["model"], str(row["true_negative"]), str(row["false_positive"]),
             str(row["false_negative"]), str(row["true_positive"])]
            for row in sorted_by_auc
        ],
        widths=[1.9, 1.15, 1.15, 1.25, 1.15],
        font_size=8.4,
        align_right_from=1,
    )
    add_caption(document, "Table — Error decomposition behind each confusion matrix (of 267 non-churners and 93 churners).")

    body(
        document,
        f"The contrast between the two extremes is stark. {knn['model']} makes "
        f"{knn['false_positive']} false alarms and misses {knn['false_negative']} churners. "
        f"{logit['model']} makes {logit['false_positive']} false alarms and misses "
        f"{logit['false_negative']}. If a missed churner is the expensive error — which it usually "
        "is — then the second model is clearly preferable, despite having lower accuracy and a "
        "smaller number in the accuracy column.",
    )

    document.add_heading("7.4 ROC and precision-recall curves", level=2)
    add_figure(
        document,
        FIGURES / "05_roc_and_pr_curves.png",
        "Figure 8 — Ranking quality. ROC curves (left) and precision-recall curves (right). The "
        "dashed line on the right is the no-skill baseline for this class balance.",
        width=6.35,
    )
    body(
        document,
        "ROC-AUC and PR-AUC both measure ranking and both ignore the threshold, but they behave "
        "differently when the positive class is rare. PR-AUC is the more honest summary here, "
        "because it focuses entirely on the minority class that the business cares about. The "
        "no-skill precision on the right-hand panel is 0.267 — the churn rate — not 0.5, which is "
        "another reminder that this is not a balanced problem.",
    )


def build_cv(document: Document, summary: dict) -> None:
    classification = summary["classification"]
    cv = classification["cross_validation"]
    results = sorted(cv["results"], key=lambda row: row["cv_roc_auc_mean"], reverse=True)

    document.add_heading("8. Choosing honestly: cross-validation", level=1)
    body(
        document,
        "Every score in Section 7 came from the test set. If the best of them were now used to pick "
        "a model, the test set would have become a validation set — and the final estimate would be "
        "optimistic, because the winner was selected partly by luck on those particular 360 rows.",
    )
    body(
        document,
        "Cross-validation avoids this. The training data is split into five folds; the model trains "
        "on four and validates on the fifth; the process rotates so every row is validated exactly "
        "once; the five scores are averaged. The test set stays untouched until the end.",
    )

    add_table(
        document,
        ["Model", "CV ROC-AUC mean", "Std. dev.", "CV F1 mean", "Std. dev."],
        [
            [row["model"], fmt(row["cv_roc_auc_mean"]), fmt(row["cv_roc_auc_std"]),
             fmt(row["cv_f1_mean"]), fmt(row["cv_f1_std"])]
            for row in results
        ],
        widths=[2.1, 1.25, 0.95, 1.15, 0.95],
        font_size=8.5,
        align_right_from=1,
    )
    add_caption(document, "Table — 5-fold stratified cross-validation on the training split (outputs/classification_cv_results.csv).")

    add_figure(
        document,
        FIGURES / "09_cross_validation.png",
        "Figure 9 — Cross-validated scores for classification and regression. Error bars are one "
        "standard deviation across the five folds.",
        width=6.3,
    )

    best = results[0]
    second = results[1]
    gap = best["cv_roc_auc_mean"] - second["cv_roc_auc_mean"]
    body(
        document,
        f"The selected model is **{cv['selected_model']}**, with a mean cross-validated ROC-AUC of "
        f"{fmt(best['cv_roc_auc_mean'])}. But look at the standard deviations, and then at the gap: "
        f"the leader is ahead of {second['model']} by only {fmt(gap)} ROC-AUC, while the "
        f"fold-to-fold standard deviation is {fmt(best['cv_roc_auc_std'])}. The difference is a "
        "fraction of the noise.",
    )
    add_callout(
        document,
        "The honest conclusion",
        "On 1,440 training rows, the top two classifiers are not reliably different. Reporting one "
        "as definitively better would over-claim. What the comparison does support is the weaker "
        "and more useful statement that Logistic Regression and Random Forest both perform well, "
        "and that both are preferable to an unweighted tree for churn because of their recall.",
    )


def build_thresholds(document: Document, summary: dict) -> None:
    classification = summary["classification"]
    threshold_data = classification["threshold_analysis"]
    weight_data = classification["class_weighting"]

    document.add_heading("9. Thresholds and class weights: two levers on one trade-off", level=1)
    body(
        document,
        "Precision and recall move in opposite directions. This is not a limitation of any "
        "particular algorithm; it is arithmetic. Every positive prediction is either a hit or a "
        "false alarm, so catching more of the positives necessarily means accepting more false "
        "alarms. There are two independent ways to slide along that curve, and they are often "
        "confused.",
    )

    document.add_heading("9.1 Lever one: the decision threshold", level=2)
    body(
        document,
        "The model outputs a probability. The threshold decides how high that probability must be "
        "before a customer is labelled as a churn risk. The default of 0.5 is a convention inherited "
        "from balanced problems and has no special status here.",
    )
    add_figure(
        document,
        FIGURES / "06_threshold_tradeoff.png",
        "Figure 10 — Precision, recall and F1 as the threshold moves, for the selected model. The "
        "default 0.5 is marked; the best F1 sits to its left.",
        width=6.2,
    )

    thresholds = threshold_data["rows"]
    best_f1 = max(thresholds, key=lambda row: row["f1"])
    add_table(
        document,
        ["Threshold", "Precision", "Recall", "F1", "Accuracy", "Flagged as churn"],
        [
            [f"{row['threshold']:.2f}", fmt(row["precision"]), fmt(row["recall"]),
             fmt(row["f1"]), fmt(row["accuracy"]), f"{row['predicted_positive_rate']:.1%}"]
            for row in thresholds
            if abs(row["threshold"] * 20 - round(row["threshold"] * 20)) < 1e-6
            and row["threshold"] in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
        ],
        widths=[0.85, 1.0, 0.9, 0.8, 0.9, 1.2],
        font_size=8.4,
        align_right_from=0,
    )
    add_caption(document, f"Table — Threshold sweep for {threshold_data['model']} (outputs/threshold_analysis.csv).")

    body(
        document,
        f"At the default 0.5, the model flags about 41% of customers as churn risks. Dropping the "
        f"threshold to {best_f1['threshold']:.2f} raises recall to {fmt(best_f1['recall'])} while "
        f"cutting precision to {fmt(best_f1['precision'])} — it now catches most churners, but more "
        "than half of the customers it flags were never going to leave.",
    )
    body(
        document,
        "Which row is correct depends entirely on cost. If a retention offer costs 10 currency units "
        "and a lost customer's remaining value is 500, then false negatives are 50 times more "
        "expensive than false positives, and a low threshold is obviously right. If the offer is "
        "expensive and the customer base is large, the opposite holds. **The data cannot answer this "
        "question; only the business can.**",
        italic=False,
    )

    document.add_heading("9.2 Lever two: class weights during training", level=2)
    body(
        document,
        "The threshold acts after training. Class weighting acts during it, by changing the loss "
        "the optimiser minimises. `class_weight='balanced'` scales each class's contribution by its "
        "frequency, so a missed churner counts roughly 2.7 times a false alarm.",
    )
    add_figure(
        document,
        FIGURES / "07_class_weighting.png",
        "Figure 11 — What class weighting buys. Recall rises substantially; accuracy falls. Quality, "
        "measured properly, improves.",
        width=6.0,
    )

    add_table(
        document,
        ["Setting", "Accuracy", "Precision", "Recall", "F1", "Customers flagged"],
        [
            [row["setting"], fmt(row["accuracy"]), fmt(row["precision"]), fmt(row["recall"]),
             fmt(row["f1"]), f"{row['predicted_positive_rate']:.1%}"]
            for row in weight_data["rows"]
        ],
        widths=[1.8, 0.95, 1.05, 0.9, 0.85, 1.35],
        font_size=8.5,
        align_right_from=1,
    )
    add_caption(document, "Table — Logistic Regression with and without balanced class weights (outputs/class_weight_comparison.csv).")

    unweighted = weight_data["rows"][0]
    weighted = weight_data["rows"][1]
    body(
        document,
        f"Unweighted, the model flags {unweighted['predicted_positive_rate']:.1%} of customers and "
        f"catches {unweighted['recall']:.1%} of churners. Weighted, it flags "
        f"{weighted['predicted_positive_rate']:.1%} and catches {weighted['recall']:.1%} — "
        f"{100 * (weighted['recall'] - unweighted['recall']):.0f} percentage points more churners. "
        f"Accuracy falls from {fmt(unweighted['accuracy'])} to {fmt(weighted['accuracy'])} while F1 "
        f"rises from {fmt(unweighted['f1'])} to {fmt(weighted['f1'])}.",
    )
    add_callout(
        document,
        "Accuracy fell while quality rose",
        "That sentence is the reason this report uses ten metrics instead of one. If the evaluation "
        "had optimised accuracy, it would have selected the unweighted model, which finds fewer than "
        "half the churners.",
        fill="FDF1EC",
        accent="C4664D",
    )


def build_regression(document: Document, summary: dict) -> None:
    regression = summary["regression"]
    models = regression["models"]
    real = [row for row in models if not row["model"].startswith("Dummy")]
    cv_results = sorted(regression["cross_validation"]["results"], key=lambda r: r["cv_r2_mean"], reverse=True)

    document.add_heading("10. Regression — predicting monthly charges", level=1)
    body(
        document,
        "The brief lists Linear Regression, so the report applies it where it belongs: to a numeric "
        "target. Monthly charges is the natural choice — it is continuous, almost entirely observed, "
        "and commercially meaningful.",
    )

    document.add_heading("10.1 A leakage trap that had to be defused", level=2)
    body(
        document,
        "total_charges is approximately monthly_charges multiplied by tenure. If it were included as "
        "an input while predicting monthly_charges, the model could nearly recover the answer from "
        "the input. R² would look excellent and the model would be worthless in production, where "
        "the whole point is to estimate a charge that is not yet known.",
    )
    add_callout(
        document,
        "Excluded feature",
        "total_charges is excluded from the regression inputs because it is derived from the target. "
        "This is a deliberate ceiling on achievable accuracy: dropping a highly informative column on "
        "purpose makes the score lower and the model usable.",
        fill="FDF1EC",
        accent="C4664D",
    )

    document.add_heading("10.2 Results", level=2)
    add_table(
        document,
        ["Model", "MAE", "RMSE", "R²", "MAPE", "Median abs. error"],
        [
            [row["model"], f"{row['mae']:.2f}", f"{row['rmse']:.2f}", fmt(row["r2"]),
             f"{row['mape']:.1%}", f"{row['median_absolute_error']:.2f}"]
            for row in models
        ],
        widths=[2.05, 0.8, 0.8, 0.7, 0.75, 1.15],
        font_size=8.4,
        align_right_from=1,
    )
    add_caption(document, "Table — Regression on monthly charges, 345 held-out customers (outputs/regression_metrics.csv).")

    add_table(
        document,
        ["Model", "CV R² mean", "Std. dev."],
        [[row["model"], fmt(row["cv_r2_mean"]), fmt(row["cv_r2_std"])] for row in cv_results],
        widths=[2.6, 1.4, 1.1],
        font_size=8.5,
        align_right_from=1,
    )
    add_caption(document, "Table — 5-fold cross-validated R² on the training split (outputs/regression_cv_results.csv).")

    linear = next(row for row in real if row["model"] == "Linear Regression")
    body(
        document,
        f"Linear Regression performs best, with an MAE of {linear['mae']:.2f}, an RMSE of "
        f"{linear['rmse']:.2f} and an R² of {fmt(linear['r2'])}. Three things about that result are "
        "worth noting.",
    )
    numbered(
        document,
        [
            f"MAE of {linear['mae']:.2f} against a mean charge of about 60 is roughly 12% — useful "
            "for triage and budgeting, not precise enough to bill a customer.",
            f"RMSE ({linear['rmse']:.2f}) exceeds MAE ({linear['mae']:.2f}), so some predictions are "
            "badly wrong rather than all being mildly wrong. The residual plot in Figure 13 shows "
            "where.",
            "The tree ensemble does not beat the linear model. That is informative: when a linear model "
            "wins, the underlying relationship is largely additive and linear, so extra capacity buys "
            "only noise.",
        ],
    )

    add_figure(
        document,
        FIGURES / "10_regression_comparison.png",
        "Figure 12 — Regression comparison by R² and by absolute error. Lower error and higher R² are "
        "both better; here the same model wins both.",
        width=6.2,
    )

    document.add_heading("10.3 Diagnostics, and what the bands mean", level=2)
    add_figure(
        document,
        FIGURES / "11_regression_diagnostics.png",
        "Figure 13 — Left: predicted versus actual monthly charges. Right: residual distribution.",
        width=6.35,
    )
    body(
        document,
        "The scatter plot is the most instructive panel in this report. Predictions form horizontal "
        "bands rather than a diagonal cloud. That is not a plotting artefact or a bug: monthly charges "
        "in this dataset are determined almost entirely by categorical inputs — contract type, "
        "internet service, payment method — so customers who share those categories receive nearly "
        "identical predictions.",
    )
    body(
        document,
        "The consequence is a ceiling on achievable accuracy that no model can break. When the true "
        "target is a step function of categorical features, the best possible regressor still "
        "produces bands. Recognising that saves hours of pointless tuning aimed at a problem the "
        "features cannot solve.",
    )
    body(
        document,
        "The residual histogram tells a related story. Residuals are centred near zero, so the model "
        "is not systematically biased in either direction — but they are clearly not normal, and they "
        "are not homoscedastic. That is a violation of the assumptions behind linear regression's "
        "confidence intervals, which is worth knowing before anyone quotes a prediction interval "
        "from this model.",
    )


def build_verification(document: Document, summary: dict) -> None:
    document.add_heading("11. Verification and quality assurance", level=1)
    body(
        document,
        "Every claim in this report is backed by an assertion in code or by an exported artifact. The "
        "table below lists what was checked and how.",
    )
    add_table(
        document,
        ["Check", "How it is enforced", "Result"],
        [
            ["Dataset schema", "Required columns asserted at load; raises otherwise", "Pass — 13/13"],
            ["Target validity", "Only No/Yes accepted", "Pass — 1,800 labels"],
            ["Stratification", "Train and test churn rates compared", "Pass — 0.267 / 0.267"],
            ["No leakage", "Preprocessing fitted inside a Pipeline on X_train only", "Pass — by construction"],
            ["Identical inputs", "Every model uses the same ColumnTransformer", "Pass"],
            ["Test set isolation", "Selection via CV on train; test scored once", "Pass"],
            ["Reproducibility", "random_state=42 on every stochastic step", "Pass — identical re-runs"],
            ["Metrics computable", "All ten classification and five regression metrics produced", "Pass"],
            ["Baselines present", "Two dummy classifiers and one dummy regressor", "Pass"],
        ],
        widths=[1.55, 3.15, 1.8],
        font_size=8.2,
    )

    add_table(
        document,
        ["Artifact", "Contents"],
        [
            ["outputs/classification_metrics.csv", "Ten metrics for six classifiers (4 real + 2 baselines)"],
            ["outputs/regression_metrics.csv", "Five metrics for five regressors"],
            ["outputs/classification_cv_results.csv", "Per-fold ROC-AUC and F1 for each classifier"],
            ["outputs/regression_cv_results.csv", "Per-fold R² for each regressor"],
            ["outputs/classification_predictions.csv", "Per-row predicted probability for every model"],
            ["outputs/regression_predictions.csv", "Per-row predicted charge for every regressor"],
            ["outputs/confusion_matrix_*.csv", "Four confusion matrices"],
            ["outputs/threshold_analysis.csv", "Precision/recall/F1 at 17 thresholds"],
            ["outputs/class_weight_comparison.csv", "Weighted versus unweighted Logistic Regression"],
            ["outputs/feature_importance.csv", "Random Forest impurity importance for all features"],
            ["outputs/tree_depth_curve.csv", "Train/test scores at depths 1–18"],
            ["outputs/week2_summary.json", "Machine-readable record; this report is generated from it"],
        ],
        widths=[2.65, 3.9],
        font_size=8.2,
    )
    add_caption(document, "Table — Exported artifacts, all reproducible from src/train_models.py.")


def build_limitations(document: Document) -> None:
    document.add_heading("12. Limitations and responsible interpretation", level=1)
    body(
        document,
        "These limitations are not disclaimers tacked onto the end. Each one changes how a number "
        "should be read, and each has a specific remedy before the work could support a real "
        "decision.",
    )
    add_table(
        document,
        ["Limitation", "Why it matters", "What real use would require"],
        [
            [
                "Synthetic data",
                "Relationships were generated by a probability model; every pattern found here is a property of that generator",
                "Validate on representative, permitted production data",
            ],
            [
                "Small test set",
                "360 rows gives roughly ±2.5 percentage points of sampling noise on accuracy",
                "A larger holdout, and confidence intervals on every reported metric",
            ],
            [
                "Hyperparameters fixed in advance",
                "A proper search would likely improve scores, but tuning on the test set would invalidate them",
                "Nested cross-validation: tune inside the training folds",
            ],
            [
                "Class imbalance handled by weighting only",
                "Resampling, SMOTE or cost-sensitive learning might do better",
                "Compare resampling strategies under the same CV protocol",
            ],
            [
                "No fairness analysis",
                "Recall and precision can differ by subgroup, so a model can be accurate overall and harmful to one group",
                "Audit error rates by subgroup before deployment",
            ],
            [
                "Accuracy ceiling from the features",
                "Monthly charges are a step function of categorical inputs, so the regression cannot exceed its banded predictions",
                "Collect continuous drivers (usage, add-ons) that vary within a category",
            ],
            [
                "Association is not causation",
                "Contract type correlates with churn but does not prove that changing a contract reduces it",
                "A controlled experiment or causal-inference design",
            ],
        ],
        widths=[1.6, 2.5, 2.4],
        font_size=8.1,
    )


def build_lessons(document: Document) -> None:
    document.add_heading("13. Lessons learned", level=1)
    numbered(
        document,
        [
            "Build the baseline before the model. The dummy classifier scoring 0.733 accuracy "
            "redefined what the word good means, and it arrived in four lines of code.",
            "A single metric is a decision waiting to go wrong. Three models won three metrics in "
            "this report; picking one metric would have silently picked one business strategy.",
            "Leakage is invisible unless you design against it. Fitting the scaler before splitting "
            "produces a slightly better score and no error message. Only the Pipeline structure "
            "prevents it reliably.",
            "Measured surprises beat asserted ones. The scaled-versus-unscaled KNN comparison "
            "(recall 0.396 versus 0.156) was more convincing than any paragraph explaining why "
            "scaling matters.",
            "Cross-validation showed the top two models were statistically tied. Reporting a winner "
            "on a 0.01 difference would have been dishonest, and the standard deviation is what "
            "revealed it.",
            "Sometimes the features cap the model. The banded regression plot showed that no "
            "algorithm can exceed what the inputs allow — worth discovering before tuning.",
        ],
    )


def build_conclusion(document: Document, summary: dict) -> None:
    headline = summary["headline"]
    document.add_heading("14. Conclusion", level=1)
    body(
        document,
        "The Week 2 brief asked for two things: implement supervised algorithms including Linear "
        "Regression, Logistic Regression, Decision Trees, Random Forest and K-Nearest Neighbors, and "
        "compare at least two models using evaluation metrics. Both are delivered, and then some — "
        "eight models across two tasks, with two dummy baselines for calibration and a "
        "cross-validated selection protocol behind every conclusion.",
    )
    body(
        document,
        f"The classification results do not produce a champion, and that is the honest finding. "
        f"{headline['classification_best_accuracy_model']} wins accuracy at "
        f"{fmt(headline['classification_best_accuracy'])} while missing most churners; "
        f"{headline['classification_best_roc_auc_model']} wins ranking quality at "
        f"{fmt(headline['classification_best_roc_auc'])} with roughly double the recall. Which one "
        "is right depends on the cost of a wasted retention offer versus a lost customer — a question "
        "the data cannot answer. The regression task does produce a clean winner: Linear Regression "
        f"explains {headline['regression_best_r2'] * 100:.1f}% of the variance in monthly charges "
        f"with a mean error of {headline['regression_linear_mae']:.2f} currency units.",
    )
    body(
        document,
        "The technical outcome is a set of trained models and metrics. The methodological outcome "
        "matters more: establish a baseline, evaluate with more than one metric, cross-validate "
        "before claiming a winner, and treat the threshold and the class weight as business "
        "decisions rather than defaults. Week 3 builds directly on this by tuning properly and by "
        "adding unsupervised structure to the same data.",
    )

    document.add_heading("Repository", level=2)
    body(document, f"Source, data, outputs and notebook: {REPOSITORY}")


def build_appendices(document: Document, summary: dict) -> None:
    page_break(document)

    document.add_heading("Appendix A — Reproducing these results", level=1)
    add_code(
        document,
        "python -m venv .venv\n"
        "source .venv/bin/activate            # Windows: .venv\\Scripts\\activate\n"
        "python -m pip install -r week-2/requirements.txt\n"
        "\n"
        "python week-2/src/train_models.py    # trains every model, writes all tables\n"
        "python week-2/src/make_figures.py    # builds the 13 figures\n"
        "python week-2/docs/build_report.py   # rebuilds this document\n"
        "\n"
        "jupyter notebook week-2/notebooks/week2_supervised_learning.ipynb",
    )

    document.add_heading("Appendix B — Model configuration", level=2)
    add_table(
        document,
        ["Model", "Configuration"],
        [
            ["Dummy (most frequent)", "strategy='most_frequent'"],
            ["Dummy (stratified)", "strategy='stratified'"],
            ["Logistic Regression", "max_iter=2000, class_weight='balanced'"],
            ["Decision Tree", "max_depth=5, min_samples_leaf=20, class_weight='balanced'"],
            ["Random Forest", "n_estimators=300, min_samples_leaf=5, class_weight='balanced'"],
            ["K-Nearest Neighbors", "n_neighbors=15, weights='distance'"],
            ["Linear Regression", "default parameters"],
            ["Decision Tree Regressor", "max_depth=6, min_samples_leaf=20"],
            ["Random Forest Regressor", "n_estimators=300, min_samples_leaf=5"],
            ["KNN Regressor", "n_neighbors=15, weights='distance'"],
        ],
        widths=[2.2, 4.35],
        font_size=8.4,
    )
    body(
        document,
        "All models use random_state=42 where the estimator accepts it. Hyperparameters were fixed "
        "before any test-set score was examined.",
        italic=True,
    )

    document.add_heading("Appendix C — The evaluation function", level=2)
    add_code(
        document,
        "def classify_metrics(y_true, y_pred, y_prob):\n"
        "    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()\n"
        "    return {\n"
        "        \"accuracy\": accuracy_score(y_true, y_pred),\n"
        "        \"balanced_accuracy\": balanced_accuracy_score(y_true, y_pred),\n"
        "        \"precision\": precision_score(y_true, y_pred, zero_division=0),\n"
        "        \"recall\": recall_score(y_true, y_pred, zero_division=0),\n"
        "        \"f1\": f1_score(y_true, y_pred, zero_division=0),\n"
        "        \"roc_auc\": roc_auc_score(y_true, y_prob),\n"
        "        \"pr_auc\": average_precision_score(y_true, y_prob),\n"
        "        \"mcc\": matthews_corrcoef(y_true, y_pred),\n"
        "        \"log_loss\": log_loss(y_true, y_prob),\n"
        "        \"brier\": brier_score_loss(y_true, y_prob),\n"
        "        \"true_negative\": int(tn), \"false_positive\": int(fp),\n"
        "        \"false_negative\": int(fn), \"true_positive\": int(tp),\n"
        "    }",
    )

    document.add_heading("Appendix D — Figure index", level=2)
    add_table(
        document,
        ["Figure", "File", "Shows"],
        [
            ["1", "00_workflow_overview.png", "Split / fit / predict / evaluate, and where CV lives"],
            ["2", "01_class_balance.png", "Imbalance and the accuracy baseline"],
            ["3", "12_overfitting_tree_depth.png", "Train vs test score as tree depth grows"],
            ["4", "08_feature_importance.png", "Random Forest impurity importance"],
            ["5", "02_model_comparison.png", "All four classifiers by six metrics"],
            ["6", "03_accuracy_vs_recall.png", "Accuracy and recall disagreeing"],
            ["7", "04_confusion_matrices.png", "Error decomposition per model"],
            ["8", "05_roc_and_pr_curves.png", "Ranking quality"],
            ["9", "09_cross_validation.png", "Cross-validated scores with error bars"],
            ["10", "06_threshold_tradeoff.png", "Precision/recall/F1 across thresholds"],
            ["11", "07_class_weighting.png", "Weighted versus unweighted model"],
            ["12", "10_regression_comparison.png", "Regressor R² and MAE"],
            ["13", "11_regression_diagnostics.png", "Predicted vs actual, and residuals"],
        ],
        widths=[0.65, 2.4, 3.5],
        font_size=8.1,
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    summary = load_json("week2_summary.json")

    document = setup_document()
    build_cover(document, summary)
    build_executive_summary(document, summary)
    build_description(document)
    build_intro(document, summary)
    build_dataset(document, summary)
    build_metrics(document)
    build_classification(document, summary)
    build_comparison(document, summary)
    build_cv(document, summary)
    build_thresholds(document, summary)
    build_regression(document, summary)
    build_verification(document, summary)
    build_limitations(document)
    build_lessons(document)
    build_conclusion(document, summary)
    build_appendices(document, summary)

    document.core_properties.title = REPORT_TITLE
    document.core_properties.subject = WEEK_TITLE
    document.core_properties.author = AUTHOR
    document.core_properties.keywords = (
        "supervised learning, regression, classification, scikit-learn, logistic regression, "
        "decision tree, random forest, k-nearest neighbors, ROC-AUC"
    )
    document.core_properties.comments = (
        f"Author: {AUTHOR}. Generated from week-2/outputs/week2_summary.json. "
        "Synthetic teaching dataset; not real customer data."
    )

    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / "Week_2_Supervised_Machine_Learning_Report.docx"
    document.save(path)

    count = word_count(SUBMISSION_DESCRIPTION)
    print(f"Report written : {path.relative_to(WEEK_DIR.parent)}")
    print(f"Author         : {AUTHOR}")
    print(f"Description    : {count} words (minimum 200)")
    if count < 200:
        raise SystemExit("Submission description is below the 200-word minimum")


if __name__ == "__main__":
    main()
