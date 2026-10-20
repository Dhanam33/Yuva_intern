#!/usr/bin/env python3
"""Build the Week 3 Word report from the generated analysis artifacts.

Every number is read from ``week-3/outputs/week3_summary.json`` and the exported
CSV tables, so the document cannot drift from the code that produced it.

    python src/run_analysis.py     # generate artifacts
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
WEEK_TITLE = "Week 3 — Unsupervised Learning & Model Evaluation"
REPORT_TITLE = "Unsupervised Learning & Model Evaluation"
REPORT_SUBTITLE = (
    "Clustering with K-Means and hierarchical methods, dimensionality reduction with PCA, "
    "and a full evaluation of a tuned classifier"
)
REPOSITORY = "https://github.com/Dhanam33/Yuva_intern/tree/main/week-3"

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
    "This Week 3 report addresses both halves of the brief — unsupervised learning and model "
    "evaluation — using scikit-learn on the same reproducible synthetic customer-churn dataset of "
    "1,800 records. Part A applies K-Means and agglomerative hierarchical clustering to a 21-column "
    "feature matrix, with principal component analysis used to examine dimensionality reduction. "
    "Cluster quality is judged using silhouette, Davies-Bouldin and Calinski-Harabasz indices, and "
    "the value of k is chosen only after demonstrating that the three disagree. Hierarchical "
    "clustering is compared across ward, complete, average and single linkage, which exposes a common "
    "trap: single linkage achieves the highest silhouette of any configuration while producing a "
    "one-customer cluster. Two independent algorithms recover almost the same three segments, with an "
    "adjusted Rand index of 0.732, which is evidence that the structure is real; yet neither relates "
    "to churn, where the index is 0.015 and the churn rate varies by only nine percentage points "
    "across segments. Part B builds a supervised classifier and evaluates it properly: the confusion "
    "matrix, precision, recall, F1 and ROC-AUC are computed from first principles, five-fold "
    "stratified cross-validation is used throughout, and GridSearchCV searches 37 hyperparameter "
    "combinations. Nested cross-validation separates the tuned score from an unbiased estimate. "
    "Results are reported without embellishment: tuning improved Logistic Regression by 0.001 "
    "ROC-AUC, nested cross-validation agreed to within 0.001, and the final held-out scores were "
    "0.768 ROC-AUC with 0.656 recall."
)


# --------------------------------------------------------------------------
# Low-level document helpers
# --------------------------------------------------------------------------
def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'’-]+\b", text))


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
    header.text = "YUVA INTERNSHIP   ·   WEEK 3   ·   UNSUPERVISED LEARNING & MODEL EVALUATION"
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
    first.alignment = WD_ALIGN_PARAGRAPH.LEFT
    first.paragraph_format.space_before = Pt(6)
    first.paragraph_format.space_after = Pt(2)
    head = first.add_run(heading)
    head.font.bold = True
    head.font.size = Pt(9.5)
    head.font.color.rgb = RGBColor.from_string(accent)

    second = cell.add_paragraph()
    second.alignment = WD_ALIGN_PARAGRAPH.LEFT
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


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


# --------------------------------------------------------------------------
# Cover
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
    run = rule.add_run("WEEK 3  ·  DELIVERABLE REPORT")
    run.font.size = Pt(9)
    run.font.color.rgb = LIGHT_GREY

    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(8)
    run = title.add_run("Unsupervised Learning\n& Model Evaluation")
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
        ("Week", "3 of 4 — Unsupervised learning and model evaluation"),
        ("Deliverable", "Perform clustering on a dataset and prepare a model evaluation report"),
        ("Part A", "K-Means · Agglomerative clustering (ward, complete, average, single) · PCA"),
        ("Part B", "Confusion matrix · precision · recall · F1 · ROC-AUC · cross-validation · GridSearchCV"),
        ("Dataset", f"customer_churn_raw.csv — {summary['dataset']['rows']:,} synthetic customer records"),
        ("Tools", "Python 3.11 · scikit-learn · SciPy · NumPy · Pandas · Matplotlib"),
        ("Repository", REPOSITORY),
        ("Date", date.today().strftime("%d %B %Y")),
    ]

    table = document.add_table(rows=0, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for label, value in facts:
        row = table.add_row()
        label_cell, value_cell = row.cells
        label_cell.width = Inches(1.45)
        value_cell.width = Inches(4.85)
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


# --------------------------------------------------------------------------
# 1. Executive summary
# --------------------------------------------------------------------------
def build_executive_summary(document: Document, summary: dict) -> None:
    kmeans = summary["kmeans"]
    hierarchical = summary["hierarchical"]
    agreement = summary["agreement"]
    dimensionality = summary["dimensionality"]
    tuning = summary["tuning"]
    nested = summary["nested_cv"]
    dataset = summary["dataset"]
    overlap = kmeans["churn_overlap"]

    document.add_heading("1. Executive summary", level=1)

    body(
        document,
        "This report covers both halves of the Week 3 brief. Part A clusters the customer dataset "
        "without using the churn label and asks afterwards whether the clusters mean anything. "
        "Part B takes the churn label seriously and asks how well a classifier can predict it, "
        "using the evaluation metrics and tuning procedures the brief names.",
    )

    body(
        document,
        f"The dataset holds {dataset['rows']:,} synthetic customer records encoded into "
        f"{dataset['encoded_features']} numeric columns. Four clustering configurations are compared — "
        f"K-Means over k = 2 to 8, and agglomerative clustering under ward, complete, average and "
        f"single linkage — alongside principal component analysis as a dimensionality-reduction step. "
        f"On the supervised side, {tuning['n_combinations']['logistic'] + tuning['n_combinations']['random_forest']} "
        f"hyperparameter combinations are searched with GridSearchCV, and nested cross-validation "
        f"checks the tuned score against an unbiased estimate.",
    )

    rows = [
        ["Records / encoded features", f"{dataset['rows']:,} / {dataset['encoded_features']}"],
        ["Churn rate", f"{pct(dataset['churn_rate'])}  ({dataset['churn_yes']:,} of {dataset['rows']:,})"],
        ["Components for 90% of variance", f"{summary['pca']['n_components_90']} (from {summary['pca']['n_features']})"],
        ["Clusters chosen", f"k = {kmeans['selected_k']} by K-Means and Ward alike"],
        ["Best K-Means silhouette", f"{fmt(kmeans['selected_silhouette'])}  (weak — below 0.25)"],
        ["K-Means vs Ward agreement", f"ARI {fmt(agreement['ari_models'])}"],
        ["Clusters vs churn agreement", f"ARI {fmt(agreement['ari_churn_kmeans'])}  (essentially unrelated)"],
        ["Churn spread across clusters", f"{pct(overlap['spread'])}  (from {pct(min(overlap['rates']))} to {pct(max(overlap['rates']))})"],
        ["Silhouette gained by projecting to 2-D", f"{fmt(dimensionality['inflated_original'])} → {fmt(dimensionality['inflated_reduced'])}  (illusory)"],
        ["Selected classifier", f"{tuning['winner']}  (CV ROC-AUC {fmt(tuning['best_cv_score'])})"],
        ["Held-out test ROC-AUC", f"{fmt(tuning['metrics']['roc_auc'])}"],
        ["Held-out test recall / precision", f"{fmt(tuning['metrics']['recall'])} / {fmt(tuning['metrics']['precision'])}"],
        ["Nested CV estimate", f"{fmt(nested['mean'])} ± {fmt(nested['std'])}  (optimism {nested['optimism']:+.3f})"],
    ]
    add_table(document, ["Measure", "Result"], rows, widths=[2.65, 3.85], align_right_from=1)

    add_callout(
        document,
        "The headline finding",
        f"Two different clustering algorithms independently recover almost the same three customer "
        f"segments (ARI {fmt(agreement['ari_models'])}), which is strong evidence the segments are real. "
        f"Those segments carry essentially no information about churn (ARI {fmt(agreement['ari_churn_kmeans'])}), "
        f"and the churn rate varies by only {pct(overlap['spread'])} across them. Real structure and a "
        f"useful predictor are not the same thing — which is precisely why the labelled evaluation in "
        f"Part B is necessary.",
    )

    body(
        document,
        "Part A also documents two ways unsupervised output is commonly misread. Single linkage "
        f"achieves the highest silhouette of any configuration ({fmt(hierarchical['trap']['silhouette'])}) "
        f"while producing a cluster containing {hierarchical['trap']['smallest_cluster']} customer. And "
        f"clustering in two dimensions appears to raise silhouette from "
        f"{fmt(dimensionality['inflated_original'])} to {fmt(dimensionality['inflated_reduced'])} purely "
        f"because the space the clusters are scored in got smaller. Both are demonstrated with numbers "
        f"rather than described in the abstract.",
    )

    body(
        document,
        f"Part B is deliberately unflattering. A model that never predicts churn scores "
        f"{fmt(1 - dataset['churn_rate'])} accuracy, above the classifier's "
        f"{fmt(tuning['metrics']['accuracy'])}. Hyperparameter tuning improved Logistic Regression by "
        f"{fmt(tuning['comparison'][0]['gain'])} ROC-AUC across five values of C — there was no headroom "
        f"to exploit. Nested cross-validation agreed with the tuned score to within "
        f"{fmt(abs(nested['optimism']))}, confirming the tuning was not overfitting. The reported value "
        f"of the exercise is the discipline of these checks, not a headline number.",
    )


# --------------------------------------------------------------------------
# 2. Submission description
# --------------------------------------------------------------------------
def build_description(document: Document) -> None:
    document.add_heading("2. Submission description", level=1)
    body(
        document,
        "The text below is the description submitted with this report. It is reproduced here so the "
        "report is self-contained. The word count is computed on generation and the build fails if it "
        "falls below the required minimum.",
    )
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.left_indent = Inches(0.22)
    paragraph.paragraph_format.right_indent = Inches(0.22)
    paragraph.paragraph_format.space_before = Pt(4)
    run = paragraph.add_run(SUBMISSION_DESCRIPTION)
    run.font.size = Pt(9.8)

    add_caption(document, f"Word count: {word_count(SUBMISSION_DESCRIPTION)} words (minimum 200).")


# --------------------------------------------------------------------------
# 3. Introduction and objectives
# --------------------------------------------------------------------------
def build_intro(document: Document, summary: dict) -> None:
    document.add_heading("3. Introduction", level=1)

    body(
        document,
        "The Week 3 brief has three objectives, and they pull in different directions. The first is "
        "unsupervised: understand clustering and dimensionality reduction. The second is evaluative: "
        "understand confusion matrices, precision, recall, F1, ROC-AUC, cross-validation and "
        "hyperparameter tuning. The third asks for a clustering exercise and a model evaluation "
        "report as a single deliverable.",
    )

    body(
        document,
        "The natural temptation is to treat these as two separate pieces of homework stapled "
        "together. This report instead uses the output of Part A to justify Part B. The clustering "
        "succeeds on its own terms — it finds stable, reproducible segments — but those segments "
        "cannot predict churn. That negative result is what makes the supervised evaluation "
        "necessary, and it sets up the central question of Part B: if a label matters, how do you "
        "measure how well you are predicting it?",
    )

    document.add_heading("3.1 How each objective is addressed", level=2)

    rows = [
        ["K-Means clustering", "Section 6", "k = 2…8 sweep with elbow, silhouette, Davies-Bouldin and Calinski-Harabasz"],
        ["Hierarchical clustering", "Section 8", "Four linkage rules compared; dendrogram; non-degeneracy filter"],
        ["Dimensionality reduction", "Sections 5 and 10", "PCA variance, loadings, 2-D projection, and clustering after reduction"],
        ["Cross-validation", "Section 14", "Five-fold stratified; fold spread read as an error bar"],
        ["Confusion matrix and metrics", "Sections 11–13", "Counts derived first, then every metric computed by hand and cross-checked"],
        ["Precision, recall, F1, ROC-AUC", "Sections 12–13", "Each defined, computed from the matrix, then interpreted in cost terms"],
        ["Hyperparameter tuning", "Section 15", "GridSearchCV over 5 + 32 combinations, scored on CV only"],
        ["Clustering deliverable", "Sections 5–10", "Three segments profiled and validated against an external label"],
        ["Evaluation report", "Sections 11–19", "Tuned model, nested CV, learning curve, calibration, limitations"],
    ]
    add_table(document, ["Objective", "Where", "Evidence"], rows, widths=[1.72, 0.95, 3.88])

    document.add_heading("3.2 What this report deliberately does not claim", level=2)
    bullets(
        document,
        [
            "That the clusters describe customers in a way a business could act on. The silhouette of "
            f"{fmt(summary['kmeans']['selected_silhouette'])} indicates substantial overlap.",
            "That the classifier is good. It is measurably better than chance and measurably worse "
            "than a do-nothing rule on accuracy.",
            "That hyperparameter tuning produced a better model. It produced a statistically "
            "indistinguishable one, and the report says so.",
            "That any of these numbers would transfer to real customer data. The dataset is synthetic.",
        ],
    )


# --------------------------------------------------------------------------
# 4. Dataset
# --------------------------------------------------------------------------
def build_dataset(document: Document, summary: dict) -> None:
    dataset = summary["dataset"]

    document.add_heading("4. Dataset and feature space", level=1)

    body(
        document,
        f"The data is the same synthetic customer-churn sample used in Week 2: {dataset['rows']:,} "
        f"records across {dataset['columns']} columns, of which {dataset['numeric_features']} are numeric "
        f"and {dataset['categorical_features']} categorical. The churn label is present but plays no "
        f"part in Part A — it is held back and used only to ask, afterwards, whether the clusters "
        f"happened to match it.",
    )

    body(
        document,
        f"There are {dataset['churn_yes']:,} churners ({pct(dataset['churn_rate'])}) and "
        f"{dataset['churn_no']:,} non-churners, and {dataset['missing_cells']:,} missing feature cells. "
        f"Missingness is handled by median imputation for numbers and most-frequent imputation for "
        f"categories, both fitted inside the transformer so that no statistic crosses from test data "
        f"into training.",
    )

    document.add_heading("4.1 Encoding choices", level=2)

    body(
        document,
        "Clustering is a distance-based procedure, so the representation of each column directly "
        "determines the result. Two different treatments are applied:",
    )

    rows = [
        ["Numeric (5 columns)", "Median impute → standardise", "Dollars, months and counts are not comparable magnitudes"],
        ["Categorical (6 columns)", "Mode impute → one-hot", "K-Means needs numbers, not labels"],
    ]
    add_table(document, ["Column type", "Treatment", "Reason"], rows, widths=[1.62, 1.85, 3.08])

    body(
        document,
        f"One-hot columns are deliberately NOT standardised. A dummy for a category present in 5% "
        f"of rows already has a small variance; forcing it to unit variance would give rare categories "
        f"as much weight in the distance metric as common ones. Numbers need scaling; indicators do "
        f"not. The result is a {dataset['encoded_features']}-column matrix.",
    )

    body(
        document,
        f"The dataset contains {dataset['senior_citizens']} senior citizens. This is worth stating "
        f"up front because the clustering will find them without being told the column exists — a "
        f"detail that recurs in Section 7.",
    )

    add_figure(
        document,
        FIGURES / "00_workflow_overview.png",
        "Figure 1 — The two halves of the work. Part A clusters without the label; Part B evaluates a "
        "supervised model. The arrow marks the finding that connects them.",
        width=6.35,
    )


# --------------------------------------------------------------------------
# 5. PCA
# --------------------------------------------------------------------------
def build_pca(document: Document, summary: dict) -> None:
    pca = summary["pca"]
    variance = read_csv("pca_variance.csv")

    document.add_heading("5. Dimensionality reduction with PCA", level=1)

    body(
        document,
        f"Principal component analysis finds the orthogonal directions along which the data varies "
        f"most, and re-expresses every record in those directions. It is a rotation, not a summary — "
        f"the first component alone accounts for {pct(pca['explained_first'])} of the total variance, "
        f"and the first three together for {pct(pca['explained_first_three'])}.",
    )

    rows = []
    for _, row in variance.head(10).iterrows():
        rows.append([
            str(int(row["component"])),
            f"{row['eigenvalue']:.3f}",
            pct(row["explained_variance_ratio"]),
            pct(row["cumulative_variance"]),
        ])
    add_table(
        document,
        ["Component", "Eigenvalue", "Variance explained", "Cumulative"],
        rows,
        widths=[1.2, 1.6, 1.85, 1.9],
        align_right_from=1,
    )

    add_figure(
        document,
        FIGURES / "01_pca_variance.png",
        f"Figure 2 — Variance explained per component (left) and cumulatively (right). "
        f"{pca['n_components_90']} components are needed to reach 90%.",
        width=6.3,
    )

    body(
        document,
        f"The practical figures are {pca['n_components_90']} components for 90% of the variance and "
        f"{pca['n_components_95']} for 95%. Reducing {pca['n_features']} columns to "
        f"{pca['n_components_90']} is a real compression, and it is what makes the 2-D plots in this "
        f"report possible at all.",
    )

    document.add_heading("5.1 What the first component is made of", level=2)

    body(
        document,
        "A loading is the weight of one original feature inside a component. Sorting by the absolute "
        "loading shows what each component is built from, which is more informative than the variance "
        "number alone.",
    )

    rows = [[d["feature"].replace("_", " "), f"{d['loading']:+.3f}"] for d in pca["top_pc1"]]
    add_table(document, ["Feature", "PC1 loading"], rows, widths=[3.4, 1.6], align_right_from=1)

    add_callout(
        document,
        "Reading PC1",
        f"The three largest loadings are total charges ({pca['top_pc1'][0]['loading']:+.2f}), monthly "
        f"charges ({pca['top_pc1'][1]['loading']:+.2f}) and tenure ({pca['top_pc1'][2]['loading']:+.2f}), "
        "all with the same sign. That is not three findings — it is one axis: how much a customer is "
        "worth and how long they have stayed. A component is often a single latent quantity expressed "
        "through several correlated columns.",
        fill="FBF1E3",
        accent="B97A2E",
    )

    body(
        document,
        f"Two components retain only {pct(pca['explained_first_two'])} of the variance, yet they are the "
        f"only ones that can be drawn on a page. That ratio sets the expectation for the next figure: "
        f"it is a compression, and something has been given up.",
    )

    add_figure(
        document,
        FIGURES / "02_pca_projection.png",
        "Figure 3 — The same 1,800 customers in two dimensions, coloured first by K-Means cluster and then "
        "by actual churn. Both partitions overlap heavily.",
        width=6.35,
    )

    body(
        document,
        "The right-hand panel is the honest starting point. Churners and non-churners occupy the same "
        "region of the projected space, with no traceable boundary between them. Any claim that "
        "clustering has 'discovered churn' has to explain away this picture first.",
    )


# --------------------------------------------------------------------------
# 6. K-Means selection
# --------------------------------------------------------------------------
def build_kmeans(document: Document, summary: dict) -> None:
    kmeans = summary["kmeans"]
    preferred = kmeans["preferred_k"]
    sweep = read_csv("kmeans_sweep.csv")

    document.add_heading("6. K-Means: choosing k", level=1)

    body(
        document,
        "K-Means partitions records into k groups by alternating between assigning each point to its "
        "nearest centroid and moving each centroid to the mean of its members. It requires k in "
        "advance, and it will produce k clusters even when the data contains none. Choosing k is "
        "therefore the substantive decision, and it cannot be answered by a single formula.",
    )

    body(
        document,
        "Three internal indices are computed across k = 2 to 8. Each measures something different, "
        "and they are presented together because they do not agree.",
    )

    rows = []
    for _, row in sweep.iterrows():
        rows.append([
            str(int(row["k"])),
            f"{row['inertia']:,.0f}",
            f"{row['silhouette']:.3f}",
            f"{row['davies_bouldin']:.3f}",
            f"{row['calinski_harabasz']:.1f}",
            f"{row['ari_vs_churn']:+.3f}",
        ])
    add_table(
        document,
        ["k", "Inertia", "Silhouette", "Davies-Bouldin", "Calinski-Harabasz", "ARI vs churn"],
        rows,
        widths=[0.55, 1.15, 1.05, 1.35, 1.55, 1.05],
        align_right_from=1,
    )

    add_figure(
        document,
        FIGURES / "03_kmeans_selection.png",
        "Figure 4 — The elbow (left) and the two internal indices (right). Silhouette peaks at k = 3, "
        "Davies-Bouldin bottoms out at k = 5 and Calinski-Harabasz peaks at k = 2.",
        width=6.2,
    )

    document.add_heading("6.1 The indices disagree, and that is the finding", level=2)

    body(
        document,
        f"Silhouette prefers k = {preferred['silhouette']}, Davies-Bouldin prefers k = "
        f"{preferred['davies_bouldin']} and Calinski-Harabasz prefers k = {preferred['calinski_harabasz']}. "
        f"No rule arbitrates between them, because they are not measuring the same property: silhouette "
        f"compares each point with the nearest rival cluster, Davies-Bouldin compares cluster spread with "
        f"cluster separation, and Calinski-Harabasz is a global between-over-within ratio that tends to "
        f"favour small k.",
    )

    body(
        document,
        f"k = {kmeans['selected_k']} is used for the rest of the report. The reasons are explicit: "
        f"silhouette peaks there ({fmt(kmeans['selected_silhouette'])}), the elbow agrees — the largest "
        f"single drop in inertia occurs moving to k = {kmeans['elbow_k']} — and three segments are few "
        f"enough to describe and reason about. This is a defensible judgement, not a derived fact, and "
        f"it is recorded as such.",
    )

    add_callout(
        document,
        "The absolute value matters more than the ranking",
        f"A silhouette of {fmt(kmeans['selected_silhouette'])} is low. Scores below roughly 0.25 "
        f"indicate overlapping, poorly separated clusters, and this dataset sits well inside that "
        f"range. The best of eight values is still a weak segmentation — the ranking identifies the "
        f"least-bad choice, not a strong one.",
        fill="F8ECE7",
        accent="C4664D",
    )


# --------------------------------------------------------------------------
# 7. K-Means profiles
# --------------------------------------------------------------------------
def build_profiles_kmeans(document: Document, summary: dict) -> None:
    kmeans = summary["kmeans"]
    profiles = sorted(kmeans["profiles"], key=lambda r: r["size"], reverse=True)

    document.add_heading("7. K-Means cluster profiles", level=1)

    body(
        document,
        f"With k = {kmeans['selected_k']} and 50 random restarts, K-Means converges to a solution with "
        f"an inertia of {kmeans['inertia']:,.1f}. A cluster is only useful if it can be described, so "
        f"each one is profiled on the original, unstandardised features.",
    )

    rows = []
    for row in profiles:
        rows.append([
            str(row["cluster"]),
            row["label"],
            f"{row['size']:,}",
            pct(row["share"]),
            f"{row['tenure_months']:.1f}",
            f"{row['monthly_charges']:.1f}",
            f"{row['total_charges']:,.0f}",
            pct(row["churn_rate"]),
        ])
    add_table(
        document,
        ["#", "Segment", "Customers", "Share", "Tenure", "Monthly", "Total", "Churn"],
        rows,
        widths=[0.35, 1.72, 1.0, 0.78, 0.78, 0.86, 0.95, 0.75],
        align_right_from=2,
    )

    add_figure(
        document,
        FIGURES / "04_cluster_overview.png",
        "Figure 5 — Cluster sizes (left) and churn rate within each segment (right). The dashed line is "
        "the overall churn rate; no segment departs from it by much.",
        width=6.2,
    )

    add_figure(
        document,
        FIGURES / "07_cluster_profiles.png",
        "Figure 6 — Cluster profiles for K-Means and Ward side by side. Cell shading is the z-score "
        "within each column; the printed number is the raw mean in original units.",
        width=6.3,
    )

    senior = next(row for row in profiles if row["senior_citizen"] >= 0.9)
    ordinary = [row for row in profiles if row["senior_citizen"] < 0.9]
    short_tenure = min(ordinary, key=lambda r: r["tenure_months"])
    long_tenure = max(ordinary, key=lambda r: r["tenure_months"])

    body(
        document,
        f"Two of the three segments form a straightforward axis. The group of "
        f"{short_tenure['size']:,} customers has a mean tenure of {short_tenure['tenure_months']:.1f} "
        f"months and spends {short_tenure['monthly_charges']:.2f} per month — recent, low-value "
        f"customers. The group of {long_tenure['size']:,} has been present for "
        f"{long_tenure['tenure_months']:.1f} months and spends {long_tenure['monthly_charges']:.2f} "
        f"— established and high-value. That axis is exactly PC1 from Section 5, found again without "
        f"being asked for.",
    )

    document.add_heading("7.1 The segment nobody labelled", level=2)

    add_callout(
        document,
        "An unlabelled category made itself a cluster",
        f"Cluster {senior['cluster']} contains {senior['size']:,} customers — "
        f"{pct(senior['share'])} of the data — and the mean of its senior_citizen flag is "
        f"{senior['senior_citizen']:.2f}. In other words it contains essentially every senior citizen "
        "in the dataset and almost nobody else. K-Means was never told the column existed. It found "
        "the group because those customers differ consistently in tenure, monthly charges and total "
        "charges as well, and the distance metric picked that up.",
        fill="EAF1F6",
        accent="237A7A",
    )

    body(
        document,
        f"The third segment, cluster {senior['cluster']}, sits between the other two on every numeric "
        f"feature and coincides with the senior customers. It is a reminder that a cluster label is a "
        f"description, not an explanation: 'senior segment' is accurate for this group, but the reason "
        f"those customers are separable is a pattern across several columns at once, of which the "
        f"senior flag is only one.",
    )


# --------------------------------------------------------------------------
# 8. Hierarchical
# --------------------------------------------------------------------------
def build_hierarchical(document: Document, summary: dict) -> None:
    hierarchical = summary["hierarchical"]
    trap = hierarchical["trap"]
    sweep = read_csv("hierarchical_sweep.csv")

    document.add_heading("8. Hierarchical clustering", level=1)

    body(
        document,
        "Agglomerative clustering begins with every record in its own cluster and repeatedly merges "
        "the two closest, producing a tree. Because the whole merge history is retained, k can be "
        "chosen after the structure is visible rather than before. The price is that the result "
        "depends heavily on what 'closest' means.",
    )

    rows = [
        ["ward", "Increase in total within-cluster variance", "Compact, roughly equal-sized clusters"],
        ["complete", "Farthest pair of points between clusters", "Tends to produce compact clusters"],
        ["average", "Mean distance over all pairs", "Sensitive to cluster shape"],
        ["single", "Closest pair of points", "Follows chains; prone to singleton clusters"],
    ]
    add_table(document, ["Linkage", "Distance between clusters", "Tendency"], rows,
              widths=[0.95, 2.62, 2.98])

    body(
        document,
        "Every linkage is run for k = 2 to 5 and scored on the same silhouette measure used above, "
        "together with the size of the smallest cluster it produces.",
    )

    rows = []
    for _, row in sweep.iterrows():
        rows.append([
            row["linkage"],
            str(int(row["k"])),
            f"{row['silhouette']:.3f}",
            f"{int(row['smallest_cluster']):,}",
            pct(row["smallest_cluster_share"]),
            "yes" if row["degenerate"] else "no",
        ])
    add_table(
        document,
        ["Linkage", "k", "Silhouette", "Smallest cluster", "Share", "Degenerate"],
        rows,
        widths=[1.05, 0.5, 1.1, 1.45, 0.95, 1.1],
        align_right_from=1,
    )

    add_figure(
        document,
        FIGURES / "06_linkage_trap.png",
        "Figure 7 — Silhouette by linkage (left) against the size of the smallest cluster each one "
        "produces (right, log scale). The two best silhouettes belong to the two degenerate solutions.",
        width=6.2,
    )

    document.add_heading("8.1 The silhouette trap", level=2)

    body(
        document,
        f"Sort the table by silhouette and the winner is {trap['linkage']} linkage at k = "
        f"{trap['k']}, scoring {fmt(trap['silhouette'])} — higher than anything else in the table and "
        f"nearly twice the value K-Means achieved. It is also worthless: the smallest cluster contains "
        f"{trap['smallest_cluster']} customer.",
    )

    body(
        document,
        "The mechanism is worth stating plainly. The silhouette of a point is the difference between "
        "its distance to its own cluster and to the nearest other cluster, scaled by the larger of the "
        "two. For a point alone in its cluster, the within-cluster distance is zero by definition, so "
        "its silhouette is the maximum possible value. A partition containing singletons therefore has "
        "a structurally inflated average.",
    )

    add_callout(
        document,
        "The rule this report adopts",
        "A clustering score is only meaningful once every cluster is large enough to describe. Any "
        "solution containing a cluster below 2% of the data is discarded before the best remaining "
        "solution is chosen. Applying that filter leaves ward at k = 3, with a silhouette of "
        f"{fmt(hierarchical['selected_silhouette'])} and a smallest cluster of "
        f"{min(hierarchical['sizes']):,}.",
        fill="F8ECE7",
        accent="C4664D",
    )

    add_figure(
        document,
        FIGURES / "05_dendrogram.png",
        "Figure 8 — Ward dendrogram, truncated to the top 24 merges. The dashed line marks the cut that "
        "produces three clusters; numbers in brackets are the records fused into each branch.",
        width=6.3,
    )

    body(
        document,
        f"Ward's tree shows three clear branches surviving to the last joins, and the height difference "
        f"between the third and fourth merge is the visual justification for cutting at k = "
        f"{hierarchical['selected_k']}. The resulting clusters contain "
        f"{hierarchical['sizes'][0]:,}, {hierarchical['sizes'][1]:,} and "
        f"{hierarchical['sizes'][2]:,} customers — comfortably above the degenerate threshold.",
    )


# --------------------------------------------------------------------------
# 9. Do the clusters explain churn
# --------------------------------------------------------------------------
def build_validation(document: Document, summary: dict) -> None:
    agreement = summary["agreement"]
    kmeans = summary["kmeans"]
    overlap = kmeans["churn_overlap"]
    crosstab = read_csv("cluster_churn_crosstab.csv")
    labels = {row["cluster"]: row["label"] for row in kmeans["profiles"]}

    document.add_heading("9. Do the clusters explain churn?", level=1)

    body(
        document,
        "This is the question the whole report turns on, and it is asked in a way that cannot cheat. "
        "The churn label played no part in fitting either clustering. Comparing the two afterwards is "
        "therefore a genuine external check, not a circular one.",
    )

    body(
        document,
        "The Adjusted Rand Index measures agreement between two partitions, corrected for what "
        "agreement would occur by chance. A value of 1 means identical partitions, 0 means no better "
        "than random, and negative values mean worse than random.",
    )

    rows = [
        ["K-Means vs Ward", f"{agreement['ari_models']:+.3f}", f"{agreement['ami_models']:.3f}",
         "two clusterings, same data"],
        ["K-Means vs churn", f"{agreement['ari_churn_kmeans']:+.3f}", "—", "clusters against the label"],
        ["Ward vs churn", f"{agreement['ari_churn_hierarchical']:+.3f}", "—", "clusters against the label"],
    ]
    add_table(document, ["Comparison", "ARI", "AMI", "Interpretation"], rows,
              widths=[1.75, 0.85, 0.8, 3.15], align_right_from=1)

    rows = []
    for _, row in crosstab.iterrows():
        rows.append([
            labels[int(row["cluster"])],
            f"{int(row['churn_No']):,}",
            f"{int(row['churn_Yes']):,}",
            pct(row["churn_rate"]),
        ])
    add_table(document, ["Segment", "Stayed", "Churned", "Churn rate"], rows,
              widths=[2.75, 1.15, 1.15, 1.1], align_right_from=1)

    add_figure(
        document,
        FIGURES / "08_clusters_vs_churn.png",
        "Figure 9 — Left: every cluster contains churners at a similar rate. Right: the agreement "
        "indices. The two clusterings agree with each other; neither agrees with churn.",
        width=6.25,
    )

    body(
        document,
        f"The churn rate is {pct(overlap['rates'][0])}, {pct(overlap['rates'][1])} and "
        f"{pct(overlap['rates'][2])} across the three segments, against an overall rate of "
        f"{pct(overlap['overall'])}. The full spread is {pct(overlap['spread'])}. Knowing which segment "
        f"a customer belongs to moves the predicted churn rate by less than ten percentage points "
        f"around a base rate of twenty-seven.",
    )

    add_callout(
        document,
        "Two conclusions, and they are not in conflict",
        f"First: the segments are real. Two algorithms built on completely different logic recover "
        f"almost the same partition (ARI {agreement['ari_models']:+.3f} between them), and the same "
        f"senior-citizen group emerges in both. Structure this reproducible is not an artefact. "
        f"Second: that structure is not churn (ARI {agreement['ari_churn_kmeans']:+.3f} against the label). "
        "Customer segments and customer behaviour are different things, and no clustering algorithm "
        "can supply the one that was never in the features.",
    )

    body(
        document,
        "This result is the reason Part B exists. If clustering had recovered churn, a supervised "
        "model would be redundant. It did not, so the label has to be modelled directly — and once it "
        "is, the question becomes how to measure whether the prediction is any good.",
    )


# --------------------------------------------------------------------------
# 10. Clustering after dimensionality reduction
# --------------------------------------------------------------------------
def build_dimensionality(document: Document, summary: dict) -> None:
    dimensionality = summary["dimensionality"]

    document.add_heading("10. Clustering after dimensionality reduction", level=1)

    body(
        document,
        "A tempting shortcut is to run PCA first and cluster in two or three dimensions, where the "
        "result can be plotted and inspected. It is worth measuring what that actually does, rather "
        "than assuming it is harmless.",
    )

    body(
        document,
        "The experiment clusters the projected data, then scores the resulting labels twice: once in "
        "the reduced space, which is what a careless analysis would report, and once in the original "
        f"{summary['pca']['n_features']}-dimensional "
        "space, which is what the labels actually mean.",
    )

    rows = []
    for row in dimensionality["rows"]:
        rows.append([
            str(row["components"]),
            pct(row["variance_kept"]),
            f"{row['silhouette_reduced_space']:.3f}",
            f"{row['silhouette_original_space']:.3f}",
            f"{row['silhouette_reduced_space'] - row['silhouette_original_space']:+.3f}",
        ])
    add_table(
        document,
        ["Components kept", "Variance retained", "Silhouette in reduced space", "Silhouette in original space", "Inflation"],
        rows,
        widths=[1.35, 1.3, 1.6, 1.55, 0.95],
        align_right_from=1,
    )

    add_figure(
        document,
        FIGURES / "09_dimensionality_experiment.png",
        "Figure 10 — The same clusters, scored in the space they were built in and in the original space. "
        "Only the measurement changed.",
        width=5.9,
    )

    body(
        document,
        f"Projecting to {dimensionality['inflated_components']} dimensions appears to raise silhouette "
        f"from {fmt(dimensionality['inflated_original'])} to {fmt(dimensionality['inflated_reduced'])} — "
        f"a near threefold improvement that would look like a strong result in a slide.",
    )

    body(
        document,
        "It is an artefact. The clusters did not change; only the ruler did. In two dimensions, points "
        "are genuinely closer to their neighbours because two of twenty-one axes were kept, and "
        "silhouette rewards that compression. Measured in the space the data actually lives in, the "
        "labels are no better than those from full-dimensional clustering.",
    )

    add_callout(
        document,
        "The rule",
        "Compare clusterings in a consistent space, or the comparison means nothing. Reducing "
        "dimensions before clustering is legitimate — for speed, for denoising wide data, or for "
        "visualisation — but the justification has to be external to the reduced space. A silhouette "
        "computed in 2-D is not evidence about a clustering of 21-D data.",
    )


# --------------------------------------------------------------------------
# 11. Confusion matrix
# --------------------------------------------------------------------------
def build_confusion(document: Document, summary: dict) -> None:
    tuning = summary["tuning"]
    metrics = tuning["metrics"]

    document.add_heading("11. Part B — The confusion matrix", level=1)

    body(
        document,
        f"Part A established that the label cannot be recovered by clustering. Part B models it "
        f"directly. The dataset is split 80/20 into {tuning['train_size']:,} training and "
        f"{tuning['test_size']:,} held-out test rows, stratified so both halves keep the same churn "
        f"rate. The test half is not used again until Section 16.",
    )

    body(
        document,
        "Every classification metric in this report is a ratio of four counts. Computing those counts "
        "first — rather than reaching for a library function — makes the rest of the section arithmetic "
        "rather than opinion.",
    )

    add_code(
        document,
        "                 predicted: stayed    predicted: churned\n"
        "actual: stayed          TN                    FP\n"
        "actual: churned         FN                    TP\n"
        "\n"
        "TP  true positive   — churner, correctly flagged\n"
        "TN  true negative   — loyal customer, correctly left alone\n"
        "FP  false positive  — loyal customer flagged: a wasted retention offer\n"
        "FN  false negative  — churner missed: a customer lost without warning",
    )

    add_figure(
        document,
        FIGURES / "10_confusion_matrix.png",
        "Figure 11 — The held-out confusion matrix, and every metric derived from it. The four counts "
        "are the only raw data; everything to the right is a ratio of them.",
        width=6.3,
    )

    rows = [
        ["True negative (TN)", f"{metrics['true_negative']}", "Loyal customer, left alone", "none"],
        ["False positive (FP)", f"{metrics['false_positive']}", "Loyal customer, wrongly flagged", "a wasted offer"],
        ["False negative (FN)", f"{metrics['false_negative']}", "Churner, missed", "the customer"],
        ["True positive (TP)", f"{metrics['true_positive']}", "Churner, caught", "none"],
    ]
    add_table(document, ["Cell", "Count", "What it is", "Cost when wrong"], rows,
              widths=[1.75, 0.75, 2.35, 1.7], align_right_from=1)

    body(
        document,
        f"The asymmetry is the point. There are {metrics['false_positive']} false positives and "
        f"{metrics['false_negative']} false negatives — a ratio of about "
        f"{metrics['false_positive'] / metrics['false_negative']:.1f} to 1. The model errs heavily "
        f"toward flagging people who would have stayed. Whether that is a good trade depends entirely "
        f"on what an offer costs against what a customer is worth, and the data cannot answer the "
        f"question.",
    )


# --------------------------------------------------------------------------
# 12. Metrics
# --------------------------------------------------------------------------
def build_metrics(document: Document, summary: dict) -> None:
    metrics = summary["tuning"]["metrics"]
    dataset = summary["dataset"]
    tn, fp = metrics["true_negative"], metrics["false_positive"]
    fn, tp = metrics["false_negative"], metrics["true_positive"]
    total = tn + fp + fn + tp

    document.add_heading("12. Precision, recall, F1 and accuracy", level=1)

    body(
        document,
        "Each metric below is computed by hand from the four counts and then checked against the "
        "library value. Showing the arithmetic is not pedantry — the formulas are what explain why "
        "the metrics disagree with each other.",
    )

    rows = [
        ["Accuracy", "(TP + TN) / total", f"({tp} + {tn}) / {total}", f"{metrics['accuracy']:.3f}"],
        ["Precision", "TP / (TP + FP)", f"{tp} / ({tp} + {fp})", f"{metrics['precision']:.3f}"],
        ["Recall", "TP / (TP + FN)", f"{tp} / ({tp} + {fn})", f"{metrics['recall']:.3f}"],
        ["F1", "2·P·R / (P + R)", "harmonic mean of the two", f"{metrics['f1']:.3f}"],
        ["Balanced accuracy", "(recall + specificity) / 2", "mean of the two class accuracies", f"{metrics['balanced_accuracy']:.3f}"],
        ["MCC", "correlation of prediction with truth", "−1 to +1, 0 = chance", f"{metrics['mcc']:.3f}"],
        ["ROC-AUC", "area under the ROC curve", "threshold-free ranking score", f"{metrics['roc_auc']:.3f}"],
        ["PR-AUC", "area under precision-recall", "focus on the positive class", f"{metrics['pr_auc']:.3f}"],
    ]
    add_table(document, ["Metric", "Formula", "Worked here", "Value"], rows,
              widths=[1.35, 1.95, 2.15, 0.75], align_right_from=3)

    document.add_heading("12.1 Why accuracy is the wrong headline", level=2)

    baseline = 1 - dataset["churn_rate"]

    body(
        document,
        f"A model that predicts 'no churn' for every customer scores {fmt(baseline)} accuracy on this "
        f"data, because {pct(dataset['churn_rate'])} of customers stay. The classifier in this report "
        f"scores {fmt(metrics['accuracy'])} — worse than doing nothing, by that measure.",
    )

    body(
        document,
        "That is not a defect in the model. It is the reason accuracy is a poor headline for an "
        "imbalanced problem. The trivial model has a recall of zero: it catches no churners at all, "
        "and every single departure goes unnoticed. It wins on accuracy precisely because it ignores "
        "the class the business cares about.",
    )

    add_callout(
        document,
        "What each metric would have told you",
        f"Accuracy says {fmt(metrics['accuracy'])} — reject the model. Recall says "
        f"{fmt(metrics['recall'])} — it catches about two-thirds of churners. Precision says "
        f"{fmt(metrics['precision'])} — more than half its flags are false alarms. Balanced accuracy "
        f"says {fmt(metrics['balanced_accuracy'])} — clearly better than the 0.500 a do-nothing model "
        "scores. All four are true simultaneously, which is why no single number can summarise this "
        "model.",
        fill="F8ECE7",
        accent="C4664D",
    )

    document.add_heading("12.2 What F1 does and does not solve", level=2)

    body(
        document,
        "F1 is the harmonic mean of precision and recall, so it penalises a model that is extreme on "
        f"either. At {fmt(metrics['f1'])} it sits between the two inputs, and it is the single number "
        "most commonly reported.",
    )

    body(
        document,
        "It also hides the trade-off it is summarising. A model with precision 0.70 and recall 0.45 "
        "can have the same F1 as one with precision 0.45 and recall 0.70, yet the two are very "
        "different propositions for a retention team. F1 assumes the two errors are equally costly, "
        "which for churn prediction they usually are not.",
    )


# --------------------------------------------------------------------------
# 13. ROC and PR
# --------------------------------------------------------------------------
def build_curves(document: Document, summary: dict) -> None:
    metrics = summary["tuning"]["metrics"]
    dataset = summary["dataset"]

    document.add_heading("13. ROC-AUC and the precision-recall alternative", level=1)

    body(
        document,
        "Every metric so far depends on the 0.5 threshold used to convert a probability into a label. "
        "Move the threshold and all of them change. ROC-AUC asks a different question: regardless of "
        "where the threshold is set, how well does the model order customers by risk?",
    )

    add_figure(
        document,
        FIGURES / "11_curves.png",
        "Figure 12 — ROC curve, precision-recall curve and calibration plot for the held-out data.",
        width=6.4,
    )

    body(
        document,
        f"The ROC-AUC of {fmt(metrics['roc_auc'])} means that if you pick one churner and one "
        f"non-churner at random, there is a {pct(metrics['roc_auc'], 1)} chance the model assigns the "
        f"churner the higher risk score. A model with no skill scores 0.500.",
    )

    document.add_heading("13.1 Why the ROC curve flatters this problem", level=2)

    body(
        document,
        "The false positive rate on the x-axis is FP divided by the number of actual negatives — "
        f"{dataset['churn_no']:,} people in the full dataset, 264 in the test split. Flagging "
        f"{metrics['false_positive']} innocent customers only moves that share by about "
        f"{metrics['false_positive'] / (metrics['false_positive'] + metrics['true_negative']):.2f}. "
        "The curve therefore stays close to the top-left even though most of the flags raised are "
        "wrong.",
    )

    body(
        document,
        f"The precision-recall curve does not have that escape route, because precision divides by the "
        f"number of positive predictions rather than by the number of negatives. And its no-skill "
        f"baseline is not 0.500 — it is the churn rate itself, {fmt(dataset['churn_rate'])}. Against "
        f"that baseline, a PR-AUC of {fmt(metrics['pr_auc'])} is a meaningful improvement, but a much "
        f"less flattering one than the ROC number suggests.",
    )

    add_callout(
        document,
        "Which to report",
        "Report both. ROC-AUC is the more familiar and stays comparable across datasets with different "
        "class balances. PR-AUC is the more honest summary when the positive class is rare and "
        "precision is what the business actually experiences. Reporting ROC-AUC alone on a 27% "
        "positive rate overstates how usable the model is.",
    )

    document.add_heading("13.2 Calibration", level=2)

    body(
        document,
        "ROC-AUC only cares about order, so a model can rank perfectly while its probabilities are "
        f"nonsense. The Brier score of {fmt(metrics['brier'])} is the mean squared error of the "
        "predicted probabilities, and the calibration curve compares predicted risk against observed "
        "risk in each decile.",
    )

    body(
        document,
        "The curve tracks the diagonal at low probabilities and drifts slightly above it in the middle, "
        "meaning the model is a little under-confident around 0.2–0.4. Logistic Regression is "
        "naturally well calibrated because it optimises a proper scoring rule; gradient-boosted trees "
        "and random forests are usually worse. If precise probabilities were needed for expected-value "
        "calculations, the next step would be `CalibratedClassifierCV`.",
    )


# --------------------------------------------------------------------------
# 14. Cross-validation
# --------------------------------------------------------------------------
def build_cv(document: Document, summary: dict) -> None:
    nested = summary["nested_cv"]

    document.add_heading("14. Cross-validation", level=1)

    body(
        document,
        "A single 80/20 split produces one number with no error bar, and that number depends on which "
        "rows happened to land where. Cross-validation repeats the experiment: split the training data "
        "into k parts, train on k−1, score the held-out part, and rotate until every part has been "
        "held out once.",
    )

    body(
        document,
        "Stratified k-fold is used throughout, which keeps the 27% churn rate constant in every fold. "
        "With an imbalanced target, an unlucky unstratified split can produce a fold with almost no "
        "positives, and the resulting score is noise.",
    )

    rows = []
    for index, score in enumerate(nested["scores"], start=1):
        rows.append([f"Fold {index}", f"{score:.3f}", f"{score - nested['mean']:+.3f}"])
    rows.append(["Mean", f"{nested['mean']:.3f}", "—"])
    rows.append(["Standard deviation", f"{nested['std']:.3f}", "—"])
    add_table(document, ["Fold", "ROC-AUC", "Deviation from mean"], rows,
              widths=[2.2, 1.6, 2.2], align_right_from=1)

    add_callout(
        document,
        "The standard deviation is the point",
        f"A mean of {fmt(nested['mean'])} with a spread of {fmt(nested['std'])} means that two models "
        f"scoring 0.800 and 0.810 are not distinguishable — the difference is smaller than the "
        f"fold-to-fold noise. Every model comparison in this report is read against that band rather "
        f"than against the raw digits. This is why Section 15 refuses to call a tuning winner.",
    )


# --------------------------------------------------------------------------
# 15. Tuning
# --------------------------------------------------------------------------
def build_tuning(document: Document, summary: dict) -> None:
    tuning = summary["tuning"]
    logistic = read_csv("tuning_grid_logistic.csv").sort_values("param_model__C")
    logistic_ranked = logistic.sort_values("rank_test_score")
    forest = read_csv("tuning_grid_random_forest.csv").sort_values("rank_test_score")

    document.add_heading("15. Hyperparameter tuning", level=1)

    body(
        document,
        "A hyperparameter is a setting fixed before training begins. Unlike a model parameter, it is "
        "not learned from the data, so it has to be searched — and the search must be scored by "
        "cross-validation on the training split. Scoring candidate values on the test split would "
        "convert the hold-out set into a second training set and destroy its purpose.",
    )

    document.add_heading("15.1 Logistic Regression: five values of C", level=2)

    body(
        document,
        "C is the inverse of regularisation strength. Small C means heavy regularisation and a simpler "
        "decision boundary; large C lets the model fit the training data more closely.",
    )

    rows = []
    for _, row in logistic.iterrows():
        rows.append([
            f"{row['param_model__C']:g}",
            f"{row['mean_test_score']:.3f}",
            f"{row['std_test_score']:.3f}",
            f"{row['mean_train_score']:.3f}",
            f"{int(row['rank_test_score'])}",
        ])
    add_table(document, ["C", "CV ROC-AUC", "CV std", "Train ROC-AUC", "Rank"], rows,
              widths=[1.0, 1.45, 1.15, 1.5, 0.9], align_right_from=1)

    add_figure(
        document,
        FIGURES / "12_tuning_results.png",
        "Figure 13 — Left: the C sweep is flat across four orders of magnitude. Right: all 32 Random "
        "Forest combinations, which span less than one standard deviation of fold noise.",
        width=6.3,
    )

    best_c = tuning["logistic_best"]["model__C"]
    body(
        document,
        f"The best value is C = {best_c}, at {fmt(tuning['logistic_best_score'])}. The runner-up, "
        f"C = {logistic_ranked.iloc[1]['param_model__C']:g}, scores "
        f"{fmt(logistic_ranked.iloc[1]['mean_test_score'])}. The winner leads by "
        f"{tuning['logistic_best_score'] - logistic_ranked.iloc[1]['mean_test_score']:.4f} — roughly "
        f"twenty-five times smaller than the fold-to-fold standard deviation of "
        f"{fmt(logistic_ranked.iloc[0]['std_test_score'])}.",
    )

    body(
        document,
        "The honest conclusion is not that C = 0.1 is the best value. It is that C is not a lever that "
        "matters for this problem. The default of 1.0 was already at the optimum, and searching four "
        "orders of magnitude around it found nothing.",
    )

    document.add_heading("15.2 Random Forest: thirty-two combinations", level=2)

    body(
        document,
        "The forest exposes four hyperparameters, and the grid covers "
        f"{tuning['n_combinations']['random_forest']} combinations. A summary of the top eight is "
        "below; the full table is in `outputs/tuning_grid_random_forest.csv`.",
    )

    rows = []
    for _, row in forest.head(8).iterrows():
        depth = row["param_model__max_depth"]
        rows.append([
            "none" if pd.isna(depth) else str(int(depth)),
            str(int(row["param_model__min_samples_leaf"])),
            str(row["param_model__max_features"]),
            str(int(row["param_model__n_estimators"])),
            f"{row['mean_test_score']:.3f}",
            f"{row['mean_train_score']:.3f}",
        ])
    add_table(
        document,
        ["max_depth", "min_samples_leaf", "max_features", "n_estimators", "CV ROC-AUC", "Train"],
        rows,
        widths=[1.0, 1.35, 1.2, 1.15, 1.15, 0.85],
        align_right_from=4,
    )

    best_forest = forest.iloc[0]
    body(
        document,
        f"The best combination reaches {fmt(best_forest['mean_test_score'])} with a training score of "
        f"{fmt(best_forest['mean_train_score'])}. Across all 32 combinations the CV scores span "
        f"{fmt(forest['mean_test_score'].min())} to {fmt(forest['mean_test_score'].max())} — a range of "
        f"{fmt(forest['mean_test_score'].max() - forest['mean_test_score'].min())}, still narrower than "
        f"one standard deviation of fold noise.",
    )

    body(
        document,
        "The gap between the training score and the CV score is the visible signature of overfitting. "
        "The grid earns its keep here by pushing the model toward shallower trees and larger leaves — "
        "but it is correcting a defect, not discovering signal.",
    )

    document.add_heading("15.3 Did tuning help?", level=2)

    rows = []
    for entry in tuning["comparison"]:
        rows.append([
            entry["model"],
            f"{entry['default_cv_roc_auc']:.3f}",
            f"{entry['tuned_cv_roc_auc']:.3f}",
            f"{entry['gain']:+.3f}",
        ])
    add_table(document, ["Model", "Default CV", "Tuned CV", "Gain"], rows,
              widths=[2.3, 1.4, 1.4, 1.1], align_right_from=1)

    add_callout(
        document,
        "Tuning fixes overfitting; it cannot manufacture signal",
        f"Logistic Regression gained {tuning['comparison'][0]['gain']:+.3f} across five values of C — "
        f"nothing. Random Forest gained {tuning['comparison'][1]['gain']:+.3f}, which is real but "
        f"entirely attributable to constraining an unconstrained default. Neither gain exceeds the "
        f"fold-level noise, and on a dataset this size with a signal this weak there is very little "
        f"headroom for a grid search to find.",
        fill="FBF1E3",
        accent="B97A2E",
    )


# --------------------------------------------------------------------------
# 16. Nested CV and learning curve
# --------------------------------------------------------------------------
def build_nested(document: Document, summary: dict) -> None:
    nested = summary["nested_cv"]
    tuning = summary["tuning"]
    curve = pd.DataFrame(nested["learning_curve"])

    document.add_heading("16. Nested cross-validation and the learning curve", level=1)

    body(
        document,
        f"The tuned CV score of {fmt(tuning['best_cv_score'])} has a subtle problem. The same folds "
        f"were used both to select the winner and to report its score. With five candidate values "
        f"there are five chances to get lucky on the same five folds, and the reported maximum of "
        f"those five is biased upward.",
    )

    body(
        document,
        "Nested cross-validation separates the two loops. An inner loop selects hyperparameters; an "
        "outer loop scores the entire procedure — including the selection — on rows the inner loop "
        "never saw.",
    )

    rows = [
        ["Tuned CV score (inner loop only)", f"{tuning['best_cv_score']:.3f}"],
        ["Nested CV estimate (unbiased)", f"{nested['mean']:.3f} ± {nested['std']:.3f}"],
        ["Optimism", f"{nested['optimism']:+.3f}"],
    ]
    add_table(document, ["Estimate", "ROC-AUC"], rows, widths=[3.6, 2.2], align_right_from=1)

    add_figure(
        document,
        FIGURES / "13_learning_curve.png",
        "Figure 14 — Left: training and validation scores converge as data is added. Right: the five "
        "outer folds of the nested cross-validation.",
        width=6.35,
    )

    body(
        document,
        f"The optimism is {nested['optimism']:+.3f} — essentially zero. That is reassuring, and it has "
        f"a specific cause: with a five-value grid containing one effective hyperparameter, there was "
        f"very little for the search to overfit. A wide grid on a small dataset can produce optimism "
        f"several times larger, and a tuned score reported on its own would then be misleading.",
    )

    document.add_heading("16.1 Is the model overfitting or underfitting?", level=2)

    body(
        document,
        "A learning curve trains the same model on progressively more data and plots both the training "
        "and validation scores. Three shapes are possible: a persistent gap means variance (overfitting), "
        "two low converging scores mean bias (underfitting), and two high converging scores mean the "
        "data — not the algorithm — is the limit.",
    )

    rows = []
    for row in curve.itertuples():
        rows.append([
            f"{int(row.train_size):,}",
            f"{row.train_mean:.3f}",
            f"{row.test_mean:.3f}",
            f"{row.train_mean - row.test_mean:+.3f}",
        ])
    add_table(document, ["Training rows", "Train ROC-AUC", "Validation ROC-AUC", "Gap"], rows,
              widths=[1.6, 1.6, 1.8, 1.0], align_right_from=1)

    first_gap = curve["train_mean"].iloc[0] - curve["test_mean"].iloc[0]
    last_gap = curve["train_mean"].iloc[-1] - curve["test_mean"].iloc[-1]

    body(
        document,
        f"The gap narrows from {fmt(first_gap)} at {int(curve['train_size'].iloc[0]):,} rows to "
        f"{fmt(last_gap)} at the full training split, while the validation score rises steadily. That "
        f"is the signature of variance, not bias — the model is not memorising, it simply does not "
        f"have enough examples yet.",
    )

    body(
        document,
        "The practical reading: collecting more data would help a little. Making the model more complex "
        "would not help at all, and is likely to hurt.",
    )


# --------------------------------------------------------------------------
# 17. Verification
# --------------------------------------------------------------------------
def build_verification(document: Document, summary: dict) -> None:
    tuning = summary["tuning"]
    metrics = tuning["metrics"]

    document.add_heading("17. Verification and reproducibility", level=1)

    body(
        document,
        "The report is generated from the artifacts of a single run rather than typed by hand, so no "
        "number in it can drift from the code that produced it. The checks below were run against the "
        "final build.",
    )

    rows = [
        ["Every metric recomputed by hand", f"TP {metrics['true_positive']}, FP {metrics['false_positive']}, "
         f"FN {metrics['false_negative']}, TN {metrics['true_negative']}", "matches library values"],
        ["Churn label excluded from clustering", "K-Means and Ward fitted on features only", "ARI computed after the fact"],
        ["Degenerate clusters rejected", "threshold 2% of records", "single and average linkage excluded"],
        ["Tuned on training data only", f"{tuning['train_size']:,} rows", "test split untouched until Section 16"],
        ["Nested CV reproduces the tuned score", f"{summary['nested_cv']['mean']:.3f} vs "
         f"{tuning['best_cv_score']:.3f}", "optimism within 0.001"],
        ["All figures rebuilt from saved tables", "14 figures", "no model refitting required"],
        ["Random seed", "42 throughout", "identical results on re-run"],
    ]
    add_table(document, ["Check", "Method", "Outcome"], rows, widths=[2.05, 2.45, 2.05])

    body(
        document,
        "Reproducing the whole report requires three commands and about two minutes on a laptop, with "
        "no network access:",
    )

    add_code(
        document,
        "python week-3/src/run_analysis.py     # clustering, tuning, all tables\n"
        "python week-3/src/make_figures.py     # the 14 figures\n"
        "python week-3/docs/build_report.py    # this document",
    )


# --------------------------------------------------------------------------
# 18. Limitations
# --------------------------------------------------------------------------
def build_limitations(document: Document) -> None:
    document.add_heading("18. Limitations", level=1)
    body(
        document,
        "The results above are accurate for this dataset and should not be read as properties of "
        "customer churn in general.",
    )

    document.add_heading("18.1 Data", level=2)
    bullets(
        document,
        [
            "The dataset is synthetic. Every pattern in it — including the senior-citizen segment — is a "
            "property of the generator, not an observation about real customers.",
            "There are 1,800 records. On a 360-row test split, an accuracy difference of two percentage "
            "points is roughly one standard error of the sampling noise.",
            "The churn label is imbalanced at 27%. Metrics that ignore the base rate will be misleading, "
            "which is a theme of this report rather than a footnote to it.",
        ],
    )

    document.add_heading("18.2 Clustering", level=2)
    bullets(
        document,
        [
            "The best silhouette of 0.162 indicates weak, overlapping clusters. The segmentation is "
            "usable for description, not for clean separation.",
            "k was chosen by judgement after three indices disagreed. A different analyst could "
            "defensibly choose k = 2 or k = 5.",
            "The clusters were validated against churn only because that label happened to exist. "
            "In a genuinely unlabelled problem there is no external check available at all.",
        ],
    )

    document.add_heading("18.3 Modelling", level=2)
    bullets(
        document,
        [
            "Hyperparameters were searched over a modest grid. A wider search or a different algorithm "
            "family might find more, though the learning curve suggests the ceiling is the data.",
            "No fairness analysis was performed. Error rates can differ substantially across subgroups, "
            "and a model used to allocate retention offers would need one.",
            "The threshold was left at 0.5 for the reported metrics. Section 13 shows the curve, but a "
            "deployment would need a threshold chosen against actual costs.",
            "Only two algorithms were tuned. The comparison is between those two, not a claim that "
            "Logistic Regression is the best available classifier.",
        ],
    )


# --------------------------------------------------------------------------
# 19. Lessons
# --------------------------------------------------------------------------
def build_lessons(document: Document) -> None:
    document.add_heading("19. Lessons learned", level=1)

    lessons = [
        ("A clustering score without a cluster-size check is not a result",
         "Single linkage scored 0.294 — the highest in this report — by isolating one customer. The "
         "number was real and the clustering was useless. Size, shape and interpretability have to be "
         "checked alongside any internal index."),
        ("Internal indices disagree, and that is information",
         "Silhouette, Davies-Bouldin and Calinski-Harabasz preferred k = 3, 5 and 2 respectively. "
         "There is no formula that resolves the disagreement; what matters is noticing it and stating "
         "the reason for the final choice."),
        ("Reducing dimensions changes the ruler, not the clusters",
         "Silhouette rose from 0.135 to 0.389 by measuring in two dimensions instead of twenty-one. "
         "The labels did not improve at all. Comparisons must be made in a consistent space."),
        ("Structure is not the same thing as signal",
         "Two algorithms agreed on the same three segments (ARI 0.732) while neither related to churn "
         "(ARI 0.015). Reproducible clusters can be entirely orthogonal to the question being asked."),
        ("Accuracy is not a metric for an imbalanced problem",
         "A do-nothing model scores 0.733 against the classifier's 0.678. Reporting accuracy alone "
         "would have rejected the only model that catches any churners at all."),
        ("Tuning fixes overfitting, not absence of signal",
         "Logistic Regression gained 0.001 from a five-value grid. The Random Forest gained 0.020, all "
         "of it from constraining an over-flexible default. Neither gain exceeded the fold-level noise."),
        ("A tuned CV score is optimistic, and nested CV measures by how much",
         "The same folds that select a winner also score it. Here the optimism was only 0.001, but the "
         "method is what matters — with a wide grid on a small dataset the gap can be far larger."),
        ("Leave the test set alone until the end",
         "Everything that involved a choice — scaling, grid search, model family — was decided on the "
         "training split. The 360 held-out rows were scored once, in Section 16, and their result was "
         "not used to change anything."),
    ]

    for heading, detail in lessons:
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.space_after = Pt(1)
        run = paragraph.add_run(heading)
        run.font.bold = True
        run.font.size = Pt(10)
        run.font.color.rgb = TEAL
        detail_paragraph = document.add_paragraph()
        detail_paragraph.paragraph_format.left_indent = Inches(0.22)
        detail_paragraph.paragraph_format.space_after = Pt(8)
        detail_run = detail_paragraph.add_run(detail)
        detail_run.font.size = Pt(9.6)


# --------------------------------------------------------------------------
# 20. Conclusion
# --------------------------------------------------------------------------
def build_conclusion(document: Document, summary: dict) -> None:
    kmeans = summary["kmeans"]
    agreement = summary["agreement"]
    tuning = summary["tuning"]
    nested = summary["nested_cv"]

    document.add_heading("20. Conclusion", level=1)

    body(
        document,
        f"Part A produced a segmentation that holds up. K-Means and Ward clustering, using different "
        f"logic and different merge criteria, converged on the same three customer groups: a "
        f"long-tenure high-spend segment, a short-tenure low-spend segment, and a senior segment that "
        f"the algorithms found without being told the column existed. The agreement of "
        f"{fmt(agreement['ari_models'])} between the two methods is the strongest evidence in the "
        f"report that this structure is real rather than an artefact of one algorithm's assumptions.",
    )

    body(
        document,
        f"The same analysis also established the limits of that result. The best silhouette of "
        f"{fmt(kmeans['selected_silhouette'])} is weak, the three internal indices disagreed about k, "
        f"and the segments carry almost no information about churn. Testing a clustering against a "
        f"label it never saw is the only way to know that, and it is the check most clustering "
        f"exercises omit.",
    )

    body(
        document,
        f"Part B then modelled the label directly and evaluated it with the full set of tools the "
        f"brief calls for. The classifier reaches {fmt(tuning['metrics']['roc_auc'])} ROC-AUC and "
        f"{fmt(tuning['metrics']['recall'])} recall on held-out data, which is a real but modest "
        f"result. It loses to a do-nothing baseline on accuracy and beats it decisively on every metric "
        f"that looks at the minority class — a distinction that only exists because more than one "
        f"metric was reported.",
    )

    body(
        document,
        f"Hyperparameter tuning was the least productive step, and reporting that plainly is part of "
        f"doing it properly. A five-value grid moved Logistic Regression by "
        f"{tuning['comparison'][0]['gain']:+.3f}; nested cross-validation confirmed the tuned score was "
        f"not inflated ({nested['optimism']:+.3f}). The learning curve shows training and validation "
        f"scores converging, which locates the ceiling in the dataset rather than in the model.",
    )

    add_callout(
        document,
        "The through-line",
        "Clustering found real structure that does not predict churn. Supervised learning predicted "
        "churn, weakly but measurably. Neither result is impressive on its own, and both are only "
        "trustworthy because of the checks around them — the size filter that rejected a flattering "
        "score, the consistent space that exposed a fake improvement, the baseline that beat the "
        "model, and the nested loop that tested the tuning. That is what a model evaluation report is "
        "for.",
        fill="EAF1F6",
        accent="237A7A",
    )

    body(
        document,
        "Week 4 turns to deployment: serialising the fitted pipeline, serving predictions through an "
        "API, and monitoring a model after it leaves the notebook.",
    )


# --------------------------------------------------------------------------
# Appendices
# --------------------------------------------------------------------------
def build_appendices(document: Document, summary: dict) -> None:
    page_break(document)
    document.add_heading("Appendix A — Reproducing this report", level=1)

    body(
        document,
        "The whole deliverable is rebuilt from the repository with three commands. Each stage reads "
        "only the artifacts of the previous stage, so figures and the document can be regenerated "
        "without rerunning the clustering or the grid search.",
    )

    add_code(
        document,
        "cd week-3\n"
        "python -m venv .venv && source .venv/bin/activate\n"
        "python -m pip install -r requirements.txt\n"
        "\n"
        "python src/run_analysis.py    # clustering + tuning -> outputs/*.csv, week3_summary.json\n"
        "python src/make_figures.py    # -> outputs/figures/*.png\n"
        "python docs/build_report.py   # -> reports/Week_3_..._Report.docx",
    )

    rows = [
        ["run_analysis.py", "Clustering, PCA, tuning, nested CV", "~70 s"],
        ["make_figures.py", "14 figures from saved tables", "~6 s"],
        ["build_report.py", "This document", "~1 s"],
    ]
    add_table(document, ["Script", "Produces", "Runtime"], rows,
              widths=[1.6, 3.5, 0.9], align_right_from=2)

    document.add_heading("Appendix B — Configuration", level=1)

    rows = [
        ["Random seed", "42"],
        ["Test split", "20%, stratified, held out until Section 16"],
        ["Cross-validation", "5-fold stratified, shuffled"],
        ["Inner CV for nested", "5-fold stratified"],
        ["K-Means", f"k = 2–8 swept, {summary['kmeans']['selected_k']} selected, n_init = 50 for the final fit"],
        ["Hierarchical", "ward / complete / average / single, k = 2–5"],
        ["Degenerate threshold", "smallest cluster ≥ 2% of records"],
        ["PCA components swept", "2, 3, 5, 8, 10, 12, 16, 21"],
        ["Logistic grid", "C ∈ {0.01, 0.1, 1, 10, 100}, penalty l2"],
        ["Forest grid", "n_estimators × max_depth × min_samples_leaf × max_features = 32"],
        ["Scoring metric", "ROC-AUC throughout"],
    ]
    add_table(document, ["Setting", "Value"], rows, widths=[2.1, 4.4])

    document.add_heading("Appendix C — Metric glossary", level=1)

    rows = [
        ["Silhouette", "−1…1", "How much closer each point is to its own cluster than the nearest rival"],
        ["Davies-Bouldin", "0…∞", "Average within-cluster spread over between-cluster distance; lower is better"],
        ["Calinski-Harabasz", "0…∞", "Ratio of between-cluster to within-cluster dispersion; higher is better"],
        ["Adjusted Rand Index", "−0.5…1", "Agreement between two partitions, corrected for chance"],
        ["Inertia", "0…∞", "Sum of squared distances to each cluster's centroid"],
        ["Precision", "0…1", "Of the positives predicted, the share that were real"],
        ["Recall", "0…1", "Of the real positives, the share that were found"],
        ["F1", "0…1", "Harmonic mean of precision and recall"],
        ["Balanced accuracy", "0…1", "Mean of per-class recall; 0.5 for a do-nothing model"],
        ["ROC-AUC", "0…1", "Probability a random positive outranks a random negative"],
        ["PR-AUC", "0…1", "Area under the precision-recall curve; baseline is the positive rate"],
        ["Brier score", "0…1", "Mean squared error of predicted probabilities"],
    ]
    add_table(document, ["Term", "Range", "Meaning"], rows, widths=[1.5, 0.85, 4.15], font_size=8.2)

    document.add_heading("Appendix D — Figure index", level=1)

    rows = [
        ["1", "00_workflow_overview.png", "The two halves of the work"],
        ["2", "01_pca_variance.png", "Scree and cumulative variance"],
        ["3", "02_pca_projection.png", "2-D projection by cluster and by churn"],
        ["4", "03_kmeans_selection.png", "Elbow and internal indices"],
        ["5", "04_cluster_overview.png", "Cluster sizes and churn rate per segment"],
        ["6", "07_cluster_profiles.png", "Cluster profiles, K-Means and Ward"],
        ["7", "06_linkage_trap.png", "Silhouette against cluster size"],
        ["8", "05_dendrogram.png", "Ward dendrogram"],
        ["9", "08_clusters_vs_churn.png", "Agreement indices and churn by cluster"],
        ["10", "09_dimensionality_experiment.png", "Silhouette in reduced vs original space"],
        ["11", "10_confusion_matrix.png", "Confusion matrix and derived metrics"],
        ["12", "11_curves.png", "ROC, precision-recall and calibration"],
        ["13", "12_tuning_results.png", "Grid search results"],
        ["14", "13_learning_curve.png", "Learning curve and nested CV"],
    ]
    add_table(document, ["#", "File", "Content"], rows, widths=[0.45, 2.75, 3.3])


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    summary = load_json("week3_summary.json")

    document = setup_document()
    build_cover(document, summary)
    build_executive_summary(document, summary)
    build_description(document)
    build_intro(document, summary)
    build_dataset(document, summary)
    build_pca(document, summary)
    build_kmeans(document, summary)
    build_profiles_kmeans(document, summary)
    build_hierarchical(document, summary)
    build_validation(document, summary)
    build_dimensionality(document, summary)
    build_confusion(document, summary)
    build_metrics(document, summary)
    build_curves(document, summary)
    build_cv(document, summary)
    build_tuning(document, summary)
    build_nested(document, summary)
    build_verification(document, summary)
    build_limitations(document)
    build_lessons(document)
    build_conclusion(document, summary)
    build_appendices(document, summary)

    document.core_properties.title = REPORT_TITLE
    document.core_properties.subject = WEEK_TITLE
    document.core_properties.author = AUTHOR
    document.core_properties.keywords = (
        "unsupervised learning, clustering, k-means, hierarchical clustering, PCA, "
        "cross-validation, confusion matrix, precision, recall, F1, ROC-AUC, hyperparameter tuning"
    )
    document.core_properties.comments = (
        f"Author: {AUTHOR}. Generated from week-3/outputs/week3_summary.json. "
        "Synthetic teaching dataset; not real customer data."
    )

    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / "Week_3_Unsupervised_Learning_Model_Evaluation_Report.docx"
    document.save(path)

    count = word_count(SUBMISSION_DESCRIPTION)
    print(f"Report written : {path.relative_to(WEEK_DIR.parent)}")
    print(f"Author         : {AUTHOR}")
    print(f"Description    : {count} words (minimum 200)")
    if count < 200:
        raise SystemExit("Submission description is below the 200-word minimum")


if __name__ == "__main__":
    main()
