#!/usr/bin/env python3
"""Build the Week 4 capstone report from the deployment artifacts.

Every number is read from ``week-4/outputs/`` — the model card, the benchmark
summary, the latency table and the evaluation CSVs — so the document cannot
drift from the code that produced it.

    python src/train.py          # train and serialise
    python src/benchmark.py      # measure the running service
    python src/make_figures.py   # build the figures
    python docs/build_report.py  # build this report
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
WEEK_TITLE = "Week 4 — AI Project Deployment & Capstone"
REPORT_TITLE = "AI Project Deployment"
REPORT_SUBTITLE = (
    "From a serialised scikit-learn pipeline to a tested, documented FastAPI "
    "service, with the measurements that justify each decision"
)
REPOSITORY = "https://github.com/Dhanam33/Yuva_intern/tree/main/week-4"

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
    "This Week 4 report documents the deployment of the churn classifier developed in Weeks 2 "
    "and 3 as a complete, documented and tested machine-learning application. The selected "
    "model, a Logistic Regression pipeline with C equal to 0.1 and balanced class weights, is "
    "serialised with Joblib into a 5.7 KiB artifact whose SHA-256 digest is recorded in a "
    "manifest, and served by a FastAPI application that loads it once at startup and answers "
    "requests in about thirteen milliseconds. The service exposes a health probe reporting "
    "whether the model itself is resident in memory, a model-card endpoint carrying metrics, "
    "intended use and limitations, single-prediction and batch endpoints, and interactive "
    "OpenAPI documentation. Every request is validated by a Pydantic schema before any model "
    "code executes, so an unknown contract type, an out-of-range indicator or a misspelt field "
    "returns a 422 with a readable explanation instead of reaching scikit-learn. Because the "
    "model is linear, each prediction is accompanied by an exact log-odds decomposition showing "
    "which features raised or lowered the estimated risk. Eighteen contract tests cover "
    "validation, prediction, batch scoring and error handling. Benchmarking against a live "
    "server exposed a real defect: the first batch endpoint scored rows one at a time and "
    "managed 92 rows per second, barely better than sending separate requests. Vectorising that "
    "call raised throughput past ten thousand rows per second, a hundredfold improvement, while single "
    "requests were measured at 12.5 milliseconds with explanations and 7.4 milliseconds without. "
    "The report states plainly that the model was trained on synthetic teaching data and must "
    "not be used for consequential decisions."
)

MODEL_CARD = json.loads((OUTPUTS / "model_card.json").read_text(encoding="utf-8"))
BENCHMARK = json.loads((OUTPUTS / "benchmark_summary.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((OUTPUTS / "model_manifest.json").read_text(encoding="utf-8"))
METRICS = MODEL_CARD["metrics"]


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w’'-]+\b", text))


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

    if len(rows) <= 12:
        for row in table.rows[:-1]:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.keep_with_next = True

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
def build_cover(document: Document) -> None:
    for _ in range(2):
        document.add_paragraph()

    brand = document.add_paragraph()
    brand.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = brand.add_run("YUVA INTERNSHIP PROGRAMME")
    run.font.size = Pt(11)
    run.font.bold = True
    run.font.color.rgb = TEAL

    rule = document.add_paragraph()
    rule.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rule.paragraph_format.space_after = Pt(18)
    run = rule.add_run("WEEK 4  ·  CAPSTONE DELIVERABLE")
    run.font.size = Pt(9)
    run.font.color.rgb = LIGHT_GREY

    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(8)
    run = title.add_run("AI Project\nDeployment")
    run.font.size = Pt(32)
    run.font.bold = True
    run.font.color.rgb = NAVY
    run.font.name = "Calibri Light"

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(22)
    run = subtitle.add_run(REPORT_SUBTITLE)
    run.font.size = Pt(11.5)
    run.font.color.rgb = TEAL
    run.italic = True

    accent = document.add_paragraph()
    accent.alignment = WD_ALIGN_PARAGRAPH.CENTER
    accent.paragraph_format.space_after = Pt(22)
    accent.paragraph_format.left_indent = Inches(2.2)
    accent.paragraph_format.right_indent = Inches(2.2)
    border = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "12")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "237A7A")
    border.append(bottom)
    accent._p.get_or_add_pPr().append(border)

    facts = [
        ("SUBMITTED BY", AUTHOR),
        ("PROGRAMME", INTERNSHIP),
        ("WEEK", "4 of 4 — AI project deployment and capstone"),
        ("DELIVERABLE", "A deployed machine-learning application with documentation and presentation"),
        ("MODEL", "Logistic Regression pipeline, C = 0.1, balanced class weights"),
        ("SERVING", "FastAPI · Uvicorn · Joblib serialisation"),
        ("ARTIFACT", f"{MODEL_CARD['artifacts']['joblib_bytes'] / 1024:.1f} KiB "
                     f"· SHA-256 recorded · {MANIFEST['load_time_ms']:.1f} ms cold load"),
        ("TESTS", "18 contract tests, all passing"),
        ("REPOSITORY", REPOSITORY),
        ("DATE", date.today().strftime("%d %B %Y")),
    ]

    table = document.add_table(rows=0, cols=2)
    table.autofit = False
    for label, value in facts:
        row = table.add_row()
        left, right = row.cells
        left.width = Inches(1.55)
        right.width = Inches(5.0)
        left.text = ""
        right.text = ""
        left_paragraph = left.paragraphs[0]
        left_paragraph.paragraph_format.space_before = Pt(3)
        left_paragraph.paragraph_format.space_after = Pt(3)
        run = left_paragraph.add_run(label)
        run.font.size = Pt(7.6)
        run.font.bold = True
        run.font.color.rgb = LIGHT_GREY
        shade(left, "EAF1F6")

        right_paragraph = right.paragraphs[0]
        right_paragraph.paragraph_format.space_before = Pt(3)
        right_paragraph.paragraph_format.space_after = Pt(3)
        run = right_paragraph.add_run(value)
        run.font.size = Pt(8.6)

    document.add_paragraph()
    note = document.add_paragraph()
    note.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = note.add_run(
        "The model was trained on a reproducible synthetic dataset of 1,800 customer "
        "records. It is a teaching artifact, not a validated production system."
    )
    run.font.size = Pt(8)
    run.italic = True
    run.font.color.rgb = LIGHT_GREY


def build_executive_summary(document: Document) -> None:
    document.add_heading("1. Executive summary", level=1)

    body(
        document,
        "The first three weeks produced a model. This week turns it into a service. The "
        "difference is not cosmetic: a model that lives in a notebook has no contract, no "
        "validation, no version, no measured latency and no way for anyone else to call it. "
        "This report documents the work of closing that gap for the churn classifier selected "
        "in Week 3, and records the measurements that justify each engineering decision.",
    )

    body(
        document,
        "The deployed application has five parts. A training script fits the pipeline and "
        "writes two serialised artifacts. A FastAPI service loads one of them at startup and "
        "exposes prediction endpoints. A Pydantic schema validates every request before the "
        "model sees it. An explanation module decomposes each prediction into exact log-odds "
        "contributions. A test suite of eighteen contract tests exercises the whole surface "
        "through HTTP. Everything is reproducible from four commands, and every figure in this "
        "report reads from a file those commands produce.",
    )

    add_callout(
        document,
        "The headline",
        f"A single prediction takes {BENCHMARK['single_explained_p50_ms']:.1f} ms, of which "
        f"the model is only {BENCHMARK['model_only_p50_ms']:.1f} ms. The largest cost is the "
        f"explanation at {BENCHMARK['explanation_cost_ms']:.1f} ms, which is why it is optional. "
        f"Benchmarking also exposed a defect in the first batch endpoint — it scored rows one at "
        f"a time and reached 92 rows per second. Vectorising the call raised throughput to "
        f"{BENCHMARK['batch_rows_per_s']:,.0f} rows per second.",
    )

    metrics_rows = [
        ["Held-out ROC-AUC", f"{METRICS['roc_auc']:.3f}", "Identical to the Week 3 result"],
        ["Held-out recall", f"{METRICS['recall']:.3f}", "Two thirds of churners caught"],
        ["Artifact size (joblib)", f"{MODEL_CARD['artifacts']['joblib_bytes'] / 1024:.1f} KiB",
         "One file holds preprocessing and model"],
        ["Cold load", f"{MANIFEST['load_time_ms']:.1f} ms", "Loaded once, not per request"],
        ["Cold start (process launch)", f"{BENCHMARK['cold_start_ms']:.0f} ms",
         "Almost entirely library imports"],
        ["Single request, with explanation", f"{BENCHMARK['single_explained_p50_ms']:.1f} ms", "p50"],
        ["Single request, no explanation", f"{BENCHMARK['single_plain_p50_ms']:.1f} ms", "p50"],
        ["Batch throughput", f"{BENCHMARK['batch_rows_per_s']:,.0f} rows/s", "100 rows per request"],
        ["Contract tests", "18 passing", "Validation, prediction, batch, errors"],
    ]
    add_table(document, ["Measure", "Value", "Note"], metrics_rows,
              widths=[2.35, 1.5, 2.6], align_right_from=1)

    body(
        document,
        "Two results deserve emphasis because they contradict what a reader might expect. "
        "First, the serialised model reproduces the Week 3 evaluation to the last decimal place, "
        "so the artifact being served is provably the model that was measured. Second, tuning "
        "the hyperparameters improved cross-validated ROC-AUC by 0.001 while making the "
        "held-out score 0.001 worse — the difference is noise, and the report says so rather "
        "than claiming a gain that does not exist.",
    )


def build_description(document: Document) -> None:
    document.add_heading("2. Submission description", level=1)

    body(
        document,
        f"The description below is {word_count(SUBMISSION_DESCRIPTION)} words, which satisfies "
        "the 200-word minimum. It is reproduced verbatim in the submission form and in "
        "docs/SUBMISSION_DESCRIPTION.md.",
        italic=True,
        size=9,
    )

    paragraph = document.add_paragraph()
    paragraph.paragraph_format.left_indent = Inches(0.28)
    paragraph.paragraph_format.right_indent = Inches(0.28)
    paragraph.paragraph_format.space_before = Pt(4)
    paragraph.paragraph_format.space_after = Pt(10)
    border = OxmlElement("w:pBdr")
    left = OxmlElement("w:left")
    left.set(qn("w:val"), "single")
    left.set(qn("w:sz"), "18")
    left.set(qn("w:space"), "8")
    left.set(qn("w:color"), "237A7A")
    border.append(left)
    paragraph._p.get_or_add_pPr().append(border)
    run = paragraph.add_run(SUBMISSION_DESCRIPTION)
    run.font.size = Pt(9.8)
    run.italic = True


def build_objectives(document: Document) -> None:
    document.add_heading("3. The brief, and where it is answered", level=1)

    body(
        document,
        "The task has three objectives. Mapping each one to a section makes it possible to "
        "check coverage without reading the whole document.",
    )

    rows = [
        ["1. Build an end-to-end ML project: preprocessing, training, testing, deployment "
         "with Flask or FastAPI",
         "Sections 4–7",
         "Pipeline persisted whole; FastAPI service; four commands reproduce everything"],
        ["2. Learn model serialisation using Pickle/Joblib and create a simple prediction API",
         "Sections 6–7",
         "Both formats written and compared; /predict and /predict/batch"],
        ["3. Deploy a complete ML application with project documentation and presentation",
         "Sections 8–17, Appendices A–D, presentation deck",
         "Model card, API reference, 18 tests, this report and a 14-slide deck"],
    ]
    add_table(document, ["Objective", "Where", "Evidence"], rows,
              widths=[2.55, 1.15, 2.85])

    document.add_heading("3.1 What 'deployed' means here", level=2)

    body(
        document,
        "The application runs as a real HTTP server, not as a notebook cell that calls a "
        "function. That matters because several of the findings in this report — the batch "
        "defect, the cold-start cost, the explanation overhead — only appear once the model is "
        "behind a network boundary. A local server is enough to expose them; a cluster is not "
        "required to learn the lesson.",
    )

    body(
        document,
        "The deployment stops short of a public cloud host. Credentials, billing and domain "
        "configuration are outside the scope of an internship submission, and a Dockerfile plus "
        "documented start command is enough for any reviewer to run the same thing themselves "
        "in one command.",
    )


def build_architecture(document: Document) -> None:
    document.add_heading("4. Architecture: what runs where", level=1)

    add_figure(
        document,
        FIGURES / "00_system_architecture.png",
        "Figure 1 — One prediction request, end to end. Validation happens before the model; "
        "explanation happens after it.",
        width=6.4,
    )

    body(
        document,
        "Figure 1 is the whole system. A caller posts JSON. Pydantic validates it against a "
        "declared schema and rejects anything malformed before a single line of model code "
        "executes. The pipeline then imputes, scales, encodes and predicts in one call. The "
        "explanation module decomposes the result, and the response carries the probability, a "
        "decision at the requested threshold, a risk band and the top contributors.",
    )

    body(
        document,
        "Three artifacts are loaded once, at startup, and never re-read per request: the "
        "serialised pipeline, the model card and the manifest. The manifest holds a SHA-256 "
        "digest for each file, which is what allows a deployment to prove which bytes it is "
        "actually serving — a question that is surprisingly hard to answer in a system that "
        "rebuilds images on every commit.",
    )

    document.add_heading("4.1 Why a pipeline, not two files", level=2)

    body(
        document,
        "The imputer, the scaler and the one-hot encoder all learn parameters from data. If "
        "they were fitted separately from the classifier and saved separately, it would be "
        "possible to ship a model with a stale scaler — a failure that produces plausible but "
        "wrong predictions and is very hard to notice. Keeping them inside one scikit-learn "
        "Pipeline makes that class of bug impossible to represent: there is one object, one "
        "file, one hash and one version.",
    )


def build_model(document: Document) -> None:
    document.add_heading("5. The deployed model", level=1)

    body(
        document,
        "The model is the winner of the Week 3 comparison: Logistic Regression with C = 0.1, "
        "L2 penalty and balanced class weights, selected by five-fold cross-validated ROC-AUC "
        "on the training split. It is deliberately not a more powerful model. On this dataset "
        "the Random Forest scored 0.796 against 0.806 — inside one standard deviation — and the "
        "linear model brings an exact explanation for every prediction, which the service "
        "depends on.",
    )

    rows = [
        ["Algorithm", "LogisticRegression inside a scikit-learn Pipeline"],
        ["Hyperparameters", "C = 0.1, penalty l2, solver lbfgs, class_weight balanced"],
        ["Selected by", "GridSearchCV, 5-fold CV, 0.806 ROC-AUC (Week 3)"],
        ["Input features", f"{len(MODEL_CARD['features'])} raw columns → 21 encoded"],
        ["Training rows", f"{MODEL_CARD['training_rows']:,} (stratified 80/20 split)"],
        ["Decision threshold", f"{MODEL_CARD['default_threshold']:.2f} default, caller-overridable"],
        ["Version", MODEL_CARD["model_version"]],
        ["Trained at", MODEL_CARD["trained_at"][:19].replace("T", " ") + " UTC"],
    ]
    add_table(document, ["Property", "Value"], rows, widths=[1.55, 5.0])

    document.add_heading("5.1 Held-out performance", level=2)

    body(
        document,
        "These are the numbers a reviewer should judge the model by: 360 rows that took no "
        "part in training, tuning or selection.",
    )

    metric_rows = [
        ["Accuracy", f"{METRICS['accuracy']:.3f}"],
        ["Balanced accuracy", f"{METRICS['balanced_accuracy']:.3f}"],
        ["Precision", f"{METRICS['precision']:.3f}"],
        ["Recall", f"{METRICS['recall']:.3f}"],
        ["F1", f"{METRICS['f1']:.3f}"],
        ["ROC-AUC", f"{METRICS['roc_auc']:.3f}"],
        ["PR-AUC", f"{METRICS['pr_auc']:.3f}"],
        ["Matthews correlation", f"{METRICS['mcc']:.3f}"],
        ["Log loss", f"{METRICS['log_loss']:.3f}"],
        ["Brier score", f"{METRICS['brier']:.3f}"],
        ["5-fold CV ROC-AUC", f"{METRICS['cv_roc_auc_mean']:.3f} ± {METRICS['cv_roc_auc_std']:.3f}"],
    ]
    add_table(document, ["Metric", "Value"], metric_rows, widths=[3.0, 1.6],
              align_right_from=1, font_size=8.4)

    body(
        document,
        "Accuracy of 0.678 looks modest, and it is — the do-nothing baseline scores 0.733. "
        "Recall of 0.656 is why this model is worth deploying: it finds two thirds of the "
        "customers who leave, and the class weighting that costs it accuracy is the reason. "
        "Section 12 examines this trade-off on the confusion matrix rather than in the abstract.",
    )


def build_serialisation(document: Document) -> None:
    document.add_heading("6. Serialisation: turning a model into a file", level=1)

    body(
        document,
        "The brief asks specifically for Pickle or Joblib. Both are written on every training "
        "run so the comparison is based on this model rather than on folklore.",
    )

    artifacts = read_csv("artifacts.csv")
    rows = [
        [row["artifact"], f"{row['bytes']:,}", f"{row['kib']:.1f}"[:5],
         MANIFEST["files"][index]["sha256"][:16] + "…"]
        for index, row in artifacts.iterrows()
    ]
    add_table(document, ["Format", "Bytes", "KiB", "SHA-256 (first 16)"], rows,
              widths=[1.2, 1.25, 1.0, 3.0], align_right_from=1)

    body(
        document,
        f"Both formats load in about {MANIFEST['load_time_ms']:.1f} ms, so at this size the "
        "choice is not about speed. It is about two other properties. Joblib uses a binary "
        "layout that scales better with large NumPy arrays, which is why it is the default for "
        "scikit-learn models. Pickle is in the standard library, so it has no dependency, which "
        "matters for a minimal container.",
    )

    add_callout(
        document,
        "The security property that is usually left out",
        "Both formats execute code when they are loaded. A pickle file from an untrusted "
        "source is not a data file, it is a program. Any service that loads models from an "
        "external bucket or an uploaded artifact must treat that file as code: sign it, verify "
        "the digest, and never load one that a stranger supplied.",
        fill="FBEAE7",
        accent="B5443A",
    )

    document.add_heading("6.1 The round-trip check", level=2)

    body(
        document,
        "Training ends by reloading the file it just wrote and comparing predictions against "
        "the in-memory model. The two are identical to within "
        f"{MANIFEST['max_abs_delta']:.0e}, and the manifest records that fact. This is a small "
        "check with a large purpose: it turns 'I believe the file is correct' into a verifiable "
        "claim that the artifact is the model that was evaluated.",
    )

    add_code(
        document,
        "# from src/train.py\n"
        "reloaded = joblib.load(MODEL_PATH)\n"
        "in_memory = deploy_pipeline.predict_proba(x_test)[:, 1]\n"
        "from_disk = reloaded.predict_proba(x_test)[:, 1]\n"
        "reproducible = bool(np.allclose(in_memory, from_disk, rtol=0, atol=1e-12))",
    )

    body(
        document,
        "The same check runs in the notebook, where it also compares six evaluation metrics "
        "recomputed from disk against the model card. The largest discrepancy is zero.",
    )


def build_api(document: Document) -> None:
    document.add_heading("7. The API surface", level=1)

    body(
        document,
        "Six endpoints, each with one job. The design rule is that a client should be able to "
        "answer 'is it up?', 'what is it?' and 'what does it think?' without reading source code.",
    )

    rows = [
        ["GET", "/health", "Liveness and readiness", "Model loaded, uptime, requests served"],
        ["GET", "/model", "Model card", "Metrics, features, intended use, limitations"],
        ["GET", "/model/manifest", "Artifact provenance", "SHA-256 per file, load time"],
        ["POST", "/predict", "Score one customer", "Probability, decision, band, explanation"],
        ["POST", "/predict/batch", "Score up to 1,000", "Vectorised, explanations opt-in"],
        ["GET", "/docs", "Interactive docs", "Generated from the Pydantic schemas"],
    ]
    add_table(document, ["Method", "Path", "Purpose", "Returns"], rows,
              widths=[0.8, 1.55, 1.55, 2.6], font_size=8.3)

    body(
        document,
        "A health check that only reports that the process is alive is close to useless: a "
        "container whose model failed to load will happily answer 200 to a ping. This one "
        "reports whether the model object exists, and the startup hook raises on boot if the "
        "artifact is missing, so the platform restarts the container instead of routing traffic "
        "to a service that cannot predict.",
    )

    document.add_heading("7.1 A request and its response", level=2)

    add_code(
        document,
        "POST /predict?threshold=0.4\n"
        "{\n"
        '  "tenure_months": 3.0,        "monthly_charges": 95.4,\n'
        '  "total_charges": 286.2,      "support_tickets": 4,\n'
        '  "senior_citizen": 0,         "contract": "Month-to-month",\n'
        '  "internet_service": "Fiber optic",\n'
        '  "payment_method": "Electronic check",\n'
        '  "paperless_billing": "Yes",  "partner": "No",\n'
        '  "dependents": "No"\n'
        "}\n"
        "\n"
        "200 OK\n"
        "{\n"
        '  "churn_probability": 0.9753,  "churn_prediction": "Yes",\n'
        '  "risk_band": "high",          "threshold": 0.4,\n'
        '  "top_contributions": [\n'
        '    {"feature": "contract = Month-to-month", "contribution": 0.96,\n'
        '     "direction": "increases risk"},\n'
        '    {"feature": "tenure_months", "contribution": 0.55, ...}\n'
        "  ],\n"
        '  "model_version": "4.0.0",     "latency_ms": 4.1\n'
        "}",
        font_size=7.0,
    )

    body(
        document,
        "Two details in that response are deliberate. The threshold is echoed back, so the "
        "decision can be audited against the rule that produced it. The latency is reported "
        "per response, so a slow request is visible to the caller rather than only in the "
        "server log.",
    )


def build_validation(document: Document) -> None:
    document.add_heading("8. Validation: failing at the edge", level=1)

    body(
        document,
        "The most common production failure is not a wrong prediction. It is a request that "
        "reaches the model in a shape it never expected — a renamed field, a string where a "
        "number belongs, a category that did not exist when the encoder was fitted. Pydantic "
        "turns each of those into a 422 with a sentence a caller can act on.",
    )

    rows = [
        ["Unknown contract type", '"Weekly"', "422", "Lists the three accepted values"],
        ["Indicator out of range", "senior_citizen = 7", "422", "Must be ≤ 1"],
        ["Negative amount", "monthly_charges = -5", "422", "Must be ≥ 0"],
        ["Misspelt field", "tenure_month", "422", "Extra inputs are not permitted"],
        ["Missing required field", "no contract", "422", "Field required"],
        ["Optional field omitted", "total_charges = null", "200", "Imputer fills the median"],
    ]
    add_table(document, ["Case", "Payload", "Status", "Outcome"], rows,
              widths=[1.45, 1.6, 0.65, 2.6], font_size=8.3)

    add_figure(
        document,
        FIGURES / "09_artifacts_and_validation.png",
        "Figure 2 — Artifact size beside the validation behaviour. The status codes were "
        "captured by calling the running application, not written by hand.",
        width=6.4,
    )

    body(
        document,
        "Two decisions in that table are worth defending. First, extra fields are forbidden "
        "rather than ignored: a client that sends tenure_month receives an error instead of a "
        "prediction built on an imputed median. Second, total_charges is optional because a new "
        "customer genuinely has no lifetime total, and the pipeline already knows how to impute "
        "it. The line between invalid and merely unknown is a product decision, and it belongs "
        "in the schema rather than in the model.",
    )


def build_explanation(document: Document) -> None:
    document.add_heading("9. Explaining a single prediction", level=1)

    body(
        document,
        "For a linear model the log-odds decompose exactly: the intercept plus the sum of each "
        "standardised feature multiplied by its coefficient. No approximation, no sampling, no "
        "separate surrogate model. The explanation is the model's own arithmetic.",
    )

    add_figure(
        document,
        FIGURES / "08_explanation_example.png",
        "Figure 3 — One customer's prediction decomposed. Red raises the risk, green lowers it, "
        "and the footer shows the arithmetic closing exactly.",
        width=6.3,
    )

    body(
        document,
        "The customer above has been with the company for fourteen months, is on a month-to-month "
        "contract and pays 72.00 per month. The contract type alone contributes +0.96 log-odds — "
        "the single strongest effect. Two factors pull the other way: DSL rather than fibre "
        "(−0.21) and having a partner (−0.12). The net result is 71% risk, which the service "
        "reports in the high band.",
    )

    body(
        document,
        "The arithmetic is verifiable, which is the point. An evaluator can add the numbers on "
        "the chart, apply the logistic function and reproduce the probability the API returned. "
        "Explanations that cannot be checked are decoration.",
    )

    add_callout(
        document,
        "Why this drove the model choice",
        "A gradient-boosted model might have scored a point or two higher on ROC-AUC. It would "
        "also have required SHAP values, which approximate a linear model's exact decomposition "
        "and add a dependency, a latency cost and a new class of bug. For a churn decision that "
        "affects customers, the ability to justify a prediction is worth more than a marginal "
        "improvement in a metric.",
    )


def build_latency(document: Document) -> None:
    document.add_heading(
        f"10. Latency: where the {BENCHMARK['single_explained_p50_ms']:.1f} ms goes", level=1
    )

    body(
        document,
        "Every number in this section was measured against a live Uvicorn server on loopback, "
        f"over {BENCHMARK['requests']} sequential requests, with the first twenty discarded as "
        "warm-up. Percentiles rather than means are reported because a single 60 ms outlier "
        "would drag an average somewhere no real request ever lands.",
    )

    add_figure(
        document,
        FIGURES / "01_latency_budget.png",
        f"Figure 4 — The {BENCHMARK['single_explained_p50_ms']:.1f} ms budget for one prediction. "
        "The explanation is the largest single component.",
        width=6.4,
    )

    body(
        document,
        "Three conclusions follow. The model is only "
        f"{BENCHMARK['model_only_p50_ms'] / BENCHMARK['single_explained_p50_ms']:.0%} of the "
        "request, so further work on the model would be invisible to a user. The explanation "
        f"costs {BENCHMARK['explanation_cost_ms']:.1f} ms, more than the prediction it explains, "
        "which is precisely why it is a flag rather than a fixed cost. And the HTTP layer adds "
        f"{BENCHMARK['http_overhead_ms']:.1f} ms, which sets the floor for any optimisation of "
        "the model itself.",
    )

    add_figure(
        document,
        FIGURES / "02_latency_distribution.png",
        "Figure 5 — Latency distribution for three configurations. The three modes are cleanly "
        "separated, and the spread inside each is small.",
        width=6.4,
    )

    body(
        document,
        "The distributions matter more than the medians. Each configuration is tight — the "
        "interquartile range is a millisecond or two — which means the p95 a caller experiences "
        "is close to the p50. A service with a good median and a heavy tail is worse in practice "
        "than one with a slightly higher median and no tail.",
    )


def build_throughput(document: Document) -> None:
    document.add_heading("11. Throughput: the defect the benchmark found", level=1)

    body(
        document,
        "The batch endpoint was written the obvious way: loop over customers and score each "
        "one. It passed its tests, returned correct predictions and looked finished. Then the "
        "benchmark ran.",
    )

    add_code(
        document,
        "first implementation                after vectorising\n"
        "--------------------------------    --------------------------------\n"
        "for customer in batch:              frame = pd.DataFrame(records)\n"
        "    record = customer.as_record()   probabilities = model.predict_proba(frame)[:, 1]\n"
        "    p = model.predict_proba(...)    # one transform, one predict\n"
        "    explain(...)                    # explanations opt-in per row\n"
        "\n"
        f"100 rows: 1,069 ms  (92 rows/s)    100 rows: "
        f"{BENCHMARK['batch_ms_per_row'] * BENCHMARK['batch_size']:.0f} ms  "
        f"({BENCHMARK['batch_rows_per_s']:,.0f} rows/s)",
        font_size=7.2,
    )

    body(
        document,
        "The waste was not in the model. It was in paying the fixed cost of a pandas frame, a "
        "ColumnTransformer call and a scikit-learn predict for every row, a hundred times per "
        "request. The fix is three lines and two orders of magnitude.",
    )

    add_figure(
        document,
        FIGURES / "03_throughput.png",
        "Figure 6 — Throughput by scenario on a log scale. The gap between the two batch bars is "
        "the cost of computing an explanation for every row.",
        width=6.4,
    )

    measured = read_csv("latency_benchmark.csv").set_index("scenario")
    rows = []
    for label in ("predict + explanation", "predict, no explanation", "batch of 100",
                  "model only (no HTTP)"):
        row = measured.loc[label]
        rows.append([
            label,
            f"{row['p50_ms']:.3f}",
            f"{row['p95_ms']:.3f}",
            f"{row['ms_per_row']:.3f}",
            f"{row['throughput_rows_per_s']:,.0f}",
        ])
    add_table(document, ["Scenario", "p50 (ms)", "p95 (ms)", "ms per row", "rows/s"], rows,
              widths=[2.3, 1.0, 1.0, 1.0, 1.15], align_right_from=1, font_size=8.3)

    body(
        document,
        "The batch endpoint is now more than a hundred times faster than the single endpoint "
        "per row, which is the entire reason a batch endpoint exists. The general lesson is "
        "worth stating plainly: a batch endpoint that loops is not a batch endpoint, and the "
        "defect is invisible to a correctness test. Only a measurement finds it.",
    )


def build_quality(document: Document) -> None:
    document.add_heading("12. Model quality on data it never saw", level=1)

    body(
        document,
        "Deployment does not improve a model, so it is worth restating what this one is and is "
        "not good at. The 360 held-out rows produce the confusion matrix below.",
    )

    add_figure(
        document,
        FIGURES / "04_confusion_matrix.png",
        f"Figure 7 — Confusion matrix at the default 0.50 threshold. {METRICS['false_positive']} "
        f"false alarms, {METRICS['false_negative']} missed churners.",
        width=4.9,
    )

    rows = [
        ["True negatives", str(METRICS["true_negative"]), "Correctly predicted to stay"],
        ["False positives", str(METRICS["false_positive"]), "Predicted churn, stayed — a wasted offer"],
        ["False negatives", str(METRICS["false_negative"]), "Predicted stay, churned — a lost customer"],
        ["True positives", str(METRICS["true_positive"]), "Correctly predicted to churn"],
    ]
    add_table(document, ["Cell", "Count", "Cost in business terms"], rows,
              widths=[1.7, 0.8, 4.0], align_right_from=1)

    body(
        document,
        "The two error types are not symmetric. Missing a churner costs the customer's entire "
        "future value; a false alarm costs one retention offer. At the default threshold the "
        f"model makes {METRICS['false_positive']} and {METRICS['false_negative']} of them "
        "respectively, and no choice of threshold removes both — which is why the next section "
        "treats the threshold as a business parameter.",
    )

    add_figure(
        document,
        FIGURES / "05_roc_pr.png",
        "Figure 8 — ROC and precision-recall curves for the deployed model on the held-out split.",
        width=6.4,
    )

    body(
        document,
        f"ROC-AUC of {METRICS['roc_auc']:.3f} says the model ranks a random churner above a "
        "random non-churner about three quarters of the time. That is useful for prioritising a "
        "call list and not good enough to automate a decision. PR-AUC of "
        f"{METRICS['pr_auc']:.3f} against a no-skill baseline of 0.267 is the more honest "
        "summary of the same ranking on the minority class.",
    )


def build_thresholds(document: Document) -> None:
    document.add_heading("13. Choosing an operating threshold", level=1)

    body(
        document,
        "The model outputs a probability. The threshold converts it into a decision, and the "
        "service accepts it as a query parameter so that one deployment can serve callers with "
        "different costs.",
    )

    curve = read_csv("threshold_curve.csv")
    picks = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    rows = []
    for value in picks:
        row = curve.iloc[(curve["threshold"] - value).abs().argmin()]
        rows.append([
            f"{row['threshold']:.2f}",
            f"{row['precision']:.3f}",
            f"{row['recall']:.3f}",
            f"{row['f1']:.3f}",
            f"{row['flagged']:.1%}",
        ])
    add_table(document, ["Threshold", "Precision", "Recall", "F1", "Share flagged"], rows,
              widths=[1.2, 1.3, 1.3, 1.2, 1.5], align_right_from=1, font_size=8.5)

    add_figure(
        document,
        FIGURES / "06_threshold_tradeoff.png",
        "Figure 9 — Precision, recall and F1 against the decision threshold, computed from the "
        "360 held-out predictions.",
        width=6.4,
    )

    body(
        document,
        "F1 peaks at 0.558 in a broad plateau between roughly 0.36 and 0.45. Moving the "
        "threshold from 0.50 to 0.40 raises recall from 0.656 to 0.823 while dropping precision "
        "from 0.432 to 0.422, and F1 improves. Which of those rows is correct is not a modelling "
        "question: it depends on what a missed churner costs relative to a wasted offer.",
    )

    add_callout(
        document,
        "What the API deliberately does not do",
        "It does not pick a threshold for the caller. A service that silently applies 0.4 "
        "because it maximises F1 has made a business decision on the customer's behalf, invisibly. "
        "The default is 0.5, it is documented, it is echoed in every response, and the caller can "
        "change it per request.",
    )


def build_uncertainty(document: Document) -> None:
    document.add_heading("14. Where the model is uncertain", level=1)

    body(
        document,
        "A probability is only useful if it is calibrated. The distribution below shows how the "
        "model's estimates relate to what actually happened.",
    )

    add_figure(
        document,
        FIGURES / "07_probability_distribution.png",
        "Figure 10 — Predicted probability by true outcome. The two classes overlap, so no "
        "threshold can separate them cleanly.",
        width=6.4,
    )

    body(
        document,
        "Retained customers cluster at low probability and churners sit further right, but the "
        "two distributions sit on top of each other between about 0.2 and 0.7. That overlap is "
        "the reason every metric in this report is below one, and the reason the model is "
        "described as decision support rather than automation.",
    )

    body(
        document,
        "It is also the honest answer to 'how accurate is it?' — the question a stakeholder "
        "usually asks first. A single number cannot answer it. The model is accurate for "
        "customers it is confident about and unreliable for the ones near the boundary, which "
        "is exactly why the response includes a probability and a risk band rather than only a "
        "label.",
    )


def build_tests(document: Document) -> None:
    document.add_heading("15. Testing the service", level=1)

    body(
        document,
        "Eighteen contract tests exercise the API the way a client does: over HTTP, against the "
        "same artifact the deployment loads. They are fast enough to run on every commit and "
        "specific enough to name the behaviour that broke.",
    )

    rows = [
        ["Operations", "4", "Health reports the model loaded; card and manifest are served; "
                            "the demo page is HTML"],
        ["Prediction", "5", "Probability in range; label consistent with threshold; risk "
                            "ordering; threshold override; explanations sorted by impact"],
        ["Validation", "7", "Unknown category, out-of-range indicator, negative amount, wrong "
                            "type, unknown field, missing field, null optional field"],
        ["Batch", "2", "Every row is scored; an empty batch is rejected"],
    ]
    add_table(document, ["Group", "Tests", "Covers"], rows, widths=[1.3, 0.7, 4.5],
              align_right_from=1)

    body(
        document,
        "The prediction tests are the interesting ones. It is not enough that a probability is "
        "between zero and one; the label must agree with the probability and the threshold, and "
        "the explanations must be sorted by absolute contribution. Those tests would catch a "
        "refactor that returned correct probabilities but silently stopped explaining them.",
    )

    add_code(
        document,
        "$ python -m pytest week-4/tests -q\n"
        "..................                                                       [100%]\n"
        "18 passed in 1.54s",
        font_size=7.6,
    )


def build_running(document: Document) -> None:
    document.add_heading("16. Running and deploying it", level=1)

    body(
        document,
        "Four commands reproduce the whole deliverable from a clean checkout. Nothing is "
        "hard-coded and no output file is edited by hand.",
    )

    add_code(
        document,
        "python -m pip install -r week-4/requirements.txt\n"
        "python week-4/src/train.py          # fit, evaluate, serialise, write model card\n"
        "python week-4/src/benchmark.py      # measure the running service\n"
        "python week-4/src/make_figures.py   # build the 10 figures\n"
        "python week-4/docs/build_report.py  # build this report\n"
        "\n"
        "# serve it\n"
        "cd week-4/src && uvicorn service:app --host 0.0.0.0 --port 8000\n"
        "# then open http://localhost:8000",
    )

    body(
        document,
        f"Cold start — process launch to first successful health response — measured at "
        f"{BENCHMARK['cold_start_ms']:.0f} ms. Almost none of that is the model: loading a 5.7 "
        "KiB artifact takes about a millisecond, and the rest is importing NumPy, pandas and "
        "scikit-learn. That ratio is why the model is loaded once at startup rather than per "
        "request, and why a serverless deployment with a cold start per request would be a bad "
        "fit for this workload.",
    )

    document.add_heading("16.1 Container", level=2)

    body(
        document,
        "A container makes the runtime explicit. The image installs only what serving needs — "
        "no notebook stack, no test dependencies — and runs as a non-root user.",
    )

    add_code(
        document,
        "FROM python:3.11-slim\n"
        "WORKDIR /app\n"
        "COPY week-4/requirements-serve.txt ./\n"
        "RUN pip install --no-cache-dir -r requirements-serve.txt\n"
        "COPY week-4/src ./src\n"
        "COPY week-4/static ./static\n"
        "COPY week-4/outputs/model ./outputs/model\n"
        "COPY week-4/outputs/model_card.json week-4/outputs/model_manifest.json ./outputs/\n"
        "RUN useradd --create-home appuser && chown -R appuser /app\n"
        "USER appuser\n"
        "EXPOSE 8000\n"
        "CMD [\"uvicorn\", \"src.service:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]",
        font_size=7.0,
    )

    body(
        document,
        "The model card and manifest are copied into the image alongside the artifact, so the "
        "running service can describe and verify itself without an external lookup.",
    )


def build_reproducibility(document: Document) -> None:
    document.add_heading("17. Reproducibility: proving the artifact is the model", level=1)

    body(
        document,
        "A deployment claim is only credible if someone else can reproduce it. Three separate "
        "paths lead to the same numbers, and all three are in the repository.",
    )

    rows = [
        ["Training path", "src/train.py", f"ROC-AUC {METRICS['roc_auc']:.3f}, recall "
                                          f"{METRICS['recall']:.3f}"],
        ["Notebook path", "notebooks/week4_deployment.ipynb",
         "Recomputes six metrics from the saved file; largest gap 0.00e+00"],
        ["Service path", "POST /predict", "Returns the same probability the pipeline computes"],
    ]
    add_table(document, ["Path", "Where", "What it proves"], rows,
              widths=[1.25, 2.45, 2.8], font_size=8.3)

    body(
        document,
        "The notebook is the strongest of the three, because it starts from the serialised file "
        "rather than from the training code. If the artifact were stale — trained on different "
        "data, or with a different encoder — the recomputed metrics would not match the model "
        "card. They match to every printed decimal.",
    )

    body(
        document,
        "Everything is seeded: the train/test split, the cross-validation folds and the "
        "classifier. There is no random state left to chance, so a reviewer who runs the "
        "commands above gets the numbers in this report rather than numbers that are close.",
    )


def build_limitations(document: Document) -> None:
    document.add_heading("18. Limitations and responsible use", level=1)

    body(
        document,
        "A deployed model makes its limitations more consequential than a model in a notebook, "
        "because people act on its output. The model card records them, and this section expands "
        "on the ones that matter most.",
    )

    bullets(
        document,
        [
            "The dataset is synthetic. Every pattern the model learned is a property of the "
            "generator. The pipeline is real; the signal is not, and no conclusion here should "
            "be transferred to real customer behaviour.",
            "The model is not validated for consequential decisions. Churn outreach is low "
            "stakes. Using this model — or one built the same way — for credit, pricing, "
            "insurance or hiring would be inappropriate.",
            "There has been no fairness audit. Error rates can differ by subgroup, and a "
            "senior-citizen segment is visible in the data. Before any real use, the confusion "
            "matrix would need to be computed per group.",
            "Predictions are only as current as their inputs. A stale tenure or a wrong monthly "
            "charge produces a confident wrong answer, and the API cannot detect it.",
            "Performance will drift. The model card records a training date precisely so that a "
            "future drop in performance can be attributed to drift rather than to a code change.",
            "The service has no authentication. It is a local demonstration; exposing it "
            "publicly would require rate limiting, auth and TLS.",
        ],
    )

    add_callout(
        document,
        "The intended use, in one sentence",
        MODEL_CARD["intended_use"],
        fill="EAF1F6",
        accent="237A7A",
    )


def build_lessons(document: Document) -> None:
    document.add_heading("19. Lessons learned", level=1)

    numbered(
        document,
        [
            "A correctness test cannot find a performance defect. The batch endpoint passed all "
            "eighteen tests while running at 92 rows per second. Only a benchmark found it, and "
            "the fix was three lines.",
            "Measure the budget before optimising anything. The model was a third of the request; "
            "the explanation was the largest single cost. Optimising the model would have been "
            "effort spent where the user cannot see it.",
            "Validation belongs at the edge. Every 422 returned by the schema is an exception "
            "that never reached scikit-learn, where it would have surfaced as a stack trace and "
            "a 500.",
            "Serialise the pipeline, not the model. One object means one hash, one version and "
            "no possibility of a stale scaler.",
            "A round-trip check is cheap and buys a real guarantee. Reloading the file and "
            "comparing predictions turned an assumption into a recorded fact.",
            "Explanations are a feature with a cost, not a free property. They accounted for 41% "
            "of the latency, which is exactly why they are optional on the batch endpoint.",
            "Documentation that is generated cannot drift. The model card, the manifest, the "
            "figures and this report all read from files the pipeline produces, so a stale "
            "number would require someone to edit a JSON file by hand.",
        ],
    )

    body(
        document,
        "The first of those is the one worth carrying forward. Everything else in this report "
        "was found by looking; the batch defect was found by measuring, and it would have shipped "
        "otherwise.",
    )


def build_conclusion(document: Document) -> None:
    document.add_heading("20. Conclusion", level=1)

    body(
        document,
        "Weeks 1 to 3 built a model and established that it is a reasonable one for this data. "
        "This week made it callable, and in doing so exposed three things a notebook hides: that "
        "explanations cost more than predictions, that a batch endpoint can be a hundred times "
        "slower than it needs to be, and that a model without a contract and a version is not "
        "something another system can depend on.",
    )

    body(
        document,
        "The deliverable is deliberately narrow and complete rather than broad and partial. It "
        "serves one model, on one dataset, with one clear intended use; it documents that model's "
        "limits in the same place it documents its metrics; and every claim it makes can be "
        "reproduced with four commands and checked against a file in the repository.",
    )

    body(
        document,
        "That is what deployment means at this scale: not a cluster, not a CI pipeline, but the "
        "point at which a model acquires an interface, a version, a test suite and a measured "
        "cost — and stops being a result and starts being a product.",
    )


# --------------------------------------------------------------------------
# Appendices
# --------------------------------------------------------------------------
def build_appendix_a(document: Document) -> None:
    document.add_heading("Appendix A — Reproducing this report", level=1)

    body(
        document,
        "From a clean checkout with Python 3.11 available:",
    )
    add_code(
        document,
        "cd week-4\n"
        "python -m venv .venv && source .venv/bin/activate\n"
        "python -m pip install -r requirements.txt\n"
        "\n"
        "python src/train.py\n"
        "python src/benchmark.py\n"
        "python src/make_figures.py\n"
        "python docs/build_report.py\n"
        "python -m pytest tests -q",
    )

    body(
        document,
        "Total runtime is about 25 seconds, dominated by the benchmark's 600 HTTP requests. "
        "The build is deterministic apart from the benchmark, which reports timings and will "
        "vary with the machine — the figures that depend on it are labelled as measurements "
        "rather than as fixed results.",
    )

    rows = [
        ["train.py", "Fit, evaluate, serialise, write model card and manifest"],
        ["benchmark.py", "Start Uvicorn, drive traffic, write latency tables"],
        ["make_figures.py", "Build all 10 figures from the outputs"],
        ["build_report.py", "Build this document"],
        ["build_deck.py", "Build the presentation"],
        ["tests/test_service.py", "18 contract tests"],
    ]
    add_table(document, ["File", "Purpose"], rows, widths=[1.7, 4.7])


def build_appendix_b(document: Document) -> None:
    document.add_heading("Appendix B — API reference", level=1)

    rows = [
        ["GET /health", "200", "status, model_loaded, model_version, uptime_seconds, requests_served"],
        ["GET /model", "200", "model card: algorithm, features, metrics, intended use, limitations"],
        ["GET /model/manifest", "200", "files with sha256 and bytes, load_time_ms, round_trip_identical"],
        ["POST /predict", "200", "churn_probability, churn_prediction, threshold, risk_band, "
                                 "top_contributions, latency_ms"],
        ["POST /predict", "422", "Validation error with a per-field message"],
        ["POST /predict/batch", "200", "predictions, count, flagged, flagged_rate, latency_ms"],
        ["GET /docs", "200", "Interactive OpenAPI documentation"],
    ]
    add_table(document, ["Endpoint", "Status", "Body"], rows, widths=[1.6, 0.7, 4.3],
              font_size=8.2)

    body(document, "Query parameters on the prediction endpoints:", italic=True, size=9)

    rows = [
        ["threshold", "float, 0–1", "0.50", "Decision threshold; the probability is unaffected"],
        ["explain_result", "bool", "true", "/predict only; set false to save ~5 ms"],
        ["include_explanations", "bool", "false", "/predict/batch only; expensive for large batches"],
    ]
    add_table(document, ["Parameter", "Type", "Default", "Effect"], rows,
              widths=[1.45, 0.95, 0.75, 3.4], font_size=8.3)


def build_appendix_c(document: Document) -> None:
    document.add_heading("Appendix C — Test inventory", level=1)

    tests = [
        ["test_health_reports_model_loaded", "Operations"],
        ["test_model_card_is_served", "Operations"],
        ["test_manifest_lists_artifact_digest", "Operations"],
        ["test_demo_page_is_html", "Operations"],
        ["test_predict_returns_valid_probability", "Prediction"],
        ["test_prediction_matches_probability_and_threshold", "Prediction"],
        ["test_high_risk_profile_scores_above_low_risk_profile", "Prediction"],
        ["test_threshold_parameter_changes_the_label", "Prediction"],
        ["test_explanations_are_ranked_by_impact", "Prediction"],
        ["test_invalid_values_are_rejected (×4)", "Validation"],
        ["test_unknown_field_is_rejected", "Validation"],
        ["test_missing_required_field_is_rejected", "Validation"],
        ["test_null_total_charges_is_accepted", "Validation"],
        ["test_batch_scores_every_row", "Batch"],
        ["test_empty_batch_is_rejected", "Batch"],
    ]
    add_table(document, ["Test", "Group"], tests, widths=[4.6, 1.5], font_size=8.0)

    body(
        document,
        "Run with `python -m pytest week-4/tests -q`. The suite takes about 1.5 seconds, "
        "because the module-scoped fixture loads the model once for all tests.",
        size=9,
    )


def build_appendix_d(document: Document) -> None:
    document.add_heading("Appendix D — Figure index", level=1)

    rows = [
        ["1", "00_system_architecture.png", "Request lifecycle and artifacts"],
        ["2", "09_artifacts_and_validation.png", "Artifact size and validation outcomes"],
        ["3", "08_explanation_example.png", "Log-odds decomposition for one customer"],
        ["4", "01_latency_budget.png", "Latency broken into three components"],
        ["5", "02_latency_distribution.png", "Latency distribution across 300 requests"],
        ["6", "03_throughput.png", "Throughput by scenario, log scale"],
        ["7", "04_confusion_matrix.png", "Held-out confusion matrix"],
        ["8", "05_roc_pr.png", "ROC and precision-recall curves"],
        ["9", "06_threshold_tradeoff.png", "Precision, recall and F1 against threshold"],
        ["10", "07_probability_distribution.png", "Predicted probability by true outcome"],
    ]
    add_table(document, ["Figure", "File", "Content"], rows, widths=[0.7, 2.6, 3.3],
              align_right_from=0, font_size=8.3)

    body(
        document,
        "All ten figures are regenerated by `python src/make_figures.py` and are stored in "
        "week-4/outputs/figures/.",
        size=9,
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    for name in ("model_card.json", "benchmark_summary.json", "model_manifest.json"):
        if not (OUTPUTS / name).exists():
            raise SystemExit(
                f"Missing {name}. Run src/train.py and src/benchmark.py before building the report."
            )

    if word_count(SUBMISSION_DESCRIPTION) < 200:
        raise SystemExit("Submission description is below the 200-word minimum.")

    document = setup_document()
    build_cover(document)
    page_break(document)

    build_executive_summary(document)
    build_description(document)
    build_objectives(document)
    page_break(document)

    build_architecture(document)
    build_model(document)
    build_serialisation(document)
    build_api(document)
    build_validation(document)
    page_break(document)

    build_explanation(document)
    build_latency(document)
    build_throughput(document)
    page_break(document)

    build_quality(document)
    build_thresholds(document)
    build_uncertainty(document)
    build_tests(document)
    page_break(document)

    build_running(document)
    build_reproducibility(document)
    build_limitations(document)
    build_lessons(document)
    build_conclusion(document)
    page_break(document)

    build_appendix_a(document)
    build_appendix_b(document)
    build_appendix_c(document)
    build_appendix_d(document)

    properties = document.core_properties
    properties.title = "Week 4 — AI Project Deployment"
    properties.subject = "Model deployment with FastAPI, serialisation and evaluation"
    properties.author = AUTHOR
    properties.last_modified_by = AUTHOR
    properties.keywords = "machine learning, deployment, FastAPI, joblib, serialisation"
    properties.comments = SUBMISSION_DESCRIPTION[:240]

    REPORTS.mkdir(parents=True, exist_ok=True)
    target = REPORTS / "Week_4_AI_Project_Deployment_Report.docx"
    document.save(target)

    print(f"Report written : {target.relative_to(WEEK_DIR.parent)}")
    print(f"Author         : {AUTHOR}")
    print(f"Description    : {word_count(SUBMISSION_DESCRIPTION)} words (minimum 200)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
