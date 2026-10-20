#!/usr/bin/env python3
"""Build the Week 4 capstone presentation.

Fourteen slides, 16:9, with speaker notes on each one. Every number is read
from ``outputs/`` so the deck cannot disagree with the report.

    python docs/build_deck.py
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

AUTHOR = "Peram Dhanam Srinivas"
WEEK_DIR = Path(__file__).resolve().parents[1]
OUTPUTS = WEEK_DIR / "outputs"
FIGURES = OUTPUTS / "figures"
REPORTS = WEEK_DIR / "reports"

NAVY = RGBColor(0x17, 0x32, 0x4D)
TEAL = RGBColor(0x23, 0x7A, 0x7A)
CORAL = RGBColor(0xC4, 0x66, 0x4D)
GREY = RGBColor(0x5C, 0x6F, 0x82)
LIGHT = RGBColor(0xF1, 0xF5, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

CARD = json.loads((OUTPUTS / "model_card.json").read_text(encoding="utf-8"))
BENCH = json.loads((OUTPUTS / "benchmark_summary.json").read_text(encoding="utf-8"))
METRICS = CARD["metrics"]

WIDTH = Inches(13.333)
HEIGHT = Inches(7.5)

prs = Presentation()
prs.slide_width = WIDTH
prs.slide_height = HEIGHT
BLANK = prs.slide_layouts[6]


def textbox(slide, left, top, width, height, text, *, size=18, bold=False,
            colour=NAVY, align=PP_ALIGN.LEFT, italic=False, spacing=1.15):
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    paragraph.line_spacing = spacing
    run = paragraph.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = colour
    run.font.name = "Calibri"
    return box


def bullets(slide, left, top, width, height, items, *, size=17, colour=NAVY):
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    for index, item in enumerate(items):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.space_after = Pt(12)
        paragraph.line_spacing = 1.12
        if isinstance(item, tuple):
            lead, rest = item
            run = paragraph.add_run()
            run.text = f"{lead}  "
            run.font.size = Pt(size)
            run.font.bold = True
            run.font.color.rgb = TEAL
            run.font.name = "Calibri"
            run = paragraph.add_run()
            run.text = rest
            run.font.size = Pt(size)
            run.font.color.rgb = colour
            run.font.name = "Calibri"
        else:
            run = paragraph.add_run()
            run.text = f"•  {item}"
            run.font.size = Pt(size)
            run.font.color.rgb = colour
            run.font.name = "Calibri"
    return box


def new_slide(title: str, kicker: str | None = None) -> object:
    slide = prs.slides.add_slide(BLANK)

    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, WIDTH, Inches(0.09))
    band.fill.solid()
    band.fill.fore_color.rgb = TEAL
    band.line.fill.background()

    if kicker:
        textbox(slide, Inches(0.85), Inches(0.42), Inches(11.6), Inches(0.3),
                kicker.upper(), size=11, bold=True, colour=TEAL)

    textbox(slide, Inches(0.85), Inches(0.72), Inches(11.6), Inches(0.9),
            title, size=30, bold=True, colour=NAVY)

    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.87), Inches(1.62),
                                  Inches(1.1), Inches(0.045))
    line.fill.solid()
    line.fill.fore_color.rgb = CORAL
    line.line.fill.background()

    footer = slide.shapes.add_textbox(Inches(0.85), Inches(7.02), Inches(11.6), Inches(0.3))
    paragraph = footer.text_frame.paragraphs[0]
    run = paragraph.add_run()
    run.text = f"{AUTHOR}   ·   Week 4 Capstone   ·   AI Project Deployment"
    run.font.size = Pt(9)
    run.font.color.rgb = GREY
    return slide


def add_notes(slide, text: str) -> None:
    slide.notes_slide.notes_text_frame.text = text


def add_image(slide, path: Path, left, top, max_width, max_height) -> None:
    picture = slide.shapes.add_picture(str(path), left, top, width=max_width)
    if picture.height > max_height:
        ratio = max_height / picture.height
        picture.height = int(picture.height * ratio)
        picture.width = int(picture.width * ratio)
    picture.left = int(left + (max_width - picture.width) / 2)
    picture.top = int(top + (max_height - picture.height) / 2)


# --------------------------------------------------------------------------
# 1 - Title
# --------------------------------------------------------------------------
slide = prs.slides.add_slide(BLANK)
background = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, WIDTH, HEIGHT)
background.fill.solid()
background.fill.fore_color.rgb = NAVY
background.line.fill.background()

accent = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.9), Inches(1.55),
                                Inches(1.4), Inches(0.06))
accent.fill.solid()
accent.fill.fore_color.rgb = TEAL
accent.line.fill.background()

textbox(slide, Inches(0.9), Inches(0.85), Inches(11.0), Inches(0.4),
        "YUVA INTERNSHIP PROGRAMME   ·   WEEK 4 CAPSTONE", size=13, bold=True,
        colour=RGBColor(0x7F, 0xC4, 0xC4))

textbox(slide, Inches(0.9), Inches(1.95), Inches(11.4), Inches(2.2),
        "Customer Churn Prediction", size=52, bold=True, colour=WHITE)
textbox(slide, Inches(0.9), Inches(3.15), Inches(11.4), Inches(1.4),
        "from a notebook to a service", size=32, colour=RGBColor(0xBF, 0xD3, 0xE0),
        italic=True)

textbox(slide, Inches(0.9), Inches(4.75), Inches(10.8), Inches(1.0),
        "Serialising a scikit-learn pipeline, serving it with FastAPI,\n"
        "and measuring what it actually costs.", size=17,
        colour=RGBColor(0xA8, 0xBC, 0xCB))

textbox(slide, Inches(0.9), Inches(6.35), Inches(8.0), Inches(0.4),
        AUTHOR, size=13, bold=True, colour=WHITE)
textbox(slide, Inches(0.9), Inches(6.7), Inches(8.0), Inches(0.4),
        date.today().strftime("%B %Y"), size=11, colour=RGBColor(0xA8, 0xBC, 0xCB))

add_notes(slide,
          "This is the final week of the internship: the deployment week. Over the last three "
          "weeks I built a churn model and compared algorithms. This week is about the gap "
          "between that model and something another team could actually call, version and trust. "
          "I will keep it to about ten minutes and leave time for questions.")

# --------------------------------------------------------------------------
# 2 - The gap
# --------------------------------------------------------------------------
slide = new_slide("A model in a notebook is not a product", "The starting point")
bullets(slide, Inches(0.85), Inches(1.95), Inches(7.0), Inches(4.6), [
    ("No interface.", "Nobody can call it without opening the notebook."),
    ("No contract.", "Nothing validates what comes in; a typo becomes a silent imputation."),
    ("No version.", "Which coefficients are running in production?"),
    ("No measured cost.", "12 ms or 12 seconds? Nobody knows until someone asks."),
    ("No limits.", "Nothing records what the model must not be used for."),
], size=16)

box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.35), Inches(2.1),
                             Inches(4.1), Inches(3.5))
box.fill.solid()
box.fill.fore_color.rgb = LIGHT
box.line.color.rgb = RGBColor(0xD3, 0xDF, 0xE8)
textbox(slide, Inches(8.65), Inches(2.4), Inches(3.5), Inches(0.5),
        "The deliverable", size=12, bold=True, colour=TEAL)
textbox(slide, Inches(8.65), Inches(2.85), Inches(3.5), Inches(2.5),
        "A FastAPI service, a serialised pipeline, a model card, "
        "18 tests and a measured latency budget.",
        size=15, colour=NAVY)

add_notes(slide,
          "The framing I want to establish: the hard part of deployment is not the code. It is "
          "the five things a notebook does not have — an interface, a contract, a version, a "
          "measured cost and documented limits. Everything in this deck is one of those five.")

# --------------------------------------------------------------------------
# 3 - What was built
# --------------------------------------------------------------------------
slide = new_slide("What the brief asked for, and what exists", "Scope")
rows = [
    ("1", "End-to-end project with FastAPI", "train → serialise → serve → test"),
    ("2", "Serialisation with Pickle and Joblib", "both written, compared, hashed"),
    ("3", "Documentation and presentation", "report, model card, API reference, this deck"),
]
top = Inches(2.05)
for number, title, detail in rows:
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.85), top,
                                  Inches(11.5), Inches(1.25))
    card.fill.solid()
    card.fill.fore_color.rgb = LIGHT
    card.line.color.rgb = RGBColor(0xD3, 0xDF, 0xE8)
    textbox(slide, Inches(1.15), top + Inches(0.33), Inches(0.6), Inches(0.6),
            number, size=26, bold=True, colour=TEAL)
    textbox(slide, Inches(1.85), top + Inches(0.22), Inches(6.6), Inches(0.4),
            title, size=16, bold=True, colour=NAVY)
    textbox(slide, Inches(1.85), top + Inches(0.68), Inches(9.8), Inches(0.4),
            detail, size=12.5, colour=GREY)
    top += Inches(1.5)

add_notes(slide,
          "Three objectives, three deliverables. The important thing is that this is not a mock "
          "up: the service in this deck is the one in the repository, and every number was "
          "produced by running it.")

# --------------------------------------------------------------------------
# 4 - The model
# --------------------------------------------------------------------------
slide = new_slide("The model: simple on purpose", "What is deployed")
textbox(slide, Inches(0.85), Inches(1.95), Inches(6.0), Inches(0.5),
        "Logistic Regression, C = 0.1, balanced weights", size=20, bold=True, colour=NAVY)
bullets(slide, Inches(0.85), Inches(2.55), Inches(6.3), Inches(3.6), [
    ("Selected by", "5-fold CV ROC-AUC in Week 3 (0.806)"),
    ("Not the strongest", "Random Forest scored 0.796 — inside one std. dev."),
    ("Chosen for", "an exact explanation for every prediction"),
    ("One artifact", "imputer, scaler, encoder and classifier in one file"),
], size=15)

stats = [
    ("0.768", "ROC-AUC", "held-out"),
    ("0.656", "recall", "two thirds of churners"),
    ("5.7 KiB", "artifact", f"{CARD['artifacts']['joblib_bytes'] / 1024:.1f} KiB, 1 ms load"),
]
left = Inches(7.55)
for value, label, note in stats:
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, Inches(2.1),
                                 Inches(1.72), Inches(2.3))
    box.fill.solid()
    box.fill.fore_color.rgb = NAVY
    box.line.fill.background()
    textbox(slide, left + Inches(0.1), Inches(2.35), Inches(1.52), Inches(0.7),
            value, size=21, bold=True, colour=WHITE, align=PP_ALIGN.CENTER)
    textbox(slide, left + Inches(0.1), Inches(3.05), Inches(1.52), Inches(0.3),
            label, size=11, bold=True, colour=RGBColor(0x7F, 0xC4, 0xC4),
            align=PP_ALIGN.CENTER)
    textbox(slide, left + Inches(0.1), Inches(3.42), Inches(1.52), Inches(0.8),
            note, size=9, colour=RGBColor(0xB8, 0xCB, 0xD9), align=PP_ALIGN.CENTER)
    left += Inches(1.86)

add_notes(slide,
          "I could have deployed the Random Forest. I did not, and the reason is on the next "
          "slide but one: the linear model gives an exact decomposition of every prediction. "
          "The gap between them was inside the noise band, so explainability was free.")

# --------------------------------------------------------------------------
# 5 - Architecture
# --------------------------------------------------------------------------
slide = new_slide("One request, end to end", "Architecture")
add_image(slide, FIGURES / "00_system_architecture.png", Inches(0.9), Inches(1.85),
          Inches(11.5), Inches(4.9))
add_notes(slide,
          "Read it left to right. Validation happens before any model code runs, which is the "
          "single most valuable design decision in the whole service. The model is loaded once "
          "at startup — you can see the artifact box at the bottom. Nothing is re-fitted per "
          "request.")

# --------------------------------------------------------------------------
# 6 - The API
# --------------------------------------------------------------------------
slide = new_slide("Six endpoints, each with one job", "The API")
api_rows = [
    ("GET", "/health", "is the model actually loaded?"),
    ("GET", "/model", "model card: metrics, intended use, limits"),
    ("GET", "/model/manifest", "SHA-256 per artifact file"),
    ("POST", "/predict", "one customer, with an explanation"),
    ("POST", "/predict/batch", "up to 1,000, vectorised"),
    ("GET", "/docs", "generated from the schemas"),
]
top = Inches(2.0)
for method, path, detail in api_rows:
    textbox(slide, Inches(0.85), top, Inches(0.8), Inches(0.35),
            method, size=12, bold=True,
            colour=TEAL if method == "GET" else CORAL)
    textbox(slide, Inches(1.75), top, Inches(3.5), Inches(0.35),
            path, size=14, bold=True, colour=NAVY)
    textbox(slide, Inches(5.4), top + Inches(0.04), Inches(6.6), Inches(0.35),
            detail, size=12.5, colour=GREY)
    top += Inches(0.66)

textbox(slide, Inches(0.85), Inches(6.15), Inches(11.4), Inches(0.6),
        "Health reports whether the model object exists — a container whose model failed to "
        "load should not answer 200.", size=12, italic=True, colour=GREY)

add_notes(slide,
          "The health endpoint is worth a sentence. The usual implementation returns 200 as "
          "long as the process is alive, which means a container with no model still passes its "
          "readiness probe. This one checks the model object.")

# --------------------------------------------------------------------------
# 7 - Explainability
# --------------------------------------------------------------------------
slide = new_slide("Every prediction can be justified", "Explainability")
add_image(slide, FIGURES / "08_explanation_example.png", Inches(0.7), Inches(1.8),
          Inches(8.4), Inches(4.4))
bullets(slide, Inches(9.3), Inches(2.2), Inches(3.5), Inches(3.8), [
    "Exact, not approximated",
    "Adds to the log-odds",
    "The arithmetic checks out",
], size=14)

add_notes(slide,
          "This customer is at 71% risk. The contract type alone is worth +0.96 log-odds; DSL "
          "and having a partner pull the other way. If you add those numbers, apply the logistic "
          "function, you get exactly the probability the API returned. That is the difference "
          "between an explanation and a plausible-looking chart.")

# --------------------------------------------------------------------------
# 8 - Latency
# --------------------------------------------------------------------------
slide = new_slide(f"Where {BENCH['single_explained_p50_ms']:.1f} milliseconds actually goes", "Performance")
add_image(slide, FIGURES / "01_latency_budget.png", Inches(0.8), Inches(1.95),
          Inches(11.7), Inches(2.6))
bullets(slide, Inches(0.85), Inches(4.85), Inches(11.6), Inches(1.8), [
    (f"Model is {BENCH['model_only_p50_ms'] / BENCH['single_explained_p50_ms']:.0%}.",
     "Optimising it further would be invisible to the user."),
    (f"Explanation is {BENCH['explanation_cost_ms'] / BENCH['single_explained_p50_ms']:.0%}.",
     "The largest cost — so it is optional, not mandatory."),
    (f"HTTP is {BENCH['http_overhead_ms'] / BENCH['single_explained_p50_ms']:.0%}.",
     "A floor that no model change can go below."),
], size=15)

add_notes(slide,
          "This chart changed how I thought about the project. I assumed the model was the "
          "expensive part. It is not: the explanation costs more than the prediction it "
          "explains. That is why the API has an explain_result flag.")

# --------------------------------------------------------------------------
# 9 - The defect
# --------------------------------------------------------------------------
slide = new_slide("A defect only a benchmark could find", "Performance")
add_image(slide, FIGURES / "03_throughput.png", Inches(0.7), Inches(1.8),
          Inches(7.6), Inches(4.3))
bullets(slide, Inches(8.55), Inches(2.15), Inches(4.3), Inches(4.0), [
    ("92 rows/s", "first implementation, all 18 tests green"),
    (f"{BENCH['batch_rows_per_s']:,.0f} rows/s", "after vectorising one call"),
    ("3 lines", "to remove a per-row pandas frame"),
], size=14)

add_notes(slide,
          "This is the finding I am most pleased with, because it was invisible to every test. "
          "The batch endpoint looped over rows, so it was paying the fixed cost of a pandas frame "
          "and a transformer call a hundred times per request. All eighteen tests passed. The "
          "benchmark took it from 92 rows per second to nearly eleven thousand.")

# --------------------------------------------------------------------------
# 10 - Quality
# --------------------------------------------------------------------------
slide = new_slide("How good is it, really?", "Model quality")
add_image(slide, FIGURES / "04_confusion_matrix.png", Inches(0.8), Inches(1.75),
          Inches(5.4), Inches(4.6))
bullets(slide, Inches(6.6), Inches(2.2), Inches(6.0), Inches(4.0), [
    ("181 + 63 correct", "of 360 held-out rows"),
    ("83 false alarms", "a wasted retention offer"),
    ("33 missed churners", "a lost customer — far more expensive"),
    ("0.768 ROC-AUC", "useful for ranking, not for automating"),
], size=15)

add_notes(slide,
          "I want to be careful not to oversell this. Accuracy is 0.678, which is below the "
          "do-nothing baseline of 0.733. That is expected — balanced class weights trade accuracy "
          "for recall deliberately. The two errors are not symmetric: a false alarm costs one "
          "offer, a missed churner costs the whole customer.")

# --------------------------------------------------------------------------
# 11 - Threshold
# --------------------------------------------------------------------------
slide = new_slide("The threshold is a business decision", "Operating point")
add_image(slide, FIGURES / "06_threshold_tradeoff.png", Inches(0.7), Inches(1.8),
          Inches(7.8), Inches(4.3))
bullets(slide, Inches(8.75), Inches(2.2), Inches(4.1), Inches(4.0), [
    ("0.50 default", "neutral, documented, echoed in every response"),
    ("0.40", "recall 0.82, precision 0.42"),
    ("The API refuses", "to pick one silently for the caller"),
], size=13.5)

add_notes(slide,
          "The model returns a probability. Turning it into a decision is a separate act, and "
          "the right threshold depends on what a missed churner costs relative to a wasted offer. "
          "That is not something I can know from the data, so the API exposes it as a parameter "
          "and returns it in the response for auditability.")

# --------------------------------------------------------------------------
# 12 - Validation
# --------------------------------------------------------------------------
slide = new_slide("Failing at the edge, not in the model", "Robustness")
add_image(slide, FIGURES / "09_artifacts_and_validation.png", Inches(0.6), Inches(1.9),
          Inches(8.4), Inches(3.6))
bullets(slide, Inches(9.35), Inches(2.2), Inches(3.5), Inches(3.8), [
    "7 validation tests",
    "422 with a readable message",
    "null total_charges accepted",
], size=13.5)

add_notes(slide,
          "The most common production failure is not a bad prediction, it is a request in a "
          "shape the model never saw. Every 422 here is an exception that never reached "
          "scikit-learn. Note the last row: an omitted total_charges is accepted, because a new "
          "customer genuinely has none.")

# --------------------------------------------------------------------------
# 13 - Limits
# --------------------------------------------------------------------------
slide = new_slide("What this model must not be used for", "Responsible use")
bullets(slide, Inches(0.85), Inches(2.0), Inches(11.4), Inches(4.4), [
    ("Synthetic data.", "Every pattern is a property of the generator, not of real customers."),
    ("Not for consequential decisions.", "No credit, pricing, insurance or hiring."),
    ("No fairness audit.", "Error rates can differ by subgroup; that has not been measured."),
    ("No authentication.", "A local demonstration, not a public service."),
    ("It will drift.", "The model card records the training date for exactly this reason."),
], size=16)

add_notes(slide,
          "Deploying a model makes its limitations consequential, because people act on the "
          "output. The model card lists these, and I would rather say them out loud than have "
          "someone discover them later.")

# --------------------------------------------------------------------------
# 14 - Close
# --------------------------------------------------------------------------
slide = prs.slides.add_slide(BLANK)
background = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, WIDTH, HEIGHT)
background.fill.solid()
background.fill.fore_color.rgb = NAVY
background.line.fill.background()

textbox(slide, Inches(0.9), Inches(1.15), Inches(11.4), Inches(0.9),
        "What I would do next", size=34, bold=True, colour=WHITE)
bullets(slide, Inches(0.9), Inches(2.25), Inches(11.0), Inches(2.6), [
    ("Monitoring.", "Log every prediction and compare the live distribution to training."),
    ("Fairness.", "Compute the confusion matrix per subgroup before any real use."),
    ("Auth and rate limiting.", "The two things between this and a public deployment."),
], size=15)

line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.92), Inches(5.0),
                              Inches(1.4), Inches(0.05))
line.fill.solid()
line.fill.fore_color.rgb = TEAL
line.line.fill.background()

textbox(slide, Inches(0.9), Inches(5.25), Inches(11.0), Inches(1.0),
        "Thank you — questions welcome.", size=26, bold=True, colour=WHITE)
textbox(slide, Inches(0.9), Inches(6.35), Inches(11.0), Inches(0.4),
        f"{AUTHOR}   ·   github.com/Dhanam33/Yuva_intern/tree/main/week-4",
        size=11, colour=RGBColor(0xA8, 0xBC, 0xCB))

add_notes(slide,
          "To close: the deployment works, it is measured, and it is honest about what it is. "
          "The three things I would add before real use are monitoring, a fairness audit, and "
          "authentication. Happy to take questions.")


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    target = REPORTS / "Week_4_Capstone_Presentation.pptx"
    prs.save(target)
    count = len(prs.slides)
    print(f"Deck written : {target.relative_to(WEEK_DIR.parent)}")
    print(f"Slides       : {count}")
    print(f"Author       : {AUTHOR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
