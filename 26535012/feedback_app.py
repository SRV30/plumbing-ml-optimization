
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from feedback_analyzer import FeedbackAnalyzer

MODEL_PATH = "models/feedback_models.joblib"
DATA_PATH = "plumbing_reviews.csv"
LOG_PATH = "feedback_log.csv"

SENTIMENT_EMOJI = {"positive": "😊", "neutral": "😐", "negative": "😠"}
PRIORITY_TEXT = {
    "urgent": "🔴 URGENT - act within 24 hours",
    "follow_up": "🟠 Follow up with the customer",
    "monitor": "🟡 Monitor - minor issue",
    "none": "🟢 No action needed",
}
EXAMPLES = [
    "Technician never showed up for my appointment and nobody called me back. Avoid.",
    "Charged me $820 when the estimate was $670, and added a trip fee I was never told about.",
    "The leak is back two days after the repair. They said it was fixed but it was not.",
    "Great service! Luis was friendly, fixed the pipe quickly and cleaned up everything. Highly recommend!",
    "The job was okay. Work got done but the technician kept pushing add-ons I did not need.",
]


# loaders
@st.cache_resource(show_spinner=False)
def load_analyzer(model_path: str = MODEL_PATH) -> FeedbackAnalyzer:
    """Load the trained models once and reuse them across reruns."""
    return FeedbackAnalyzer(model_path)


@st.cache_data(show_spinner=False)
def load_data(data_path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(data_path)


# helpers
def log_feedback(log_path: str | None, r: dict) -> None:
    """Append every submitted review + prediction to a CSV (used by the log tab)."""
    if not log_path:
        return
    path = Path(log_path)
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "review_text", "sentiment", "category", "severity", "priority"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), r["text"], r["sentiment"]["label"],
                    r["complaint"]["category"], r["severity"] or "", r["priority"]])


def build_insights_figure(df: pd.DataFrame):
    """Two simple charts for managers: complaint mix and complaint rate by branch."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    complaints = df[df["complaint_category"] != "none"]
    counts = complaints["complaint_category"].value_counts().sort_values()
    axes[0].barh([c.replace("_", " ") for c in counts.index], counts.values, color="#3b6ea5")
    axes[0].set_title("Complaints by category")
    axes[0].set_xlabel("Number of reviews")

    if "branch_id" in df.columns:
        rate = (df["complaint_category"] != "none").groupby(df["branch_id"]).mean().sort_values(ascending=False) * 100
        axes[1].bar(rate.index.astype(str), rate.values, color="#c0504d")
        axes[1].set_title("Share of reviews with a complaint, by branch")
        axes[1].set_ylabel("% of reviews")
        axes[1].tick_params(axis="x", rotation=45)
    else:
        axes[1].axis("off")
    fig.tight_layout()
    return fig


def _load_example():
    """Button callback: copy the chosen example into the text box."""
    choice = st.session_state.get("fb_example")
    if choice and choice != "(choose an example)":
        st.session_state["fb_text"] = choice


# UI sections
def render_result(r: dict) -> None:
    """Show one analysis result (internal / manager view)."""
    s, c = r["sentiment"], r["complaint"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Sentiment", f"{SENTIMENT_EMOJI[s['label']]} {s['label'].title()}")
    col1.caption(f"{s['confidence']:.0%} confident")
    col2.metric("Complaint category", c["category_label"])
    if c["has_complaint"]:
        col2.caption(f"{c['confidence']:.0%} confident")
    col3.metric("Severity", r["severity"].title() if r["severity"] else "-")

    priority = r["priority"]
    box = {"urgent": st.error, "follow_up": st.warning, "monitor": st.info, "none": st.success}[priority]
    box(f"**Priority:** {PRIORITY_TEXT[priority]}" + (f"  \n**Owner team:** {r['team']}" if r["team"] else ""))

    if r["needs_human_review"]:
        st.warning("The models are not very confident about this review - please have a person check it.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Sentiment probabilities**")
        for label, p in sorted(s["probabilities"].items(), key=lambda kv: -kv[1]):
            st.progress(float(p), text=f"{label.title()} - {p:.0%}")
        if c["alternatives"]:
            alt = ", ".join(a["category"].replace("_", " ") for a in c["alternatives"])
            st.caption(f"Other possible categories: {alt}")
    with right:
        st.markdown("**Recommended actions for the business**")
        st.markdown("\n".join(f"{i}. {rec}" for i, rec in enumerate(r["recommendations"], 1)))

    # No key on purpose: a new reply text creates a fresh box for each new analysis.
    st.text_area("Suggested reply to the customer (editable)", value=r["suggested_reply"], height=110)


def feedback_tab(analyzer: FeedbackAnalyzer, log_path: str | None = LOG_PATH) -> None:
    """The feedback form. Call inside any page / tab."""
    st.subheader("Share your feedback")
    st.write("Tell us about your plumbing service. We read every review.")

    ex_col, btn_col = st.columns([4, 1])
    ex_col.selectbox("Try an example", ["(choose an example)"] + EXAMPLES, key="fb_example",
                     label_visibility="collapsed")
    btn_col.button("Load example", on_click=_load_example)

    text = st.text_area("Your review", key="fb_text", height=140, placeholder="Describe your experience ...")

    if st.button("Submit feedback", type="primary", key="fb_submit"):
        try:
            result = analyzer.analyze(text)
        except ValueError as e:
            st.error(str(e))
            st.session_state.pop("fb_result", None)
        else:
            log_feedback(log_path, result)
            st.session_state["fb_result"] = result

    result = st.session_state.get("fb_result")
    if result:
        st.divider()
        st.subheader("Analysis (internal view)")
        render_result(result)


def insights_tab(data_path: str | None = DATA_PATH) -> None:
    st.subheader("Complaint insights (historical data)")
    if data_path and Path(data_path).exists():
        df = load_data(data_path)
        st.pyplot(build_insights_figure(df))
        if "service_type" in df.columns:
            st.markdown("**Complaint rate by service type**")
            rate = ((df["complaint_category"] != "none").groupby(df["service_type"]).mean() * 100).sort_values()
            st.bar_chart(rate)
    else:
        st.info("Dataset file not found. Put plumbing_reviews.csv next to feedback_app.py.")


def log_tab(log_path: str | None = LOG_PATH) -> None:
    st.subheader("Reviews submitted through the form")
    if log_path and Path(log_path).exists():
        log = pd.read_csv(log_path)
        st.dataframe(log.iloc[::-1], use_container_width=True)
        st.download_button("Download log as CSV", log.to_csv(index=False), "feedback_log.csv", "text/csv")
    else:
        st.info("No reviews submitted yet.")


def model_info(analyzer: FeedbackAnalyzer) -> None:
    """Small 'about the models' panel (shown in the sidebar)."""
    st.markdown("### About the models")
    st.write("TF-IDF features + calibrated Linear SVM, trained separately for sentiment, "
             "complaint category and severity.")
    metrics = analyzer.meta.get("metrics", {})
    if metrics:
        rows = [{"model": k, "macro-F1": v["macro_f1"], "accuracy": v["accuracy"]} for k, v in metrics.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("Scores on a held-out 20% test set.")


# main
def main() -> None:
    st.set_page_config(page_title="Plumbing Feedback Analyzer", page_icon="🔧", layout="wide")
    st.title("🔧 Plumbing Service - Customer Feedback Analyzer")

    if not Path(MODEL_PATH).exists():
        st.error("Trained models not found. Run `python feedback_train.py --data plumbing_reviews.csv` first.")
        st.stop()
    analyzer = load_analyzer(MODEL_PATH)

    with st.sidebar:
        model_info(analyzer)

    tab1, tab2, tab3 = st.tabs(["Feedback form", "Complaint insights", "Submitted reviews"])
    with tab1:
        feedback_tab(analyzer, LOG_PATH)
    with tab2:
        insights_tab(DATA_PATH)
    with tab3:
        log_tab(LOG_PATH)


if __name__ == "__main__":
    main()
