
import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             f1_score)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from feedback_analyzer import clean_text

REQUIRED = ["review_text", "sentiment", "complaint_category", "complaint_severity"]
SEED = 42


def make_classifier():
    
    return CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced"), cv=3)


def evaluate(name, model, X_test, y_test, out_dir, metrics):
    pred = model.predict(X_test)
    macro = f1_score(y_test, pred, average="macro")
    acc = float((pred == y_test).mean())
    metrics[name] = {"macro_f1": round(float(macro), 4), "accuracy": round(acc, 4), "n_test": int(len(y_test))}
    report = classification_report(y_test, pred, digits=3)
    (out_dir / f"{name}_report.txt").write_text(report)
    print(f"\n=== {name.upper()}  macro-F1={macro:.3f}  accuracy={acc:.3f} ===\n{report}")

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"{name} - confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_confusion.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="plumbing_reviews.csv")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--reports-dir", default="feedback_reports")
    ap.add_argument("--no-refit", action="store_true",
                    help="keep models trained on the 80%% split instead of refitting on all data")
    args = ap.parse_args()

    models_dir, reports_dir = Path(args.models_dir), Path(args.reports_dir)
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    df = pd.read_csv(args.data)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV is missing required columns: {missing}")
    df = df.dropna(subset=["review_text", "sentiment", "complaint_category"]).copy()
    df["text"] = df["review_text"].map(clean_text)
    print(f"Loaded {len(df):,} reviews")

    
    train, test = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["complaint_category"])

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_train = vec.fit_transform(train["text"])
    X_test = vec.transform(test["text"])

    metrics = {}
    sent = make_classifier().fit(X_train, train["sentiment"])
    evaluate("sentiment", sent, X_test, test["sentiment"], reports_dir, metrics)

    cat = make_classifier().fit(X_train, train["complaint_category"])
    evaluate("category", cat, X_test, test["complaint_category"], reports_dir, metrics)

    
    tr_c = train[train["complaint_category"] != "none"].dropna(subset=["complaint_severity"])
    te_c = test[test["complaint_category"] != "none"].dropna(subset=["complaint_severity"])
    sev = make_classifier().fit(vec.transform(tr_c["text"]), tr_c["complaint_severity"])
    evaluate("severity", sev, vec.transform(te_c["text"]), te_c["complaint_severity"], reports_dir, metrics)

    if not args.no_refit:
        
        print("\nRefitting final models on the full dataset ...")
        X_all = vec.fit_transform(df["text"])
        sent = make_classifier().fit(X_all, df["sentiment"])
        cat = make_classifier().fit(X_all, df["complaint_category"])
        all_c = df[df["complaint_category"] != "none"].dropna(subset=["complaint_severity"])
        sev = make_classifier().fit(vec.transform(all_c["text"]), all_c["complaint_severity"])

    meta = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "sklearn_version": sklearn.__version__,
        "n_rows": int(len(df)),
        "refit_on_all_data": not args.no_refit,
        "metrics": metrics,
    }
    joblib.dump({"vectorizer": vec, "sentiment": sent, "category": cat, "severity": sev, "meta": meta},
                models_dir / "feedback_models.joblib")
    (reports_dir / "metrics.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved models/feedback_models.joblib and feedback_reports/ (scikit-learn {sklearn.__version__})")


if __name__ == "__main__":
    main()
