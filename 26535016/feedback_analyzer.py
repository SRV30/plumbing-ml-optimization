
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List

import joblib

# Text cleaning (shared by training and prediction - must stay identical!)
_URL = re.compile(r"https?://\S+")
_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:

    text = str(text).lower()
    text = _URL.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text



PLAYBOOK: Dict[str, dict] = {
    "late_arrival_no_show": {
        "label": "Late arrival / no-show",
        "team": "Dispatch & Scheduling",
        "actions": [
            "Send proactive ETA and delay alerts (SMS) as soon as a technician is running late.",
            "Offer a service credit or waive the trip fee for the late or missed visit.",
            "Review dispatch routing and technician workload for this branch and time slot.",
        ],
        "high": [
            "Call the customer within 24 hours and book the earliest available slot.",
            "Escalate to the branch manager for a root-cause review of the missed appointment.",
        ],
        "reply": "We are sorry our technician was not there on time. We are reviewing what went wrong with our dispatch team.",
    },
    "pricing_billing": {
        "label": "Pricing / billing",
        "team": "Billing & Pricing",
        "actions": [
            "Give a written, itemised quote and get approval before any price change.",
            "Show every fee (trip fee, disposal, parts) upfront, before work starts.",
            "Compare the final invoice with the original quote and refund unjustified charges.",
        ],
        "high": [
            "Have a manager contact the customer within 24 hours to explain or correct the bill.",
        ],
        "reply": "We are sorry the final bill did not match your expectations. We are reviewing your invoice against the original quote.",
    },
    "repair_not_fixed": {
        "label": "Repair not fixed",
        "team": "Service Quality",
        "actions": [
            "Offer a free return visit with a senior technician.",
            "Record the root cause and run a post-job quality check on this job type.",
            "Offer or extend a warranty on the repair.",
        ],
        "high": [
            "Schedule a same-day follow-up visit at no charge.",
            "Review the technician's recent repeat-visit rate.",
        ],
        "reply": "We are sorry the problem was not resolved. We would like to send a senior technician back at no charge.",
    },
    "poor_communication": {
        "label": "Poor communication",
        "team": "Customer Support",
        "actions": [
            "Send automatic appointment confirmations and status updates.",
            "Guarantee a callback from the office within one hour of a missed call.",
            "Ask technicians to explain the problem, the fix and the cost before starting.",
        ],
        "high": ["Assign a named contact to the customer until the issue is closed."],
        "reply": "We are sorry it was hard to get updates from us. We are improving how we keep customers informed.",
    },
    "upselling": {
        "label": "Upselling / pressure",
        "team": "Sales Practices",
        "actions": [
            "Train technicians to present add-ons as optional, with clear pricing.",
            "Require customer approval before any extra work begins.",
            "Track add-on rate per technician to spot aggressive selling.",
        ],
        "high": ["Review the technician's recent invoices for unapproved add-ons."],
        "reply": "We are sorry you felt pressured about extra services. Add-ons should always be your choice.",
    },
    "scheduling_cancellation": {
        "label": "Scheduling / cancellation",
        "team": "Dispatch & Scheduling",
        "actions": [
            "Confirm appointments 24 hours ahead and notify customers early about any change.",
            "Give priority rescheduling to customers whose visit we cancelled.",
            "Review cancellation reasons for this branch (capacity, parts, staffing).",
        ],
        "high": ["Call the customer today and offer the next available priority slot."],
        "reply": "We are sorry about the scheduling problem. We would like to rebook you with priority.",
    },
    "technician_behavior": {
        "label": "Technician behavior",
        "team": "Technician Management",
        "actions": [
            "Share the feedback with the technician's supervisor for coaching.",
            "Refresh customer-service training (courtesy, listening, respect for the home).",
            "Follow up with the customer to apologise.",
        ],
        "high": ["Review the technician's recent ratings before assigning more visits."],
        "reply": "We are sorry about the way our technician behaved. This is not our standard and we are following up internally.",
    },
    "property_damage_mess": {
        "label": "Property damage / mess",
        "team": "Service Quality & Claims",
        "actions": [
            "Add a cleanup and floor-protection checklist to the end of every job.",
            "Photograph the damage and start the claim process quickly.",
            "Contact the customer to arrange repair or compensation.",
        ],
        "high": ["Open a damage claim and call the customer within 24 hours."],
        "reply": "We are sorry about the damage or mess left at your home. We will contact you to put it right.",
    },
}

NO_COMPLAINT = "none"
SEVERITY_ORDER = ["low", "medium", "high"]


def _build_reply(sentiment: str, category: str) -> str:
    if category != NO_COMPLAINT:
        return (
            "Thank you for your feedback. "
            + PLAYBOOK[category]["reply"]
            + " A member of our team will contact you shortly."
        )
    if sentiment == "positive":
        return "Thank you for the kind words! We will share your feedback with the team. We would really appreciate a public review."
    if sentiment == "neutral":
        return "Thank you for your feedback. We would love to hear what would have made your visit better."
    return "Thank you for your feedback. We are sorry your visit fell short and a team member will contact you."


class FeedbackAnalyzer:
    """Loads the trained models and turns review text into a structured result."""

    def __init__(self, model_path: str | Path = "models/feedback_models.joblib"):
        bundle = joblib.load(model_path)
        self.vectorizer = bundle["vectorizer"]
        self.sentiment_model = bundle["sentiment"]
        self.category_model = bundle["category"]
        self.severity_model = bundle["severity"]
        self.meta = bundle.get("meta", {})
        self.model_version = self.meta.get("trained_at", "unknown")

    # public
    def analyze(self, text: str) -> dict:
        """Analyse ONE review."""
        return self.analyze_many([text])[0]

    def analyze_many(self, texts: Iterable[str]) -> List[dict]:
        """Analyse many reviews at once (much faster than a Python loop)."""
        texts = [self._validate(t) for t in texts]
        if not texts:
            return []
        X = self.vectorizer.transform([clean_text(t) for t in texts])
        sent_p = self.sentiment_model.predict_proba(X)
        cat_p = self.category_model.predict_proba(X)
        sev_p = self.severity_model.predict_proba(X)

        s_classes = list(self.sentiment_model.classes_)
        c_classes = list(self.category_model.classes_)
        v_classes = list(self.severity_model.classes_)

        out = []
        for i, text in enumerate(texts):
            s_idx = int(sent_p[i].argmax())
            sentiment = s_classes[s_idx]
            sent_conf = float(sent_p[i][s_idx])

            c_idx = int(cat_p[i].argmax())
            category = c_classes[c_idx]
            cat_conf = float(cat_p[i][c_idx])
            has_complaint = category != NO_COMPLAINT

            alternatives = sorted(
                (
                    {"category": c, "probability": round(float(p), 3)}
                    for c, p in zip(c_classes, cat_p[i])
                    if c != category and c != NO_COMPLAINT and p >= 0.10  
                ),
                key=lambda d: d["probability"],
                reverse=True,
            )[:2]

            severity = None
            if has_complaint:
                severity = v_classes[int(sev_p[i].argmax())]

            out.append(
                {
                    "text": text,
                    "sentiment": {
                        "label": sentiment,
                        "confidence": round(sent_conf, 3),
                        "probabilities": {c: round(float(p), 3) for c, p in zip(s_classes, sent_p[i])},
                    },
                    "complaint": {
                        "has_complaint": has_complaint,
                        "category": category,
                        "category_label": PLAYBOOK[category]["label"] if has_complaint else "No complaint",
                        "confidence": round(cat_conf, 3),
                        "alternatives": alternatives if has_complaint else [],
                    },
                    "severity": severity,
                    "priority": self._priority(sentiment, has_complaint, severity),
                    "team": PLAYBOOK[category]["team"] if has_complaint else None,
                    "recommendations": self._recommendations(sentiment, category, severity),
                    "suggested_reply": _build_reply(sentiment, category),
                    "needs_human_review": bool(sent_conf < 0.5 or cat_conf < 0.6),
                    "model_version": self.model_version,
                }
            )
        return out

    # helpers
    @staticmethod
    def _validate(text) -> str:
        if not isinstance(text, str):
            raise ValueError("Review text must be a string.")
        text = text.strip()
        if len(text) < 3:
            raise ValueError("Please enter a longer review.")
        return text[:2000]  # guard against huge inputs

    @staticmethod
    def _priority(sentiment: str, has_complaint: bool, severity: str | None) -> str:
        if has_complaint and severity == "high":
            return "urgent"
        if has_complaint and (severity == "medium" or sentiment == "negative"):
            return "follow_up"
        if has_complaint:
            return "monitor"
        if sentiment == "negative":
            return "follow_up"
        return "none"

    @staticmethod
    def _recommendations(sentiment: str, category: str, severity: str | None) -> List[str]:
        if category == NO_COMPLAINT:
            if sentiment == "positive":
                return [
                    "Thank the customer and invite them to post a public review.",
                    "Share the praise with the technician and branch team.",
                ]
            if sentiment == "neutral":
                return ["Follow up with the customer to ask what would have made the visit better."]
            return ["Negative feedback with no clear cause - have a manager call the customer to find out why."]
        info = PLAYBOOK[category]
        recs = list(info["actions"])
        if severity == "high":
            recs = info["high"] + recs
        elif severity == "low":
            recs = recs[:2] 
        return recs
