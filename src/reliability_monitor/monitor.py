"""Monitoring orchestration and interpretable alert rules."""
from dataclasses import dataclass
import numpy as np
from .metrics import labeled_metrics, unlabeled_metrics


@dataclass(frozen=True)
class AlertRule:
    metric: str
    threshold: float
    comparison: str = "gt"
    message: str = "reliability threshold exceeded"


DEFAULT_RULES = (
    AlertRule("feature_psi_max", .25, "gt", "input feature distribution shifted"),
    AlertRule("embedding_shift", .5, "gt", "embedding distribution shifted"),
    AlertRule("ood_rate", .10, "gt", "OOD rate exceeded operational threshold"),
    AlertRule("ece", .10, "gt", "confidence calibration degraded"),
    AlertRule("worst_group_coverage", .88, "lt", "conditional reliability issue"),
)


def _columns(events):
    return {
        "features": [e["features"] for e in events],
        "embeddings": [e["embedding"] for e in events],
        "predicted_class": [e["predicted_class"] for e in events],
        "confidence": [e["confidence"] for e in events],
        "set_size": [len(e["prediction_set"]) for e in events],
        "ood": [e["ood"] for e in events],
    }


class ReliabilityMonitor:
    def __init__(self, reference_events, *, target_coverage=.9, rules=DEFAULT_RULES):
        if not reference_events:
            raise ValueError("reference_events cannot be empty")
        self.reference = _columns(reference_events)
        self.target_coverage = target_coverage
        self.rules = tuple(rules)

    def evaluate(self, events):
        if not events:
            return {"status": "empty", "metrics": {}, "alerts": []}
        metrics = unlabeled_metrics(self.reference, _columns(events))
        labeled = [e for e in events if e.get("label") is not None]
        if labeled:
            labeled_input = {
                "label": [e["label"] for e in labeled],
                "predicted_class": [e["predicted_class"] for e in labeled],
                "confidence": [e["confidence"] for e in labeled],
                "covered": [e["label"] in e["prediction_set"] for e in labeled],
                "group": [e.get("subgroup", "unknown") for e in labeled],
            }
            metrics.update(labeled_metrics(labeled_input, self.target_coverage))
        alerts = []
        for rule in self.rules:
            value = metrics.get(rule.metric)
            fired = value is not None and ((rule.comparison == "gt" and value > rule.threshold) or
                                            (rule.comparison == "lt" and value < rule.threshold))
            if fired:
                alerts.append({"metric": rule.metric, "value": value, "threshold": rule.threshold,
                               "message": rule.message})
        return {"status": "alert" if alerts else "ok", "metrics": metrics, "alerts": alerts,
                "label_availability": {"received": len(labeled), "pending": len(events) - len(labeled)}}

