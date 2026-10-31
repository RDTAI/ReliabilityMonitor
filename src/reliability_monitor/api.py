"""FastAPI service for versioned prediction logging and reliability monitoring."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from prometheus_client import Gauge, make_asgi_app

from .monitor import ReliabilityMonitor
from .storage import EventStore

app = FastAPI(title="ReliabilityMonitor", version="0.2.0")
store = EventStore(os.getenv("MONITOR_DB", "state/events.duckdb"))

GAUGES = {
    "ood_rate": Gauge("reliability_ood_rate", "Observed OOD rate", ["model", "version"]),
    "ece": Gauge("reliability_ece", "Expected calibration error", ["model", "version"]),
    "worst_group_coverage": Gauge("reliability_worst_group_coverage", "Worst subgroup coverage", ["model", "version"]),
    "feature_psi_max": Gauge("reliability_feature_psi_max", "Maximum feature PSI", ["model", "version"]),
    "confidence_psi": Gauge("reliability_confidence_psi", "Confidence PSI", ["model", "version"]),
    "embedding_shift": Gauge("reliability_embedding_shift", "Normalized embedding mean shift", ["model", "version"]),
    "mean_set_size": Gauge("reliability_mean_set_size", "Mean conformal set size", ["model", "version"]),
    "accuracy": Gauge("reliability_accuracy", "Delayed-label accuracy", ["model", "version"]),
    "coverage": Gauge("reliability_coverage", "Observed conformal coverage", ["model", "version"]),
    "label_completion_rate": Gauge("reliability_label_completion_rate", "Fraction of predictions with labels", ["model", "version"]),
}
app.mount("/prometheus", make_asgi_app())


class PredictionEvent(BaseModel):
    request_id: str
    model_name: str = "default"
    model_version: str = "unknown"
    features: list[float] = Field(min_length=1)
    embedding: list[float] = Field(min_length=1)
    predicted_class: int
    confidence: float = Field(ge=0, le=1)
    prediction_set: list[int]
    ood: bool = False
    subgroup: str = "unknown"
    event_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LabelEvent(BaseModel):
    request_id: str
    label: int
    label_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ReferenceProfile(BaseModel):
    reference_id: str = Field(default_factory=lambda: str(uuid4()))
    model_name: str
    model_version: str
    target_coverage: float = Field(default=.9, gt=0, lt=1)
    events: list[PredictionEvent] = Field(min_length=1)


@app.get("/health")
def health():
    return {"status": "ok", "version": app.version}


@app.post("/predictions", status_code=201)
def prediction(event: PredictionEvent):
    store.log_prediction(event.model_dump(mode="json"))
    return {"logged": event.request_id, "model": event.model_name,
            "version": event.model_version, "label_status": "pending"}


@app.post("/labels", status_code=201)
def label(event: LabelEvent):
    try:
        store.attach_label(event.request_id, event.label, event.label_time.isoformat())
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"updated": event.request_id}


@app.post("/references", status_code=201)
def set_reference(profile: ReferenceProfile):
    events = [event.model_dump(mode="json") for event in profile.events]
    mismatched = [event["request_id"] for event in events
                  if event["model_name"] != profile.model_name or
                  event["model_version"] != profile.model_version]
    if mismatched:
        raise HTTPException(422, f"reference events have mismatched model identity: {mismatched[:3]}")
    store.save_reference(profile.reference_id, profile.model_name, profile.model_version,
                         events, profile.target_coverage)
    return {"reference_id": profile.reference_id, "reference_size": len(events),
            "model": profile.model_name, "version": profile.model_version,
            "persisted": True}


@app.get("/references")
def references():
    return {"references": store.list_references()}


def _export_prometheus(metrics: dict, model_name: str, model_version: str,
                       received: int, pending: int):
    labels = {"model": model_name, "version": model_version}
    for name, gauge in GAUGES.items():
        if name == "label_completion_rate":
            total = received + pending
            gauge.labels(**labels).set(received / total if total else 0)
        elif metrics.get(name) is not None:
            gauge.labels(**labels).set(metrics[name])


@app.get("/report")
def report(
    model_name: str = "default",
    model_version: str = "unknown",
    since: datetime | None = None,
    window_hours: float | None = Query(default=24, gt=0),
    min_group_size: int = Query(default=20, ge=1),
):
    reference = store.get_reference(model_name, model_version)
    if reference is None:
        raise HTTPException(409, f"no active reference for {model_name}:{model_version}")
    effective_since = since
    if effective_since is None and window_hours is not None:
        effective_since = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    events = store.window(effective_since.isoformat() if effective_since else None,
                          model_name=model_name, model_version=model_version)
    result = ReliabilityMonitor(reference["events"],
                                target_coverage=reference["target_coverage"],
                                min_group_size=min_group_size).evaluate(events)
    result["window"] = {"since": effective_since.isoformat() if effective_since else None,
                        "hours": window_hours, "model": model_name, "version": model_version,
                        "reference_id": reference["reference_id"]}
    availability = result.get("label_availability", {"received": 0, "pending": len(events)})
    _export_prometheus(result["metrics"], model_name, model_version,
                       availability["received"], availability["pending"])
    return result
