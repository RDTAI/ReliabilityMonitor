"""FastAPI service for prediction logging, delayed labels, and monitoring."""
from __future__ import annotations
from datetime import datetime, timezone
import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Gauge, make_asgi_app
from .monitor import ReliabilityMonitor
from .storage import EventStore

app = FastAPI(title="ReliabilityMonitor", version="0.1.0")
store = EventStore(os.getenv("MONITOR_DB", "state/events.duckdb"))
reference_events: list[dict] = []

OOD_RATE = Gauge("reliability_ood_rate", "Observed OOD rate")
ECE = Gauge("reliability_ece", "Expected calibration error")
COVERAGE = Gauge("reliability_worst_group_coverage", "Worst subgroup conformal coverage")
app.mount("/prometheus", make_asgi_app())


class PredictionEvent(BaseModel):
    request_id: str
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


@app.get("/health")
def health(): return {"status": "ok"}


@app.post("/predictions", status_code=201)
def prediction(event: PredictionEvent):
    store.log_prediction(event.model_dump(mode="json"))
    return {"logged": event.request_id, "label_status": "pending"}


@app.post("/labels", status_code=201)
def label(event: LabelEvent):
    try:
        store.attach_label(event.request_id, event.label, event.label_time.isoformat())
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"updated": event.request_id}


@app.post("/reference")
def set_reference(events: list[PredictionEvent]):
    reference_events.clear(); reference_events.extend(e.model_dump(mode="json") for e in events)
    return {"reference_size": len(reference_events)}


@app.get("/report")
def report(since: str | None = None):
    if not reference_events:
        raise HTTPException(409, "post a reference window first")
    result = ReliabilityMonitor(reference_events).evaluate(store.window(since))
    metrics = result["metrics"]
    if "ood_rate" in metrics: OOD_RATE.set(metrics["ood_rate"])
    if "ece" in metrics: ECE.set(metrics["ece"])
    if metrics.get("worst_group_coverage") is not None: COVERAGE.set(metrics["worst_group_coverage"])
    return result

