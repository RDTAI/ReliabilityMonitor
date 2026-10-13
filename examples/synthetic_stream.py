"""Simulate a healthy reference window, drift, and delayed labels."""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
import json
import numpy as np
from reliability_monitor import ReliabilityMonitor


def make_window(n, *, seed, shift=0.0, label_fraction=0.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(shift, 1, size=(n, 4))
    embedding = np.c_[x[:, :2], x[:, 0] * x[:, 1]]
    logits = np.c_[x[:, 0], x[:, 1], -x[:, 0] - x[:, 1]]
    exp = np.exp(logits - logits.max(1, keepdims=True)); p = exp / exp.sum(1, keepdims=True)
    pred, confidence = p.argmax(1), p.max(1)
    true = np.argmax(logits + rng.normal(0, .8 + abs(shift), logits.shape), axis=1)
    threshold = .24
    sets = [np.flatnonzero(row >= threshold).tolist() for row in p]
    ood = np.linalg.norm(x, axis=1) > 3.2
    now = datetime.now(timezone.utc)
    events = []
    for i in range(n):
        event = {"request_id": f"{seed}-{i}", "event_time": (now + timedelta(seconds=i)).isoformat(),
                 "features": x[i].tolist(), "embedding": embedding[i].tolist(),
                 "predicted_class": int(pred[i]), "confidence": float(confidence[i]),
                 "prediction_set": sets[i], "ood": bool(ood[i]),
                 "subgroup": "device_b" if i % 4 == 0 else "device_a"}
        if i < int(n * label_fraction): event["label"] = int(true[i])
        events.append(event)
    return events


reference = make_window(1000, seed=1)
production = make_window(400, seed=2, shift=.75, label_fraction=0)
monitor = ReliabilityMonitor(reference)

print("=== Phase 1: labels unavailable ===")
print(json.dumps(monitor.evaluate(production), indent=2))
for event, labeled in zip(production, make_window(400, seed=2, shift=.75, label_fraction=1)):
    event["label"] = labeled["label"]
print("\n=== Phase 2: delayed labels received ===")
print(json.dumps(monitor.evaluate(production), indent=2))

