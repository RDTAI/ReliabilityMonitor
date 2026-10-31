from datetime import datetime, timezone

from fastapi.testclient import TestClient

import reliability_monitor.api as api
from reliability_monitor.storage import EventStore


def payload(request_id, version="v2", label_class=0):
    return {"request_id": request_id, "model_name": "classifier", "model_version": version,
            "event_time": datetime.now(timezone.utc).isoformat(), "features": [0.1, 0.2],
            "embedding": [0.3, 0.4], "predicted_class": label_class, "confidence": .9,
            "prediction_set": [label_class], "ood": False, "subgroup": "site_a"}


def test_versioned_reference_and_sliding_report(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "store", EventStore(tmp_path / "api.duckdb"))
    client = TestClient(api.app)
    reference_events = [payload(f"ref-{i}") for i in range(30)]
    response = client.post("/references", json={"reference_id": "stable-v2",
                           "model_name": "classifier", "model_version": "v2",
                           "target_coverage": .9, "events": reference_events})
    assert response.status_code == 201
    assert response.json()["persisted"] is True

    assert client.post("/predictions", json=payload("case-1")).status_code == 201
    assert client.post("/predictions", json=payload("other-version", version="v1")).status_code == 201
    assert client.post("/labels", json={"request_id": "case-1", "label": 0}).status_code == 201
    report = client.get("/report", params={"model_name": "classifier", "model_version": "v2",
                                           "window_hours": 24, "min_group_size": 20})
    assert report.status_code == 200
    body = report.json()
    assert body["metrics"]["n_unlabeled"] == 1
    assert body["metrics"]["accuracy"] == 1
    assert body["window"]["reference_id"] == "stable-v2"
    assert body["metrics"]["worst_group_coverage"] is None
