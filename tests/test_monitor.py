from reliability_monitor import ReliabilityMonitor
from reliability_monitor.storage import EventStore


def event(i, shift=0, label=None, group="a"):
    out = {"request_id": str(i), "event_time": "2026-01-01T00:00:00Z",
           "features": [i / 20 + shift, 0.0], "embedding": [i / 20 + shift, 0.0],
           "predicted_class": i % 2, "confidence": .8, "prediction_set": [i % 2],
           "ood": shift > 1, "subgroup": group}
    if label is not None: out["label"] = label
    return out


def test_metrics_expand_when_labels_arrive():
    reference = [event(i) for i in range(40)]
    unlabeled = ReliabilityMonitor(reference).evaluate([event(i + 100) for i in range(10)])
    assert "accuracy" not in unlabeled["metrics"]
    assert unlabeled["label_availability"]["pending"] == 10
    labeled = ReliabilityMonitor(reference).evaluate([event(i + 100, label=i % 2) for i in range(10)])
    assert labeled["metrics"]["accuracy"] == 1
    assert labeled["label_availability"]["received"] == 10


def test_ood_alert_fires():
    reference = [event(i) for i in range(40)]
    result = ReliabilityMonitor(reference).evaluate([event(i, shift=2) for i in range(20)])
    assert result["status"] == "alert"
    assert "ood_rate" in {a["metric"] for a in result["alerts"]}


def test_duckdb_joins_delayed_label(tmp_path):
    store = EventStore(tmp_path / "events.duckdb")
    row = event(1)
    store.log_prediction(row)
    assert store.window(labeled_only=True) == []
    store.attach_label("1", 1, "2026-01-02T00:00:00Z")
    joined = store.window(labeled_only=True)
    assert len(joined) == 1
    assert joined[0]["label"] == 1
