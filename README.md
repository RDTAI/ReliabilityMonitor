# ReliabilityMonitor

A reproducible post-deployment monitoring service for classification systems.
It separates signals available immediately after inference from reliability
metrics that become available only when delayed labels arrive.

```text
model/API -> prediction event log -> drift monitor -> delayed labels
                                            |              |
                                            v              v
                                  unlabeled metrics   coverage/calibration
                                            \              /
                                             alerts + report
```

## v0.2 highlights

- Reference profiles are persisted and versioned in DuckDB instead of being
  lost when the API process restarts.
- Prediction events carry `model_name` and `model_version`, and reports never
  mix metrics from different deployed versions.
- `GET /report` uses a configurable sliding window (24 hours by default).
- Subgroup coverage includes 95% Wilson intervals and a minimum sample-size
  guard; undersized groups are reported but excluded from worst-group alerts.
- Prometheus exports drift, workload, delayed-label, accuracy, and coverage
  gauges with model/version labels.

## Why two monitoring phases?

**Before labels arrive**, the service monitors feature PSI, predicted-class
ratio shift, confidence PSI, normalized embedding mean shift, OOD rate,
conformal set size, and empty-set rate. These are warning signals, not proof
that accuracy has degraded.

**After labels arrive**, the same request IDs are joined with outcomes in
DuckDB. The report then adds accuracy, ECE, correctness Brier score, conformal
coverage, subgroup accuracy/coverage, worst-group coverage, and a binned
confidence-versus-error curve.

## Components

- `FastAPI`: prediction logging, delayed-label ingestion, reference registration,
  and JSON reports.
- `DuckDB`: persistent prediction/label event store with request-ID joins.
- `Prometheus`: exports OOD rate, ECE, and worst-group coverage.
- `Grafana`: provisioned starter dashboard for the exported metrics.
- `Docker Compose`: launches all three services locally.
- `pytest` and GitHub Actions: verify drift, delayed-label, and alert behavior.

The statistical metrics are implemented directly with NumPy so their
definitions are auditable. Evidently can be added later as a comparison layer,
but is not required by the core service.

## Alert policy

The default rules are deliberately readable:

| Metric | Alert condition | Interpretation |
|---|---:|---|
| Maximum feature PSI | `> 0.25` | input feature distribution shifted |
| Embedding shift | `> 0.50` | representation distribution shifted |
| OOD rate | `> 0.10` | too many unfamiliar inputs |
| ECE | `> 0.10` | confidence calibration degraded |
| Worst-group coverage | `< 0.88` | conditional reliability issue |

Thresholds are operational policy parameters, not universal scientific
constants. They should be selected using a stable validation period and the
cost of false alarms versus missed failures.

## Quick start

```bash
python -m pip install -e '.[dev]'
pytest -q
python examples/synthetic_stream.py
uvicorn reliability_monitor.api:app --reload
```

Interactive API documentation is then available at
`http://localhost:8000/docs`.

## Event workflow

Register a stable reference window with `POST /references`, then log each model
response with `POST /predictions`:

```json
{
  "request_id": "case-1042",
  "model_name": "pathology-classifier",
  "model_version": "2026.10",
  "features": [0.2, -0.4, 1.1],
  "embedding": [0.14, -0.71],
  "predicted_class": 2,
  "confidence": 0.84,
  "prediction_set": [2, 3],
  "ood": false,
  "subgroup": "device_b"
}
```

When the outcome becomes available, send it independently:

```json
{"request_id": "case-1042", "label": 3}
```

References contain the same model identity and are persisted with a unique
`reference_id` and target coverage. `GET /references` lists active and historic
profiles.

`GET /report?model_name=pathology-classifier&model_version=2026.10&window_hours=24`
joins the available labels and explicitly returns counts for
`received` and `pending` labels. It never treats missing labels as errors or
correct predictions.

## Docker stack

```bash
docker compose up --build
```

- API and OpenAPI: `http://localhost:8000/docs`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

## Metric definitions

- Feature and confidence drift use Population Stability Index with bins fitted
  only on the reference window.
- Class-ratio drift uses total variation distance.
- Embedding drift is the L2 distance between feature-wise standardized means.
- ECE compares mean confidence and empirical correctness in confidence bins.
- Conformal coverage is the fraction of delayed true labels contained in the
  prediction set logged at inference time.
- Worst-group coverage is the minimum observed coverage across supplied groups.
- Each subgroup receives a 95% Wilson coverage interval. Groups smaller than
  `min_group_size` (default 20) are marked `insufficient_data` and cannot fire a
  worst-group alert.

## Production limitations

This repository is an engineering and research prototype. PSI thresholds are
heuristics; repeated window testing needs multiplicity-aware governance; and
labels may be missing non-randomly. A production rollout should also add
authentication, alert deduplication, retention policies, privacy controls,
scheduled evaluation, and a real notification transport.

