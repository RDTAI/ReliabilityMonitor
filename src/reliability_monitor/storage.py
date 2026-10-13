"""DuckDB event log supporting labels that arrive after predictions."""
from __future__ import annotations
import json
from pathlib import Path
import duckdb


class EventStore:
    def __init__(self, path="state/events.duckdb"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = duckdb.connect(str(path))
        self.connection.execute("""CREATE TABLE IF NOT EXISTS predictions (
            request_id VARCHAR PRIMARY KEY, event_time TIMESTAMP, features JSON, embedding JSON,
            predicted_class INTEGER, confidence DOUBLE, prediction_set JSON, ood BOOLEAN,
            subgroup VARCHAR)""")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS labels (
            request_id VARCHAR PRIMARY KEY, label INTEGER, label_time TIMESTAMP)""")

    def log_prediction(self, event: dict):
        self.connection.execute("INSERT OR REPLACE INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", [
            event["request_id"], event["event_time"], json.dumps(event["features"]),
            json.dumps(event["embedding"]), event["predicted_class"], event["confidence"],
            json.dumps(event["prediction_set"]), event["ood"], event.get("subgroup", "unknown")])

    def attach_label(self, request_id: str, label: int, label_time: str):
        exists = self.connection.execute("SELECT count(*) FROM predictions WHERE request_id=?", [request_id]).fetchone()[0]
        if not exists:
            raise KeyError(f"unknown request_id: {request_id}")
        self.connection.execute("INSERT OR REPLACE INTO labels VALUES (?, ?, ?)", [request_id, label, label_time])

    def window(self, since: str | None = None, labeled_only=False) -> list[dict]:
        query = """SELECT p.*, l.label, l.label_time FROM predictions p
                   LEFT JOIN labels l USING(request_id) WHERE 1=1"""
        params = []
        if since:
            query += " AND p.event_time >= ?"; params.append(since)
        if labeled_only:
            query += " AND l.label IS NOT NULL"
        query += " ORDER BY p.event_time"
        columns = [x[0] for x in self.connection.execute(query, params).description]
        rows = self.connection.execute(query, params).fetchall()
        result = []
        for row in rows:
            item = dict(zip(columns, row))
            for key in ("features", "embedding", "prediction_set"):
                item[key] = json.loads(item[key])
            result.append(item)
        return result

