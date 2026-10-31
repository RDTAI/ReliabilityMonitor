"""DuckDB event log with persistent references and delayed-label joins."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb


class EventStore:
    def __init__(self, path="state/events.duckdb"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = duckdb.connect(str(path))
        self.connection.execute("""CREATE TABLE IF NOT EXISTS predictions (
            request_id VARCHAR PRIMARY KEY, event_time TIMESTAMP, features JSON, embedding JSON,
            predicted_class INTEGER, confidence DOUBLE, prediction_set JSON, ood BOOLEAN,
            subgroup VARCHAR, model_name VARCHAR DEFAULT 'default',
            model_version VARCHAR DEFAULT 'unknown')""")
        existing = {row[1] for row in self.connection.execute("PRAGMA table_info('predictions')").fetchall()}
        if "model_name" not in existing:
            self.connection.execute("ALTER TABLE predictions ADD COLUMN model_name VARCHAR DEFAULT 'default'")
        if "model_version" not in existing:
            self.connection.execute("ALTER TABLE predictions ADD COLUMN model_version VARCHAR DEFAULT 'unknown'")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS labels (
            request_id VARCHAR PRIMARY KEY, label INTEGER, label_time TIMESTAMP)""")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS reference_profiles (
            reference_id VARCHAR PRIMARY KEY, model_name VARCHAR, model_version VARCHAR,
            target_coverage DOUBLE, events JSON, created_at TIMESTAMP, active BOOLEAN)""")

    def log_prediction(self, event: dict):
        self.connection.execute("""INSERT OR REPLACE INTO predictions
            (request_id, event_time, features, embedding, predicted_class, confidence,
             prediction_set, ood, subgroup, model_name, model_version)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", [
            event["request_id"], event["event_time"], json.dumps(event["features"]),
            json.dumps(event["embedding"]), event["predicted_class"], event["confidence"],
            json.dumps(event["prediction_set"]), event["ood"], event.get("subgroup", "unknown"),
            event.get("model_name", "default"), event.get("model_version", "unknown")])

    def attach_label(self, request_id: str, label: int, label_time: str):
        exists = self.connection.execute(
            "SELECT count(*) FROM predictions WHERE request_id=?", [request_id]
        ).fetchone()[0]
        if not exists:
            raise KeyError(f"unknown request_id: {request_id}")
        self.connection.execute("INSERT OR REPLACE INTO labels VALUES (?, ?, ?)", [request_id, label, label_time])

    def save_reference(self, reference_id: str, model_name: str, model_version: str,
                       events: list[dict], target_coverage: float = .9):
        self.connection.execute(
            "UPDATE reference_profiles SET active=false WHERE model_name=? AND model_version=?",
            [model_name, model_version],
        )
        created_at = datetime.now(timezone.utc).isoformat()
        self.connection.execute("""INSERT OR REPLACE INTO reference_profiles
            VALUES (?, ?, ?, ?, ?, ?, true)""", [reference_id, model_name, model_version,
            target_coverage, json.dumps(events), created_at])

    def get_reference(self, model_name: str, model_version: str) -> dict | None:
        row = self.connection.execute("""SELECT reference_id, target_coverage, events, created_at
            FROM reference_profiles WHERE model_name=? AND model_version=? AND active=true
            ORDER BY created_at DESC LIMIT 1""", [model_name, model_version]).fetchone()
        if row is None:
            return None
        return {"reference_id": row[0], "model_name": model_name, "model_version": model_version,
                "target_coverage": row[1], "events": json.loads(row[2]),
                "created_at": row[3].isoformat() if hasattr(row[3], "isoformat") else str(row[3])}

    def list_references(self) -> list[dict]:
        rows = self.connection.execute("""SELECT reference_id, model_name, model_version,
            target_coverage, created_at, active, json_array_length(events) AS n
            FROM reference_profiles ORDER BY created_at DESC""").fetchall()
        return [{"reference_id": r[0], "model_name": r[1], "model_version": r[2],
                 "target_coverage": r[3], "created_at": r[4].isoformat(),
                 "active": r[5], "n": r[6]} for r in rows]

    def window(self, since: str | None = None, labeled_only=False, *,
               model_name: str | None = None, model_version: str | None = None) -> list[dict]:
        query = """SELECT p.*, l.label, l.label_time FROM predictions p
                   LEFT JOIN labels l USING(request_id) WHERE 1=1"""
        params = []
        if since:
            query += " AND p.event_time >= ?"; params.append(since)
        if model_name:
            query += " AND p.model_name = ?"; params.append(model_name)
        if model_version:
            query += " AND p.model_version = ?"; params.append(model_version)
        if labeled_only:
            query += " AND l.label IS NOT NULL"
        query += " ORDER BY p.event_time"
        cursor = self.connection.execute(query, params)
        columns = [x[0] for x in cursor.description]
        rows = cursor.fetchall()
        result = []
        for row in rows:
            item = dict(zip(columns, row))
            for key in ("features", "embedding", "prediction_set"):
                item[key] = json.loads(item[key])
            result.append(item)
        return result
