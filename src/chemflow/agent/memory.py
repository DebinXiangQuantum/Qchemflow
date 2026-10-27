from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from chemflow.ids import canonical_json, content_hash
from chemflow.types import EnergyCurve, Estimate, SpectralEstimate


class ResultsDB:
    def __init__(self, path: str | Path = ":memory:"):
        self.connection = sqlite3.connect(str(path))
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("""CREATE TABLE IF NOT EXISTS results (
            id TEXT PRIMARY KEY, task_id TEXT NOT NULL, record_hash TEXT NOT NULL,
            plan_hash TEXT NOT NULL, node_id TEXT NOT NULL, operator TEXT NOT NULL,
            target_identity TEXT NOT NULL, relation TEXT NOT NULL, value REAL NOT NULL,
            unit TEXT NOT NULL, uncertainty TEXT NOT NULL, evidence TEXT NOT NULL,
            cost TEXT NOT NULL, provenance TEXT NOT NULL)""")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS cost_observations (
            template TEXT NOT NULL, feature TEXT NOT NULL, size REAL NOT NULL,
            dimension TEXT NOT NULL, measured REAL NOT NULL)""")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS replay_records (
            screen TEXT NOT NULL, role TEXT NOT NULL, id TEXT NOT NULL,
            record TEXT NOT NULL, PRIMARY KEY (screen, role, id))""")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS report_records (
            result_id TEXT PRIMARY KEY, headline TEXT NOT NULL, critic TEXT)""")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS refuted_results (
            result_id TEXT PRIMARY KEY, report_id TEXT NOT NULL)""")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS qcl_runs (
            id TEXT PRIMARY KEY, record_hash TEXT NOT NULL, plan_hash TEXT NOT NULL,
            target_kind TEXT NOT NULL, decision TEXT NOT NULL, result TEXT NOT NULL,
            profile TEXT NOT NULL, spent TEXT NOT NULL)""")

    def add_qcl_run(self, *, record_hash: str, plan_hash: str, target_kind: str,
                    decision: str, result: object, profile: list[dict],
                    spent: dict) -> str:
        data = (record_hash, plan_hash, target_kind, decision,
                canonical_json(result), canonical_json(profile), canonical_json(spent))
        identifier = content_hash(data)
        self.connection.execute("INSERT OR IGNORE INTO qcl_runs VALUES (?,?,?,?,?,?,?,?)",
                                (identifier, *data))
        self.connection.commit()
        return identifier

    def qcl_run(self, identifier: str) -> dict | None:
        row = self.connection.execute("SELECT * FROM qcl_runs WHERE id=?",
                                      (identifier,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        for key in ("result", "profile", "spent"):
            data[key] = json.loads(data[key])
        return data

    def refute(self, plan_hash: str, node_id: str, report_id: str) -> None:
        self.connection.execute("INSERT OR REPLACE INTO refuted_results "
                                "SELECT id, ? FROM results WHERE plan_hash=? AND node_id=?",
                                (report_id, plan_hash, node_id))
        self.connection.commit()

    def _refutation(self, identifier):
        row = self.connection.execute("SELECT report_id FROM refuted_results WHERE result_id=?",
                                      (identifier,)).fetchone()
        return {"relation_status": "refuted", "falsification_report": row[0]} if row else {}

    def record_critic(self, result_id: str, verdict: dict) -> None:
        if self.headline(result_id) is None:
            raise ValueError("Critic verdict requires a recorded result")
        self.connection.execute("UPDATE report_records SET critic=? WHERE result_id=?",
                                (canonical_json(verdict), result_id))
        self.connection.commit()

    def headline(self, result_id: str) -> dict | None:
        row = self.connection.execute("SELECT headline, critic FROM report_records WHERE result_id=?",
                                      (result_id,)).fetchone()
        if row is None:
            return None
        return {"headline": {**json.loads(row[0]), **self._refutation(result_id)},
                "critic": json.loads(row[1]) if row[1] else None}

    def seed_replay(self, screen: str, role: str, records: list[dict]) -> None:
        self.connection.executemany("INSERT OR REPLACE INTO replay_records VALUES (?,?,?,?)",
                                    [(screen, role, row["id"], canonical_json(row))
                                     for row in records])
        self.connection.commit()

    def replay_records(self, screen: str, role: str) -> list[dict]:
        return [json.loads(row[0]) for row in self.connection.execute(
            "SELECT record FROM replay_records WHERE screen=? AND role=? ORDER BY id",
            (screen, role))]

    def add_cost(self, template: str, feature: str, size: float,
                 dimension: str, measured: float) -> None:
        if size <= 0 or measured <= 0:
            raise ValueError("Cost observations must be positive")
        self.connection.execute("INSERT INTO cost_observations VALUES (?,?,?,?,?)",
                                (template, feature, size, dimension, measured))
        self.connection.commit()

    def cost_observations(self, template: str, feature: str, dimension: str) -> list[tuple]:
        return [tuple(row) for row in self.connection.execute(
            "SELECT size, measured FROM cost_observations WHERE template=? AND feature=? "
            "AND dimension=?", (template, feature, dimension))]

    def add(self, *, task_id: str, record_hash: str, plan_hash: str, node_id: str,
            operator: str, estimate: Estimate | SpectralEstimate, headline: dict | None = None) -> str:
        spectral = isinstance(estimate, SpectralEstimate)
        curve = isinstance(estimate, EnergyCurve)
        data = (task_id, record_hash, plan_hash, node_id, operator,
                canonical_json({"input_hashes": estimate.provenance.input_hashes,
                                "omega": estimate.omega, "spectral": estimate.spectral}
                               if spectral else dict(estimate.context.keys)), estimate.relation.value,
                estimate.diagnostics["sum_rule"] if spectral else
                len(estimate.values_hartree) if curve else estimate.value.value,
                "dimensionless" if spectral else "points" if curve else estimate.value.unit,
                canonical_json({} if spectral or curve else estimate.uncertainty),
                canonical_json([{"status": "diagnostic_only"}] if spectral or curve else
                               [item.model_dump(mode="json") for item in estimate.evidence]),
                canonical_json(estimate.cost), canonical_json(estimate.provenance))
        identifier = content_hash((data, headline)) if headline is not None else content_hash(data)
        self.connection.execute("INSERT OR IGNORE INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                (identifier, *data))
        if headline is not None:
            self.connection.execute("INSERT OR IGNORE INTO report_records VALUES (?,?,NULL)",
                (identifier, canonical_json({**headline, "result_id": identifier})))
        self.connection.commit()
        return identifier

    def get(self, identifier: str) -> dict | None:
        row = self.connection.execute("SELECT * FROM results WHERE id=?", (identifier,)).fetchone()
        if row:
            return self._decode(row)
        qcl_row = self.qcl_run(identifier)
        return {"kind": "qcl", **qcl_row} if qcl_row is not None else None

    def query(self, **filters: str) -> list[dict]:
        legacy_columns = {"task_id", "record_hash", "plan_hash", "node_id",
                          "operator", "relation"}
        qcl_columns = {"record_hash", "plan_hash", "target_kind"}
        allowed = legacy_columns | qcl_columns
        if set(filters) - allowed:
            raise ValueError("Unsupported results filter")
        results = []
        if set(filters) <= legacy_columns:
            where = " AND ".join(f"{key}=?" for key in filters)
            sql = "SELECT * FROM results" + (f" WHERE {where}" if where else "") + " ORDER BY rowid"
            results.extend(self._decode(row) for row in self.connection.execute(
                sql, tuple(filters.values())))
        if set(filters) <= qcl_columns:
            where = " AND ".join(f"{key}=?" for key in filters)
            sql = "SELECT * FROM qcl_runs" + (f" WHERE {where}" if where else "") + " ORDER BY rowid"
            for row in self.connection.execute(sql, tuple(filters.values())):
                data = dict(row)
                for key in ("result", "profile", "spent"):
                    data[key] = json.loads(data[key])
                results.append({"kind": "qcl", **data})
        return results

    def _decode(self, row: sqlite3.Row) -> dict:
        data = dict(row)
        for key in ("target_identity", "uncertainty", "evidence", "cost", "provenance"):
            data[key] = json.loads(data[key])
        return {**data, **self._refutation(data["id"])}
