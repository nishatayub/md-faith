"""SQLite persistence for experiment runs: explanations, extracted claims and verdicts.

Plain `sqlite3`, no ORM. One connection guarded by a lock so the API server can share it across threads.
API keys are never stored; only the model *name* is recorded.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterable

from .claims import Claim, Verdict
from .metrics import counts

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at REAL NOT NULL,
    is_demo INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'running',
    config_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS explanations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    task_id TEXT NOT NULL,
    artifact_kind TEXT NOT NULL,
    condition TEXT NOT NULL,
    model TEXT NOT NULL,
    seed INTEGER NOT NULL DEFAULT 0,
    text TEXT NOT NULL,
    n_tool_calls INTEGER NOT NULL DEFAULT 0,
    n_model_calls INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    explanation_id INTEGER NOT NULL REFERENCES explanations(id) ON DELETE CASCADE,
    claim_id TEXT NOT NULL,
    text TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    label TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_expl_run ON explanations(run_id);
CREATE INDEX IF NOT EXISTS idx_claims_expl ON claims(explanation_id);
"""


class Store:
    def __init__(self, path: str = ":memory:"):
        self.path = path
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        with self._lock:
            self._db.executescript(SCHEMA)

    # ---- runs --------------------------------------------------------------------------------
    def create_run(self, name: str, config: dict | None = None, is_demo: bool = False) -> str:
        rid = uuid.uuid4().hex[:12]
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO runs (id, name, created_at, is_demo, config_json) VALUES (?,?,?,?,?)",
                (rid, name, time.time(), int(is_demo), json.dumps(config or {})),
            )
        return rid

    def set_status(self, run_id: str, status: str) -> None:
        self.finish_run(run_id, status)

    def finish_run(self, run_id: str, status: str = "done") -> None:
        with self._lock, self._db:
            self._db.execute("UPDATE runs SET status=? WHERE id=?", (status, run_id))

    def delete_run(self, run_id: str) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM runs WHERE id=?", (run_id,))

    def list_runs(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT r.*, (SELECT COUNT(*) FROM explanations e WHERE e.run_id=r.id) AS n_explanations "
                "FROM runs r ORDER BY created_at DESC"
            ).fetchall()
        return [self._run_dict(r) for r in rows]

    def get_run(self, run_id: str) -> dict | None:
        with self._lock:
            r = self._db.execute(
                "SELECT r.*, (SELECT COUNT(*) FROM explanations e WHERE e.run_id=r.id) AS n_explanations "
                "FROM runs r WHERE id=?",
                (run_id,),
            ).fetchone()
        return self._run_dict(r) if r else None

    @staticmethod
    def _run_dict(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["is_demo"] = bool(d["is_demo"])
        d["config"] = json.loads(d.pop("config_json"))
        return d

    # ---- explanations, claims, verdicts -----------------------------------------------------
    def add_explanation(
        self,
        run_id: str,
        task_id: str,
        artifact_kind: str,
        condition: str,
        model: str,
        text: str,
        claims: Iterable[Claim],
        verdicts: Iterable[Verdict],
        seed: int = 0,
        n_tool_calls: int = 0,
        n_model_calls: int = 1,
    ) -> int:
        vmap = {v.claim_id: v for v in verdicts}
        with self._lock, self._db:
            cur = self._db.execute(
                "INSERT INTO explanations (run_id, task_id, artifact_kind, condition, model, seed, text, "
                "n_tool_calls, n_model_calls, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    task_id,
                    artifact_kind,
                    condition,
                    model,
                    seed,
                    text,
                    n_tool_calls,
                    n_model_calls,
                    time.time(),
                ),
            )
            eid = cur.lastrowid
            for c in claims:
                v = vmap.get(c.id)
                self._db.execute(
                    "INSERT INTO claims (explanation_id, claim_id, text, kind, payload_json, label, detail) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (
                        eid,
                        c.id,
                        c.text,
                        c.kind.value,
                        json.dumps(c.payload),
                        v.label.value if v else "unverifiable",
                        v.detail if v else "",
                    ),
                )
        return eid

    def explanations(self, run_id: str, with_claims: bool = False) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM explanations WHERE run_id=? ORDER BY id", (run_id,)).fetchall()
            out = [dict(r) for r in rows]
            if with_claims:
                for e in out:
                    e["claims"] = [
                        {**dict(c), "payload": json.loads(c["payload_json"])}
                        for c in self._db.execute("SELECT * FROM claims WHERE explanation_id=? ORDER BY id", (e["id"],))
                    ]
                    for c in e["claims"]:
                        c.pop("payload_json", None)
        return out

    def per_explanation_counts(self, run_id: str) -> list[dict]:
        """One row per explanation with its claim-label counts (input to aggregation and bootstrap)."""
        out = []
        with self._lock:
            for e in self._db.execute("SELECT * FROM explanations WHERE run_id=? ORDER BY id", (run_id,)).fetchall():
                labels = [
                    r["label"] for r in self._db.execute("SELECT label FROM claims WHERE explanation_id=?", (e["id"],))
                ]
                c = counts([_L(label) for label in labels])
                out.append(
                    {
                        "explanation_id": e["id"],
                        "task_id": e["task_id"],
                        "artifact_kind": e["artifact_kind"],
                        "condition": e["condition"],
                        "model": e["model"],
                        "seed": e["seed"],
                        **c,
                    }
                )
        return out

    def close(self) -> None:
        with self._lock:
            self._db.close()


class _L:
    """Minimal verdict-like wrapper so `metrics.counts` can be reused on stored labels."""

    def __init__(self, label: str):
        from .claims import Label

        self.label = Label(label)
