"""
store.py — Capa de persistencia de Fluxia (jobs, resultados, proyectos).

Reemplaza los diccionarios en memoria de server.py (JOBS, RESULTS) por:
    - SQLite (stdlib) para metadatos: jobs, results (meta_json), projects.
    - Archivos .npz en disco para los arrays grandes de "snapshots"
      (reutiliza exporters.to_npz, ya usado por la función de exportación).

Resolución de la carpeta de datos, consciente de Hugging Face Spaces:
    - Si existe y es escribible /data (addon de pago "Persistent Storage"
      de HF Spaces, o FLUXIA_DATA_DIR apuntando ahí), se usa — sobrevive
      a reinicios del Space.
    - Si no, cae a pinn-maxwell/data/ dentro del contenedor — sobrevive a
      caídas del proceso pero no a un rebuild completo del Space sin el
      addon activado.
"""
from __future__ import annotations

import os
import json
import time
import sqlite3
import threading
from collections import OrderedDict

import numpy as np

import exporters

_LOCK = threading.Lock()
_CONN: sqlite3.Connection | None = None
_DATA_DIR: str | None = None

_LRU_CAP = 8
_lru: "OrderedDict[str, dict]" = OrderedDict()
_lru_lock = threading.Lock()


# ═════════════════════════════════════════════════════════════════════════════
# Rutas
# ═════════════════════════════════════════════════════════════════════════════
def data_dir() -> str:
    global _DATA_DIR
    if _DATA_DIR:
        return _DATA_DIR
    for cand in [os.environ.get("FLUXIA_DATA_DIR"), "/data"]:
        if cand and os.path.isdir(cand) and os.access(cand, os.W_OK):
            _DATA_DIR = cand
            return _DATA_DIR
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(base, exist_ok=True)
    _DATA_DIR = base
    return _DATA_DIR


def _results_dir() -> str:
    d = os.path.join(data_dir(), "results")
    os.makedirs(d, exist_ok=True)
    return d


def _db_path() -> str:
    return os.path.join(data_dir(), "app.db")


def _npz_path(rid: str) -> str:
    return os.path.join(_results_dir(), f"{rid}.npz")


# ═════════════════════════════════════════════════════════════════════════════
# Conexión / esquema
# ═════════════════════════════════════════════════════════════════════════════
def _conn() -> sqlite3.Connection:
    global _CONN
    if _CONN is None:
        _CONN = sqlite3.connect(_db_path(), check_same_thread=False)
        _CONN.row_factory = sqlite3.Row
    return _CONN


def init_db():
    with _LOCK:
        c = _conn()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                id          TEXT PRIMARY KEY,
                type        TEXT NOT NULL,
                status      TEXT NOT NULL,
                cancel      INTEGER NOT NULL DEFAULT 0,
                result_id   TEXT,
                error       TEXT,
                created     REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS results (
                id          TEXT PRIMARY KEY,
                mode        TEXT NOT NULL,
                created     REAL NOT NULL,
                meta_json   TEXT NOT NULL,
                npz_path    TEXT
            );
            CREATE TABLE IF NOT EXISTS projects (
                id           TEXT PRIMARY KEY,
                name         TEXT NOT NULL,
                created      REAL NOT NULL,
                updated      REAL NOT NULL,
                project_json TEXT NOT NULL
            );
        """)
        c.commit()


def reconcile_startup():
    """Marca como 'interrupted' cualquier job que haya quedado a mitad de
    ejecución por una caída/reinicio previo del proceso."""
    with _LOCK:
        c = _conn()
        c.execute(
            "UPDATE jobs SET status='interrupted', "
            "error='Servidor reiniciado durante la ejecución' "
            "WHERE status IN ('queued','running')"
        )
        c.commit()


# ═════════════════════════════════════════════════════════════════════════════
# Jobs
# ═════════════════════════════════════════════════════════════════════════════
def job_create(job_id: str, jtype: str):
    with _LOCK:
        c = _conn()
        c.execute(
            "INSERT INTO jobs (id, type, status, cancel, result_id, error, created) "
            "VALUES (?, ?, 'queued', 0, NULL, NULL, ?)",
            (job_id, jtype, time.time()),
        )
        c.commit()


def job_set_status(job_id: str, status: str, result_id: str | None = None,
                    error: str | None = None):
    with _LOCK:
        c = _conn()
        c.execute(
            "UPDATE jobs SET status=?, result_id=COALESCE(?, result_id), "
            "error=COALESCE(?, error) WHERE id=?",
            (status, result_id, error, job_id),
        )
        c.commit()


def job_set_cancel(job_id: str):
    with _LOCK:
        c = _conn()
        c.execute("UPDATE jobs SET cancel=1 WHERE id=?", (job_id,))
        c.commit()


def job_get(job_id: str) -> dict | None:
    with _LOCK:
        row = _conn().execute(
            "SELECT id, type, status, cancel, result_id, error FROM jobs WHERE id=?",
            (job_id,),
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["cancel"] = bool(d["cancel"])
    return d


# ═════════════════════════════════════════════════════════════════════════════
# Resultados
# ═════════════════════════════════════════════════════════════════════════════
def _lru_put(rid: str, full: dict):
    with _lru_lock:
        _lru[rid] = full
        _lru.move_to_end(rid)
        while len(_lru) > _LRU_CAP:
            _lru.popitem(last=False)


def _lru_get(rid: str) -> dict | None:
    with _lru_lock:
        v = _lru.get(rid)
        if v is not None:
            _lru.move_to_end(rid)
        return v


def _lru_drop(rid: str):
    with _lru_lock:
        _lru.pop(rid, None)


def result_store(res: dict) -> str:
    import uuid
    rid = uuid.uuid4().hex[:12]
    snapshots = res.pop("snapshots", None)

    npz_rel = None
    if snapshots:
        data = exporters.to_npz(snapshots)
        with open(_npz_path(rid), "wb") as f:
            f.write(data)
        npz_rel = f"{rid}.npz"
        res["snapshot_fields"] = list(snapshots.keys())
    else:
        res["snapshot_fields"] = []

    meta_json = json.dumps(res)
    with _LOCK:
        c = _conn()
        c.execute(
            "INSERT INTO results (id, mode, created, meta_json, npz_path) "
            "VALUES (?, ?, ?, ?, ?)",
            (rid, res.get("mode", "unknown"), time.time(), meta_json, npz_rel),
        )
        c.commit()

    # restaurar snapshots en el dict que el llamador pueda seguir usando
    if snapshots is not None:
        res["snapshots"] = snapshots

    retention_sweep()
    return rid


def result_get_meta(rid: str) -> dict | None:
    with _LOCK:
        row = _conn().execute(
            "SELECT meta_json FROM results WHERE id=?", (rid,)
        ).fetchone()
    if row is None:
        return None
    meta = json.loads(row["meta_json"])
    meta["fields"] = meta.pop("snapshot_fields", [])
    return meta


def result_get_frames(rid: str, field: str) -> np.ndarray | None:
    cached = _lru_get(rid)
    if cached is not None:
        snaps = cached.get("snapshots", {})
        return snaps.get(field)

    with _LOCK:
        row = _conn().execute(
            "SELECT npz_path FROM results WHERE id=?", (rid,)
        ).fetchone()
    if row is None or not row["npz_path"]:
        return None
    path = os.path.join(_results_dir(), row["npz_path"])
    try:
        with np.load(path) as npz:
            return np.asarray(npz[field]) if field in npz else None
    except (FileNotFoundError, OSError, KeyError):
        return None


def result_get_full(rid: str) -> dict | None:
    cached = _lru_get(rid)
    if cached is not None:
        return cached

    meta = result_get_meta(rid)
    if meta is None:
        return None
    fields = meta.pop("fields", [])

    with _LOCK:
        row = _conn().execute(
            "SELECT npz_path FROM results WHERE id=?", (rid,)
        ).fetchone()

    snapshots = {}
    if row and row["npz_path"]:
        path = os.path.join(_results_dir(), row["npz_path"])
        try:
            with np.load(path) as npz:
                snapshots = {k: np.asarray(npz[k]) for k in fields if k in npz}
        except (FileNotFoundError, OSError):
            snapshots = {}

    full = dict(meta)
    full["snapshots"] = snapshots
    _lru_put(rid, full)
    return full


def result_list() -> list[dict]:
    with _LOCK:
        rows = _conn().execute(
            "SELECT id, mode, meta_json FROM results ORDER BY created DESC"
        ).fetchall()
    out = []
    for r in rows:
        meta = json.loads(r["meta_json"])
        out.append({
            "id": r["id"],
            "mode": r["mode"],
            "grid": meta.get("grid"),
            "n_frames": len(meta.get("times", [])),
        })
    return out


def result_delete(rid: str):
    with _LOCK:
        c = _conn()
        row = c.execute("SELECT npz_path FROM results WHERE id=?", (rid,)).fetchone()
        c.execute("DELETE FROM results WHERE id=?", (rid,))
        c.commit()
    _lru_drop(rid)
    if row and row["npz_path"]:
        path = os.path.join(_results_dir(), row["npz_path"])
        try:
            os.remove(path)
        except OSError:
            pass


def retention_sweep(max_results: int | None = None):
    if max_results is None:
        max_results = int(os.environ.get("FLUXIA_MAX_RESULTS", 200))
    with _LOCK:
        n = _conn().execute("SELECT COUNT(*) AS n FROM results").fetchone()["n"]
    if n <= max_results:
        return
    with _LOCK:
        rows = _conn().execute(
            "SELECT id FROM results ORDER BY created ASC LIMIT ?",
            (n - max_results,),
        ).fetchall()
    for r in rows:
        result_delete(r["id"])


# ═════════════════════════════════════════════════════════════════════════════
# Proyectos (helpers internos — sin endpoint HTTP todavía, ver mejora #4)
# ═════════════════════════════════════════════════════════════════════════════
def save_project(project: dict, pid: str | None = None) -> str:
    import uuid
    now = time.time()
    name = project.get("name", "Proyecto")
    project_json = json.dumps(project)
    with _LOCK:
        c = _conn()
        if pid:
            c.execute(
                "UPDATE projects SET name=?, updated=?, project_json=? WHERE id=?",
                (name, now, project_json, pid),
            )
            if c.total_changes:
                c.commit()
                return pid
        pid = pid or uuid.uuid4().hex[:12]
        c.execute(
            "INSERT INTO projects (id, name, created, updated, project_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (pid, name, now, now, project_json),
        )
        c.commit()
    return pid


def load_project(pid: str) -> dict | None:
    with _LOCK:
        row = _conn().execute(
            "SELECT project_json FROM projects WHERE id=?", (pid,)
        ).fetchone()
    return json.loads(row["project_json"]) if row else None
