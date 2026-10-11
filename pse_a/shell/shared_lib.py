"""SHARED_LIB -- orkestrirajuca inteligencija ([PSE §6], [PSE-A] Sec 8.4).

NIJE guardian/shared_lib.py klasa Hub. Hub je sistem za potpisivanje i
revokaciju kapabiliteta; ovo je skladiste znanja, artefakata i dosijea.

Invarijante koje su UGRADJENE, ne proverene naknadno:
  AZ-5  zapecaceno je nepromenljivo -- ne postoji metod koji menja artefakt
  AZ-6  beleske su append-only, atribuirane, vremenski zigosane, usidrene za hes
  AZ-4  artefakt bez (model_hash, config_hash, seeds) nije rezultat
  I-9   Welford je IZVEDEN iz per-chunk trojki, ne azuriran u mestu

Nema update_artifact, delete_run ni edit_note. Ispravka je NOV run sa
supersedes=<run_id>; oba ostaju vidljiva i dosije prikazuje lozu.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contribution import GateResult, contribution_gate

SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    model_hash TEXT PRIMARY KEY, display_name TEXT, architecture TEXT,
    canonical_map TEXT, created REAL);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, model_hash TEXT, kind TEXT, config_hash TEXT,
    seeds TEXT, started REAL, sealed REAL, supersedes TEXT,
    budget_spent REAL, state TEXT);
CREATE TABLE IF NOT EXISTS artifacts (
    run_id TEXT, name TEXT, content_hash TEXT, n_bytes INTEGER, media TEXT,
    PRIMARY KEY (run_id, name));
CREATE TABLE IF NOT EXISTS notes (
    note_id TEXT PRIMARY KEY, run_id TEXT, author TEXT, anchor_hash TEXT,
    text TEXT, created REAL);
CREATE TABLE IF NOT EXISTS priors (
    key TEXT PRIMARY KEY, n INTEGER, mean REAL, m2 REAL, updated REAL);
CREATE TABLE IF NOT EXISTS registry (
    entry_id TEXT PRIMARY KEY, run_id TEXT, kind TEXT, prior_key TEXT,
    value REAL, z REAL, created REAL);
CREATE TABLE IF NOT EXISTS progress (
    run_id TEXT, step INTEGER, total INTEGER, theta REAL, label TEXT, t REAL,
    PRIMARY KEY (run_id, step));
CREATE TABLE IF NOT EXISTS deaths (
    run_id TEXT PRIMARY KEY, reason TEXT, registry INTEGER, t REAL);
CREATE TABLE IF NOT EXISTS inbox (
    item_id TEXT PRIMARY KEY, kind TEXT, model_hash TEXT, run_kind TEXT,
    text TEXT, t REAL, seen INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS idx_runs_model ON runs(model_hash);
CREATE INDEX IF NOT EXISTS idx_reg_run ON registry(run_id);
"""


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def config_hash(cfg: dict) -> str:
    return sha256_bytes(json.dumps(cfg, sort_keys=True, default=str).encode())


@dataclass
class Welford:
    """I-9: trojka (n, mean, M2) koja se spaja PAIRWISE (Chan).

    Ovo je razlog zasto baseline moze biti izveden a ne primaran: svaki chunk
    nosi svoju trojku, globalni baseline je njihov merge nad done-skupom iz
    manifesta, pa je uvek tacno konzistentan sa onim sto je na disku.
    """
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def push(self, x: float) -> "Welford":
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)
        return self

    @staticmethod
    def merge(a: "Welford", b: "Welford") -> "Welford":
        if a.n == 0:
            return Welford(b.n, b.mean, b.m2)
        if b.n == 0:
            return Welford(a.n, a.mean, a.m2)
        n = a.n + b.n
        d = b.mean - a.mean
        return Welford(n, a.mean + d * b.n / n, a.m2 + b.m2 + d * d * a.n * b.n / n)

    @property
    def var(self) -> float:
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.var)


class SealRefused(Exception):
    """Tri sloja verifikacije nisu prosla ([PSE-A] 11.4)."""


@dataclass
class RunRecord:
    run_id: str
    model_hash: str
    kind: str
    config_hash: str
    seeds: dict
    started: float
    state: str = "working"            # working -> sealed; nikad nazad
    sealed: float | None = None
    supersedes: str | None = None
    artifacts: dict[str, str] = field(default_factory=dict)   # name -> content_hash


class SharedLib:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "artifacts").mkdir(parents=True, exist_ok=True)
        self._path = self.root / "shared_lib.sqlite"
        self._tls = threading.local()
        self.db.executescript(SCHEMA)
        self.db.commit()

    @property
    def db(self) -> sqlite3.Connection:
        """Jedna konekcija PO NITI. Web sloj izvrsava endpoint-e u threadpool-u,
        a sqlite konekcija ne sme preci granicu niti. WAL dozvoljava vise
        istovremenih citalaca -- i dok misija u drugom procesu pise."""
        c = getattr(self._tls, "conn", None)
        if c is None:
            c = sqlite3.connect(self._path, timeout=30)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA busy_timeout=30000")
            self._tls.conn = c
        return c

    # ------------------------------------------------------------- artefakti
    def _blob_path(self, h: str) -> Path:
        d = self.root / "artifacts" / h[:2]
        d.mkdir(parents=True, exist_ok=True)
        return d / h

    def put_blob(self, data: bytes) -> str:
        """Content-addressed upis. Isti sadrzaj -> isti hes -> jedan fajl.

        tmp -> fsync -> rename -> fsync(dir): pad ne sme ostaviti krnj artefakt
        koji hes-verifikacija posle proglasi ispravnim (I-10).
        """
        import os
        h = sha256_bytes(data)
        p = self._blob_path(h)
        if p.exists():
            return h
        tmp = p.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
        fd = os.open(p.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return h

    def get_blob(self, h: str) -> bytes:
        p = self._blob_path(h)
        data = p.read_bytes()
        if sha256_bytes(data) != h:
            raise IOError(f"artefakt {h[:12]} ne prolazi hes-verifikaciju")
        return data

    # ------------------------------------------------------------- modeli
    def register_model(self, model_hash: str, display_name: str,
                       architecture: str, canonical_map: dict) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO models VALUES (?,?,?,?,?)",
            (model_hash, display_name, architecture,
             json.dumps(canonical_map, default=str), time.time()))
        self.db.commit()

    # ------------------------------------------------------------- run-ovi
    def open_run(self, model_hash: str, kind: str, config: dict,
                 seeds: dict | None = None, supersedes: str | None = None) -> RunRecord:
        r = RunRecord(run_id=uuid.uuid4().hex[:16], model_hash=model_hash, kind=kind,
                      config_hash=config_hash(config), seeds=seeds or {},
                      started=time.time(), supersedes=supersedes)
        self.db.execute(
            "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?)",
            (r.run_id, r.model_hash, r.kind, r.config_hash, json.dumps(r.seeds),
             r.started, None, supersedes, 0.0, "working"))
        self.db.commit()
        return r

    def attach(self, r: RunRecord, name: str, data: bytes, media: str = "json") -> str:
        if r.state == "sealed":
            raise SealRefused(f"run {r.run_id} je zapecacen; ispravka je NOV run")
        h = self.put_blob(data)
        r.artifacts[name] = h
        self.db.execute("INSERT OR REPLACE INTO artifacts VALUES (?,?,?,?,?)",
                        (r.run_id, name, h, len(data), media))
        self.db.commit()
        return h

    # ------------------------------------------------------------- pecacenje
    def seal(self, r: RunRecord, mission_outcome: dict, mission_spec: dict,
             theta_series: list[float], registry_entries: int,
             registry_baseline: float | None = None,
             pending_priors: list[tuple[str, float]] | None = None) -> GateResult:
        """Tro-slojna verifikacija zavrsetka misije ([PSE-A] 11.4).

          sloj 1: ishod odgovara specifikaciji misije
          sloj 2: artefakti prolaze hes-verifikaciju
          sloj 3: cetvorouslovna kapija doprinosa prihvata

        Pre ovog trenutka run je RADNO STANJE i ne sme se pojaviti u dosijeu.
        """
        missing = [k for k in mission_spec if k not in mission_outcome]
        if missing:
            raise SealRefused(f"sloj 1: ishod ne pokriva specifikaciju, fali {missing}")
        for k, want in mission_spec.items():
            if want is not None and mission_outcome[k] != want:
                raise SealRefused(
                    f"sloj 1: '{k}' = {mission_outcome[k]!r}, misija trazi {want!r}")

        for name, h in r.artifacts.items():
            try:
                self.get_blob(h)
            except Exception as e:
                raise SealRefused(f"sloj 2: artefakt '{name}' ne verifikuje: {e}")

        gate = contribution_gate(theta_series, registry_entries, registry_baseline)
        if not gate.passed:
            raise SealRefused("sloj 3: kapija doprinosa odbija — "
                              + ", ".join(gate.failed_conditions()))

        r.state = "sealed"
        r.sealed = time.time()
        self.db.execute("UPDATE runs SET state='sealed', sealed=? WHERE run_id=?",
                        (r.sealed, r.run_id))
        # tek sada doprinos ulazi u znanje -- i sve u istoj transakciji
        if pending_priors:
            self._commit_priors(pending_priors)
        self.db.commit()
        return gate

    # ------------------------------------------------------------- napredak
    def progress(self, run_id: str, step: int, total: int, theta: float, label: str) -> None:
        """Radno stanje za status traku. NIJE rezultat i nikad ne ulazi u dosije."""
        self.db.execute("INSERT OR REPLACE INTO progress VALUES (?,?,?,?,?,?)",
                        (run_id, step, total, theta, label, time.time()))
        self.db.commit()

    def active_runs(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT r.run_id, r.kind, m.display_name, r.started FROM runs r "
            "LEFT JOIN models m ON m.model_hash = r.model_hash "
            "WHERE r.state='working' AND r.started > ? ORDER BY r.started DESC",
            (time.time() - 12 * 3600,)).fetchall()
        out = []
        for run_id, kind, name, started in rows:
            pts = self.db.execute("SELECT step, total, theta, label, t FROM progress "
                                  "WHERE run_id=? ORDER BY step", (run_id,)).fetchall()
            if not pts:
                continue
            last = pts[-1]
            out.append({"run_id": run_id, "kind": kind, "model": name, "step": last[0],
                        "total": last[1], "theta": [p[2] for p in pts][-200:],
                        "label": last[3], "age_s": time.time() - last[4]})
        return out

    # ------------------------------------------------------------- smrt i inbox
    def record_death(self, run_id: str, reason: str, registry: int) -> None:
        """Mission Failure = Death ([PSE §13]): odbijen run se BELEZI kao mrtav,
        ne ostaje vecno "radno stanje". I dalje ne ulazi u dosije (11.4)."""
        self.db.execute("UPDATE runs SET state='died' WHERE run_id=? AND state='working'",
                        (run_id,))
        self.db.execute("INSERT OR REPLACE INTO deaths VALUES (?,?,?,?)",
                        (run_id, reason, registry, time.time()))
        self.db.commit()

    def post_inbox(self, kind: str, model_hash: str, run_kind: str, text: str) -> str:
        """Odluke koje sistem ne sme da donese sam idu ovde kao posta, ne kao
        iskacuci prozor ([SPEC] 10.3). Isti tekst se ne salje dvaput dok je necitan."""
        dup = self.db.execute("SELECT item_id FROM inbox WHERE model_hash=? AND run_kind=? "
                              "AND text=? AND seen=0", (model_hash, run_kind, text)).fetchone()
        if dup:
            return dup[0]
        iid = uuid.uuid4().hex[:16]
        self.db.execute("INSERT INTO inbox VALUES (?,?,?,?,?,?,0)",
                        (iid, kind, model_hash, run_kind, text, time.time()))
        self.db.commit()
        return iid

    def inbox(self, unseen_only: bool = False) -> list[dict]:
        q = "SELECT item_id, kind, model_hash, run_kind, text, t, seen FROM inbox"
        q += " WHERE seen=0" if unseen_only else ""
        q += " ORDER BY t DESC LIMIT 100"
        return [dict(zip(("item_id", "kind", "model_hash", "run_kind", "text", "t", "seen"), r))
                for r in self.db.execute(q).fetchall()]

    def mark_seen(self, item_id: str) -> None:
        """Procitano nije isto sto i obrisano: stavka ostaje, samo nije vise nova."""
        self.db.execute("UPDATE inbox SET seen=1 WHERE item_id=?", (item_id,))
        self.db.commit()

    def run_history(self, model_hash: str, kind: str, last: int = 10) -> list[dict]:
        rows = self.db.execute(
            "SELECT run_id, state, started FROM runs WHERE model_hash=? AND kind=? "
            "AND state IN ('sealed','died') ORDER BY started DESC LIMIT ?",
            (model_hash, kind, last)).fetchall()
        return [{"run_id": r[0], "state": r[1], "started": r[2],
                 "registry": self.registry_count(r[0])} for r in rows][::-1]

    # ------------------------------------------------------------- beleske
    def add_note(self, run_id: str, author: str, anchor_hash: str, text: str) -> str:
        """AZ-6. Nema update_note ni delete_note -- i nece ih biti."""
        row = self.db.execute(
            "SELECT 1 FROM artifacts WHERE run_id=? AND content_hash=?",
            (run_id, anchor_hash)).fetchone()
        if not row:
            raise ValueError("beleska mora biti usidrena za hes artefakta OVOG run-a")
        if self.db.execute("SELECT state FROM runs WHERE run_id=?",
                           (run_id,)).fetchone()[0] != "sealed":
            raise ValueError("beleske se kace samo na zapecacen run")
        nid = uuid.uuid4().hex[:16]
        self.db.execute("INSERT INTO notes VALUES (?,?,?,?,?,?)",
                        (nid, run_id, author, anchor_hash, text, time.time()))
        self.db.commit()
        return nid

    # ------------------------------------------------------------- priori
    def prior(self, key: str) -> Welford:
        row = self.db.execute("SELECT n, mean, m2 FROM priors WHERE key=?",
                              (key,)).fetchone()
        return Welford(*row) if row else Welford()

    def test_prior(self, key: str, value: float, z_thresh: float = 2.0,
                   local: "Welford | None" = None) -> tuple[bool, float]:
        """§3.2: da li bi ova vrednost POMERILA ocekivanje? SAMO CITA.

        Prior se NE azurira ovde. [PSE §6]: "Before any unit's contribution is
        incorporated into the SHARED_LIB, a four-condition ... check is
        performed". Ranija verzija je azurirala odmah, pa je odbijen run vec bio
        upisao svoje vrednosti -- ponovljeni odbijeni run-ovi bi vestacki suzili
        varijansu priora. Azuriranje ide kroz seal(), posle sva tri sloja.

        local: ono sto je jedinica sama naucila TOKOM svog zivota (nije jos
        predato). Bez toga je u prvom run-u na novom modelu svaka izmerena
        vrednost "nova", pa je rast registra opet srazmeran compute-u -- tacno
        zamka iz §3.2. Jedinica uci od sebe; SHARED_LIB uci od jedinice tek
        posle kapije.
        """
        w = self.prior(key)
        if local is not None:
            w = Welford.merge(w, local)
        if w.n < 3:
            return True, float("inf")
        sd = w.std
        z = abs(value - w.mean) / sd if sd > 0 else float("inf")
        return z > z_thresh, z

    def _commit_priors(self, pending: list[tuple[str, float]]) -> None:
        for key, value in pending:
            w2 = Welford.merge(self.prior(key), Welford().push(value))
            self.db.execute(
                "INSERT INTO priors VALUES (?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET "
                "n=excluded.n, mean=excluded.mean, m2=excluded.m2, updated=excluded.updated",
                (key, w2.n, w2.mean, w2.m2, time.time()))

    def test_and_update_prior(self, key: str, value: float, z_thresh: float = 2.0
                              ) -> tuple[bool, float]:
        """Test + trenutni upis. SAMO za vec prihvacen doprinos (npr. testovi)."""
        r = self.test_prior(key, value, z_thresh)
        self._commit_priors([(key, value)])
        self.db.commit()
        return r

    def record_registry(self, run_id: str, kind: str, prior_key: str,
                        value: float, z: float) -> None:
        self.db.execute("INSERT INTO registry VALUES (?,?,?,?,?,?,?)",
                        (uuid.uuid4().hex[:16], run_id, kind, prior_key,
                         value, z, time.time()))
        self.db.commit()

    def registry_count(self, run_id: str) -> int:
        return self.db.execute("SELECT COUNT(*) FROM registry WHERE run_id=?",
                               (run_id,)).fetchone()[0]

    def registry_baseline(self, model_hash: str, kind: str, last: int = 5) -> float | None:
        rows = self.db.execute(
            "SELECT r.run_id FROM runs r WHERE r.model_hash=? AND r.kind=? "
            "AND r.state='sealed' ORDER BY r.sealed DESC LIMIT ?",
            (model_hash, kind, last)).fetchall()
        if not rows:
            return None
        counts = [self.registry_count(x[0]) for x in rows]
        return sum(counts) / len(counts)

    # ------------------------------------------------------------- dosije
    def dossier(self, model_hash: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT run_id, kind, config_hash, started, sealed, supersedes, state "
            "FROM runs WHERE model_hash=? AND state='sealed' ORDER BY sealed",
            (model_hash,)).fetchall()
        out = []
        for run_id, kind, ch, st, se, sup, state in rows:
            arts = self.db.execute(
                "SELECT name, content_hash, n_bytes FROM artifacts WHERE run_id=?",
                (run_id,)).fetchall()
            notes = self.db.execute(
                "SELECT author, text, created FROM notes WHERE run_id=? ORDER BY created",
                (run_id,)).fetchall()
            out.append({"run_id": run_id, "kind": kind, "config_hash": ch,
                        "started": st, "sealed": se, "supersedes": sup,
                        "artifacts": {a[0]: a[1] for a in arts},
                        "n_registry": self.registry_count(run_id),
                        "notes": [{"author": n[0], "text": n[1]} for n in notes]})
        return out
