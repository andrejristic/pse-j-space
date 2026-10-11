"""Harvest aktivacija -- [SPEC] Sec 5, sa ispravkama I-9, I-10, I-11.

Tri kvara koja [SPEC]-ov pseudokod pravi u jednoj liniji (§6 plana, faza 5):
  1. append u deljeni fajl po sloju -> pad usred append-a unistava i ranije
     dobre chunk-ove (krsi AZ-13: "pad gubi najvise jedan chunk")
  2. baferisan upis pa mark_done -> manifest kaze "gotovo", fajl je krnj
  3. Welford u mestu -> baseline i skup chunk-ova se razidju posle pada

Zato:
  - JEDAN FAJL PO (sloj, chunk); nikad append
  - tmp -> fsync(fajl) -> rename -> fsync(direktorijum) -> mark_done
  - uz svaki chunk ide NJEGOVA Welford trojka (n, mean, M2); globalni baseline
    se IZVODI pairwise merge-om nad done-skupom iz manifesta (I-9)
  - chunk_id = hes(corpus_hash, offset), ne brojac: izmena korpusa ne mesa
    stare i nove aktivacije
  - crc32 u zaglavlju; citanje odbija ostecen chunk
  - slobodan prostor se proverava PRE svakog chunk-a: pun disk usred harvest-a
    je cista smrt sa logom, ne krnj fajl
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import struct
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = b"PSEACT1\0"
HEADER = struct.Struct("<8s32sIIIIII")   # magic, model_hash[:32], layer, chunk_no, n, d, dtype, crc
DT_F16, DT_F32 = 1, 2


def chunk_id(corpus_hash: str, offset: int) -> str:
    return hashlib.sha256(f"{corpus_hash}:{offset}".encode()).hexdigest()[:16]


@dataclass
class VWelford:
    """Vektorski Welford: po jedna (mean, M2) po dimenziji, merge po Chan-u."""
    n: int
    mean: np.ndarray
    m2: np.ndarray

    @staticmethod
    def of(x: np.ndarray) -> "VWelford":
        x = x.astype(np.float64)
        mu = x.mean(0)
        return VWelford(x.shape[0], mu, ((x - mu) ** 2).sum(0))

    @staticmethod
    def merge(a: "VWelford", b: "VWelford") -> "VWelford":
        if a.n == 0:
            return b
        if b.n == 0:
            return a
        n = a.n + b.n
        d = b.mean - a.mean
        return VWelford(n, a.mean + d * b.n / n, a.m2 + b.m2 + d * d * a.n * b.n / n)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.m2 / max(self.n - 1, 1))


class HarvestStore:
    def __init__(self, root: str | Path, model_hash: str):
        self.root = Path(root) / "harvest" / model_hash[:16]
        self.root.mkdir(parents=True, exist_ok=True)
        self.model_hash = model_hash
        self.db = sqlite3.connect(self.root / "manifest.sqlite", timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS chunks (layer INTEGER, cid TEXT, "
                        "n INTEGER, path TEXT, crc INTEGER, done REAL, "
                        "PRIMARY KEY (layer, cid))")
        self.db.commit()

    # --------------------------------------------------------------- upis
    def _path(self, layer: int, cid: str) -> Path:
        d = self.root / f"L{layer:03d}"
        d.mkdir(exist_ok=True)
        return d / f"{cid}.act"

    def is_done(self, layer: int, cid: str) -> bool:
        return self.db.execute("SELECT 1 FROM chunks WHERE layer=? AND cid=?",
                               (layer, cid)).fetchone() is not None

    def write(self, layer: int, cid: str, chunk_no: int, acts: np.ndarray,
              min_free_bytes: int = 2 << 30) -> None:
        free = shutil.disk_usage(self.root).free
        if free < min_free_bytes + acts.nbytes * 2:
            raise OSError(f"premalo prostora ({free/1e9:.1f} GB) -- harvest staje cisto, "
                          f"bez krnjeg chunk-a")
        a16 = np.ascontiguousarray(acts, dtype=np.float16)
        w = VWelford.of(acts)
        body = (a16.tobytes() + np.asarray([w.n], np.int64).tobytes()
                + w.mean.astype(np.float64).tobytes() + w.m2.astype(np.float64).tobytes())
        crc = zlib.crc32(body) & 0xFFFFFFFF
        head = HEADER.pack(MAGIC, self.model_hash[:32].encode(), layer, chunk_no,
                           a16.shape[0], a16.shape[1], DT_F16, crc)
        p = self._path(layer, cid)
        tmp = p.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            f.write(head)
            f.write(body)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)                                   # atomicno na istom fs
        fd = os.open(p.parent, os.O_RDONLY)                  # ovaj svi zaborave
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        self.db.execute("INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?)",
                        (layer, cid, int(a16.shape[0]), str(p), crc, time.time()))
        self.db.commit()                                     # TEK SADA je "gotovo"

    # --------------------------------------------------------------- citanje
    def read(self, layer: int, cid: str) -> tuple[np.ndarray, VWelford]:
        p = self._path(layer, cid)
        raw = p.read_bytes()
        magic, mh, L, _, n, d, dt, crc = HEADER.unpack_from(raw, 0)
        if magic != MAGIC:
            raise IOError(f"{p.name}: nije PSE-A chunk")
        body = raw[HEADER.size:]
        if zlib.crc32(body) & 0xFFFFFFFF != crc:
            raise IOError(f"{p.name}: crc ne odgovara -- chunk je ostecen, ne koristi se")
        na = n * d * 2
        acts = np.frombuffer(body[:na], dtype=np.float16).reshape(n, d)
        off = na
        wn = int(np.frombuffer(body[off:off + 8], np.int64)[0]); off += 8
        mean = np.frombuffer(body[off:off + 8 * d], np.float64); off += 8 * d
        m2 = np.frombuffer(body[off:off + 8 * d], np.float64)
        return acts, VWelford(wn, mean.copy(), m2.copy())

    def write_tokens(self, cid: str, ids: np.ndarray) -> None:
        """Tokeni chunk-a (posle stride-a) -- da kartica feature-a moze da pokaze
        GDE se pali. Isti atomski redosled kao aktivacije."""
        p = self.root / "tokens" / f"{cid}.tok"
        p.parent.mkdir(exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            f.write(np.asarray(ids, dtype=np.uint32).tobytes())
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)

    def read_tokens(self, cid: str) -> np.ndarray:
        return np.frombuffer((self.root / "tokens" / f"{cid}.tok").read_bytes(), dtype=np.uint32)

    def done_chunks(self, layer: int) -> list[str]:
        return [r[0] for r in self.db.execute(
            "SELECT cid FROM chunks WHERE layer=? ORDER BY done", (layer,)).fetchall()]

    def baseline(self, layer: int) -> VWelford:
        """I-9: IZVEDEN iz per-chunk trojki nad done-skupom -- uvek tacno
        konzistentan sa onim sto je na disku."""
        acc = VWelford(0, np.zeros(1), np.zeros(1))
        for cid in self.done_chunks(layer):
            _, w = self.read(layer, cid)
            acc = VWelford.merge(acc, w)
        return acc


def run_harvest(rt, store: HarvestStore, corpus_ids: list[int], corpus_hash: str,
                layers: list[int], chunk_tokens: int, position_stride: int = 1,
                budget_chunks: int | None = None, progress=print) -> dict:
    """[SPEC] HarvestJob. Nastavlja tamo gde je stao: done chunk-ovi se preskacu."""
    offsets = list(range(0, len(corpus_ids), chunk_tokens))
    done = skipped = 0
    for no, off in enumerate(offsets):
        if budget_chunks is not None and done >= budget_chunks:
            break
        cid = chunk_id(corpus_hash, off)
        if all(store.is_done(L, cid) for L in layers):
            skipped += 1
            continue
        ids = corpus_ids[off:off + chunk_tokens]
        if len(ids) < 2:
            continue
        acts = rt.forward_with_hooks(ids, {f"resid_pre.{L}" for L in layers})
        store.write_tokens(cid, np.asarray(ids)[::position_stride])
        for L in layers:
            a = acts[f"resid_pre.{L}"].float().numpy()[::position_stride]
            store.write(L, cid, no, a)
        done += 1
        if progress:
            progress(f"  chunk {no+1}/{len(offsets)}  ({len(ids)} tokena, {len(layers)} slojeva)")
    return {"written": done, "skipped_done": skipped, "total_chunks": len(offsets)}
