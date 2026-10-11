"""L1 sweep kao PSE misija: jedinica, budzet, priori, registar, pecat.

[PSE-A] 8.1: svaki L1-L4 posao je misija jedne jedinice, lifespan u compute
koracima. Za L1 je jedan korak = jedna matrica.

Tok po matrici:
  dequantizuj (I-5) -> layer_health (core, cist) -> za svaku velicinu testiraj
  prior (§3.2) -> nalaz koji je pomerio ocekivanje ide u registar -> korak
  jedinice sa brojem takvih nalaza -> oslobodi matricu

Na kraju: artefakt (JSON izvestaj + ESD histogrami) -> tri sloja pecacenja.
Ako kapija odbije, run ostaje RADNO STANJE i ne ulazi u dosije -- to nije
greska programa nego odluka sistema, i vraca se kao takva.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from ..core.source.gguf import open_gguf
from ..core.source.canonical import build_map
from ..core.source import dequant as dq
from ..core.spectral.esd import layer_health, ESD_BINS
from .shared_lib import SharedLib, SealRefused, Welford, config_hash
from .unit import Unit, Budget

QUANTITIES = ("alpha", "tail_frac", "log_sigma2", "log_stable_rank")


def _prior_key(family: str, role: str, depth: float | None, q: str) -> str:
    """Kljuc priora = STA JE MODEL, ne u kojoj je preciznosti zapisan.

    ggml_type je kovarijata u izvestaju (I-8), ali NIJE deo kljuca: sa tipom u
    kljucu svaka precka lestvice postaje zaseban silos, pa ponovno merenje istog
    modela u Q8_0 (korelacija 0.99998 sa f16) daje stotine "novih" nalaza --
    rast registra srazmeran broju precki, ne ucenju. Stetu kvantizacije meri
    posebna qdiff misija, ciji priori tip ukljucuju.

    family = arhitektura + sirina: 0.5B i 7B su obe 'qwen2', ali d_model 896
    naspram 3584 menja i sigma² i alfa, pa ne smeju deliti priore.
    """
    d = "g" if depth is None else f"d{round(depth * 4) / 4:.2f}"   # 5 pojaseva dubine
    return f"{family}/{role}/{d}/{q}"


def _values(h) -> dict[str, float]:
    return {
        "alpha": h.alpha,
        "tail_frac": h.tail_frac,
        "log_sigma2": math.log(h.sigma2) if h.sigma2 > 0 else float("nan"),
        "log_stable_rank": math.log(h.stable_rank) if h.stable_rank > 0 else float("nan"),
    }


def run_l1(model_path: str | Path, lib: SharedLib, max_matrices: int | None = None,
           include_embeddings: bool = False, seed: int = 0,
           progress=print) -> dict:
    model_path = Path(model_path)
    g = open_gguf(model_path)
    mm = build_map(g)
    mh = g.model_hash()
    lib.register_model(mh, model_path.name, mm.architecture,
                       {n: vars(r) for n, r in mm.roles.items()})

    names = [n for n in mm.weight_matrices()
             if include_embeddings or mm.roles[n].role_class != "embedding"]
    names.sort(key=lambda n: (mm.roles[n].layer if mm.roles[n].layer is not None else -1, n))
    if max_matrices:
        names = names[:max_matrices]

    cfg = {"kind": "l1_sweep", "esd_bins": ESD_BINS, "n": len(names),
           "embeddings": include_embeddings, "quantities": QUANTITIES}
    run = lib.open_run(mh, "l1_sweep", cfg, {"seed": seed})
    unit = Unit(f"l1-{run.run_id}", {"state": "sweep", "model": mh,
                                     "action": "l1", "outcome": "report"},
                Budget(total=len(names)), seed=seed)

    family = f"{mm.architecture}-d{mm.d_model}"
    rows, hists = [], {}
    pending: list[tuple[str, float]] = []
    local: dict[str, Welford] = {}
    t0 = time.perf_counter()
    for i, nm in enumerate(names):
        r = mm.roles[nm]
        tname = g.tensors[nm].type_name
        W = dq.tensor(g, nm)
        h = layer_health(W, nm, tname, r.role, r.role_class, r.depth)
        del W

        moved = 0
        for q, v in _values(h).items():
            if not np.isfinite(v):
                continue
            key = _prior_key(family, r.role, r.depth, q)
            novel, z = lib.test_prior(key, float(v), local=local.get(key))
            pending.append((key, float(v)))
            local.setdefault(key, Welford()).push(float(v))
            if novel:
                lib.record_registry(run.run_id, f"layer_health/{q}", key, float(v),
                                    z if np.isfinite(z) else 1e9)
                unit.absorb(key)
                moved += 1
        unit.step(moved)
        lib.progress(run.run_id, i + 1, len(names), unit.theta, nm)

        rows.append({
            "name": nm, "role": r.role, "role_class": r.role_class, "layer": r.layer,
            "depth": r.depth, "ggml_type": tname, "shape": list(h.shape), "q": h.q,
            "sigma2": h.sigma2, "sigma_method": h.sigma_method, "edge": h.edge,
            "regime": h.regime, "n_tail": h.n_tail, "tail_frac": h.tail_frac,
            "alpha": h.alpha, "alpha_lo": h.alpha_lo, "alpha_hi": h.alpha_hi,
            "alpha_xmin": h.alpha_xmin, "stable_rank": h.stable_rank,
            "spectral_norm": h.spectral_norm, "moved_priors": moved,
            "mp_ks_n": h.mp_ks_n, "mp_fit_ok": h.mp_fit_ok,
            "theta": unit.theta,
        })
        hists[nm] = {"counts": h.esd_hist.tolist(), "edges": h.esd_edges.tolist()}
        if progress and (i % 10 == 0 or i == len(names) - 1):
            el = time.perf_counter() - t0
            eta = el / (i + 1) * (len(names) - i - 1)
            progress(f"  [{i+1:3}/{len(names)}] {nm:28} {h.regime:13} "
                     f"theta={unit.theta:5.2f} registar={len(unit.registry):3} "
                     f"eta {eta/60:4.1f} min")

    elapsed = time.perf_counter() - t0
    report = {"model_hash": mh, "model": model_path.name, "architecture": mm.architecture,
              "n_layers": mm.n_layers, "d_model": mm.d_model, "d_ff": mm.d_ff,
              "kquant_clean": mm.kquant_clean, "type_inventory": g.type_inventory(),
              "config_hash": config_hash(cfg), "elapsed_s": elapsed, "matrices": rows}
    lib.attach(run, "l1_report.json", json.dumps(report, default=float).encode())
    lib.attach(run, "esd_histograms.json", json.dumps(hists).encode())
    lib.attach(run, "theta_series.json", json.dumps(unit.theta_series).encode())

    outcome = {"state": "sweep", "model": mh, "action": "l1", "outcome": "report",
               "n_matrices": len(rows)}
    spec = {"state": "sweep", "model": mh, "action": "l1", "outcome": "report",
            "n_matrices": len(names)}
    baseline = lib.registry_baseline(mh, "l1_sweep")
    result = {"run_id": run.run_id, "model_hash": mh, "elapsed_s": elapsed,
              "n_matrices": len(rows), "registry": len(unit.registry),
              "theta_min": min(unit.theta_series), "theta_max": max(unit.theta_series),
              "theta_mean": sum(unit.theta_series) / len(unit.theta_series)}
    try:
        gate = lib.seal(run, outcome, spec, unit.theta_series,
                        len(unit.registry), baseline, pending_priors=pending)
        result.update(sealed=True, gate=gate.detail)
    except SealRefused as e:
        result.update(sealed=False, refused=str(e), death=unit.die(str(e)))
        lib.record_death(run.run_id, str(e), len(unit.registry))
    from .physician import review
    result["prognoses"] = review(lib, run.model_hash if hasattr(run, "model_hash") else "", run.kind)
    return result
