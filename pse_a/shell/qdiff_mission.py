"""qdiff kao PSE misija: poredjenje dve preciznosti istog modela.

Zatvara rupu iz §13.6: L1 sweep kvantizovane kopije vec zapecacenog modela
nije doprinos o modelu nego o KVANTIZACIJI, pa ovde pripada. Priori ovde
UKLJUCUJU par tipova (I-19 obrnuto): pitanje je "sta Q4_K radi ffn_down-u na
dubini d", ne "kakav je ffn_down".

Osnova za uslov 3 kapije se racuna po PARU (ref, test) i vrsti misije, pa
ponovljeni qdiff istog para ne moze da izbegne sunk-cost detekciju.

Izvestaj je STRATIFIKOVAN PO ULOZI (§13.7): shema i uloga su u K_M mesavinama
vezane po konstrukciji, pa agregat preko uloga meri mesavinu, ne shemu.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from ..core.source.gguf import open_gguf
from ..core.source.canonical import build_map
from ..core.source import dequant as dq
from ..core.spectral.qdiff import compare_full
from .shared_lib import SharedLib, SealRefused, Welford, config_hash
from .unit import Unit, Budget

QUANTITIES = ("sigma2_shift", "alpha_shift", "energy_change", "noise_ratio",
              "kept_q")   # kept_q = zadrzano na pojasu m/4


def _pair_hash(a: str, b: str) -> str:
    return hashlib.sha256(f"{a}|{b}".encode()).hexdigest()


def _values(c: dict, m: int) -> dict[str, float]:
    kq = c["bands"].get(str(m // 4), {}).get("energy_kept", float("nan"))
    return {"sigma2_shift": c["sigma2_shift"], "alpha_shift": c["alpha_shift"],
            "energy_change": c["energy_change"],
            "noise_ratio": math.log(c["noise_ratio"]) if c["noise_ratio"] > 0 else float("nan"),
            "kept_q": math.log1p(-kq) if kq < 1 else float("nan")}


def run_qdiff(ref_path, test_path, lib: SharedLib, max_matrices: int | None = None,
              seed: int = 0, progress=print) -> dict:
    ref_path, test_path = Path(ref_path), Path(test_path)
    gr, gt = open_gguf(ref_path), open_gguf(test_path)
    mr, mt = build_map(gr), build_map(gt)
    if (mr.architecture, mr.d_model, mr.n_layers) != (mt.architecture, mt.d_model, mt.n_layers):
        raise ValueError("qdiff trazi isti model u dve preciznosti")
    pair = _pair_hash(gr.model_hash(), gt.model_hash())
    lib.register_model(pair, f"{ref_path.name} ↔ {test_path.name}", mr.architecture, {})

    names = [n for n in mr.weight_matrices() if mr.roles[n].role_class != "embedding"]
    names.sort(key=lambda n: (mr.roles[n].layer if mr.roles[n].layer is not None else -1, n))
    if max_matrices:
        names = names[:max_matrices]

    family = f"{mr.architecture}-d{mr.d_model}"
    cfg = {"kind": "qdiff", "ref": gr.model_hash(), "test": gt.model_hash(), "n": len(names)}
    run = lib.open_run(pair, "qdiff", cfg, {"seed": seed})
    unit = Unit(f"qd-{run.run_id}", {}, Budget(len(names)), seed=seed)

    rows, pending, local = [], [], {}
    t0 = time.perf_counter()
    for i, nm in enumerate(names):
        r = mr.roles[nm]
        tr, tt = gr.tensors[nm].type_name, gt.tensors[nm].type_name
        c = compare_full(dq.tensor(gr, nm), dq.tensor(gt, nm), nm, tr, tt,
                         r.role, r.role_class, r.depth)
        m = c["shape"][1]
        moved = 0
        d = "g" if r.depth is None else f"d{round(r.depth * 4) / 4:.2f}"
        for qn, v in _values(c, m).items():
            if not np.isfinite(v):
                continue
            key = f"{family}/{tr}->{tt}/{r.role}/{d}/{qn}"
            novel, z = lib.test_prior(key, float(v), local=local.get(key))
            pending.append((key, float(v)))
            local.setdefault(key, Welford()).push(float(v))
            if novel:
                lib.record_registry(run.run_id, f"quant_damage/{qn}", key, float(v),
                                    z if np.isfinite(z) else 1e9)
                unit.absorb(key)
                moved += 1
        unit.step(moved)
        lib.progress(run.run_id, i + 1, len(names), unit.theta, nm)
        c["theta"] = unit.theta
        c["moved_priors"] = moved
        rows.append(c)
        if progress and (i % 10 == 0 or i == len(names) - 1):
            el = time.perf_counter() - t0
            progress(f"  [{i+1:3}/{len(names)}] {nm:28} {tt:6} "
                     f"kept={c['bands'].get(str(m//4),{}).get('energy_kept',0):.4f} "
                     f"theta={unit.theta:5.2f} eta {el/(i+1)*(len(names)-i-1)/60:5.1f} min")

    # stratifikacija po ulozi i paru tipova (§13.7)
    strata: dict[str, dict] = {}
    for c in rows:
        k = f"{c['role']}|{c['type_ref']}->{c['type_test']}"
        s = strata.setdefault(k, {"n": 0, "noise": [], "kept_q": [], "dE": [], "ds2": [], "da": [],
                                  "mp_ok": 0})
        m = c["shape"][1]
        s["n"] += 1
        s["noise"].append(c["noise_ratio"])
        s["kept_q"].append(c["bands"].get(str(m // 4), {}).get("energy_kept", float("nan")))
        s["dE"].append(c["energy_change"])
        if c.get("mp_fit_ok_ref"):
            s["ds2"].append(c["sigma2_shift"])       # samo gde je sigma² stvarno sumni pod
            s["mp_ok"] += 1
        if np.isfinite(c["alpha_shift"]):
            s["da"].append(c["alpha_shift"])
    summary = {k: {"n": v["n"],
                   "noise_ratio": float(np.nanmean(v["noise"])),
                   "kept_quarter": float(np.nanmean(v["kept_q"])),
                   "energy_change": float(np.nanmean(v["dE"])),
                   "sigma2_shift": float(np.nanmean(v["ds2"])) if v["ds2"] else float("nan"),
                   "mp_ok": v["mp_ok"],
                   "alpha_shift": float(np.nanmean(v["da"])) if v["da"] else float("nan")}
               for k, v in sorted(strata.items())}

    elapsed = time.perf_counter() - t0
    report = {"pair": pair, "ref": ref_path.name, "test": test_path.name,
              "ref_hash": gr.model_hash(), "test_hash": gt.model_hash(),
              "family": family, "config_hash": config_hash(cfg), "elapsed_s": elapsed,
              "strata": summary, "matrices": rows}
    lib.attach(run, "qdiff_report.json", json.dumps(report, default=float).encode())
    lib.attach(run, "theta_series.json", json.dumps(unit.theta_series).encode())

    spec = {"action": "qdiff", "n_matrices": len(names)}
    outcome = {"action": "qdiff", "n_matrices": len(rows)}
    result = {"run_id": run.run_id, "pair": pair, "elapsed_s": elapsed,
              "registry": len(unit.registry), "strata": summary,
              "theta_mean": sum(unit.theta_series) / len(unit.theta_series),
              "theta_min": min(unit.theta_series)}
    try:
        lib.seal(run, outcome, spec, unit.theta_series, len(unit.registry),
                 lib.registry_baseline(pair, "qdiff"), pending_priors=pending)
        result["sealed"] = True
    except SealRefused as e:
        result.update(sealed=False, refused=str(e), death=unit.die(str(e)))
        lib.record_death(run.run_id, str(e), len(unit.registry))
    from .physician import review
    result["prognoses"] = review(lib, run.model_hash if hasattr(run, "model_hash") else "", run.kind)
    return result
