"""L2 (logit lens) i L4 (J-prostor) kao PSE misije -- isti tok kao L1/qdiff:
jedinica, budzet, priori, registar, artefakti, tri sloja pecacenja.

Runtime se pravi iz ISTOG GGUF-a koji L1 cita (§13.10). Za sada samo qwen2.
Capability semafor (AZ-2): ako torch nije tu ili model ne staje u RAM, misija
se ne pokrece -- nikad ne padaj usred run-a iz razloga koji je poznat na L0.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from ..core.source.gguf import open_gguf
from ..core.source.canonical import build_map
from .shared_lib import SharedLib, SealRefused, Welford, config_hash
from .unit import Unit, Budget


def _check_runtime(model_path: Path) -> tuple[object, str, str]:
    g = open_gguf(model_path)
    mm = build_map(g)
    if mm.architecture != "qwen2":
        raise RuntimeError(f"L2/L4 runtime za '{mm.architecture}' jos ne postoji (samo qwen2)")
    try:
        import torch  # noqa: F401
    except Exception:
        raise RuntimeError("torch nije instaliran -- pse-a setup-torch")
    import os
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    need = sum(t.n_elements for t in g.tensors.values()) * 4
    if need > 0.55 * ram:
        raise RuntimeError(f"model u fp32 trazi {need/1e9:.1f} GB, RAM {ram/1e9:.1f} GB")
    return g, mm, f"{mm.architecture}-d{mm.d_model}"


def _seal(lib, run, unit, pending, kind, mh, n_steps):
    spec = {"action": kind, "steps": n_steps}
    out = {"action": kind, "steps": len(unit.theta_series)}
    try:
        lib.seal(run, out, spec, unit.theta_series, len(unit.registry),
                 lib.registry_baseline(mh, kind), pending_priors=pending)
        return {"sealed": True, "run_id": run.run_id, "registry": len(unit.registry)}
    except SealRefused as e:
        lib.record_death(run.run_id, str(e), len(unit.registry))
        from .physician import review
        return {"sealed": False, "run_id": run.run_id, "refused": str(e),
                "registry": len(unit.registry), "death": unit.die(str(e)),
                "prognoses": review(lib, mh, kind)}


def run_logit_lens(model_path, lib: SharedLib, prompts: list[str], seed: int = 0,
                   progress=print) -> dict:
    from ..core.runtime.qwen2 import Qwen2Runtime
    from ..core.lens.logit import waterfall, pack
    model_path = Path(model_path)
    g, mm, family = _check_runtime(model_path)
    mh = g.model_hash()
    lib.register_model(mh, model_path.name, mm.architecture, {})
    rt = Qwen2Runtime(model_path)
    run = lib.open_run(mh, "l2_lens", {"kind": "l2_lens", "prompts": prompts}, {"seed": seed})
    unit = Unit(f"l2-{run.run_id}", {}, Budget(len(prompts)), seed=seed)
    pending, local, out = [], {}, []
    for i, p in enumerate(prompts):
        ids = rt.encode(p)
        wf = waterfall(rt, ids)
        # prvi sloj na kome lens na poslednjoj poziciji pogadja konacni izbor
        final_top = int(wf["rows"][-1]["ids"][-1][0])
        commit = next((r["layer"] for r in wf["rows"]
                       if int(r["ids"][-1][0]) == final_top), "final")
        commit_depth = (commit / (rt.n_layers - 1)) if isinstance(commit, int) else 1.0
        key = f"{family}/lens/commit_depth"
        novel, z = lib.test_prior(key, commit_depth, local=local.get(key))
        pending.append((key, commit_depth))
        local.setdefault(key, Welford()).push(commit_depth)
        moved = 0
        if novel:
            lib.record_registry(run.run_id, "lens/commit_depth", key, commit_depth,
                                z if math.isfinite(z) else 1e9)
            unit.absorb(key)
            moved = 1
        unit.step(moved)
        lib.progress(run.run_id, i + 1, len(prompts), unit.theta, p[:40])
        out.append({"prompt": p, "tokens": [rt.decode([t]) for t in ids],
                    "layers": [str(x) for x in wf["layers"]], "commit_layer": str(commit),
                    "top": [[[rt.decode([int(t)]) for t in r["ids"][pos][:3]]
                             for pos in range(len(ids))] for r in wf["rows"]],
                    "p": [[float(r["probs"][pos][0]) for pos in range(len(ids))]
                          for r in wf["rows"]],
                    "entropy": [[float(x) for x in r["entropy"]] for r in wf["rows"]]})
        lib.attach(run, f"waterfall_{i:03d}.bin", pack(wf), media="bin")
        if progress:
            progress(f"  [{i+1}/{len(prompts)}] '{p[:40]}' -> '{out[-1]['top'][-1][-1][0]}' "
                     f"(opredeljen na sloju {commit})")
    lib.attach(run, "lens_report.json",
               json.dumps({"model": model_path.name, "n_layers": rt.n_layers,
                           "items": out}).encode())
    lib.attach(run, "theta_series.json", json.dumps(unit.theta_series).encode())
    return _seal(lib, run, unit, pending, "l2_lens", mh, len(prompts))


def run_jspace(model_path, lib: SharedLib, layers: list[int], n_probes: int = 48,
               prompts: list[str] | None = None, seed: int = 0, progress=print) -> dict:
    from ..core.runtime.qwen2 import Qwen2Runtime
    from ..core.lens.jlens import jspace_map, load_ambiguous
    model_path = Path(model_path)
    g, mm, family = _check_runtime(model_path)
    mh = g.model_hash()
    lib.register_model(mh, model_path.name, mm.architecture, {})
    rt = Qwen2Runtime(model_path)
    prompts = prompts or load_ambiguous()
    corpus = " ".join(prompts)
    cfg = {"kind": "l4_jspace", "layers": layers, "n_probes": n_probes, "prompts": prompts}
    run = lib.open_run(mh, "l4_jspace", cfg, {"seed": seed})
    unit = Unit(f"l4-{run.run_id}", {}, Budget(len(layers)), seed=seed)
    pending, local = [], {}
    step = {"i": 0}

    def on_layer(msg):
        step["i"] += 1
        if progress:
            progress(msg)

    m = jspace_map(rt, prompts, layers, n_probes, corpus, progress=on_layer)
    for L in layers:
        pl = m["per_layer"][L]
        d = f"d{round(L / (rt.n_layers - 1) * 4) / 4:.2f}"
        moved = 0
        for qn in ("var_frac", "entropy", "motor_kl"):
            v = float(pl[qn]) if qn != "motor_kl" else math.log1p(float(pl[qn]))
            key = f"{family}/jspace/{d}/{qn}"
            novel, z = lib.test_prior(key, v, local=local.get(key))
            pending.append((key, v))
            local.setdefault(key, Welford()).push(v)
            if novel:
                lib.record_registry(run.run_id, f"jspace/{qn}", key, v,
                                    z if math.isfinite(z) else 1e9)
                unit.absorb(key)
                moved += 1
        unit.step(moved)
        lib.progress(run.run_id, len(unit.theta_series), len(layers), unit.theta, f"sloj {L}")
    m["model"] = model_path.name
    m["n_layers"] = rt.n_layers
    m["per_layer"] = {str(k): v for k, v in m["per_layer"].items()}
    m["chips"] = {str(k): v for k, v in m["chips"].items()}
    lib.attach(run, "jspace_map.json", json.dumps(m, default=float).encode())
    lib.attach(run, "theta_series.json", json.dumps(unit.theta_series).encode())
    return _seal(lib, run, unit, pending, "l4_jspace", mh, len(layers))
