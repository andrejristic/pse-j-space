"""Perplexity lestvica -- §4.5: veza spektra i ponasanja.

"BBP prag je prag detektabilnosti, ne prag funkcije." Hipoteza iz §13.5:
na blagoj kvantizaciji perplexity se menja BEZ spektralnog potpisa. Ovde se
za svaku precku meri perplexity na istom tekstu, istim runtime-om (isti
tokenizer, iste operacije), i spaja sa qdiff merama iz SHARED_LIB-a ako
postoje za isti par (f16 -> precka).

Ograničenje koje ide uz svaki prikaz: kod modela ciji d_model nije deljiv
sa 256 (0.5B) precke se razlikuju po SASTAVU tipova (§13.1), pa je osa
"fajl", ne "sirina bita".
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

from .shared_lib import SharedLib, SealRefused, Welford
from .unit import Unit, Budget
from .lens_missions import _check_runtime
from .sae_mission import dev_corpus


def perplexity(rt, ids: list[int], window: int = 1024) -> float:
    t = rt.torch
    nll, n = 0.0, 0
    with t.no_grad():
        for s in range(0, len(ids) - 1, window):
            chunk = ids[s:s + window + 1]
            if len(chunk) < 2:
                break
            lg = rt.logits(chunk[:-1])
            nll += float(t.nn.functional.cross_entropy(
                lg.float(), t.tensor(chunk[1:]), reduction="sum"))
            n += len(chunk) - 1
    return math.exp(nll / max(n, 1))


def run_ppl_ladder(ladder_dir, lib: SharedLib, n_tokens: int = 2048,
                   eval_text: str | None = None, eval_name: str = "dev: common-licenses",
                   progress=print) -> dict:
    from ..core.runtime.qwen2 import Qwen2Runtime
    files = sorted(Path(ladder_dir).glob("*.gguf"))
    ref = next((f for f in files if "f16" in f.name.lower()), files[0])
    order = [ref] + [f for f in files if f != ref]
    g, mm, family = _check_runtime(ref)
    set_hash = hashlib.sha256("|".join(sorted(f.name for f in files)).encode()).hexdigest()
    lib.register_model(set_hash, f"lestvica: {Path(ladder_dir).name}", mm.architecture, {})
    text = eval_text if eval_text is not None else dev_corpus()
    run = lib.open_run(set_hash, "ppl_ladder",
                       {"kind": "ppl_ladder", "files": [f.name for f in order],
                        "n_tokens": n_tokens, "eval": eval_name}, {"seed": 0})
    unit = Unit(f"ppl-{run.run_id}", {}, Budget(len(order)), seed=0)
    pending, local, rows = [], {}, []
    ids = None
    for i, f in enumerate(order):
        t0 = time.perf_counter()
        rt = Qwen2Runtime(f)
        if ids is None:
            ids = rt.encode(text)[:n_tokens]
        p = perplexity(rt, ids)
        inv = {}
        from ..core.source.gguf import open_gguf
        tot = 0
        for k, v in open_gguf(f).type_inventory().items():
            inv[k] = v["elements"]
            tot += v["elements"]
        rows.append({"file": f.name, "ppl": p, "types": {k: v / tot for k, v in inv.items()},
                     "bytes": f.stat().st_size})
        key = f"{family}/ppl/{f.name.split('-')[-1].replace('.gguf','')}"
        v = math.log(p)
        novel, z = lib.test_prior(key, v, local=local.get(key))
        pending.append((key, v))
        local.setdefault(key, Welford()).push(v)
        moved = 0
        if novel:
            lib.record_registry(run.run_id, "behavior/ppl", key, v, z if math.isfinite(z) else 1e9)
            unit.absorb(key)
            moved = 1
        unit.step(moved)
        lib.progress(run.run_id, i + 1, len(order), unit.theta, f.name)
        del rt
        if progress:
            progress(f"  {f.name:40} perplexity {p:8.3f}   ({time.perf_counter()-t0:.0f} s)")
    base = rows[0]["ppl"]
    for r in rows:
        r["ppl_rel"] = r["ppl"] / base - 1.0
    # spoj sa qdiff merama za isti par, ako postoje u ovom SHARED_LIB-u
    for r in rows[1:]:
        hit = lib.db.execute(
            "SELECT m.model_hash FROM models m WHERE m.display_name = ?",
            (f"{order[0].name} ↔ {r['file']}",)).fetchone()
        if not hit:
            continue
        run_row = lib.db.execute("SELECT run_id FROM runs WHERE model_hash=? AND kind='qdiff' "
                                 "AND state='sealed' ORDER BY sealed DESC LIMIT 1",
                                 (hit[0],)).fetchone()
        if not run_row:
            continue
        ah = lib.db.execute("SELECT content_hash FROM artifacts WHERE run_id=? AND "
                            "name='qdiff_report.json'", (run_row[0],)).fetchone()
        rep = json.loads(lib.get_blob(ah[0]))
        ks = [c["bands"][str(c["shape"][1] // 4)]["energy_kept"] for c in rep["matrices"]
              if str(c["shape"][1] // 4) in c["bands"]]
        r["qdiff_run"] = run_row[0]
        r["kept_quarter_mean"] = sum(ks) / len(ks)
        r["energy_change_mean"] = sum(c["energy_change"] for c in rep["matrices"]) / len(rep["matrices"])
        r["rel_rmse_mean"] = sum(c["rel_rmse"] for c in rep["matrices"]) / len(rep["matrices"])
    report = {"family": family, "eval": eval_name, "n_tokens": len(ids), "ref": order[0].name,
              "kquant_clean": mm.kquant_clean, "rows": rows}
    lib.attach(run, "ppl_ladder.json", json.dumps(report, default=float).encode())
    lib.attach(run, "theta_series.json", json.dumps(unit.theta_series).encode())
    try:
        lib.seal(run, {"action": "ppl", "n": len(rows)}, {"action": "ppl", "n": len(order)},
                 unit.theta_series, len(unit.registry),
                 lib.registry_baseline(set_hash, "ppl_ladder"), pending_priors=pending)
        return {"sealed": True, "run_id": run.run_id, "rows": rows}
    except SealRefused as e:
        lib.record_death(run.run_id, str(e), len(unit.registry))
        return {"sealed": False, "run_id": run.run_id, "refused": str(e), "rows": rows}
