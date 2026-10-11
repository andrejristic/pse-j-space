"""pse-a CLI."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pse-a", description="PSE Analyzer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="sposobnosti masine, izmerene")
    d.add_argument("model", nargs="?", help="GGUF fajl za model card + semafor")
    d.add_argument("--no-bench", action="store_true", help="preskoci BLAS benchmark")

    i = sub.add_parser("info", help="model card i inventar tipova")
    i.add_argument("model")

    sv = sub.add_parser("serve", help="web pogled nad SHARED_LIB-om (samo 127.0.0.1)")
    sv.add_argument("--port", type=int, default=8765)

    gu = sub.add_parser("gui", help="pokreni web pogled i otvori ga u browseru")
    gu.add_argument("--port", type=int, default=8765)

    sub.add_parser("setup-torch", help="instaliraj torch (CPU) -- samo na masini gde ce L2-L4 raditi")

    st = sub.add_parser("selftest", help="proveri instalaciju: AZ-1, AZ-8, K6, SHARED_LIB [, K2]")
    st.add_argument("--k2", metavar="DIR", help="folder sa f16 i kvantizovanim GGUF-ovima istog modela")

    l1 = sub.add_parser("l1", help="L1 spektralni sweep kao misija, sa pecatom")
    l1.add_argument("model")
    l1.add_argument("--max", type=int, default=None)

    qd = sub.add_parser("qdiff", help="poredjenje dve preciznosti istog modela")
    qd.add_argument("ref", help="visa preciznost (idealno f16)")
    qd.add_argument("test")
    qd.add_argument("--max", type=int, default=None)

    l2 = sub.add_parser("lens", help="L2 logit lens nad promptovima (trazi torch)")
    l2.add_argument("model")
    l2.add_argument("prompts", nargs="*", help="podrazumevano: skup dvosmislenih promptova")

    l4 = sub.add_parser("jspace", help="L4 mapa J-prostora (trazi torch)")
    l4.add_argument("model")
    l4.add_argument("--layers", default="auto",
                    help="npr. 12,13,14,...  ili auto = svaki sloj (ignition trazi <= 2 sloja)")
    l4.add_argument("--probes", type=int, default=48)

    l3 = sub.add_parser("sae", help="L3: harvest + recnik pojmova za jedan sloj (trazi torch)")
    l3.add_argument("model")
    l3.add_argument("--layer", type=int, required=True)
    l3.add_argument("--corpus", help="tekstualni fajl; podrazumevano razvojni korpus")
    l3.add_argument("--profile", default=None, help="dev/toy/workstation/full; podrazumevano iz doctor-a")

    pp = sub.add_parser("ppl", help="perplexity svake precke u folderu -- spektar protiv ponasanja")
    pp.add_argument("folder", help="folder sa GGUF lestvicom istog modela (sa f16)")
    pp.add_argument("--tokens", type=int, default=2048)
    pp.add_argument("--text", help="tekstualni fajl za merenje; podrazumevano razvojni korpus")

    a = ap.parse_args(argv)

    if a.cmd == "gui":
        import threading, time, webbrowser
        url = f"http://127.0.0.1:{a.port}/"
        threading.Thread(target=lambda: (time.sleep(1.5), webbrowser.open(url)),
                         daemon=True).start()
        a.cmd = "serve"

    if a.cmd == "setup-torch":
        # §8: torch se NE pakuje u .deb -- instalira se na ciljnoj masini, gde se i
        # proverava. Na CPU-u bez AVX2 moze da ne radi; doctor to posle kaze.
        import subprocess
        cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
               "torch", "--index-url", "https://download.pytorch.org/whl/cpu"]
        print("instaliram torch (CPU, ~200 MB) ...")
        rc = subprocess.call(cmd)
        if rc == 0:
            # tokenizer za runtime iz GGUF-a (L2-L4)
            rc = subprocess.call([sys.executable, "-m", "pip", "install",
                                  "--disable-pip-version-check", "tokenizers==0.23.2"])
        if rc == 0:
            print("gotovo. Proveri: pse-a doctor")
        return rc

    if a.cmd == "selftest":
        from .tests import selftest
        return selftest.main(a.k2)

    if a.cmd == "serve":
        import uvicorn
        from .web.app import create_app
        # §7.4: NIKAD 0.0.0.0 -- Inbox ce autorizovati Tier-2 akcije.
        # Za pristup sa drugog uredjaja: ssh -L 8765:localhost:8765 <masina>
        print(f"PSE-A na http://127.0.0.1:{a.port}  (drugi uredjaj: ssh -L {a.port}:localhost:{a.port} ...)")
        uvicorn.run(create_app(), host="127.0.0.1", port=a.port, workers=1,
                    log_level="warning", loop="asyncio", http="h11")
        return 0

    if a.cmd == "ppl":
        from .shell.shared_lib import SharedLib
        from .shell.doctor import data_root
        from .shell.ppl_mission import run_ppl_ladder
        lib = SharedLib(data_root())
        text = Path(a.text).read_text(errors="ignore") if a.text else None
        try:
            r = run_ppl_ladder(a.folder, lib, a.tokens, text, a.text or "dev: common-licenses")
        except RuntimeError as e:
            print(f"Nije pokrenuto: {e}")
            return 2
        print("\nZapecaceno." if r["sealed"] else f"\nNije primljeno: {r['refused']}")
        return 0

    if a.cmd == "sae":
        from .shell.shared_lib import SharedLib
        from .shell.doctor import data_root, probe, select_profile
        from .shell.sae_mission import run_sae
        from .profiles import PROFILES
        lib = SharedLib(data_root())
        prof = PROFILES[a.profile] if a.profile else select_profile(probe(benchmark=True))
        text = Path(a.corpus).read_text(errors="ignore") if a.corpus else None
        name = a.corpus or "dev: /usr/share/common-licenses"
        try:
            r = run_sae(a.model, lib, a.layer, prof, text, name)
        except RuntimeError as e:
            print(f"Nije pokrenuto: {e}")
            return 2
        if r["sealed"]:
            print(f"\nZapecaceno: run {r['run_id']} · FVU {r['fvu']:.3f} · L0 {r['l0']:.1f} · "
                  f"stabilnih pojmova {r['n_stable']} · N_eff {r['n_eff']:.0f}")
        else:
            print(f"\nNije primljeno: {r['refused']}")
        return 0

    if a.cmd in ("lens", "jspace"):
        from .shell.shared_lib import SharedLib
        from .shell.doctor import data_root
        from .shell import prognostics as pg
        from .shell import lens_missions as lm
        lib = SharedLib(data_root())
        try:
            if a.cmd == "lens":
                from .core.lens.jlens import load_ambiguous
                r = lm.run_logit_lens(a.model, lib, a.prompts or load_ambiguous())
            else:
                from .core.source.gguf import open_gguf
                nl = int(open_gguf(a.model).metadata.get("qwen2.block_count", 24))
                # auto: gusto (korak 1) -- kriterijum ignition-a trazi <= 2 sloja
                layers = (list(range(1, nl)) if a.layers == "auto"
                          else [int(x) for x in a.layers.split(",")])
                r = lm.run_jspace(a.model, lib, layers, a.probes)
        except RuntimeError as e:
            print(f"Nije pokrenuto: {e}")
            return 2
        if r["sealed"]:
            print(f"\nZapecaceno: run {r['run_id']}, {r['registry']} nalaza u znanju.")
        else:
            print("\n" + pg.refusal([k.strip() for k in r["refused"].split("—")[-1].split(",")]))
        return 0

    if a.cmd in ("l1", "qdiff"):
        import json
        from .shell.shared_lib import SharedLib
        from .shell.doctor import data_root
        lib = SharedLib(data_root())
        if a.cmd == "l1":
            from .shell.l1_mission import run_l1
            r = run_l1(a.model, lib, max_matrices=a.max)
        else:
            from .shell.qdiff_mission import run_qdiff
            r = run_qdiff(a.ref, a.test, lib, max_matrices=a.max)
        from .shell import prognostics as pg
        if r.get("sealed"):
            print(f"\nZapecaceno: run {r['run_id']}, {r['registry']} nalaza u znanju.")
        else:
            failed = [k.strip() for k in r["refused"].split("—")[-1].split(",")]
            print("\n" + pg.refusal(failed))
        return 0

    if a.cmd == "doctor":
        from .shell import doctor
        r = doctor.probe(benchmark=not a.no_bench)
        print(doctor.render(r, a.model))
        return 0

    if a.cmd == "info":
        from .core.source.gguf import open_gguf
        from .core.source.canonical import build_map
        g = open_gguf(a.model)
        mm = build_map(g)
        print(mm.summary())
        print(f"model_hash {g.model_hash()}")
        print()
        inv = g.type_inventory()
        tot = sum(v["elements"] for v in inv.values())
        print(f"{'tip':9} {'tenzora':>8} {'% parametara':>13} {'MB':>9}")
        for k, v in inv.items():
            print(f"{k:9} {v['tensors']:8} {100*v['elements']/tot:12.1f}% {v['bytes']/1e6:9.0f}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
