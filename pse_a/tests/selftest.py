"""pse-a selftest -- ono sto se pokrece POSLE PRENOSA .deb-a na drugu masinu.

Proverava invarijante koje ne traze model (AZ-1, AZ-8, K6, SHARED_LIB), i
opciono K2 ako se da folder sa GGUF lestvicom. K2 je jedini test koji hvata
tihu gresku u dequantizaciji, pa ga na novoj masini vredi pustiti bar jednom.
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path


def _run(label, fn) -> bool:
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            r = fn()
        ok = (r is None or r is True or r == 0 or r == [])
    except SystemExit as e:
        ok = (e.code in (0, None))
    except Exception as e:                       # noqa: BLE001
        print(f"  {label:34} GRESKA  {type(e).__name__}: {e}")
        return False
    print(f"  {label:34} {'OK' if ok else 'PAD'}")
    if not ok:
        print("    " + buf.getvalue().strip().replace("\n", "\n    ")[-1500:])
    return ok


def main(k2_dir: str | None = None) -> int:
    from . import test_layering, test_az8, test_k6_sigma, test_shared_lib
    print("pse-a selftest")
    good = True
    good &= _run("AZ-1  cist core", lambda: test_layering.check_core_purity())
    good &= _run("AZ-8  bez imena formi i dijagnoza", lambda: test_az8.run())
    good &= _run("K6    procena sigma²", lambda: test_k6_sigma.run())
    good &= _run("SHARED_LIB invarijante", lambda: test_shared_lib.run())
    if k2_dir:
        good &= _run("K2    dequantizacija", lambda: _k2(Path(k2_dir)))
    else:
        print("  K2    dequantizacija               preskoceno (--k2 <folder sa GGUF lestvicom>)")
    print("\nSVE PROVERE PROSLE" if good else "\nIMA PROBLEMA -- vidi gore")
    return 0 if good else 1


def _k2(d: Path) -> bool:
    from ..core.source.gguf import open_gguf
    from ..core.source import dequant as dq
    from .test_k2_dequant import check_tensor
    files = sorted(d.glob("*.gguf"))
    ref = next((f for f in files if "f16" in f.name.lower() or "f32" in f.name.lower()), None)
    if ref is None:
        ref = next((f for f in files if "q8_0" in f.name.lower()), None)
    if ref is None:
        raise FileNotFoundError("nema f16 ni Q8_0 reference u folderu")
    gr = open_gguf(ref)
    names = [n for n, t in gr.tensors.items() if t.is_2d and "blk.0." in n][:4]
    ok = True
    for f in files:
        if f == ref:
            continue
        g = open_gguf(f)
        for nm in names:
            r = check_tensor(dq.tensor(g, nm), dq.tensor(gr, nm), g.tensors[nm].type_name)
            print(f"{f.name} {nm} {r['type']} {r['corr']:.6f} {r['uniformity']:.3f} "
                  f"{'OK' if r['ok'] else 'PAD'}")
            ok &= r["ok"]
    return ok
