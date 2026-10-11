"""Invarijante SHARED_LIB-a: AZ-4, AZ-5, AZ-6, I-9, i definicija iz §3.2."""
from __future__ import annotations

import random
import tempfile

from pse_a.shell.shared_lib import SharedLib, SealRefused, Welford

OK, PAD = "OK", "PAD"


def run() -> bool:
    good = True
    rnd = random.Random(0)

    print("I-9  Welford: pairwise merge == jedan prolaz")
    xs = [rnd.gauss(5, 2) for _ in range(1000)]
    single = Welford()
    for x in xs:
        single.push(x)
    parts = []
    for i in range(0, 1000, 137):                 # nejednaki delovi, kao chunk-ovi
        w = Welford()
        for x in xs[i:i + 137]:
            w.push(x)
        parts.append(w)
    merged = parts[0]
    for w in parts[1:]:
        merged = Welford.merge(merged, w)
    dm = abs(merged.mean - single.mean)
    dv = abs(merged.var - single.var)
    ok = dm < 1e-10 and dv < 1e-8
    good &= ok
    print(f"     mean delta {dm:.2e}  var delta {dv:.2e}   {OK if ok else PAD}")

    with tempfile.TemporaryDirectory() as td:
        sl = SharedLib(td)
        sl.register_model("m1", "test.gguf", "qwen2", {})
        healthy = [30 + rnd.gauss(0, 1.0) for _ in range(50)]

        print("\nAZ-5/11.4  tri sloja pecacenja")
        r = sl.open_run("m1", "l1_sweep", {"profile": "dev"}, {"seed": 0})
        h = sl.attach(r, "report", b'{"alpha": 3.2}')
        try:
            sl.seal(r, {"status": "done"}, {"status": "done", "n_matrices": 198},
                    healthy, 3)
            print(f"     sloj 1 (ishod vs misija)                        {PAD}")
            good = False
        except SealRefused as e:
            print(f"     sloj 1 odbija nepotpun ishod                    {OK}  [{str(e)[:44]}]")
        try:
            sl.seal(r, {"status": "done"}, {"status": "done"}, [40.0] * 50, 3)
            print(f"     sloj 3 (kapija)                                 {PAD}")
            good = False
        except SealRefused as e:
            print(f"     sloj 3 odbija theta van opsega                  {OK}  [{str(e)[:44]}]")
        g = sl.seal(r, {"status": "done"}, {"status": "done"}, healthy, 3)
        print(f"     zdrav run se pecati                             {OK if g.passed else PAD}")
        good &= g.passed

        print("\nAZ-5  zapecaceno je nepromenljivo")
        try:
            sl.attach(r, "drugi", b"x")
            print(f"     dodavanje artefakta posle pecata               {PAD}")
            good = False
        except SealRefused:
            print(f"     dodavanje artefakta posle pecata odbijeno      {OK}")
        has_mut = [m for m in dir(sl) if m.startswith(("update_", "delete_", "edit_"))]
        print(f"     mutirajucih metoda u API-ju: {len(has_mut)}                  "
              f"{OK if not has_mut else PAD}")
        good &= not has_mut

        print("\nAZ-6  beleske append-only, usidrene za hes")
        nid = sl.add_note(r.run_id, "andrej", h, "alfa deluje nisko za ovaj sloj")
        print(f"     beleska na zapecacen artefakt                   {OK if nid else PAD}")
        try:
            sl.add_note(r.run_id, "andrej", "deadbeef" * 8, "tekst")
            print(f"     beleska na tudji hes                           {PAD}")
            good = False
        except ValueError:
            print(f"     beleska na nepostojeci hes odbijena            {OK}")

        print("\n§3.2  rast registra = pomeranje ocekivanja, ne potroseni compute")
        key = "qwen2/ffn_gate/d0.5/alpha"
        seq = [3.60, 3.62, 3.58, 3.61, 3.59, 3.60, 3.61, 3.59]   # isti run 8 puta
        grew = []
        for v in seq:
            novel, z = sl.test_and_update_prior(key, v)
            grew.append(novel)
        print(f"     osam ponavljanja istog nalaza -> rast: {sum(grew)}/8"
              f"   (prva 3 su bootstrap)")
        ok = sum(grew[3:]) == 0
        good &= ok
        print(f"     posle bootstrap-a registar NE raste             {OK if ok else PAD}")
        novel, z = sl.test_and_update_prior(key, 5.90)            # stvarno drugaciji nalaz
        print(f"     nov nalaz alfa=5.90  ->  rast={novel}  z={z:.1f}  "
              f"{OK if novel else PAD}")
        good &= novel
    return good


if __name__ == "__main__":
    print("SHARED_LIB -- invarijante")
    print("=" * 66)
    g = run()
    print("\n" + "=" * 66)
    print("SVE INVARIJANTE DRZE" if g else "IMA PROBLEMA")
    raise SystemExit(0 if g else 1)
