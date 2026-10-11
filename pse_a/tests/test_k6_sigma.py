"""K6: mp_sigma_estimate. Obavezni test paket iz §3.1.

Bez ovoga svako alfa i svaka ivica u celom projektu su sistematski pomereni,
a greska je tiha -- ESD i dalje izgleda uverljivo.
"""
from __future__ import annotations

import numpy as np

from pse_a.core.spectral import mp
from pse_a.core.spectral.esd import esd, estimate_sigma2, MP_TOLERANCE


def gaussian(n: int, m: int, sigma: float, rng) -> np.ndarray:
    return (rng.standard_normal((n, m)) * sigma).astype(np.float32)


def with_spikes(n: int, m: int, sigma: float, k: int, theta: float, rng) -> np.ndarray:
    """Populaciona kovarijansa sigma²·(I + U diag(theta) Uᵀ) -> k BBP spike-ova."""
    Z = rng.standard_normal((n, m)) * sigma
    if k > 0:
        U, _ = np.linalg.qr(rng.standard_normal((m, k)))
        Z = Z + (Z @ U) * (np.sqrt(1.0 + theta) - 1.0) @ U.T
    return Z.astype(np.float32)


def naive_sigma2(lam, q):
    """Estimator kakav bi se dobio citanjem medijane kao 0.5 kvantila --
    to jest bez korekcije na udeo spike-ova. Sluzi za poredjenje."""
    return float(np.median(lam)) / mp.median(q)


def run() -> bool:
    rng = np.random.default_rng(7)
    ok = True

    # ---- 1. cista Gausova: sigma² unutar 1%
    print("1. cista Gausova matrica")
    for q in (0.143, 0.5, 1.0):
        m = 512; n = int(round(m / q))
        lam, qq, _, _ = esd(gaussian(n, m, 1.3, rng))
        f = estimate_sigma2(lam, qq)
        err = abs(f.sigma2 - 1.69) / 1.69
        good = err < 0.01; ok &= good
        print(f"   q={q:5.3f}  sigma2={f.sigma2:.4f} (tacno 1.6900)  greska {err*100:5.2f}%"
              f"  {'OK' if good else 'PAD'}")

    # ---- 2. planted spikes: sigma² NEPROMENJEN, svih k u repu
    print("\n2. Gaus + k planted spike-ova (sigma=1, q=0.189)")
    m = 512; q = 0.189; n = int(round(m / q))
    for k in (0, 5, 25, 60):
        W = with_spikes(n, m, 1.0, k, theta=6.0 * np.sqrt(q), rng=rng)
        lam, qq, _, _ = esd(W)
        f = estimate_sigma2(lam, qq)
        nv = naive_sigma2(lam, qq)
        n_tail = int((lam > f.edge * (1 + MP_TOLERANCE)).sum())
        good = abs(f.sigma2 - 1.0) < 0.02 and n_tail >= k
        ok &= good
        print(f"   k={k:3}  sigma2={f.sigma2:.4f}  naivni={nv:.4f}"
              f"  (naivni gresi {100*(nv-1):+5.1f}%)  u repu {n_tail:3}/{k:<3}"
              f" {'OK' if good else 'PAD'}")

    # ---- 3. skaliranje W -> cW  =>  sigma² -> c²sigma²
    print("\n3. skaliranje")
    W = gaussian(2000, 400, 1.0, rng)
    base = estimate_sigma2(*esd(W)[:2]).sigma2
    for c in (0.1, 3.0, 50.0):
        s = estimate_sigma2(*esd(W * c)[:2]).sigma2
        r = s / (base * c * c)
        good = abs(r - 1.0) < 0.01; ok &= good
        print(f"   c={c:5.1f}  sigma2/(c²·baza) = {r:.5f}  {'OK' if good else 'PAD'}")

    # ---- 4. sweep po q: nepristrasnost kroz ceo opseg
    print("\n4. sweep po q  (Qwen koristi 0.143, 0.189 i 1.0)")
    worst = 0.0
    for q in (0.05, 0.1, 0.143, 0.189, 0.25, 0.5, 0.75, 1.0):
        m = 512; n = int(round(m / q))
        lam, qq, _, _ = esd(gaussian(n, m, 1.0, rng))
        s = estimate_sigma2(lam, qq).sigma2
        worst = max(worst, abs(s - 1.0))
        print(f"   q={q:5.3f}  sigma2={s:.4f}")
    good = worst < 0.02; ok &= good
    print(f"   najveca greska {worst*100:.2f}%  {'OK' if good else 'PAD'}")

    # ---- 5. sweep po p: sigma² mora ostati RAVAN
    print("\n5. sweep po p  (test koji naivni estimator obara)")
    m = 512; q = 0.189; n = int(round(m / q))
    print(f"   {'p':>5} {'iterativni':>11} {'naivni':>9} {'naivni gresi':>13} {'metod':>11}")
    worst = 0.0
    for p in (0.01, 0.05, 0.10, 0.20, 0.30):
        k = int(round(p * m))
        W = with_spikes(n, m, 1.0, k, theta=6.0 * np.sqrt(q), rng=rng)
        lam, qq, _, _ = esd(W)
        f = estimate_sigma2(lam, qq)
        nv = naive_sigma2(lam, qq)
        worst = max(worst, abs(f.sigma2 - 1.0))
        print(f"   {p:5.2f} {f.sigma2:11.4f} {nv:9.4f} {100*(nv-1):12.1f}% {f.sigma_method if hasattr(f,'sigma_method') else f.method:>11}")
    good = worst < 0.03; ok &= good
    print(f"   najveca greska iterativnog {worst*100:.2f}%  {'OK' if good else 'PAD'}")
    return ok


if __name__ == "__main__":
    print("K6 -- mp_sigma_estimate, test paket iz §3.1")
    print("=" * 70)
    good = run()
    print("\n" + "=" * 70)
    print("K6 KAPIJA: ZATVORENA" if good else "K6 KAPIJA: OTVORENA")
    raise SystemExit(0 if good else 1)
