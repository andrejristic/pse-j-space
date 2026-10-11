"""Marchenko-Pastur: gustina, CDF, kvantilna funkcija.

Za sopstvene vrednosti od (1/n)·WᵀW, gde je W n×m sa n >= m i q = m/n <= 1:

    a = sigma2·(1-sqrt(q))²        b = sigma2·(1+sqrt(q))²
    f(x) = sqrt((b-x)(x-a)) / (2·pi·q·sigma2·x)      za x u [a,b]

CDF se ne racuna na mrezi po x -- gustina ima sqrt-singularitet na oba kraja,
sto kvari kvadraturu tacno tamo gde nam treba tacnost. Umesto toga ide smena

    x(t) = (1+q) - 2·sqrt(q)·cos(t),   t u [0, pi]

koja daje glatku podintegralnu funkciju bez singulariteta:

    f dx = 2·sin²(t) / (pi·x(t)) dt

Kvantilna funkcija se dobija interpolacijom inverza; kesira se po q, jer je
u L1 sweep-u isto q za desetine matrica (§3.1).
"""
from __future__ import annotations

import numpy as np

_CACHE: dict[tuple[float, int], tuple[np.ndarray, np.ndarray]] = {}
_GRID = 8192


def support(q: float, sigma2: float = 1.0) -> tuple[float, float]:
    r = np.sqrt(q)
    return sigma2 * (1 - r) ** 2, sigma2 * (1 + r) ** 2


def pdf(x: np.ndarray, q: float, sigma2: float = 1.0) -> np.ndarray:
    a, b = support(q, sigma2)
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    m = (x > a) & (x < b)
    out[m] = np.sqrt((b - x[m]) * (x[m] - a)) / (2 * np.pi * q * sigma2 * x[m])
    return out


def _cdf_grid(q: float) -> tuple[np.ndarray, np.ndarray]:
    """(x, F(x)) za sigma2=1, na glatkoj t-mrezi. Kesirano po q."""
    key = (round(float(q), 12), _GRID)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    t = np.linspace(0.0, np.pi, _GRID)
    r = np.sqrt(q)
    x = (1.0 + q) - 2.0 * r * np.cos(t)
    # Kod q = 1 je a = 0, pa je x(0) = 0 i izraz je oblika 0/0 -- ali je
    # singularitet UKLONJIV: sin²t / x -> (1 + cos t)/2, dakle integrand -> 2/pi.
    # Kvadratne matrice (attn_q, attn_output) sve imaju q = 1, pa ovo nije rubni slucaj.
    with np.errstate(divide="ignore", invalid="ignore"):
        integrand = 2.0 * np.sin(t) ** 2 / (np.pi * x)
    bad = ~np.isfinite(integrand)
    if bad.any():
        integrand[bad] = (2.0 / np.pi) if abs(q - 1.0) < 1e-12 else 0.0
    # kumulativna trapezna integracija po t
    F = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(t))])
    F /= F[-1]                       # normalizacija gasi rezidualnu gresku kvadrature
    _CACHE[key] = (x, F)
    return x, F


def cdf(x: np.ndarray, q: float, sigma2: float = 1.0) -> np.ndarray:
    xs, F = _cdf_grid(q)
    return np.interp(np.asarray(x, dtype=np.float64) / sigma2, xs, F, left=0.0, right=1.0)


def quantile(p: float | np.ndarray, q: float, sigma2: float = 1.0) -> float | np.ndarray:
    """Inverz MP CDF. p mora biti u (0,1)."""
    xs, F = _cdf_grid(q)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
    out = np.interp(p, F, xs) * sigma2
    return float(out) if out.ndim == 0 else out


def median(q: float, sigma2: float = 1.0) -> float:
    """M(q) iz §3.1 -- medijana MP zakona sa sigma2=1, skalirana."""
    return float(quantile(0.5, q, sigma2))


def bbp_threshold(q: float) -> float:
    """Spike postoji <=> l > 1 + sqrt(q).  ([SPEC] dopuna, §4.3)"""
    return 1.0 + np.sqrt(q)


def bbp_invert(lam: np.ndarray, q: float, sigma2: float) -> np.ndarray:
    """Iz posmatrane sopstvene vrednosti u jacinu signala l (§4.3).

        x = lam/sigma2
        l = [ (1 + x - q) + sqrt( (1 + x - q)² - 4x ) ] / 2

    Vrednosti ispod praga daju kompleksan koren -> vracamo NaN (spike ne postoji).
    """
    x = np.asarray(lam, dtype=np.float64) / sigma2
    A = 1.0 + x - q
    disc = A * A - 4.0 * x
    out = np.full_like(x, np.nan)
    ok = disc >= 0
    out[ok] = 0.5 * (A[ok] + np.sqrt(disc[ok]))
    return out


def bbp_overlap(ell: np.ndarray, q: float) -> np.ndarray:
    """Predvidjeno preklapanje sopstvenog vektora sa pravim pravcem (§4.4):

        |<u_hat, u>|² = (1 - q/(l-1)²) / (1 + q/(l-1))     -> 0 kako l -> 1+sqrt(q)

    Ovo je ono sa cim se MERENA rotacija poredi; visak preko ovoga je
    strukturirana steta, ne sum.
    """
    ell = np.asarray(ell, dtype=np.float64)
    d = ell - 1.0
    out = np.full_like(ell, np.nan)
    ok = d > 0
    num = 1.0 - q / (d[ok] ** 2)
    den = 1.0 + q / d[ok]
    out[ok] = np.clip(num / den, 0.0, 1.0)
    return out
