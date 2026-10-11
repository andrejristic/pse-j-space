"""ESD, iterativni bulk-only sigma², klasifikacija rezima, alfa.

Zatvara rupu iz §3.1 [SPEC]-a: "robust, median-based" nije formula.

Kljucni problem: medijana empirijskog spektra NIJE medijana MP zakona. Ako je
udeo spike-ova p, svi su IZNAD bulk-a, pa vazi

    F_mix(x) = (1-p)·F_MP(x)   =>   F_MP(medijana) = 0.5/(1-p)

dakle citas 0.5/(1-p) kvantil, ne 0.5. Za p=0.10 i q=0.189 to je +7.9% na
sigma², a ivica nasledjuje istu pristrasnost. Posto se p razlikuje izmedju
precki lestvice, naivni estimator kontaminira bas ono poredjenje zbog kog
eksperiment postoji (§4.2, kanal 1).

Resenje: fiksna tacka po (sigma², p), sa KS-fit rezervom za p > 0.25.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import mp

# metodoloske konstante -- NISU u MachineProfile (I-1): definisu sta se meri
MP_TOLERANCE = 0.05
ESD_BINS = 200
TAIL_MIN_COUNT = 12
ALPHA_TRAINED = (2.0, 4.0)
ALPHA_RANDOMLIKE = 6.0
# I-20: Kolmogorovljeva granica na 5 %, u jedinicama sopstvene nesigurnosti
# (KS·sqrt(n_bulk)). Posto je sigma² fitovan, stvarna nulta raspodela je uza,
# pa je prag blag -- ono sto ga predje sigurno nije MP.
MP_FIT_KS_N = 1.36

RANDOM_LIKE = "random_like"
SPIKED = "spiked"
HEAVY_TAILED = "heavy_tailed"
MIXED = "mixed"


# ---------------------------------------------------------------- ESD
def esd(W: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    """Sopstvene vrednosti od (1/n)·WᵀW preko Gram matrice MANJE dimenzije (I-3).

    Vraca (lam rastuce, q, n, m) sa n >= m.
    fp32 (I-4): Gram kvadrira kondicioni broj pa najmanje vrednosti umeju
    izaci blago negativne -> kleše se na 0.
    """
    A = np.asarray(W)
    if A.ndim != 2:
        raise ValueError("ESD se racuna samo za 2-D tenzore")
    n, m = max(A.shape), min(A.shape)
    B = A if A.shape[0] == n else A.T          # B je n×m
    G = (B.T @ B).astype(np.float32, copy=False)
    lam = np.linalg.eigvalsh(G).astype(np.float64) / n
    np.clip(lam, 0.0, None, out=lam)
    return lam, m / n, n, m


# ---------------------------------------------------------------- sigma²
@dataclass
class SigmaFit:
    sigma2: float
    p: float                  # udeo spike-ova iznad ivice
    edge: float
    iterations: int
    method: str               # "median-iter" ili "ks-fit"
    converged: bool


def estimate_sigma2(lam: np.ndarray, q: float, tol: float = MP_TOLERANCE,
                    max_iter: int = 40, atol: float = 1e-5) -> SigmaFit:
    """Iterativna bulk-only procena sigma² (§3.1).

    Cirkularnost (treba p za sigma², treba sigma² za p) se resava fiksnom
    tackom sa relaksacijom 0.5 -- bez relaksacije ume da oscilira izmedju
    dve vrednosti kad je gustina odmah iznad ivice strma.
    """
    med = float(np.median(lam))
    p = 0.0
    sigma2 = edge = 0.0
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        pc = min(p, 0.45)                      # 0.5/(1-p) mora ostati < 1
        sigma2 = med / mp.quantile(0.5 / (1.0 - pc), q)
        edge = sigma2 * (1.0 + np.sqrt(q)) ** 2
        p_new = float(np.mean(lam > edge * (1.0 + tol)))
        if abs(p_new - p) < atol:
            p = p_new
            converged = True
            break
        p = 0.5 * p + 0.5 * p_new
    fit = SigmaFit(sigma2, p, edge, it, "median-iter", converged)

    # Medijana pociva na pretpostavci da spike-ovi samo ODUZIMAJU masu iz
    # bulk-a, a da ga inace ne diraju. To vazi dok je k = O(1). Kad udeo
    # poraste, sam bulk se deformise i identitet 0.5/(1-p) prestaje da vazi --
    # izmereno: na p=0.20 iterativni prekoriguje za -5.8%, dok KS-fit pogadja
    # na 0.3%. Zato se metod NE bira pragom nego takmicenjem: oba se racunaju
    # i pobedjuje onaj sa manjim KS rastojanjem na sopstvenom bulk-u.
    if p > 0.02:
        alt = _ks_fit(lam, q, fit)
        if _ks_stat(lam, q, alt.sigma2) < _ks_stat(lam, q, fit.sigma2):
            fit = alt
    return fit


def _ks_stat(lam: np.ndarray, q: float, s2: float) -> float:
    """KS rastojanje empirijskog bulk-a od MP zakona sa datim sigma²."""
    if s2 <= 0:
        return 1e9
    edge = s2 * (1.0 + np.sqrt(q)) ** 2
    bulk = lam[lam <= edge]
    if bulk.size < 32:
        return 1e9
    ecdf = np.arange(1, bulk.size + 1) / bulk.size
    tcdf = mp.cdf(bulk, q, s2)
    tcdf = tcdf / max(tcdf[-1], 1e-12)
    return float(np.max(np.abs(ecdf - tcdf)))


def _ks_fit(lam: np.ndarray, q: float, seed: SigmaFit) -> SigmaFit:
    """Fituj sigma² minimizacijom KS rastojanja na ODSECENOM bulk-u."""
    from scipy.optimize import minimize_scalar

    def ks(s2: float) -> float:
        return _ks_stat(lam, q, s2)

    lo, hi = seed.sigma2 * 0.4, seed.sigma2 * 1.6
    r = minimize_scalar(ks, bounds=(lo, hi), method="bounded",
                        options={"xatol": seed.sigma2 * 1e-4})
    s2 = float(r.x)
    edge = s2 * (1.0 + np.sqrt(q)) ** 2
    p = float(np.mean(lam > edge * (1.0 + MP_TOLERANCE)))
    return SigmaFit(s2, p, edge, seed.iterations, "ks-fit", bool(r.success))


# ---------------------------------------------------------------- alfa
def _hill(x: np.ndarray) -> float:
    xmin = x.min()
    ssum = np.log(x / xmin).sum()
    return 1.0 + x.size / ssum if ssum > 0 else float("nan")


def _pareto_ks(x: np.ndarray, alpha: float) -> float:
    """KS izmedju empirijskog repa i fitovanog Pareto zakona."""
    xmin = x.min()
    ecdf = np.arange(1, x.size + 1) / x.size
    tcdf = 1.0 - (x / xmin) ** (1.0 - alpha)
    return float(np.max(np.abs(ecdf - tcdf)))


def powerlaw_alpha(tail: np.ndarray, n_boot: int = 200,
                   rng: np.random.Generator | None = None,
                   optimize_xmin: bool = True) -> tuple[float, float, float, float]:
    """Hill estimator sa bootstrap intervalom (I-7) i izborom xmin po KS.

    I-16: xmin se NE uzima kao MP ivica. Kod trenirane matrice rep zauzima i
    preko polovine spektra (izmereno na Qwen-7B: 6-54%), pa je MP ivica
    proizvoljna tacka duboko u telu raspodele, a Hill na njoj sistematski
    potcenjuje alfa. Standardni postupak (Clauset-Shalizi-Newman) bira xmin
    minimizacijom KS rastojanja izmedju empirijskog repa i fitovanog Pareto
    zakona. Vraca (alpha, lo, hi, xmin).
    """
    t = np.sort(np.asarray(tail, dtype=np.float64))
    t = t[t > 0]
    if t.size < TAIL_MIN_COUNT:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rng = rng or np.random.default_rng(0)

    best = (t, _hill(t))
    if optimize_xmin and t.size >= 4 * TAIL_MIN_COUNT:
        # kandidati: log-ravnomerno po broju zadrzanih tacaka, od svih do TAIL_MIN_COUNT
        keeps = np.unique(np.round(np.geomspace(TAIL_MIN_COUNT, t.size, 40)).astype(int))
        best_ks = np.inf
        for k in keeps:
            x = t[-k:]
            a = _hill(x)
            if not np.isfinite(a) or a <= 1.0:
                continue
            d = _pareto_ks(x, a)
            if d < best_ks:
                best_ks, best = d, (x, a)

    x, a = best
    boots = np.array([_hill(np.sort(rng.choice(x, x.size, replace=True)))
                      for _ in range(n_boot)])
    boots = boots[np.isfinite(boots)]
    if boots.size < 20:
        return a, float("nan"), float("nan"), float(x.min())
    return (a, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)),
            float(x.min()))


# ---------------------------------------------------------------- rezim
def classify_regime(lam: np.ndarray, fit: SigmaFit, q: float, m: int) -> str:
    """I-6: alfa i BBP su MEDJUSOBNO ISKLJUCIVI.

    [SPEC] Sec 4 trpa pod "tail" dve nespojive pojave: diskretne BBP outlier-e
    (niskorangna naucena struktura) i debeo BULK sa eksponentom alfa
    (heavy-tailed self-regularization). Fitovati power law na 12 izolovanih
    spike-ova je kategorijalna greska.
    """
    thr = fit.edge * (1.0 + MP_TOLERANCE)
    tail = lam[lam > thr]
    if tail.size == 0:
        return RANDOM_LIKE
    frac = tail.size / m
    decades = np.log10(tail.max() / thr) if thr > 0 else 0.0
    if frac < 0.02:
        return SPIKED
    if frac >= 0.02 and decades >= 0.5:
        return HEAVY_TAILED
    return MIXED


@dataclass
class LayerHealth:
    name: str
    shape: tuple[int, int]
    ggml_type: str            # I-8: kovarijata, ne ukras
    role: str
    role_class: str
    depth: float | None
    q: float
    sigma2: float
    sigma_method: str
    edge: float
    tail_frac: float
    n_tail: int
    regime: str
    alpha: float
    alpha_lo: float
    alpha_hi: float
    alpha_xmin: float
    stable_rank: float
    spectral_norm: float
    frob2_total: float
    frob2_tail: float
    esd_hist: np.ndarray = field(repr=False, default=None)
    esd_edges: np.ndarray = field(repr=False, default=None)
    mp_ks_n: float = float("nan")     # KS·sqrt(n_bulk) bulk-a protiv MP (I-20)
    mp_fit_ok: bool = False           # da li MP uopste opisuje bulk


def esd_histogram(lam: np.ndarray, bins: int = ESD_BINS) -> tuple[np.ndarray, np.ndarray]:
    """Histogram sa LOG-razmaknutim binovima.

    Linearni binovi na teskorepom spektru sabiju ceo bulk u prvih nekoliko
    binova, a rep razvuku kroz ostatak prazan -- histogram tada ne pokazuje
    upravo ono zbog cega postoji. Nulte vrednosti (kleste posle fp32 Grama)
    ulaze u prvi bin.
    """
    lam = np.asarray(lam, dtype=np.float64)
    top = float(lam.max()) if lam.size else 1.0
    pos = lam[lam > 0]
    lo = float(pos.min()) if pos.size else top * 1e-8
    lo = max(lo, top * 1e-8)
    edges = np.geomspace(lo, top * (1 + 1e-12), bins + 1)
    hist, _ = np.histogram(np.clip(lam, lo, None), bins=edges)
    return hist, edges


def layer_health_from_lam(lam: np.ndarray, q: float, n: int, m: int, name: str,
                          ggml_type: str, role: str, role_class: str,
                          depth: float | None) -> LayerHealth:
    """Isto kao layer_health, ali iz VEC izracunatih sopstvenih vrednosti.

    qdiff racuna punu dekompoziciju (sa vektorima) za reorganizaciju; ESD
    statistika se izvodi iz istih vrednosti umesto da se Gram racuna dvaput.
    """
    lam = np.sort(np.asarray(lam, dtype=np.float64))
    np.clip(lam, 0.0, None, out=lam)
    fit = estimate_sigma2(lam, q)
    thr = fit.edge * (1.0 + MP_TOLERANCE)
    tail = lam[lam > thr]
    regime = classify_regime(lam, fit, q, m)

    # alfa se racuna SAMO gde ima smisla (I-6)
    if regime in (HEAVY_TAILED, MIXED):
        a, alo, ahi, xmin = powerlaw_alpha(tail)
    else:
        a = alo = ahi = xmin = float("nan")

    hist, edges = esd_histogram(lam)
    n_bulk = int((lam <= fit.edge).sum())
    ks_n = _ks_stat(lam, q, fit.sigma2) * np.sqrt(max(n_bulk, 1))
    return LayerHealth(
        name=name, shape=(n, m), ggml_type=ggml_type, role=role,
        role_class=role_class, depth=depth, q=q,
        sigma2=fit.sigma2, sigma_method=fit.method, edge=fit.edge,
        tail_frac=tail.size / m, n_tail=int(tail.size), regime=regime,
        alpha=a, alpha_lo=alo, alpha_hi=ahi, alpha_xmin=xmin,
        stable_rank=float(lam.sum() / lam.max()) if lam.max() > 0 else 0.0,
        spectral_norm=float(np.sqrt(lam.max() * n)),
        frob2_total=float(lam.sum()), frob2_tail=float(tail.sum()),
        esd_hist=hist, esd_edges=edges,
        mp_ks_n=float(ks_n), mp_fit_ok=bool(ks_n < MP_FIT_KS_N),
    )


def layer_health(W: np.ndarray, name: str, ggml_type: str, role: str,
                 role_class: str, depth: float | None) -> LayerHealth:
    lam, q, n, m = esd(W)
    return layer_health_from_lam(lam, q, n, m, name, ggml_type, role, role_class, depth)
