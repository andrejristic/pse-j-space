"""Reorganizacija informacije pod kvantizacijom -- §4.4.

Pitanje nije "koliko je spike-ova nestalo" nego "sta se desilo sa informacijom
kad je sum porastao": da li je pravac ugasen, zarotiran, pocepan ili je energija
presla na suseda -- i sta od podprostora ostaje.

Posle merenja iz §13.3 ovo je GLAVNI alat, ne pomocni: nijedna Qwen matrica
nije spiked, pa BBP inventar nema na cemu da radi, dok glavni uglovi,
klasteri i energetski bilans rade u svakom rezimu.

Kljucna odluka (§4.4): prati se PODPROSTOR, ne pojedinacni sopstveni vektor.
Davis-Kahan: rotacija invarijantnog podprostora je ogranicena sa ||E||/gap, pa
kad je razmak mali pojedinacni vektor postaje besmislen a podprostor ostaje
stabilan. Uparivanje po rangu gresi bas tamo gde je najvaznije.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh, svd


def _tall(W: np.ndarray) -> np.ndarray:
    """Orijentacija n×m sa n >= m; Gram je tada m×m (I-3)."""
    A = np.asarray(W)
    return A if A.shape[0] >= A.shape[1] else A.T


def spectral_norm(M: np.ndarray, iters: int = 30, seed: int = 0) -> float:
    """||M||_2 stepenom iteracijom -- nekoliko matvec-ova umesto dekompozicije."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(M.shape[1]).astype(np.float32)
    v /= np.linalg.norm(v)
    s = 0.0
    for _ in range(iters):
        u = M @ v
        nu = np.linalg.norm(u)
        if nu == 0:
            return 0.0
        u /= nu
        v = M.T @ u
        s = np.linalg.norm(v)
        if s == 0:
            return 0.0
        v /= s
    return float(s)


def dominant_subspace(W: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k sopstvenih parova Gram matrice (1/n)·WᵀW, opadajuce.

    Parcijalna dekompozicija: scipy eigh(subset_by_index) umesto pune.
    Za heavy-tailed matrice ovo nije "spike podprostor" nego DOMINANTNI
    podprostor -- ista mera, sira primena.
    """
    B = _tall(W)
    n, m = B.shape
    k = int(min(k, m))
    G = (B.T @ B).astype(np.float32, copy=False)
    lam, U = eigh(G, subset_by_index=[m - k, m - 1])
    lam = np.ascontiguousarray(lam[::-1]).astype(np.float64) / n
    U = np.ascontiguousarray(U[:, ::-1])
    np.clip(lam, 0.0, None, out=lam)
    return lam, U


def cluster_by_gap(lam: np.ndarray, eps: float) -> list[tuple[int, int]]:
    """Podeli opadajuci niz na klastere: unutrasnji razmaci < eps, spoljni >= eps.

    Davis-Kahan granica ||E||/gap eksplodira unutar klastera, pa se pojedinacni
    vektori tamo ne smeju porediti -- samo podprostor klastera.
    """
    if lam.size == 0:
        return []
    out = []
    start = 0
    for i in range(1, lam.size):
        if (lam[i - 1] - lam[i]) >= eps:
            out.append((start, i))
            start = i
    out.append((start, lam.size))
    return out


def principal_angles(UA: np.ndarray, UB: np.ndarray) -> np.ndarray:
    """Glavni uglovi (radijani) izmedju span(UA) i span(UB), rastuce."""
    s = np.sort(svd(UA.T @ UB, compute_uv=False))[::-1]
    return np.arccos(np.clip(s, -1.0, 1.0))


def subspace_energy_kept(UA: np.ndarray, UB: np.ndarray) -> float:
    """Udeo energije podprostora A koji lezi u B: (1/k)·sum cos²(theta)."""
    s = svd(UA.T @ UB, compute_uv=False)
    return float((s ** 2).sum() / UA.shape[1])


def noise_concentration(D: np.ndarray, U: np.ndarray) -> dict:
    """§4.2 kanal 2: da li kvantizacioni sum gadja naucene pravce jace od proseka.

        ( ||P_dom·D||²_F / k ) / ( ||P_ostalo·D||²_F / (m-k) )

    > 1 znaci da je sum KONCENTRISAN na dominantne pravce -- sto je posledica
    skaliranja po bloku od 32 tezine kod Q4_K (veci blokovi -> veci korak).
    """
    Dt = _tall(D)
    m = Dt.shape[1]
    k = U.shape[1]
    tot = float((Dt ** 2).sum())
    dom = float(((Dt @ U) ** 2).sum())
    rest = max(tot - dom, 0.0)
    per_dom = dom / max(k, 1)
    per_rest = rest / max(m - k, 1)
    return {"total": tot, "dominant": dom, "rest": rest,
            "ratio": per_dom / per_rest if per_rest > 0 else float("inf")}


@dataclass
class ReorgReport:
    name: str
    type_ref: str
    type_test: str
    k: int
    q: float
    eps_lambda: float
    # --- podprostor
    energy_kept: float               # (1/k)·sum cos²  -- "sta ostaje"
    angles_deg: np.ndarray = field(repr=False, default=None)
    n_angles_under_10deg: int = 0
    # --- klasteri
    n_clusters_ref: int = 0
    largest_cluster: int = 0
    # --- energetski bilans (u lambda jedinicama)
    e_total_ref: float = 0.0
    e_total_test: float = 0.0
    e_head_ref: float = 0.0
    e_head_test: float = 0.0
    e_body_ref: float = 0.0
    e_body_test: float = 0.0
    # --- sum
    noise_ratio: float = 0.0
    rel_rmse: float = 0.0


def compare(W_ref: np.ndarray, W_test: np.ndarray, name: str,
            type_ref: str, type_test: str, k: int = 50,
            head_cut: float | None = None) -> ReorgReport:
    """Pun §4.4 nad jednim tenzorom izmedju dve preciznosti.

    Referenca je UVEK f16 (I-14), nikad susedna precka -- poredjenje sa susednom
    akumulira drift, poredjenje sa f16 daje monotonu putanju.
    """
    Bref, Btest = _tall(W_ref), _tall(W_test)
    if Bref.shape != Btest.shape:
        raise ValueError(f"{name}: oblici se ne poklapaju {Bref.shape} vs {Btest.shape}")
    n, m = Bref.shape
    D = Bref - Btest

    lam_r, Ur = dominant_subspace(Bref, k)
    lam_t, Ut = dominant_subspace(Btest, k)

    # Weyl: perturbacija sopstvenih vrednosti Gram matrice u lambda jedinicama
    nD = spectral_norm(D)
    nB = spectral_norm(Bref)
    eps = (2.0 * nD * nB + nD * nD) / n

    ang = principal_angles(Ur, Ut)
    clusters = cluster_by_gap(lam_r, eps)

    # energetski bilans: glava = iznad head_cut, telo = ispod
    cut = head_cut if head_cut is not None else float(lam_r[-1])
    tot_r = float((Bref ** 2).sum()) / n
    tot_t = float((Btest ** 2).sum()) / n
    head_r = float(lam_r[lam_r > cut].sum())
    head_t = float(lam_t[lam_t > cut].sum())

    nc = noise_concentration(D, Ur)
    rel = float(np.sqrt((D ** 2).mean()) / Bref.std())

    return ReorgReport(
        name=name, type_ref=type_ref, type_test=type_test, k=k, q=m / n,
        eps_lambda=eps,
        energy_kept=subspace_energy_kept(Ur, Ut),
        angles_deg=np.degrees(ang),
        n_angles_under_10deg=int((np.degrees(ang) < 10.0).sum()),
        n_clusters_ref=len(clusters),
        largest_cluster=max((b - a) for a, b in clusters) if clusters else 0,
        e_total_ref=tot_r, e_total_test=tot_t,
        e_head_ref=head_r, e_head_test=head_t,
        e_body_ref=tot_r - head_r, e_body_test=tot_t - head_t,
        noise_ratio=nc["ratio"], rel_rmse=rel,
    )
