"""SAE -- recnik pojmova, [SPEC] Sec 6, sa ispravkama I-12.

Zasto mrtvi feature-i eksplodiraju (§6 plana, faza 6.3): kod TopK SAE-a
feature koji nikad ne pobedi u top-k takmicenju dobija NULA gradijenta i
nikad se ne oporavi -- smrt je apsorbujuce stanje. Mere, po stvarnoj
delotvornosti:
  1. manji expansion na malim podacima (profil, ne kod)
  2. AuxK: pomocni gubitak koji top-k_aux MRTVIH feature-a tera da
     rekonstruisu ostatak greske -> ozivljavanje gradijentom, ne re-init
  3. resampling kako treba: dekoder na primer sa VELIKIM gubitkom, enkoder
     = transponovani dekoder, Adam stanje za te parametre RESETOVANO
  4. dekoder inicijalizovan iz podataka, jedinicne norme kolona
Sve je cista funkcija (AZ-1): podaci unutra, recnik i metrike napolje;
budzet dolazi kao povratni poziv.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SAEConfig:
    d: int
    expansion: int = 4
    k: int = 32
    k_aux: int = 256
    aux_coef: float = 1 / 32
    lr: float = 3e-4
    batch: int = 256
    steps: int = 5000
    dead_after: int = 2000          # koraka bez aktivacije -> mrtav
    resample_every: int = 1000
    seed: int = 0

    @property
    def m(self) -> int:
        return self.d * self.expansion


@dataclass
class SAEResult:
    W_enc: np.ndarray
    b_enc: np.ndarray
    W_dec: np.ndarray
    b_dec: np.ndarray
    fvu: float
    l0: float
    dead_frac: float
    history: list = field(default_factory=list)
    stopped: str = "budzet"


def train_sae(sample_batch, cfg: SAEConfig, init_data: np.ndarray,
              on_progress=None) -> SAEResult:
    """sample_batch(n) -> np.ndarray (n, d), vec normalizovan Welford statistikom.

    on_progress(step, metrics) -> "continue" | "stop"  ([PSE-A] 8.1: plato
    SKRACUJE zivot, divergencija ga produzava ili deli -- odluka nije ovde).
    """
    import torch
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    d, m = cfg.d, cfg.m

    # dekoder iz podataka, jedinicne norme; enkoder = transponovani dekoder
    idx = rng.choice(init_data.shape[0], m, replace=init_data.shape[0] < m)
    Wd = torch.tensor(init_data[idx], dtype=torch.float32)
    Wd = Wd / Wd.norm(dim=1, keepdim=True).clamp_min(1e-8)          # (m, d)
    W_dec = torch.nn.Parameter(Wd.clone())
    W_enc = torch.nn.Parameter(Wd.T.clone())                         # (d, m)
    b_enc = torch.nn.Parameter(torch.zeros(m))
    b_dec = torch.nn.Parameter(torch.tensor(init_data.mean(0), dtype=torch.float32))
    params = [W_enc, b_enc, W_dec, b_dec]
    opt = torch.optim.Adam(params, lr=cfg.lr)
    last_fired = np.zeros(m, dtype=np.int64)
    hist = []
    stopped = "budzet"

    def encode(x):
        pre = (x - b_dec) @ W_enc + b_enc
        top = torch.topk(pre, cfg.k, dim=-1)
        z = torch.zeros_like(pre).scatter_(-1, top.indices, torch.relu(top.values))
        return z, pre

    for step in range(1, cfg.steps + 1):
        x = torch.tensor(sample_batch(cfg.batch), dtype=torch.float32)
        z, pre = encode(x)
        xh = z @ W_dec + b_dec
        err = x - xh
        loss = (err ** 2).sum(-1).mean()

        fired = (z > 0).any(0).numpy()
        last_fired[fired] = step
        dead = (step - last_fired) > cfg.dead_after
        if dead.any() and cfg.aux_coef > 0:                 # AuxK
            dmask = torch.tensor(dead)
            pre_d = pre.masked_fill(~dmask, -1e9)
            ka = int(min(cfg.k_aux, dead.sum()))
            ta = torch.topk(pre_d, ka, dim=-1)
            za = torch.zeros_like(pre).scatter_(-1, ta.indices, torch.relu(ta.values))
            aux = ((err.detach() - za @ W_dec) ** 2).sum(-1).mean()
            loss = loss + cfg.aux_coef * aux

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():                               # jedinicne norme dekodera
            W_dec.div_(W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))

        if step % cfg.resample_every == 0 and dead.any():
            _resample(W_enc, b_enc, W_dec, opt, dead, sample_batch, encode, b_dec, rng)
            last_fired[dead] = step

        if step % 250 == 0 or step == cfg.steps:
            with torch.no_grad():
                xv = torch.tensor(sample_batch(2048), dtype=torch.float32)
                zv, _ = encode(xv)
                ev = xv - (zv @ W_dec + b_dec)
                fvu = float((ev ** 2).sum() / ((xv - xv.mean(0)) ** 2).sum())
                l0 = float((zv > 0).sum(-1).float().mean())
            met = {"step": step, "fvu": fvu, "l0": l0,
                   "dead_frac": float(((step - last_fired) > cfg.dead_after).mean())}
            hist.append(met)
            if on_progress and on_progress(step, met) == "stop":
                stopped = "plato"
                break

    with torch.no_grad():
        xv = torch.tensor(sample_batch(4096), dtype=torch.float32)
        zv, _ = encode(xv)
        ev = xv - (zv @ W_dec + b_dec)
        fvu = float((ev ** 2).sum() / ((xv - xv.mean(0)) ** 2).sum())
        l0 = float((zv > 0).sum(-1).float().mean())
    return SAEResult(W_enc.detach().numpy(), b_enc.detach().numpy(),
                     W_dec.detach().numpy(), b_dec.detach().numpy(),
                     fvu, l0, float(((cfg.steps - last_fired) > cfg.dead_after).mean()),
                     hist, stopped)


def _resample(W_enc, b_enc, W_dec, opt, dead, sample_batch, encode, b_dec, rng):
    """Mrtvi feature -> pravac PRIMERA SA VELIKIM GUBITKOM, i reset Adam stanja.

    Naivan nasumicni re-init ih samo ponovo ubije: nov pravac bez gradijenta u
    starom Adam momentu dobija korak skaliran tudjom istorijom.
    """
    import torch
    with torch.no_grad():
        x = torch.tensor(sample_batch(8192), dtype=torch.float32)
        z, _ = encode(x)
        e = ((x - (z @ W_dec + b_dec)) ** 2).sum(-1)
        p = (e / e.sum()).numpy().astype(np.float64)
        p /= p.sum()
        di = np.nonzero(dead)[0]
        pick = rng.choice(x.shape[0], len(di), replace=len(di) > x.shape[0], p=p)
        v = x[pick] - b_dec
        v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-8)
        di_t = torch.tensor(di)
        W_dec[di_t] = v
        alive_norm = W_enc.norm(dim=0)[~torch.tensor(dead)].mean() if (~dead).any() else 1.0
        W_enc[:, di_t] = v.T * alive_norm * 0.2
        b_enc[di_t] = 0.0
        for prm in (W_enc, b_enc, W_dec):
            st = opt.state.get(prm)
            if not st:
                continue
            for key in ("exp_avg", "exp_avg_sq"):
                if prm is W_enc:
                    st[key][:, di_t] = 0.0
                else:
                    st[key][di_t] = 0.0


def superposition_profile(res: SAEResult, sample_batch, cfg: SAEConfig,
                          window: int = 5, eps: float = 0.05, delta: float = 0.10,
                          community_tag: str = "", rel_thr: float = 0.5) -> dict:
    """[SPEC] Sec 6 export: §9 junction. SAE feature-i postaju stable_features,
    izmerene stope ko-aktivacije postaju epsilon_profile, efektivni broj
    feature-a postaje N_eff. Od prvog zavrsenog L3 run-a [PSE §9] prestaje da
    bude specifikacija i postaje merenje."""
    import torch
    x = torch.tensor(sample_batch(8192), dtype=torch.float32)
    W_enc = torch.tensor(res.W_enc); b_enc = torch.tensor(res.b_enc)
    b_dec = torch.tensor(res.b_dec)
    with torch.no_grad():
        pre = (x - b_dec) @ W_enc + b_enc
        top = torch.topk(pre, cfg.k, dim=-1)
        z = torch.zeros_like(pre).scatter_(-1, top.indices, torch.relu(top.values))
    Z = z.numpy()
    # I-22: "aktivan" u jedinicama SOPSTVENE tipicne jacine feature-a. TopK sa k
    # iznad prave retkosti puni visak slotova sitnim aktivacijama koje prate
    # glavne; one su stvarne u kodu SAE-a, ali ne u podacima, i preplave profil
    # (izmereno: k=4 na retkosti 3.4 -> 1/173 nezavisnih "stabilno"). Prag po
    # feature-u = rel_thr × 90. percentil njegovih nenultih aktivacija.
    typ = np.full(Z.shape[1], np.inf)
    live = (Z > 0).any(0)
    if live.any():
        pos = np.where(Z[:, live] > 0, Z[:, live], np.nan)
        typ[live] = np.nanpercentile(pos, 90, axis=0)
    A = ((Z > 0) & (Z > rel_thr * typ[None, :])).astype(np.float32)
    freq = A.mean(0)
    alive = freq > 0
    # ko-aktivacija u prozoru od `window` uzastopnih uzoraka
    n = (A.shape[0] // window) * window
    W = A[:n].reshape(-1, window, A.shape[1]).max(1)
    co = (W.T @ W) / max(W.shape[0], 1)
    fw = W.mean(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = np.where(fw[:, None] > 0, co / fw[:, None], 0.0)   # P(j | i)
    # I-21: VISAK iznad nezavisnosti, ne sirova stopa. U prozoru od `window`
    # uzoraka sa k aktivnih od m, SLUCAJNA stopa bilo kog para je vec ~k·window/m
    # (za 4, 5, 256 to je 0.078 > eps), pa bi apsolutni prag svaki feature
    # proglasio superponiranim. Peti put isti obrazac (I-15, 16, 17, 20).
    cond = cond - fw[None, :]
    np.fill_diagonal(cond, 0.0)
    # ...i ZNACAJNOST: max preko m-1 sumnih procena prelazi eps slucajno
    # (SE ~ sqrt(p(1-p)/n_i) ~ 0.024 za tipicne vrednosti). Par je superponiran
    # samo ako je visak i smislen (> eps) i znacajan (z > Bonferroni za m poredjenja).
    n_i = W.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.sqrt(np.maximum(fw * (1 - fw), 1e-12)[None, :] / np.maximum(n_i, 1)[:, None])
        z = np.where(se > 0, cond / se, 0.0)
    from scipy.stats import norm
    z_crit = float(norm.isf(0.05 / max(A.shape[1], 2)))
    # I-23: normalna aproksimacija ne vazi na malim brojevima. Sa ~7 pojavljivanja
    # feature-a JEDNO zajednicko pojavljivanje daje P(j|i)=0.14 i z~3.7 -> "znacajno".
    # Zato: par se broji tek uz >= MIN_CO zajednickih pojavljivanja, a feature se
    # uopste ocenjuje tek uz >= MIN_N svojih. Ostali su "nedovoljno podataka" --
    # ni stabilni ni superponirani.
    MIN_N, MIN_CO = 20, 5
    counts = W.T @ W
    sig = (cond > eps) & (z > z_crit) & (counts >= MIN_CO)
    evaluable = alive & (n_i >= MIN_N)
    stable = [int(i) for i in np.nonzero(evaluable)[0] if not sig[i].any()]
    p = freq[alive] / freq[alive].sum()
    n_eff = float(np.exp(-(p * np.log(p)).sum())) if p.size else 0.0
    exceptional = float(sig[alive][:, alive].mean()) if alive.any() else 0.0
    return {"n_eff": n_eff, "n_alive": int(alive.sum()), "n_total": int(A.shape[1]),
            "stable_features": stable, "n_stable": len(stable), "z_crit": z_crit,
            "n_evaluable": int(evaluable.sum()),
            "epsilon": eps, "delta": delta, "exceptional_share": exceptional,
            "delta_stable": exceptional < delta, "community_tag": community_tag,
            "coactivation_top": _top_pairs(cond, 50)}


def _top_pairs(cond: np.ndarray, n: int) -> list:
    iu = np.dstack(np.unravel_index(np.argsort(-cond.ravel())[:n], cond.shape))[0]
    return [(int(i), int(j), float(cond[i, j])) for i, j in iu if cond[i, j] > 0]
