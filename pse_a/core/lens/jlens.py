"""J-lens -- [SPEC] Sec 7. Kruna: kartografija J-prostora.

Pitanje: od svega sto model predstavlja, sta drzi kao VERBALIZABILNO -- spremno
da kaze ako se pita? Prosecan Jakobijan J_l = E[d h_final / d h_l] je
d_model × d_model po sloju i nikad se ne materijalizuje; redovi W_U·J_l se
procenjuju VJP-ovima po uzorkovanim pravcima nevezivanja, usrednjeno po
promptovima.

Sve ovde je cista funkcija (AZ-1). Runtime se prosledjuje; stanja nema.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

JLENS_SUPPORT_K = 25
JSPACE_VAR_EXPECTED = (0.03, 0.10)
IGNITION_ENTROPY_DROP = 0.5


# ---------------------------------------------------------------- sonde
def probe_tokens(rt, corpus: str, n: int, extra: list[str] = ()) -> list[int]:
    """Najcesci tokeni korpusa + tokeni vezani za zadatak ([SPEC] Sec 7).

    Specijalni (control) tokeni i cisti whitespace se preskacu -- njihovi redovi
    u W_U nose format, ne pojam.
    """
    ids = rt.encode(corpus)
    counts: dict[int, int] = {}
    for i in ids:
        counts[i] = counts.get(i, 0) + 1
    chosen: list[int] = []
    for w in extra:
        for i in rt.encode(w):
            if i not in chosen:
                chosen.append(i)
    for i, _ in sorted(counts.items(), key=lambda kv: -kv[1]):
        if len(chosen) >= n:
            break
        s = rt.decode([i])
        if i in chosen or not s.strip() or s.startswith("<|"):
            continue
        chosen.append(i)
    return chosen[:n]


def jlens_rows(rt, prompts: list[list[int]], layer: int, probe_ids: list[int],
               positions: str = "last") -> np.ndarray:
    """Uzorkovani redovi W_U·J_l, usrednjeno po promptovima. Oblik (P, d).

    VJP ide KROZ final norm, jer lens cita logit = W_U · norm(h_final).
    Uzima se gradijent na ISTOJ poziciji koja se cita (dijagonalni blok J).
    """
    t = rt.torch
    U = rt.W_U[probe_ids]                                   # (P, d)
    acc = None
    n = 0
    for ids in prompts:
        pos_list = [len(ids) - 1] if positions == "last" else range(len(ids))
        for pos in pos_list:
            g = rt.vjp(ids, to_layer=layer, cotangents=U, position=pos,
                       normed=True, batched=True)           # (P, T, d)
            row = g[:, pos, :].detach().to(t.float64).numpy()
            acc = row if acc is None else acc + row
            n += 1
    return acc / max(n, 1)


def lens_read(h: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Linearizovano citanje: logiti sondi = redovi · h, softmax preko sondi."""
    z = rows @ h
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


# ---------------------------------------------------------------- dekompozicija
def gradient_pursuit_nonneg(h: np.ndarray, rows: np.ndarray, k: int = JLENS_SUPPORT_K
                            ) -> np.ndarray:
    """Retka NENEGATIVNA kombinacija redova koja aproksimira h. Vraca coef (P,).

    Pohlepni izbor po korelaciji sa ostatkom, pa NNLS na izabranom nosacu posle
    svakog koraka (orthogonal matching pursuit sa projekcijom na nenegativni
    ortant). Nenegativnost je bitna: J-prostor je KONUS, ne linearni podprostor
    ([SPEC] Sec 7) -- pojam je prisutan ili nije, ne "negativno prisutan".
    """
    P = rows.shape[0]
    norms = np.linalg.norm(rows, axis=1)
    norms[norms == 0] = 1.0
    R = rows / norms[:, None]
    support: list[int] = []
    coef = np.zeros(P)
    r = h.astype(np.float64).copy()
    for _ in range(min(k, P)):
        c = R @ r
        c[support] = -np.inf
        j = int(np.argmax(c))
        if not np.isfinite(c[j]) or c[j] <= 1e-12:
            break
        support.append(j)
        A = rows[support].T                                  # (d, |S|)
        x, _ = nnls(A, h, maxiter=50 * len(support))
        coef[:] = 0.0
        coef[support] = x
        r = h - coef @ rows
    return coef


def jspace_decompose(h: np.ndarray, rows: np.ndarray, k: int = JLENS_SUPPORT_K
                     ) -> tuple[np.ndarray, float, float]:
    """(coef, udeo varijanse, entropija koeficijenata)."""
    coef = gradient_pursuit_nonneg(h, rows, k)
    span = coef @ rows
    frac = float((span @ span) / (h @ h)) if h @ h > 0 else 0.0
    w = coef[coef > 0]
    if w.size:
        p = w / w.sum()
        ent = float(-(p * np.log(p)).sum())
    else:
        ent = 0.0
    return coef, frac, ent


def depth_map(per_layer: dict[int, dict]) -> dict:
    """SENSORY / WORKSPACE / MOTOR + ignition.

    per_layer[L] = {"var_frac": .., "entropy": .., "motor_kl": ..}
    Ignition: entropija dekompozicije pada vise od IGNITION_ENTROPY_DROP unutar
    <= 2 sloja (ostro opredeljenje umesto glatke interpolacije).
    Motor: slojevi gde je lens-citanje vec prakticno jednako izlazu.
    Ako ignition-a nema, to se kaze -- [SPEC] ostavlja otvorenim da li mali
    modeli uopste imaju tro-regionsku strukturu.
    """
    Ls = sorted(per_layer)
    ent = {L: per_layer[L]["entropy"] for L in Ls}
    # Kriterijum trazi pad UNUTAR <= 2 sloja. Sa redim uzorkovanjem skok od dva
    # sloja se ne moze videti, pa negativan ishod nije dokaz odsustva.
    stride = min((b - a for a, b in zip(Ls, Ls[1:])), default=1)
    resolution_ok = stride <= 2
    ignition = None
    for i, L in enumerate(Ls):
        for j in (i + 1, i + 2):
            if j < len(Ls) and Ls[j] - L <= 2 * max(1, (Ls[1] - Ls[0]) if len(Ls) > 1 else 1):
                if ent[L] - ent[Ls[j]] > IGNITION_ENTROPY_DROP:
                    ignition = Ls[j]
                    break
        if ignition is not None:
            break
    motor = [L for L in Ls if per_layer[L].get("motor_kl", 1e9) < 0.1]
    motor_start = motor[0] if motor else None
    if ignition is None:
        note = ("nema ostrog opredeljenja -- tro-regionska struktura nije nadjena"
                if resolution_ok else
                f"nema ostrog opredeljenja NA OVOJ REZOLUCIJI (svaki {stride}. sloj); "
                "skok od <= 2 sloja se ovako ne vidi -- ishod je neodlucan")
        return {"ignition": None, "sensory": [], "workspace": [], "motor": motor,
                "note": note, "stride": stride, "resolution_ok": resolution_ok}
    sensory = [L for L in Ls if L < ignition]
    workspace = [L for L in Ls if L >= ignition and (motor_start is None or L < motor_start)]
    return {"ignition": ignition, "sensory": sensory, "workspace": workspace,
            "motor": motor, "note": "", "stride": stride, "resolution_ok": resolution_ok}


# ---------------------------------------------------------------- mapa
def load_ambiguous() -> list[str]:
    from pathlib import Path
    p = Path(__file__).with_name("prompts_ambiguous.txt")
    return [ln.strip() for ln in p.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]


def jspace_map(rt, prompts: list[str], layers: list[int], n_probes: int,
               corpus: str, k: int = JLENS_SUPPORT_K, progress=None) -> dict:
    """Cela L4 mapa za jedan model: po sloju udeo varijanse, entropija
    dekompozicije, blizina izlazu, i "cipovi" -- pojmovi koje model drzi."""
    t = rt.torch
    ids_list = [rt.encode(p) for p in prompts]
    probe_ids = probe_tokens(rt, corpus, n_probes)
    Wp = rt.W_U[probe_ids]
    finals, resid = [], {L: [] for L in layers}
    with t.no_grad():
        for ids in ids_list:
            x, acts = rt.run_resid(ids, hook_points={f"resid_pre.{L}" for L in layers})
            logit = (rt.final_norm(x[-1]) @ Wp.T).double().numpy()
            e = np.exp(logit - logit.max())
            finals.append(e / e.sum())
            for L in layers:
                resid[L].append(acts[f"resid_pre.{L}"][-1].double().numpy())
    per_layer, chips = {}, {}
    for i, L in enumerate(layers):
        rows = jlens_rows(rt, ids_list, L, probe_ids)
        fr, en, kl = [], [], []
        chips[L] = []
        for pi, h in enumerate(resid[L]):
            coef, frac, ent = jspace_decompose(h, rows, k)
            fr.append(frac)
            en.append(ent)
            q = lens_read(h, rows)
            p = finals[pi]
            kl.append(float((p * np.log(np.maximum(p, 1e-12) / np.maximum(q, 1e-12))).sum()))
            top = np.argsort(-coef)[:5]
            chips[L].append([(rt.decode([probe_ids[j]]), float(coef[j])) for j in top if coef[j] > 0])
        per_layer[L] = {"var_frac": float(np.mean(fr)), "entropy": float(np.mean(en)),
                        "motor_kl": float(np.mean(kl)),
                        "var_frac_sd": float(np.std(fr)), "entropy_sd": float(np.std(en))}
        if progress:
            progress(f"  sloj {L:3}: var {per_layer[L]['var_frac']:.3f}  "
                     f"H {per_layer[L]['entropy']:.2f}  KL do izlaza {per_layer[L]['motor_kl']:.3f}")
    return {"layers": layers, "per_layer": per_layer, "regions": depth_map(per_layer),
            "probes": [rt.decode([i]) for i in probe_ids], "chips": chips,
            "prompts": prompts, "expected_var": JSPACE_VAR_EXPECTED}
