"""K2 kapija: dequantizacija je tacna.

Dva kriterijuma, jer jedan nije dovoljan:

1. KORELACIJA iznad poda koji ZAVISI OD SIRINE BITA. Jedinstven prag od 0.99
   je kalibriran za 4+ bita; Q3_K fizicki ne moze preko ~0.99, pa bi ispravan
   kod izgledao kao kvar -- i "popravljao" bi se dok se ne pokvari.

2. STRUKTURNA RAVNOMERNOST OSTATKA. Ovo je stvarni detektor bugova i ne
   zavisi od sirine bita: greska u pakovanju bitova UVEK pravi strukturu
   (jedna grupa od 16 elemenata odskace), a kvantizacioni sum nikad.
   Ovaj kriterijum je vazniji od prvog.
"""
from __future__ import annotations

import numpy as np

# pod korelacije po tipu -- izveden iz broja nivoa, ne pogodjen
CORR_FLOOR = {
    "F16": 0.99999, "BF16": 0.9999,
    "Q8_0": 0.9999, "Q8_1": 0.9999,
    "Q6_K": 0.999, "Q5_K": 0.999, "Q5_0": 0.998, "Q5_1": 0.998,
    "Q4_K": 0.995, "Q4_0": 0.99, "Q4_1": 0.99, "IQ4_NL": 0.995,
    "Q3_K": 0.98,
    "Q2_K": 0.95,
}
UNIFORMITY_MAX = 1.35          # max/min RMSE po grupi od 16; bug daje >> 1


def residual_uniformity(err: np.ndarray, group: int = 16, block: int = 256) -> float:
    """max/min RMSE po poziciji grupe unutar bloka. ~1.0 = cist kvantizacioni sum."""
    nb = err.size // block
    if nb < 8:
        return 1.0
    g = err[:nb * block].reshape(nb, block // group, group)
    rm = np.sqrt((g ** 2).mean(axis=(0, 2)))
    rm = rm[rm > 0]
    return float(rm.max() / rm.min()) if rm.size else 1.0


def check_tensor(W: np.ndarray, R: np.ndarray, type_name: str) -> dict:
    """W = dequantizovano iz nize preciznosti, R = referenca (f16 ili Q8_0)."""
    w, r = W.ravel(), R.ravel()
    corr = float(np.corrcoef(w, r)[0, 1])
    err = w - r
    rel_rmse = float(np.sqrt((err ** 2).mean()) / r.std())
    unif = residual_uniformity(err)
    floor = CORR_FLOOR.get(type_name, 0.99)
    return {
        "type": type_name, "corr": corr, "rel_rmse": rel_rmse,
        "uniformity": unif, "floor": floor,
        "corr_ok": corr >= floor,
        "unif_ok": unif <= UNIFORMITY_MAX,
        "ok": corr >= floor and unif <= UNIFORMITY_MAX,
    }
