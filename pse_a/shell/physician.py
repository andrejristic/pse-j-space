"""Physician -- [SPEC] 8.3, [PSE §11]. Nadgleda jedinice, ne racuna nauku.

Cita istoriju run-ova (ne pojedinacnu jedinicu) i trazi obrazac koji jedna
jedinica ne moze da vidi o sebi: NIZ run-ova koji trose compute a sve manje
uce. [PSE-A] Sec 9: "monotone theta drift + NO registry growth = attention and
compute poured into a non-converging experiment = formal SUNK-COST detection.
Prognosis: decision point within ~2-3 runs."

Izlaz ide u Inbox kroz prognostics.py -- nikad ime forme (AZ-8).
Fisija (S2/S3) ovde nije potrebna: analyzer misije su vec jednonamenske
(L1, qdiff, L2, L3, L4 su odvojene), pa S2 podela po modalitetu postoji
po konstrukciji. S3 zastitna fisija trazi odobrenje coveka i nije automatska.
"""
from __future__ import annotations

from . import prognostics as pg


def review(lib, model_hash: str, kind: str) -> list[str]:
    """Posle svakog run-a: da li niz run-ova ove vrste na ovom modelu jos uci?"""
    h = lib.run_history(model_hash, kind, last=8)
    if len(h) < 3:
        return []
    regs = [x["registry"] for x in h]
    died = [x["state"] == "died" for x in h]
    posted = []
    # sunk cost: poslednja tri run-a, rast registra pada i bar dva su umrla
    last3 = regs[-3:]
    if sum(died[-3:]) >= 2 and last3[-1] <= last3[0]:
        horizon = 3 - sum(died[-3:]) + 1
        post = 0.9 if all(died[-3:]) else 0.7
        txt = pg.prognosis("war_of_attrition", post, horizon)
        lib.post_inbox("prognoza", model_hash, kind, txt)
        posted.append(txt)
    # cheap vs costly signal: pecaceni run-ovi sa jedva ikakvim ucenjem
    sealed_regs = [r for r, d in zip(regs, died) if not d]
    if len(sealed_regs) >= 3 and max(sealed_regs[-3:]) <= 2:
        txt = pg.prognosis("signalling", 0.7)
        lib.post_inbox("prognoza", model_hash, kind, txt)
        posted.append(txt)
    return posted
