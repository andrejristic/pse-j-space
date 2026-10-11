"""Jedini emiter teksta prema coveku -- AZ-8 / [PSE-A] Sec 9.

"No game form is ever named to the human -- not in text, and not in the GUI."
Sloj formi vraca (form_id, posterior, horizon); ovde se to prevodi u
PROGNOSTICKI DOMENSKI JEZIK. Nijedan drugi modul ne sme da pravi recenice o
stanju jedinica. tests/test_az8.py grep-uje sve sto ovaj modul moze da emituje,
sve sablone i sve SSE seme, protiv zabranjenog recnika.
"""
from __future__ import annotations


def _runs(h: float | None) -> str:
    if h is None:
        return ""
    n = max(1, round(h))
    return {1: "otprilike jedan run", 2: "otprilike dva run-a",
            3: "otprilike tri run-a"}.get(n, f"otprilike {n} run-ova")


def prognosis(form_id: str, posterior: float, horizon: float | None = None) -> str:
    """(forma, posterior, horizont) -> recenica. Nesigurnost ostaje u recenici."""
    sure = "verovatno" if posterior >= 0.7 else "moguće"
    if form_id == "war_of_attrition":
        return (f"Ovaj pravac {sure} ne konvergira; {_runs(horizon)} do tačke odluke — "
                "promeniti pristup ili stati.")
    if form_id == "signalling":
        return (f"Rezultati {sure} izgledaju bolje nego što su proverljivi; "
                "vredi ih ponoviti pre nego što uđu u znanje.")
    if form_id == "stag_hunt":
        return ("Poverenje u tok analize je poljuljano; pregled dnevnika i ponavljanje "
                "na zahtev su dostupni za svaki rezultat.")
    if form_id == "iterated_pd":
        return "Jedinica blizu kraja budžeta završava se ranije, da ne bi skraćivala put."
    if form_id == "ess":
        return "Podrazumevana metodologija se ne menja dok ne bude dovoljno potvrđenih run-ova."
    return "Rad teče uobičajenim ritmom."


_GATE_TEXT = {
    "theta_u_opsegu": "angažman jedinice izlazio je iz zdravog opsega",
    "varijansa_u_granicama": "angažman jedinice je bio previše nestabilan",
    "registar_ima_sadrzaj": "run nije promenio nijedno očekivanje o modelu",
    "angazman_ne_izolacija": "angažman jedinice je tokom rada opadao",
}


def refusal(failed: list[str]) -> str:
    """Zasto rezultat nije primljen u znanje -- bez klinickih i strateskih imena."""
    parts = [_GATE_TEXT.get(f, "provera nije prošla") for f in failed]
    return "Rezultat nije primljen u znanje: " + "; ".join(parts) + "."


def regime_label(regime: str) -> str:
    return {"random_like": "sličan slučajnom", "spiked": "izdvojeni pravci",
            "heavy_tailed": "težak rep", "mixed": "mešovito"}.get(regime, regime)


RUN_KIND = {"l1_sweep": "spektralna mapa", "qdiff": "poređenje preciznosti",
            "l2_lens": "misao po slojevima", "l3_sae": "rečnik pojmova",
            "l4_jspace": "mapa J-prostora"}


def run_kind_label(kind: str) -> str:
    return RUN_KIND.get(kind, kind)


def all_phrases() -> list[str]:
    """Sve sto modul moze da emituje -- za AZ-8 test."""
    out = [prognosis(f, p, h) for f in ("war_of_attrition", "signalling", "stag_hunt",
                                         "iterated_pd", "ess", "nash_substrate")
           for p in (0.5, 0.9) for h in (None, 1, 2, 5)]
    out += [refusal([k]) for k in _GATE_TEXT]
    out += [regime_label(r) for r in ("random_like", "spiked", "heavy_tailed", "mixed")]
    return out
