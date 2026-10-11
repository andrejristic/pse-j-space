"""Cetvorouslovna kapija doprinosa -- [PSE §6], analyzer citanje.

Doslovno iz [PSE]: "Before any unit's contribution is incorporated into the
SHARED_LIB, a four-condition schizoid validation check is performed: Theta
within healthy range throughout the encounter; Theta variance within normal
bounds; Organelle Registry containing patient-relevant content; and Theta
trajectory showing engagement rather than isolation."

[PSE-A] Sec 8.2: kapija se na analiticke doprinose primenjuje VERBATIM.
Prevod je samo u tome sta je "relevantan sadrzaj registra": za klinicku
jedinicu to je sadrzaj vezan za pacijenta, za analyzer jedinicu -- nalazi koji
su POMERILI PRIOR o modelu koji je analiziran (§3.2).

Zamka koju §3.2 imenuje: ako se rast registra definise kao broj zavrsenih
run-ova ili bajtova artefakata, uslov 3 postaje funkcija potrosenog compute-a,
pa kapija prestaje da razlikuje ucenje od mlacenja -- a to je tacno patologija
koju treba da hvata.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev

# [PSE §13] Bounded Extension i Time-Averaged Stability
THETA_HEALTHY = (25.00, 35.00)
THETA_MEAN_BAND = (27.50, 32.50)
THETA_VARIANCE_MAX = 4.0          # stddev; izvedeno iz sirine trake [27.5, 32.5]
ENGAGEMENT_MIN_T = -2.0           # t-statistika nagiba, ne sam nagib


@dataclass
class GateResult:
    passed: bool
    conditions: dict[str, bool] = field(default_factory=dict)
    detail: dict[str, str] = field(default_factory=dict)

    def failed_conditions(self) -> list[str]:
        return [k for k, v in self.conditions.items() if not v]


def _slope_t(xs: list[float]) -> tuple[float, float]:
    """Nagib trajektorije i njegova t-statistika.

    Apsolutan prag na nagibu je pogresno kalibrisan: za 50 uzoraka suma sa
    sigma=1 standardna greska nagiba je ~0.0098, pa bi prag od -0.002 obarao
    zdravu jedinicu u oko 40% slucajeva. "Izolacija" po [PSE §6] znaci STVARAN
    monoton pad, a to je pitanje znacajnosti a ne velicine -- isto kao sto je
    prag korelacije kod K2 morao da zavisi od sirine bita (I-15), a xmin kod
    alfa od raspodele (I-16). Vraca (nagib, t).
    """
    n = len(xs)
    if n < 5:
        return 0.0, 0.0
    mx = (n - 1) / 2
    my = mean(xs)
    den = sum((i - mx) ** 2 for i in range(n))
    if den == 0:
        return 0.0, 0.0
    b = sum((i - mx) * (x - my) for i, x in enumerate(xs)) / den
    a = my - b * mx
    resid = [x - (a + b * i) for i, x in enumerate(xs)]
    s2 = sum(r * r for r in resid) / (n - 2)
    se = (s2 / den) ** 0.5
    return b, (b / se if se > 0 else 0.0)


def contribution_gate(theta_series: list[float], registry_entries: int,
                      registry_baseline: float | None = None) -> GateResult:
    """Cetiri uslova, redom kojim ih [PSE §6] navodi.

    registry_baseline: prosecan rast registra ove jedinice u skorasnjim
    run-ovima. §3.2: sirov rast opada PO KONSTRUKCIJI kako model postaje poznat
    (na run-u #1 je sve novo, na #50 nije), pa bi apsolutan prag proglasio svaki
    zreo projekat schizoidnim. Zato se uslov 3 meri PREMA SOPSTVENOJ OSNOVI.
    """
    c: dict[str, bool] = {}
    d: dict[str, str] = {}

    lo, hi = THETA_HEALTHY
    within = [lo <= t <= hi for t in theta_series]
    c["theta_u_opsegu"] = bool(theta_series) and all(within)
    d["theta_u_opsegu"] = (f"{sum(within)}/{len(theta_series)} uzoraka u [{lo}, {hi}]"
                           if theta_series else "nema uzoraka theta")

    sd = pstdev(theta_series) if len(theta_series) > 1 else 0.0
    c["varijansa_u_granicama"] = sd <= THETA_VARIANCE_MAX
    d["varijansa_u_granicama"] = f"stddev {sd:.3f} (max {THETA_VARIANCE_MAX})"

    if registry_baseline is None:
        ok3 = registry_entries > 0
        d["registar_ima_sadrzaj"] = f"{registry_entries} upisa (nema osnove za poredjenje)"
    else:
        ok3 = registry_entries > 0 and registry_entries >= 0.25 * registry_baseline
        d["registar_ima_sadrzaj"] = (f"{registry_entries} upisa naspram osnove "
                                     f"{registry_baseline:.1f}")
    c["registar_ima_sadrzaj"] = ok3

    sl, t = _slope_t(theta_series)
    c["angazman_ne_izolacija"] = t >= ENGAGEMENT_MIN_T
    d["angazman_ne_izolacija"] = (f"nagib theta {sl:+.5f}, t={t:+.2f} "
                                  f"(prag t >= {ENGAGEMENT_MIN_T})")

    return GateResult(all(c.values()), c, d)


def schizoid_screen(theta_series: list[float], registry_entries: int) -> str | None:
    """[PSE-A §8.2]: "analize se vrte, compute gori, theta je ravna, i nista se
    nikad ne nauci -- forma bez supstance". Ravna theta + nula rasta registra."""
    if len(theta_series) < 10:
        return None
    sd = pstdev(theta_series)
    if sd < 0.5 and registry_entries == 0:
        return ("compute je potrosen, a nijedan nalaz nije promenio ocekivanje; "
                "razmotri promenu pristupa pre sledeceg run-a")
    return None
