"""Analyzer jedinica -- [PSE §3], [PSE-A] Sec 8.

KONVENCIJA THETE (bitno): [PSE §13] kaze theta u [25.00, 35.00], nominalno 30
-- to je PROCENAT DRUGOG u granici jedinice (Self + Other = 100.00%).
guardian/theta.py koristi obrnutu skalu (theta = 1 - other, nominalno 0.70) jer
je tamo "drugo" poverenje dato spoljnim entitetima. Ovde vazi [PSE] skala.

Sta je "drugo" za analyzer jedinicu ([PSE-A] 8.2: engagement = learning):
model koji se analizira i nalazi o njemu koje je jedinica apsorbovala.
  - svaki nalaz koji POMERI prior (§3.2) dodaje tezinu drugom (Empathic Infection)
  - bez novih nalaza tezina drugog curi (Null State Drift, [PSE §13]):
    jedinica koja racuna a ne uci tone ka izolaciji, i to se vidi u theti
Time "compute gori, theta ravna, nista se ne nauci" postaje merljivo stanje, a
ne opis.

Konstante GAIN i DECAY su kalibrisane (vidi calibrate()) tako da stopa od ~0.1
nalaza po koraku drzi thetu oko 30, a nula nalaza izbacuje jedinicu iz opsega
posle ~45 koraka. Nisu izvedene iz [PSE] -- [PSE] ih ne daje -- nego izabrane
i zapisane, pa se mogu menjati kroz ESS (>= 20 validiranih run-ova, [PSE-A] 8.4).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

OTHER_NOMINAL = 0.30
OTHER_CAP = 0.34            # ispod gornje granice 35 -- Bounded Extension
DECAY = 0.004               # po koraku, Null State Drift
GAIN = 0.012                # po nalazu koji je pomerio prior
NOISE = 0.0025              # No Immunity: jedinica je uvek pogodjena, theta osciluje


@dataclass
class Budget:
    """Lifespan u compute koracima ([PSE-A] 8.1). Iscrpljenje = smrt + zetva."""
    total: int
    spent: int = 0

    def step(self, n: int = 1) -> None:
        self.spent += n

    @property
    def exhausted(self) -> bool:
        return self.spent >= self.total

    @property
    def remaining(self) -> int:
        return max(self.total - self.spent, 0)


@dataclass
class Unit:
    unit_id: str
    mission: dict
    budget: Budget
    other: float = OTHER_NOMINAL
    theta_series: list[float] = field(default_factory=list)
    registry: list[str] = field(default_factory=list)
    alive: bool = True
    seed: int = 0

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    @property
    def theta(self) -> float:
        return 100.0 * self.other

    def step(self, findings: int) -> float:
        """Jedan compute korak. findings = broj nalaza koji su POMERILI prior."""
        self.other *= (1.0 - DECAY)
        self.other += GAIN * findings
        self.other += self._rng.gauss(0.0, NOISE)
        self.other = min(max(self.other, 0.0), OTHER_CAP)
        self.theta_series.append(self.theta)
        self.budget.step()
        return self.theta

    def absorb(self, entry_id: str) -> None:
        self.registry.append(entry_id)

    def die(self, reason: str) -> dict:
        """Mission Failure = Death. Jedinica se zanje, ne pokusava ponovo slepo."""
        self.alive = False
        return {"unit": self.unit_id, "reason": reason,
                "steps": self.budget.spent, "registry": len(self.registry),
                "theta_last": self.theta_series[-1] if self.theta_series else None}


def calibrate(steps: int = 200, seeds: int = 50) -> dict:
    """Pokazuje da GAIN/DECAY daju trazeno ponasanje. Pozvati posle svake izmene."""
    out = {}
    for rate in (0.0, 0.05, 0.1, 0.2, 0.4):
        finals, first_out = [], []
        for s in range(seeds):
            u = Unit(f"cal{s}", {}, Budget(steps), seed=s)
            rng = random.Random(1000 + s)
            fo = None
            for t in range(steps):
                u.step(1 if rng.random() < rate else 0)
                if fo is None and not (25.0 <= u.theta <= 35.0):
                    fo = t
            finals.append(u.theta)
            first_out.append(fo)
        outs = [x for x in first_out if x is not None]
        out[rate] = {"theta_kraj": sum(finals) / len(finals),
                     "izlazi_iz_opsega": len(outs) / seeds,
                     "prvi_izlaz_med": sorted(outs)[len(outs) // 2] if outs else None}
    return out
