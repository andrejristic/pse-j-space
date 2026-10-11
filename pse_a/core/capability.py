"""Capability semafor -- [SPEC] Sec 2, prosiren vremenom (I-2).

    ModelSource only                 -> L0, L1
    + Runtime sa hook-ovima          -> + L2, L3
    + Runtime sa autograd-om         -> + L4

Invarijanta AZ-2: nikad ne padaj usred run-a iz razloga koji je poznat na L0.
Prosirenje: procenjeno vreme JESTE takav razlog, pa ide u semafor.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .cost import MachineRates, l1_sweep_seconds, human
from .source.canonical import ModelMap
from .source.gguf import GGUFFile


@dataclass
class Capabilities:
    layers: dict[str, bool] = field(default_factory=dict)
    estimates: dict[str, str] = field(default_factory=dict)
    blockers: dict[str, str] = field(default_factory=dict)

    def available(self) -> list[str]:
        return [k for k, v in self.layers.items() if v]


def semaphore(g: GGUFFile, mm: ModelMap, rates: MachineRates,
              ram_bytes: int, has_hooks: bool = False,
              has_autograd: bool = False, profile=None) -> Capabilities:
    c = Capabilities()

    # ---- L0 / L1: dovoljan je ModelSource
    c.layers["L0"] = True
    c.layers["L1"] = True
    shapes = [tuple(reversed(g.tensors[n].shape)) for n in mm.weight_matrices()]
    nev = profile.spike_eigenvectors if profile else 0
    est = l1_sweep_seconds(shapes, rates, g._file_size, eigenvectors=False)
    c.estimates["L1 sweep (samo vrednosti)"] = f"{human(est['total_s'])}  ({est['n_matrices']} matrica)"
    if nev:
        est_v = l1_sweep_seconds(shapes, rates, g._file_size, eigenvectors=True)
        c.estimates["L1 sweep + spike vektori"] = human(est_v["total_s"])

    # ---- L2 / L3: trazi Runtime sa hook-ovima, ali pre toga mora STATI U RAM.
    # Ovo je razlog zasto je 7B L1-only (§1.3): nije compute nego memorija.
    n_params = sum(g.tensors[n].n_elements for n in g.tensors)
    fp32_bytes = n_params * 4
    fits = fp32_bytes < ram_bytes * 0.55        # tezine + aktivacije + OS
    if not has_hooks:
        c.layers["L2"] = c.layers["L3"] = False
        c.blockers["L2"] = "nema Runtime sa hook-ovima (torch nije dostupan)"
    elif not fits:
        c.layers["L2"] = c.layers["L3"] = False
        c.blockers["L2"] = (f"model u fp32 trazi {fp32_bytes/1e9:.1f} GB, "
                            f"RAM je {ram_bytes/1e9:.1f} GB — ne staje")
    else:
        c.layers["L2"] = c.layers["L3"] = True

    # ---- L4: trazi autograd
    if not c.layers.get("L2"):
        c.layers["L4"] = False
        c.blockers["L4"] = c.blockers.get("L2", "L2 nedostupan")
    elif not has_autograd:
        c.layers["L4"] = False
        c.blockers["L4"] = "Runtime nema autograd"
    else:
        c.layers["L4"] = True
    return c
