"""pse-a doctor -- AZ-2 capability semafor prosiren sa modela na SAMU MASINU.

Ovo je i alat kojim se .deb proverava posle prenosa na drugu masinu (§8.1).
Nista se ne pretpostavlja: BLAS, torch i brzina se MERE.

Najvaznija zamka koju hvata: apt-ov python3-numpy se ume linkovati na
referentni libblas3 umesto na OpenBLAS. Razlika je ~60x i potpuno je tiha.
"""
from __future__ import annotations

import os
import platform
import shutil
import sys
from dataclasses import dataclass, field


@dataclass
class MachineReport:
    cpu_model: str = "?"
    cores_physical: int = 0
    cores_logical: int = 0
    flags: list[str] = field(default_factory=list)
    ram_bytes: int = 0
    disk_free_bytes: int = 0
    data_root: str = ""
    blas_name: str = "?"
    gemm_gflops: float = 0.0
    eig_c: float = 0.0
    torch_version: str | None = None
    torch_works: bool = False
    torch_autograd: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def has_avx2(self) -> bool:
        return "avx2" in self.flags

    @property
    def has_fma(self) -> bool:
        return "fma" in self.flags


def data_root() -> str:
    """XDG. §8: koren podataka MORA biti konfigurabilan od prvog dana.

    Zamka iz stvarnog zivota: snap paketi (VS Code, Chromium...) postavljaju
    XDG_DATA_HOME unutar svog sandbox-a. Harvest od desetina GB tamo ne sme
    da zavrsi -- taj direktorijum nestaje sa revizijom snap-a.
    """
    env = os.environ.get("PSE_A_DATA")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    base = os.environ.get("XDG_DATA_HOME") or ""
    if not base or "/snap/" in base or "/.snap" in base or not os.path.isabs(base):
        base = os.path.expanduser("~/.local/share")
    return os.path.join(base, "pse-a")


def _cpu() -> tuple[str, int, int, list[str]]:
    model, flags, physical = "?", [], set()
    logical = os.cpu_count() or 0
    try:
        with open("/proc/cpuinfo") as f:
            cur_phys = cur_core = None
            for line in f:
                if line.startswith("model name") and model == "?":
                    model = line.split(":", 1)[1].strip()
                elif line.startswith("flags") and not flags:
                    want = {"avx", "avx2", "fma", "avx512f", "sse4_2", "f16c", "bf16"}
                    flags = sorted(set(line.split(":", 1)[1].split()) & want)
                elif line.startswith("physical id"):
                    cur_phys = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    cur_core = line.split(":", 1)[1].strip()
                    if cur_phys is not None:
                        physical.add((cur_phys, cur_core))
    except OSError:
        pass
    return model, len(physical) or logical, logical, flags


def _ram() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _blas_name() -> str:
    try:
        import numpy as np
        cfg = np.show_config(mode="dicts") or {}
        b = cfg.get("Build Dependencies", {}).get("blas", {})
        return f"{b.get('name', '?')} {b.get('version', '')}".strip()
    except Exception:
        return "?"


def _torch() -> tuple[str | None, bool, bool]:
    try:
        import torch
    except Exception:
        return None, False, False
    ver = getattr(torch, "__version__", "?")
    try:
        x = torch.randn(32, 32, requires_grad=True)
        y = (x @ x.T).sum()
        y.backward()
        return ver, True, x.grad is not None
    except Exception:
        return ver, False, False


def probe(benchmark: bool = True) -> MachineReport:
    r = MachineReport()
    r.cpu_model, r.cores_physical, r.cores_logical, r.flags = _cpu()
    r.ram_bytes = _ram()
    r.data_root = data_root()
    probe_dir = r.data_root if os.path.isdir(r.data_root) else os.path.expanduser("~")
    r.disk_free_bytes = shutil.disk_usage(probe_dir).free
    r.blas_name = _blas_name()

    if benchmark:
        from ..core.cost import MachineRates
        rates = MachineRates.measure()
        r.gemm_gflops, r.eig_c = rates.gemm_gflops, rates.eig_c
        if r.gemm_gflops < 2.0:
            r.warnings.append(
                "BLAS NIJE OPTIMIZOVAN — ovo je ~60x sporije nego sto treba i potpuno je tiho.\n"
                "        apt install libopenblas0\n"
                "        update-alternatives --config libblas.so.3-x86_64-linux-gnu")

    r.torch_version, r.torch_works, r.torch_autograd = _torch()

    nt = os.environ.get("OMP_NUM_THREADS")
    if r.cores_physical and nt != str(r.cores_physical):
        r.warnings.append(
            f"OMP_NUM_THREADS={nt or 'nije postavljen'}; preporuceno {r.cores_physical} "
            f"(fizicka jezgra) — hyperthreading na GEMM-u steti")
    if not r.has_avx2:
        r.warnings.append("CPU nema AVX2/FMA — torch moze da ne radi; L2-L4 su rizicni ovde")
    return r


def rates_of(r: MachineReport):
    from ..core.cost import MachineRates
    return MachineRates(gemm_gflops=r.gemm_gflops or 1.0, eig_c=r.eig_c or 1e-9)


def select_profile(r: MachineReport):
    from ..profiles import select
    return select(r.gemm_gflops, r.ram_bytes / 1e9, r.torch_works)


# ---------------------------------------------------------------- ispis
def _ok(b: bool) -> str:
    return "OK " if b else "-- "


def render(r: MachineReport, model_path: str | None = None) -> str:
    from ..core.cost import human
    L = []
    A = L.append
    A("pse-a doctor")
    A("=" * 62)
    A(f"CPU      {r.cpu_model}")
    A(f"         {r.cores_physical}c/{r.cores_logical}t   {' '.join(r.flags) or '?'}")
    A(f"RAM      {r.ram_bytes/1e9:.1f} GB")
    A(f"disk     {r.disk_free_bytes/1e9:.0f} GB slobodno u {r.data_root}")
    if r.gemm_gflops > 0:
        A(f"BLAS     {r.blas_name}   {r.gemm_gflops:.1f} GFLOPS fp32   "
          f"{_ok(r.gemm_gflops >= 2.0)}")
    else:
        A(f"BLAS     {r.blas_name}   brzina nije merena (bez --no-bench meri se)")
    if r.torch_version is None:
        A("torch    nije instaliran          --   (pse-a setup-torch)")
    else:
        A(f"torch    {r.torch_version}  forward {_ok(r.torch_works)} "
          f"autograd {_ok(r.torch_autograd)}")
    prof = select_profile(r)
    A(f"profil   -> {prof.name}")

    if model_path:
        from ..core.source.gguf import open_gguf
        from ..core.source.canonical import build_map
        from ..core.capability import semaphore
        g = open_gguf(model_path)
        mm = build_map(g)
        A("")
        A(f"model    {os.path.basename(model_path)}")
        A(f"         {mm.summary()}")
        inv = g.type_inventory()
        tot = sum(v["elements"] for v in inv.values())
        A("         " + "  ".join(f"{k} {100*v['elements']/tot:.0f}%" for k, v in inv.items()))
        A(f"         model_hash {g.model_hash()[:16]}")
        caps = semaphore(g, mm, rates_of(r), r.ram_bytes,
                         has_hooks=r.torch_works, has_autograd=r.torch_autograd,
                         profile=prof)
        A("")
        A("dostupno " + " ".join(k for k in ("L0", "L1", "L2", "L3", "L4")
                                 if caps.layers.get(k)))
        for k in ("L2", "L3", "L4"):
            if not caps.layers.get(k) and k in caps.blockers:
                A(f"         {k} nedostupan: {caps.blockers[k]}")
        A("")
        A("procene")
        for k, v in caps.estimates.items():
            A(f"         {k:32} {v}")

    if r.warnings:
        A("")
        A("upozorenja")
        for w in r.warnings:
            A(f"  !  {w}")
    return "\n".join(L)
