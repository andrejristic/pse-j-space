"""Model troska za L1 -- iz IZMERENIH konstanti masine, ne iz procene.

I-2 / AZ-2: semafor mora prijaviti PROCENJENO VREME, ne samo moze/ne moze.
Dvanaest dana harvesta jeste razlog poznat na L0.

Dve odvojene stope, jer se bitno razlikuju:
  gemm_gflops  -- Gram matrica W^T W, blizu peak BLAS-a
  eig_c        -- eigvalsh, sekundi po m^3; resavac sopstvenih vrednosti
                  je znatno manje BLAS-3 gust (izmereno ~6x sporije od GEMM-a)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MachineRates:
    gemm_gflops: float        # fp32 GEMM, izmereno
    eig_c: float              # sekundi po m^3 za eigvalsh, izmereno
    disk_mb_s: float = 120.0  # citanje GGUF-a; konzervativno za rotacioni disk

    @staticmethod
    def measure(n_gemm: int = 2000, n_eig: int = 1536) -> "MachineRates":
        import time
        import numpy as np
        A = np.random.randn(n_gemm, n_gemm).astype(np.float32)
        A @ A                                    # zagrevanje
        t = time.perf_counter(); A @ A; dt = time.perf_counter() - t
        gemm = 2 * n_gemm ** 3 / dt / 1e9
        del A
        B = np.random.randn(n_eig, n_eig).astype(np.float32)
        S = B.T @ B
        t = time.perf_counter(); np.linalg.eigvalsh(S); de = time.perf_counter() - t
        return MachineRates(gemm_gflops=gemm, eig_c=de / n_eig ** 3)


def l1_matrix_seconds(shape: tuple[int, int], rates: MachineRates,
                      eigenvectors: bool = False) -> float:
    """Trosak L1 za jednu matricu. I-3: Gram MANJE dimenzije + eigvalsh."""
    n, m = max(shape), min(shape)             # n >= m
    gram = 2 * n * m ** 2 / (rates.gemm_gflops * 1e9)
    eig = rates.eig_c * m ** 3
    if eigenvectors:
        eig *= 3.0                            # syevd sa vektorima ~3x skuplji
    return gram + eig


def l1_sweep_seconds(shapes: list[tuple[int, int]], rates: MachineRates,
                     file_bytes: int = 0, eigenvectors: bool = False) -> dict:
    compute = sum(l1_matrix_seconds(s, rates, eigenvectors) for s in shapes)
    io = file_bytes / 1e6 / rates.disk_mb_s if file_bytes else 0.0
    return {"compute_s": compute, "io_s": io, "total_s": compute + io,
            "n_matrices": len(shapes)}


def human(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds/60:.0f} min"
    if seconds < 172800:
        return f"{seconds/3600:.1f} h"
    return f"{seconds/86400:.1f} dana"
