"""MachineProfile -- I-1.

[SPEC] Sec 0 konstante se NE menjaju, nego se profilisu. Algoritam ostaje
identican; menjaju se budzeti. Bez ovoga L2-L4 na raspolozivom hardveru
nikad ne krenu.

Metodoloske konstante (MP_TOLERANCE, ESD_BINS, TAIL_MIN_COUNT, ALPHA opsezi)
NISU u profilu -- one definisu sta se meri, ne koliko se trosi.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Profile:
    name: str
    # --- L2 harvest
    harvest_tokens: int
    chunk_tokens: int
    position_stride: int          # I-11: uzorkovanje pozicija
    max_harvest_layers: int       # I-11: L1 bira koje slojeve
    # --- L3 SAE
    expansion_factor: int         # I-12
    l0_target: tuple[int, int]
    recon_fvu_max: float
    sae_batch: int
    sae_lifespan: int
    # --- L4 J-lens
    jlens_prompts: int
    vjp_probes: int
    jlens_seq_len: int
    jlens_layer_stride: int
    # --- L1
    spike_eigenvectors: int       # top-k vektora za reorg modul

    def as_dict(self) -> dict:
        return asdict(self)


# dev: jedina svrha je da ceo put kroz kod prodje za minute na Acer-u.
# Ne proizvodi rezultate, proizvodi pokrivenost koda.
DEV = Profile(
    name="dev",
    harvest_tokens=10_000, chunk_tokens=4096, position_stride=8, max_harvest_layers=2,
    expansion_factor=2, l0_target=(5, 20), recon_fvu_max=0.50, sae_batch=64,
    sae_lifespan=500,
    jlens_prompts=10, vjp_probes=8, jlens_seq_len=32, jlens_layer_stride=1,
    spike_eigenvectors=16,
)

TOY = Profile(
    name="toy",
    harvest_tokens=1_000_000, chunk_tokens=1 << 20, position_stride=4, max_harvest_layers=4,
    expansion_factor=4, l0_target=(10, 40), recon_fvu_max=0.30, sae_batch=256,
    sae_lifespan=50_000,
    jlens_prompts=100, vjp_probes=32, jlens_seq_len=64, jlens_layer_stride=3,
    spike_eigenvectors=50,
)

WORKSTATION = Profile(
    name="workstation",
    harvest_tokens=10_000_000, chunk_tokens=1 << 20, position_stride=2, max_harvest_layers=6,
    expansion_factor=8, l0_target=(20, 60), recon_fvu_max=0.20, sae_batch=1024,
    sae_lifespan=100_000,
    jlens_prompts=300, vjp_probes=64, jlens_seq_len=128, jlens_layer_stride=2,
    spike_eigenvectors=100,
)

# full = konstante iz [SPEC] Sec 0, neizmenjene. Nijedna raspoloziva masina
# ih ne dostize; stoje kao referenca i kao cilj za tudji hardver.
FULL = Profile(
    name="full",
    harvest_tokens=100_000_000, chunk_tokens=1 << 20, position_stride=1, max_harvest_layers=999,
    expansion_factor=16, l0_target=(20, 100), recon_fvu_max=0.15, sae_batch=4096,
    sae_lifespan=50_000,
    jlens_prompts=1000, vjp_probes=256, jlens_seq_len=256, jlens_layer_stride=1,
    spike_eigenvectors=200,
)

PROFILES = {p.name: p for p in (DEV, TOY, WORKSTATION, FULL)}


def select(gemm_gflops: float, ram_gb: float, has_torch: bool) -> Profile:
    """Profil se BIRA IZ IZMERENOG, ne postavlja rukom. AZ-2 / I-2."""
    if not has_torch or gemm_gflops < 20 or ram_gb < 8:
        return DEV
    if gemm_gflops < 100 or ram_gb < 24:
        return TOY
    if gemm_gflops < 400:
        return WORKSTATION
    return FULL
