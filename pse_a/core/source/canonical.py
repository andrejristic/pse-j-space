"""Kanonska mapa: imena tenzora -> uloge, nezavisno od porodice modela.

[SPEC] Sec 3: "layer/tensor naming differs across model families; L0 maps them
onto canonical roles". Kanonska mapa se cuva uz dosije da svaki kasniji run
bilo kog modela cita identicno.

Uloga i klasa uloge su kovarijate u svakom L1 izvestaju, zajedno sa ggml_type
(I-8) i relativnom dubinom. Bez njih se ne moze reci "alfa za ffn_down na
dubini 0.7 u Qwen familiji", sto je jedinica priora za registry growth (§3.2).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .gguf import GGUFFile, TensorInfo

# klase uloga -- odlucuju sta uopste ulazi u RMT analizu
ATTENTION = "attention"
MLP = "mlp"
EMBEDDING = "embedding"
NORM = "norm"
BIAS = "bias"
UNKNOWN = "unknown"

_ROLE_CLASS = {
    "attn_q": ATTENTION, "attn_k": ATTENTION, "attn_v": ATTENTION,
    "attn_output": ATTENTION, "attn_qkv": ATTENTION,
    "ffn_gate": MLP, "ffn_up": MLP, "ffn_down": MLP,
    "ffn_gate_exp": MLP, "ffn_up_exp": MLP, "ffn_down_exp": MLP,
    "token_embd": EMBEDDING, "output": EMBEDDING,
    "attn_norm": NORM, "ffn_norm": NORM, "output_norm": NORM,
    "attn_q_norm": NORM, "attn_k_norm": NORM,
}

_BLK = re.compile(r"^blk\.(\d+)\.(.+)$")


@dataclass(frozen=True)
class Role:
    name: str            # ime tenzora u fajlu
    role: str            # kanonska uloga, npr. ffn_down
    role_class: str      # attention / mlp / embedding / norm / bias
    layer: int | None    # indeks bloka, None za globalne tenzore
    depth: float | None  # relativna dubina u [0,1], None za globalne
    is_weight_matrix: bool   # ulazi li u L1 ([SPEC] Sec 4: samo 2-D)


@dataclass
class ModelMap:
    """Ono sto [SPEC] Sec 3 zove ModelMap. Cuva se uz dosije."""
    architecture: str
    n_layers: int
    d_model: int
    d_ff: int | None
    n_heads: int | None
    n_kv_heads: int | None
    vocab_size: int | None
    roles: dict[str, Role]

    # I-8 / K-kvantizacije: K-kvant trazi duzinu reda deljivu sa 256.
    # Ako nije, llama.cpp pada na blokove od 32 (Q5_0, IQ4_NL...) i lestvica
    # preciznosti vise ne meri sirinu bita nego sastav tipova (§13.1).
    @property
    def kquant_clean(self) -> bool:
        dims = [self.d_model] + ([self.d_ff] if self.d_ff else [])
        return all(d % 256 == 0 for d in dims)

    def weight_matrices(self) -> list[str]:
        return [n for n, r in self.roles.items() if r.is_weight_matrix]

    def by_role(self, role: str) -> list[str]:
        return [n for n, r in self.roles.items() if r.role == role]

    def summary(self) -> str:
        cls: dict[str, int] = {}
        for r in self.roles.values():
            if r.is_weight_matrix:
                cls[r.role_class] = cls.get(r.role_class, 0) + 1
        parts = " ".join(f"{k}:{v}" for k, v in sorted(cls.items()))
        clean = "cist" if self.kquant_clean else "K-kvant NEDOSTUPAN (dim % 256 != 0)"
        return (f"{self.architecture} · {self.n_layers} slojeva · d_model={self.d_model}"
                f" · d_ff={self.d_ff} · matrica: {parts} · {clean}")


def _meta(g: GGUFFile, arch: str, key: str, default=None):
    return g.metadata.get(f"{arch}.{key}", default)


def classify(name: str, info: TensorInfo, n_layers: int) -> Role:
    m = _BLK.match(name)
    if m:
        layer = int(m.group(1))
        rest = m.group(2)
        depth = layer / max(n_layers - 1, 1)
    else:
        layer, rest, depth = None, name, None

    suffix = rest.rsplit(".", 1)[-1] if "." in rest else ""
    role = rest[: -(len(suffix) + 1)] if suffix in ("weight", "bias") else rest

    if suffix == "bias":
        role_class = BIAS
    else:
        role_class = _ROLE_CLASS.get(role, UNKNOWN)

    # [SPEC] Sec 4: L1 gleda samo 2-D tenzore. Norme i bias-i su 1-D.
    is_wm = info.is_2d and role_class in (ATTENTION, MLP, EMBEDDING)
    return Role(name, role, role_class, layer, depth, is_wm)


def build_map(g: GGUFFile) -> ModelMap:
    arch = str(g.metadata.get("general.architecture", "unknown"))
    n_layers = int(_meta(g, arch, "block_count", 0) or 0)
    roles = {n: classify(n, t, n_layers) for n, t in g.tensors.items()}
    return ModelMap(
        architecture=arch,
        n_layers=n_layers,
        d_model=int(_meta(g, arch, "embedding_length", 0) or 0),
        d_ff=_meta(g, arch, "feed_forward_length"),
        n_heads=_meta(g, arch, "attention.head_count"),
        n_kv_heads=_meta(g, arch, "attention.head_count_kv"),
        vocab_size=len(g.metadata.get("tokenizer.ggml.tokens", [])) or None,
        roles=roles,
    )
