"""GGUF citac -- zaglavlje, metapodaci, tensor info.

AZ-1: cist core. Bez stanja, bez theta, bez SHARED_LIB.
I-5:  dequantizacija strogo na zahtev -- ovaj modul samo mapira fajl i cita
      pojedinacne tenzore kad se trazi. Nikad ceo model u RAM.
I-8:  ggml_type se cuva PO TENZORU i izlazi u izvestaj kao kovarijata.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

GGUF_MAGIC = b"GGUF"

# ---------------------------------------------------------------- GGUF KV tipovi
(UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL,
 STRING, ARRAY, UINT64, INT64, FLOAT64) = range(13)

_SCALAR = {
    UINT8: ("<B", 1), INT8: ("<b", 1), UINT16: ("<H", 2), INT16: ("<h", 2),
    UINT32: ("<I", 4), INT32: ("<i", 4), FLOAT32: ("<f", 4), BOOL: ("<?", 1),
    UINT64: ("<Q", 8), INT64: ("<q", 8), FLOAT64: ("<d", 8),
}

# ---------------------------------------------------------------- ggml tipovi
# ime, block_size (elemenata po bloku), type_size (bajtova po bloku)
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0:  ("F32",     1,   4),
    1:  ("F16",     1,   2),
    2:  ("Q4_0",   32,  18),
    3:  ("Q4_1",   32,  20),
    6:  ("Q5_0",   32,  22),
    7:  ("Q5_1",   32,  24),
    8:  ("Q8_0",   32,  34),
    9:  ("Q8_1",   32,  36),
    10: ("Q2_K",  256,  84),
    11: ("Q3_K",  256, 110),
    12: ("Q4_K",  256, 144),
    13: ("Q5_K",  256, 176),
    14: ("Q6_K",  256, 210),
    15: ("Q8_K",  256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS",  256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S",   256, 50),
    20: ("IQ4_NL",   32, 18),
    21: ("IQ3_S",   256, 110),
    22: ("IQ2_S",   256, 82),
    23: ("IQ4_XS",  256, 136),
    24: ("I8",      1,   1),
    25: ("I16",     1,   2),
    26: ("I32",     1,   4),
    27: ("I64",     1,   8),
    28: ("F64",     1,   8),
    29: ("IQ1_M",  256,  56),
    30: ("BF16",    1,   2),
}


def ggml_type_name(t: int) -> str:
    return GGML_TYPES.get(t, (f"UNKNOWN_{t}", 1, 1))[0]


# ---------------------------------------------------------------- strukture
@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]      # GGUF poredak: shape[0] je najbrza dimenzija
    ggml_type: int
    offset: int                 # relativno na pocetak data sekcije

    @property
    def type_name(self) -> str:
        return ggml_type_name(self.ggml_type)

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def n_bytes(self) -> int:
        _, blk, tsz = GGML_TYPES[self.ggml_type]
        return self.n_elements // blk * tsz

    @property
    def is_2d(self) -> bool:
        return len(self.shape) == 2

    def __repr__(self) -> str:
        s = "x".join(str(d) for d in self.shape)
        return f"<{self.name} {s} {self.type_name}>"


@dataclass
class GGUFFile:
    """Mapirani GGUF fajl. Drzi samo metapodatke -- nijedan tenzor nije ucitan."""
    path: Path
    version: int
    metadata: dict
    tensors: dict[str, TensorInfo]
    data_offset: int
    alignment: int
    _file_size: int = 0
    _model_hash: str | None = field(default=None, repr=False)

    # ---------------------------------------------------------- identitet
    def model_hash(self, quick: bool = True) -> str:
        """Identitet modela. quick=True hesira zaglavlje + uzorke, ne ceo fajl:
        na 15 GB pun SHA256 je minutima dug, a ovde sluzi kao kljuc dosijea.
        AZ-4 trazi da artefakt nosi model_hash -- ne trazi da bude pun hes fajla."""
        if self._model_hash:
            return self._model_hash
        h = hashlib.sha256()
        h.update(str(self.version).encode())
        h.update(str(self._file_size).encode())
        for name in sorted(self.tensors):
            t = self.tensors[name]
            h.update(f"{name}|{t.shape}|{t.ggml_type}|{t.offset}".encode())
        if not quick:
            with open(self.path, "rb") as f:
                while chunk := f.read(1 << 22):
                    h.update(chunk)
        self._model_hash = h.hexdigest()
        return self._model_hash

    # ---------------------------------------------------------- inventar
    def type_inventory(self) -> dict[str, dict]:
        """I-8: koliko tenzora i koliko parametara po ggml_type.
        Ovo je dokaz da Q4_K_M nije jedan format nego mesavina."""
        inv: dict[str, dict] = {}
        for t in self.tensors.values():
            e = inv.setdefault(t.type_name, {"tensors": 0, "elements": 0, "bytes": 0})
            e["tensors"] += 1
            e["elements"] += t.n_elements
            e["bytes"] += t.n_bytes
        return dict(sorted(inv.items(), key=lambda kv: -kv[1]["elements"]))

    def matrices_2d(self) -> list[TensorInfo]:
        """Samo 2-D tenzori -- jedino oni ulaze u L1 ([SPEC] Sec 4)."""
        return [t for t in self.tensors.values() if t.is_2d]

    def raw_block_data(self, name: str) -> np.ndarray:
        """Sirovi bajtovi jednog tenzora (bez dequantizacije). I-5."""
        t = self.tensors[name]
        with open(self.path, "rb") as f:
            f.seek(self.data_offset + t.offset)
            buf = f.read(t.n_bytes)
        if len(buf) != t.n_bytes:
            raise IOError(
                f"{self.path.name}: tenzor {name} trazi {t.n_bytes} B, procitano {len(buf)}. "
                f"Fajl je odsecen ili ostecen."
            )
        return np.frombuffer(buf, dtype=np.uint8)


# ---------------------------------------------------------------- parser
class _Reader:
    def __init__(self, f):
        self.f = f

    def raw(self, n: int) -> bytes:
        b = self.f.read(n)
        if len(b) != n:
            raise IOError(f"neocekivan kraj fajla (trazeno {n} B, dobijeno {len(b)})")
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.raw(8))[0]

    def string(self) -> str:
        n = self.u64()
        if n > (1 << 24):
            raise ValueError(f"nerazumno duga string duzina: {n}")
        return self.raw(n).decode("utf-8", errors="replace")

    def value(self, vtype: int):
        if vtype in _SCALAR:
            fmt, size = _SCALAR[vtype]
            return struct.unpack(fmt, self.raw(size))[0]
        if vtype == STRING:
            return self.string()
        if vtype == ARRAY:
            etype = self.u32()
            n = self.u64()
            if etype == STRING:
                return [self.string() for _ in range(n)]
            if etype in _SCALAR:
                fmt, size = _SCALAR[etype]
                buf = self.raw(size * n)
                return list(struct.unpack("<" + fmt[1] * n, buf))
            raise ValueError(f"nepodrzan tip elementa niza: {etype}")
        raise ValueError(f"nepoznat GGUF tip vrednosti: {vtype}")


def open_gguf(path: str | Path) -> GGUFFile:
    """Otvori GGUF i izgradi mapu. Ne cita nijedan tenzor (I-5)."""
    path = Path(path)
    size = path.stat().st_size
    with open(path, "rb") as f:
        r = _Reader(f)
        magic = r.raw(4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"{path.name}: nije GGUF (magic={magic!r})")
        version = r.u32()
        if version not in (2, 3):
            raise ValueError(f"{path.name}: nepodrzana GGUF verzija {version}")
        n_tensors = r.u64()
        n_kv = r.u64()

        metadata: dict = {}
        for _ in range(n_kv):
            key = r.string()
            vtype = r.u32()
            metadata[key] = r.value(vtype)

        tensors: dict[str, TensorInfo] = {}
        for _ in range(n_tensors):
            name = r.string()
            n_dims = r.u32()
            shape = tuple(r.u64() for _ in range(n_dims))
            ttype = r.u32()
            offset = r.u64()
            if ttype not in GGML_TYPES:
                raise ValueError(f"{path.name}: nepoznat ggml_type {ttype} za {name}")
            _, blk, _ = GGML_TYPES[ttype]
            n_el = 1
            for d in shape:
                n_el *= d
            if n_el % blk:
                raise ValueError(
                    f"{path.name}: {name} ima {n_el} elemenata, nije deljivo blokom {blk}"
                )
            tensors[name] = TensorInfo(name, shape, ttype, offset)

        alignment = int(metadata.get("general.alignment", 32))
        pos = f.tell()
        data_offset = (pos + alignment - 1) // alignment * alignment

    # AZ-2 u malom: kvar koji je znan pri otvaranju ne sme se pojaviti kasnije
    for t in tensors.values():
        end = data_offset + t.offset + t.n_bytes
        if end > size:
            raise IOError(
                f"{path.name}: tenzor {t.name} zavrsava na {end} B, a fajl ima {size} B. "
                f"Fajl je odsecen -- proveri velicinu preuzimanja."
            )

    return GGUFFile(path, version, metadata, tensors, data_offset, alignment, size)
