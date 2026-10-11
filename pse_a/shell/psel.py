"""PSE-L -- glas sistema prema coveku ([SPEC] 10.1, [PSE §5]).

"The conversation is the spine; panels are what the LLM summons." Model dobija
SAMO sazetke zapecacenih rezultata (price iz narrative.py i nekoliko brojeva),
nikad tezine, aktivacije ni radno stanje. Moze da otvori panel oznakom
[[panel:/run/<id>/<pogled>]] koju GUI pretvara u dugme (panel.open).

AZ-8 ne sme zavisiti od toga da li je model poslusao uputstvo: svaki odgovor
prolazi kroz ISTI sken zabranjenog recnika kao sabloni i SSE, i zabranjen izraz
se uklanja pre nego sto ga covek vidi. To je jedina garancija koja vazi za
bilo koji model, lokalni ili tudji.
"""
from __future__ import annotations

import re
import time
import uuid

from . import narrative
from .prognostics import run_kind_label

PANEL = re.compile(r"\[\[panel:(/(?:run|model)/[A-Za-z0-9_./-]+|/inbox)\]\]")
VIEW = {"l1_sweep": "spectral", "qdiff": "qdiff", "l2_lens": "lens", "l3_sae": "sae",
        "l4_jspace": "jspace", "ppl_ladder": "ppl"}

SYSTEM = """Ti si PSE-L, glas analitičkog sistema PSE-A koji proučava lokalne jezičke modele \
(spektar matrica težina, poređenje preciznosti, misao po slojevima, rečnik pojmova, J-prostor).

Pravila:
- Odgovaraj na jeziku korisnika (podrazumevano srpski, latinica), kratko i jasno.
- Oslanjaj se ISKLJUČIVO na rezultate navedene ispod. Ako nešto nije izmereno, reci da nije.
- Nesigurnost ide u samu rečenicu ("verovatno", "u ovom uzorku", "na ovoj rezoluciji").
- Kad tvrdnju zasniváš na rezultatu, navedi oznaku artefakta u uglastim zagradama, npr. [59286d39].
- Da otvoriš panel, napiši oznaku tačno ovako: [[panel:/run/<run_id>/<pogled>]] (putanje su ispod).
- Ne izmišljaj brojeve, modele ni nalaze.
- Ne koristi imena teorijskih okvira ni kliničke termine; govori o toku rada, konvergenciji, \
učenju i ponovljivosti rezultata.

ZAPEČAĆENI REZULTATI:
{context}
"""


def budget_for(pid: str) -> int:
    """Koliko znakova konteksta sme u sistemsku poruku. Lokalni serveri (Ollama
    podrazumevano 4096 tokena) TIHO odsecaju visak s POCETKA -- a tamo su pravila.
    Zato je kontekst za lokalne modele kratak, a za online sire."""
    from .llm import PROVIDERS
    grp = PROVIDERS.get(pid, ("", "online"))[1]
    return 5000 if grp == "lokalno" else 24000


def build_context(lib, max_runs: int = 12, max_chars: int = 24000) -> str:
    rows = lib.db.execute(
        "SELECT r.run_id, r.kind, m.display_name FROM runs r JOIN models m "
        "ON m.model_hash = r.model_hash WHERE r.state='sealed' "
        "ORDER BY r.sealed DESC LIMIT ?", (max_runs,)).fetchall()
    import json as _j
    load = lambda h: _j.loads(lib.get_blob(h))
    parts = []
    for run_id, kind, name in rows:
        arts = dict(lib.db.execute("SELECT name, content_hash FROM artifacts WHERE run_id=?",
                                   (run_id,)).fetchall())
        story = narrative.story(kind, arts, load)
        path = f"/run/{run_id}/{VIEW.get(kind, '')}"
        parts.append(f"- {run_kind_label(kind)} · {name} · panel {path}\n  " + "\n  ".join(story))
    inbox = lib.inbox(unseen_only=True)
    if inbox:
        parts.append("NEPROČITANA POŠTA:\n" + "\n".join(f"- {i['text']}" for i in inbox[:5]))
    out, used = [], 0
    for part in parts:                         # najnovije prvo; stari otpadaju ceo
        if used + len(part) > max_chars:
            out.append(f"(još {len(parts) - len(out)} starijih rezultata izostavljeno zbog veličine)")
            break
        out.append(part)
        used += len(part)
    return "\n".join(out) if out else "(još nema zapečaćenih rezultata)"


def enforce_az8(text: str) -> tuple[str, int]:
    """Mehanicka garancija AZ-8 za izlaz BILO KOG modela."""
    from ..tests.test_az8 import FORBIDDEN
    n = 0
    for pat in FORBIDDEN:
        text, k = re.subn(pat, "[uklonjeno]", text, flags=re.IGNORECASE)
        n += k
    return text, n


SCHEMA = """CREATE TABLE IF NOT EXISTS chat (
    msg_id TEXT PRIMARY KEY, role TEXT, text TEXT, provider TEXT, model TEXT, t REAL)"""


def history(lib, n: int = 40) -> list[dict]:
    lib.db.execute(SCHEMA)
    rows = lib.db.execute("SELECT role, text, provider, model, t FROM chat "
                          "ORDER BY t DESC LIMIT ?", (n,)).fetchall()[::-1]
    return [dict(zip(("role", "text", "provider", "model", "t"), r)) for r in rows]


def append(lib, role: str, text: str, provider: str = "", model: str = "") -> None:
    """Razgovor je append-only kao i beleske; nista se ne prepravlja."""
    lib.db.execute(SCHEMA)
    lib.db.execute("INSERT INTO chat VALUES (?,?,?,?,?,?)",
                   (uuid.uuid4().hex[:16], role, text, provider, model, time.time()))
    lib.db.commit()


def respond(lib, pid: str, entry: dict, user_text: str):
    """Generator: delovi odgovora, vec filtrirani. Na kraju cuva ceo odgovor."""
    from . import llm
    system = SYSTEM.format(context=build_context(lib))
    past = [{"role": m["role"], "content": m["text"]} for m in history(lib, 12)
            if m["role"] in ("user", "assistant")]
    msgs = past + [{"role": "user", "content": user_text}]
    append(lib, "user", user_text)
    full = ""
    try:
        for piece in llm.chat(pid, entry, system, msgs):
            full += piece
        clean, n = enforce_az8(full)
        if n:
            clean += "\n\n(Deo odgovora je uklonjen: sadržao je izraze koje ovaj sistem ne prikazuje.)"
    except llm.LLMError as e:
        clean = f"Nisam uspeo da dobijem odgovor: {e}"
    append(lib, "assistant", clean, pid, entry.get("model") or entry.get("gguf", ""))
    return clean


def render(text: str) -> list[tuple[str, str]]:
    """Tekst -> niz (vrsta, sadrzaj): obican tekst i dugmad za panele."""
    out, pos = [], 0
    for m in PANEL.finditer(text):
        if m.start() > pos:
            out.append(("text", text[pos:m.start()]))
        out.append(("panel", m.group(1)))
        pos = m.end()
    if pos < len(text):
        out.append(("text", text[pos:]))
    return out


def respond_stream(lib, pid: str, entry: dict, user_text: str):
    """Kao respond(), ali daje ("partial", tekst) dok stize, pa ("done", tekst).

    AZ-8 i za DELIMICAN tekst: salje se samo filtriran tekst do poslednje ZAVRSENE
    reci, pa se zabranjen izraz nikad ne pojavi ni na trenutak -- cak ni dok se
    odgovor jos kuca. Visereci izrazi se hvataju cim se kompletiraju, jer se
    filter uvek primenjuje na ceo dosadasnji tekst.
    """
    from . import llm
    system = SYSTEM.format(context=build_context(lib, max_chars=budget_for(pid)))
    n_hist = 4 if budget_for(pid) < 10000 else 12
    past = [{"role": m["role"], "content": m["text"]} for m in history(lib, n_hist)
            if m["role"] in ("user", "assistant")]
    append(lib, "user", user_text)
    full, clean = "", ""
    try:
        for piece in llm.chat(pid, entry, system, past + [{"role": "user", "content": user_text}]):
            full += piece
            cut = max(full.rfind(" "), full.rfind("\n"))
            if cut > 0:
                shown, _ = enforce_az8(full[:cut])
                yield "partial", shown
        clean, n = enforce_az8(full)
        if n:
            clean += "\n\n(Deo odgovora je uklonjen: sadržao je izraze koje ovaj sistem ne prikazuje.)"
    except llm.LLMError as e:
        clean = (full and enforce_az8(full)[0] + "\n\n") + f"Nisam uspeo da dovršim odgovor: {e}"
    append(lib, "assistant", clean, pid, entry.get("model") or entry.get("gguf", ""))
    yield "done", clean
