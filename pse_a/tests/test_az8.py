"""AZ-8: nijedno ime forme i nijedan klinicki termin ni u jednom pikselu.

Grep-uje sablone, JS, sve sto prognostics.py moze da emituje, i SSE tipove
dogadjaja. Dozvoljen je arhitektonski recnik (unit, mission, budget,
absorption, fission, seal) -- Inbox po [PSE-A] 10.3 prikazuje bas to.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = [r"\bnash\b", r"attrition", r"signall?ing", r"stag[ _]?hunt", r"\bchicken\b",
             r"iterated[_ ]?pd", r"prisoner", r"\bess\b", r"evolutionar", r"global[_ ]games",
             r"schizoid", r"shizoid", r"narcis", r"borderline", r"cluster[ _]?b",
             r"patholog", r"patolog", r"codependent", r"kodependent", r"defect"]


def scan_text(label: str, text: str) -> list[str]:
    bad = []
    low = text.lower()
    for pat in FORBIDDEN:
        for m in re.finditer(pat, low):
            ctx = low[max(0, m.start() - 30):m.end() + 30].replace("\n", " ")
            bad.append(f"{label}: '{m.group(0)}' u …{ctx}…")
    return bad


def run() -> list[str]:
    bad = []
    for p in list((ROOT / "web" / "templates").glob("*.html")) + \
             list((ROOT / "web" / "static").glob("*.js")):
        bad += scan_text(str(p.relative_to(ROOT)), p.read_text())
    from pse_a.shell import prognostics
    for s in prognostics.all_phrases():
        bad += scan_text("prognostics", s)
    bad += scan_text("narrative.py", (ROOT / "shell" / "narrative.py").read_text())
    app_src = (ROOT / "web" / "app.py").read_text()
    for ev in re.findall(r"event: ([\w.]+)", app_src):
        bad += scan_text("SSE dogadjaj", ev)
    return bad


if __name__ == "__main__":
    bad = run()
    if bad:
        print("AZ-8 PREKRSEN:")
        for b in bad:
            print("  " + b)
        raise SystemExit(1)
    print("AZ-8 OK -- ni sabloni, ni JS, ni prognostike, ni SSE ne imenuju formu ni dijagnozu")
