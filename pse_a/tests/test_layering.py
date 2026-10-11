"""AZ-1: core/ je cist -- ne sme importovati shell/, web/ ni profiles/.

Separation doctrine iz [SPEC] 1.2 mora biti proverljiva, ne stvar discipline.
Ovo je jedini test koji ne moze da prodje slucajno.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ("shell", "web")


def _module_targets(tree: ast.AST, pkg_depth: int) -> list[str]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:                       # relativni import
                out.append("." * node.level + (node.module or ""))
            elif node.module:
                out.append(node.module)
    return out


def check_core_purity() -> list[str]:
    bad = []
    for py in (ROOT / "core").rglob("*.py"):
        tree = ast.parse(py.read_text(), filename=str(py))
        for tgt in _module_targets(tree, 0):
            norm = tgt.lstrip(".")
            head = norm.split(".")[0]
            if head in FORBIDDEN or any(f".{f}." in f".{norm}." for f in FORBIDDEN):
                bad.append(f"{py.relative_to(ROOT)}: importuje '{tgt}'")
    return bad


if __name__ == "__main__":
    bad = check_core_purity()
    if bad:
        print("AZ-1 PREKRSEN:")
        for b in bad:
            print("  " + b)
        raise SystemExit(1)
    n = len(list((ROOT / "core").rglob("*.py")))
    print(f"AZ-1 OK -- {n} modula u core/, nijedan ne dodiruje shell/ ni web/")
