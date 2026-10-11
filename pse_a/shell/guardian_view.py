"""Stit Zastitnika u status traci -- [SPEC] 10.4, AZ-12.

"Deliberately near-invisible: a shield in the status bar." Ovaj modul SAMO
CITA audit dnevnik Zastitnika (hash-lancani jsonl iz SystemForEric/guardian),
ne uvozi njegov kod i ne pise nista. Zastitnik stiti masinu; nikad ne dira
analiticke artefakte -- i analyzer nikad ne dira njega.

Stanja:
  gray   -- Zastitnik nije pokrenut na ovoj masini (nema audit dnevnika)
  green  -- radi, nista nije eskalirano u poslednja 24 h
  yellow -- ima otkrivenih odstupanja
  red    -- eskalacija ili neuspela autentikacija vlasnika
Nikad se ne glumi zeleno kad podataka nema.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

RED = {"escalation", "owner_auth_failed", "capability_revoked"}
YELLOW = {"pathology_detected"}


def shield_state(window_s: float = 86400) -> dict:
    home = Path(os.environ.get("GUARDIAN_HOME", str(Path.home() / ".guardian")))
    logs = sorted((home / "audit").glob("*.audit.jsonl")) if (home / "audit").is_dir() else []
    if not logs:
        return {"level": "gray", "text": "Zaštitnik nije pokrenut na ovoj mašini"}
    now = time.time()
    level, n = "green", 0
    for lp in logs:
        try:
            with open(lp, "rb") as f:
                start = max(0, os.path.getsize(lp) - 256_000)
                f.seek(start)
                tail = f.read().decode("utf-8", "ignore").splitlines()
                if start > 0:
                    tail = tail[1:]            # prvi red je verovatno presecen
        except OSError:
            continue
        for line in tail:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if now - float(rec.get("ts", 0)) > window_s:
                continue
            et = str(rec.get("event_type", ""))
            if any(et.startswith(r) for r in RED):
                level, n = "red", n + 1
            elif et in YELLOW and level != "red":
                level, n = "yellow", n + 1
    text = {"green": "Zaštitnik radi, bez događaja u poslednja 24 h",
            "yellow": f"Zaštitnik je primetio odstupanja u poslednja 24 h: {n}",
            "red": f"Zaštitnik traži pažnju — događaji u poslednja 24 h: {n}"}[level]
    return {"level": level, "text": text}
