"""PSE-A web -- pogled nad SHARED_LIB-om (§7).

Nacela ugradjena u kod, ne u dogovor:
  - GUI nema sopstveno stanje: svaki panel je GET koji renderuje SHARED_LIB
  - PECAT JE HTTP KES: /artifact/<hes> sa ETag i immutable
  - jedini POST je /run/<id>/note; PUT i DELETE rute NE POSTOJE
  - jedan SSE kanal /stream, nosi samo id-jeve i skalare
  - grafici se crtaju na serveru iz zapecacenih artefakata
  - sav tekst o stanju jedinica ide kroz shell/prognostics.py (AZ-8)
  - pokrece se ISKLJUCIVO na 127.0.0.1 (§7.4) -- Inbox ce autorizovati Tier-2
"""
from __future__ import annotations

import asyncio
import json
import os
from html import escape as html_escape
import math
import time
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..shell.shared_lib import SharedLib
from ..shell.doctor import data_root
from ..shell import prognostics as pg
from ..shell import narrative
from ..shell.guardian_view import shield_state
from ..shell import llm as llmmod
from ..shell import psel
from . import svg

HERE = Path(__file__).parent
ROLES = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")


def create_app(root: str | None = None) -> FastAPI:
    lib = SharedLib(root or data_root())
    app = FastAPI(title="PSE-A", docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    app.mount("/vendor", StaticFiles(directory=HERE / "vendor"), name="vendor")
    tpl = Jinja2Templates(directory=HERE / "templates")
    tpl.env.globals.update(pg=pg, svg=svg, fmt=_fmt, isfinite=_isfin, render=psel.render)

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        """Server slusa samo na 127.0.0.1, ali to ne stiti od DRUGE web stranice
        otvorene u istom browseru: ona moze poslati POST na 127.0.0.1 (CSRF) ili
        preko DNS rebinding-a citati odgovore. Posto aplikacija cuva API kljuceve i
        salje sazetke spolja, obe provere su obavezne:
          - Host mora biti 127.0.0.1 / localhost (DNS rebinding)
          - za svaku mutaciju Origin, ako postoji, mora biti ovaj isti server"""
        host = (request.headers.get("host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost"):
            return Response("zabranjeno: nepoznat Host", status_code=403)
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin:
                oh = origin.split("://", 1)[-1].split(":")[0].split("/")[0]
                if oh not in ("127.0.0.1", "localhost"):
                    return Response("zabranjeno: zahtev sa druge stranice", status_code=403)
        return await call_next(request)

    def blob_json(h: str) -> dict:
        return json.loads(lib.get_blob(h))

    def models() -> list[dict]:
        rows = lib.db.execute(
            "SELECT m.model_hash, m.display_name, m.architecture, "
            "(SELECT COUNT(*) FROM runs r WHERE r.model_hash=m.model_hash AND r.state='sealed') "
            "FROM models m ORDER BY m.created DESC").fetchall()
        return [{"hash": h, "name": n, "arch": a, "sealed": c} for h, n, a, c in rows if c]

    def story(ri: dict) -> list[str]:
        return narrative.story(ri["kind"], ri["artifacts"], blob_json)

    def run_info(run_id: str) -> dict:
        row = lib.db.execute("SELECT model_hash, kind, state FROM runs WHERE run_id=?",
                             (run_id,)).fetchone()
        if not row or row[2] != "sealed":
            raise HTTPException(404, "nema zapecacenog run-a sa tim id-jem")
        arts = dict(lib.db.execute("SELECT name, content_hash FROM artifacts WHERE run_id=?",
                                   (run_id,)).fetchall())
        return {"run_id": run_id, "model_hash": row[0], "kind": row[1], "artifacts": arts}

    # ------------------------------------------------------------------ strane
    @app.get("/", response_class=HTMLResponse)
    def index(request: Request, open: str | None = None):
        """open=<putanja panela>: duboki link. Isti mehanizam koristice razgovor
        da prizove panel ([PSE-A] 10.1: "panels are what the LLM summons").
        Dozvoljene su samo unutrasnje putanje panela."""
        ms = models()
        start = f"/model/{ms[0]['hash']}" if ms else None
        if open and (open.startswith("/run/") or open.startswith("/model/") or open == "/inbox") \
                and ".." not in open and "//" not in open:
            start = open
        return tpl.TemplateResponse(request, "base.html", {"models": ms, "start": start,
                                                           "shield": shield_state(),
                                                           "unseen": len(lib.inbox(True))})

    def model_card(runs: list[dict]) -> dict | None:
        """Karta modela (L0) iz najnovijeg L1 izvestaja + semafor za OVU masinu.
        [SPEC] 10.3: sposobnosti masine se prijavljuju unapred -- nema iznenadjenja
        usred run-a (AZ-2), i to sa vremenom (I-2)."""
        l1 = [r for r in runs if r["kind"] == "l1_sweep"]
        if not l1:
            return None
        rep = blob_json(l1[-1]["artifacts"]["l1_report.json"])
        inv = rep.get("type_inventory", {})
        n_params = sum(v["elements"] for v in inv.values())
        import os
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        try:
            import torch  # noqa: F401
            has_torch = True
        except Exception:
            has_torch = False
        fits = n_params * 4 < 0.55 * ram
        lay = {"L0": True, "L1": True,
               "L2": has_torch and fits and rep.get("architecture") == "qwen2",
               "L3": has_torch and fits and rep.get("architecture") == "qwen2",
               "L4": has_torch and fits and rep.get("architecture") == "qwen2"}
        why = None
        if not fits:
            why = f"model u fp32 traži {n_params*4/1e9:.1f} GB, a RAM je {ram/1e9:.1f} GB"
        elif not has_torch:
            why = "torch nije instaliran (pse-a setup-torch)"
        elif rep.get("architecture") != "qwen2":
            why = f"izvršni deo za arhitekturu '{rep.get('architecture')}' još ne postoji"
        tot = sum(v["elements"] for v in inv.values()) or 1
        return {"arch": rep.get("architecture"), "n_layers": rep.get("n_layers"),
                "d_model": rep.get("d_model"), "d_ff": rep.get("d_ff"),
                "n_params": n_params, "kquant_clean": rep.get("kquant_clean", True),
                "types": [(k, v["elements"] / tot) for k, v in inv.items()],
                "layers": lay, "why": why, "l1_s": rep.get("elapsed_s")}

    @app.get("/model/{mh}", response_class=HTMLResponse)
    def dossier(request: Request, mh: str):
        runs = lib.dossier(mh)
        card = model_card(runs)
        name = lib.db.execute("SELECT display_name FROM models WHERE model_hash=?",
                              (mh,)).fetchone()
        return tpl.TemplateResponse(request, "dossier.html",
                                    {"runs": runs[::-1], "name": name[0] if name else mh[:12],
                                     "mh": mh, "card": card})

    @app.get("/run/{run_id}/spectral", response_class=HTMLResponse)
    def spectral(request: Request, run_id: str):
        ri = run_info(run_id)
        rep = blob_json(ri["artifacts"]["l1_report.json"])
        grid: dict[int, dict[str, dict]] = {}
        roles = [r for r in ROLES if any(m["role"] == r for m in rep["matrices"])]
        for m in rep["matrices"]:
            if m["layer"] is not None:
                grid.setdefault(m["layer"], {})[m["role"]] = m
        regimes: dict[str, int] = {}
        for m in rep["matrices"]:
            regimes[m["regime"]] = regimes.get(m["regime"], 0) + 1
        return tpl.TemplateResponse(request, "spectral.html",
                                    {"story": story(ri), "rep": rep, "grid": dict(sorted(grid.items())),
                                     "roles": roles, "run_id": run_id, "regimes": regimes})

    @app.get("/run/{run_id}/esd/{name}", response_class=HTMLResponse)
    def esd(request: Request, run_id: str, name: str):
        ri = run_info(run_id)
        rep = blob_json(ri["artifacts"]["l1_report.json"])
        hist = blob_json(ri["artifacts"]["esd_histograms.json"]).get(name)
        m = next((x for x in rep["matrices"] if x["name"] == name), None)
        if not m or not hist:
            raise HTTPException(404)
        chart = svg.esd_svg(hist["counts"], hist["edges"], m["sigma2"], m["q"], m["edge"],
                            m.get("tail_frac", 0.0))
        return tpl.TemplateResponse(request, "esd.html", {"m": m, "chart": chart})

    @app.get("/run/{run_id}/qdiff", response_class=HTMLResponse)
    def qdiff(request: Request, run_id: str):
        ri = run_info(run_id)
        rep = blob_json(ri["artifacts"]["qdiff_report.json"])
        # X-osa je K/m (udeo dimenzija), NE udeo energije spektra: udeo energije
        # se razlikuje po sloju, pa agregacija po njemu mesa slojeve i daje
        # cik-cak koji izgleda kao nalaz. K/m je isti za sve slojeve jedne uloge.
        series: dict[str, list] = {}
        acc: dict[tuple, list] = {}
        for c in rep["matrices"]:
            m = c["shape"][1]
            for k, b in c["bands"].items():
                acc.setdefault((c["role"], round(int(k) / m, 3)), []).append(b["energy_kept"])
        for (role, frac), vals in sorted(acc.items()):
            series.setdefault(role, []).append((frac, sum(vals) / len(vals)))
        return tpl.TemplateResponse(request, "qdiff.html",
                                    {"story": story(ri), "rep": rep, "chart": svg.bands_svg(series),
                                     "run_id": run_id})

    @app.get("/run/{run_id}/lens", response_class=HTMLResponse)
    def lens(request: Request, run_id: str):
        ri = run_info(run_id)
        rep = blob_json(ri["artifacts"]["lens_report.json"])
        return tpl.TemplateResponse(request, "lens.html", {"rep": rep, "story": story(ri)})

    @app.get("/run/{run_id}/jspace", response_class=HTMLResponse)
    def jspace(request: Request, run_id: str):
        ri = run_info(run_id)
        m = blob_json(ri["artifacts"]["jspace_map.json"])
        Ls = m["layers"]
        chart = svg.depth_svg(Ls, [m["per_layer"][str(L)]["var_frac"] for L in Ls],
                              [m["per_layer"][str(L)]["entropy"] for L in Ls],
                              m["regions"], tuple(m["expected_var"]), m["n_layers"])
        return tpl.TemplateResponse(request, "jspace.html", {"m": m, "chart": chart, "story": story(ri)})

    @app.get("/run/{run_id}/sae", response_class=HTMLResponse)
    def sae(request: Request, run_id: str):
        ri = run_info(run_id)
        rep = blob_json(ri["artifacts"]["sae_report.json"])
        return tpl.TemplateResponse(request, "sae.html", {"rep": rep, "story": story(ri)})

    @app.get("/run/{run_id}/ppl", response_class=HTMLResponse)
    def ppl(request: Request, run_id: str):
        ri = run_info(run_id)
        rep = blob_json(ri["artifacts"]["ppl_ladder.json"])
        pts = [(1 - r["kept_quarter_mean"], r["ppl_rel"], r["file"].split("-")[-1].replace(".gguf", ""))
               for r in rep["rows"][1:] if "kept_quarter_mean" in r]
        chart = svg.scatter_svg(pts, "izgubljeno iz podprostora", "pogoršanje perplexity") if pts else ""
        return tpl.TemplateResponse(request, "ppl.html", {"rep": rep, "chart": chart, "story": story(ri)})

    # ------------------------------------------------------------------ PSE-L
    def _groups():
        g = {"online": [], "lokalno": [], "prilagođeno": []}
        for pid, v in llmmod.PROVIDERS.items():
            g[v[1]].append((pid, v[0]))
        return [("Online", g["online"]), ("Lokalno", g["lokalno"]),
                ("Prilagođeno", g["prilagođeno"])]

    def _find_ggufs():
        from pathlib import Path as _P
        out = []
        for root in (_P.home() / "models", _P.home() / "Desktop", _P.home() / "MathTutor" / "models"):
            if root.is_dir():
                out += [str(p) for p in root.rglob("*.gguf") if "qwen" in p.name.lower()
                        and "instruct" in p.name.lower()][:40]
        return sorted(set(out))

    @app.get("/talk", response_class=HTMLResponse)
    def talk(request: Request):
        cfg = llmmod.load_config()
        pid = cfg.get("active")
        if not pid and not cfg.get("dismissed"):
            return setup(request)
        e = llmmod.entry(cfg, pid) if pid else {}
        return tpl.TemplateResponse(request, "talk.html", {
            "active": pid, "active_label": llmmod.PROVIDERS[pid][0] if pid else "",
            "active_model": e.get("model") or e.get("gguf", "").split("/")[-1],
            "online": bool(pid) and llmmod.PROVIDERS[pid][1] == "online",
            "history": psel.history(lib)})

    @app.get("/llm/setup", response_class=HTMLResponse)
    def setup(request: Request):
        cfg = llmmod.load_config()
        return tpl.TemplateResponse(request, "setup.html",
                                    {"groups": _groups(), "current": cfg.get("active") or "anthropic"})

    @app.get("/llm/fields", response_class=HTMLResponse)
    def fields(request: Request, pid: str = "anthropic"):
        if pid not in llmmod.PROVIDERS:
            raise HTTPException(400)
        label, grp, kind, base, env, keyurl = llmmod.PROVIDERS[pid]
        cfg = llmmod.load_config()
        e = llmmod.entry(cfg, pid)
        return tpl.TemplateResponse(request, "fields.html", {
            "kind": kind, "e": e, "env": env, "keyurl": keyurl,
            "needs_key": bool(env) or pid == "custom", "masked": llmmod.masked(e.get("key", "")),
            "show_url": pid in ("custom", "ollama", "lmstudio", "llamacpp", "vllm"),
            "ggufs": _find_ggufs() if kind == "local_gguf" else []})

    def _form_entry(pid, key, base_url, model, gguf):
        cfg = llmmod.load_config()
        e = llmmod.entry(cfg, pid)
        if key:
            e["key"] = key.strip()
        if base_url:
            e["base_url"] = base_url.strip()
        if model:
            e["model"] = model.strip()
        if gguf:
            e["gguf"] = gguf.strip()
        return cfg, e

    @app.post("/llm/models", response_class=HTMLResponse)
    def models_list(pid: str = Form(...), key: str = Form(""), base_url: str = Form(""),
                    model: str = Form(""), gguf: str = Form("")):
        _, e = _form_entry(pid, key, base_url, model, gguf)
        try:
            ms = llmmod.list_models(pid, e)
        except llmmod.LLMError as err:
            return HTMLResponse(f'<option disabled selected>{html_escape(str(err))[:120]}</option>')
        cur = e.get("model") or (llmmod.CLAUDE_DEFAULT if pid == "anthropic" else "")
        return HTMLResponse("".join(
            f'<option {"selected" if m == cur else ""}>{html_escape(m)}</option>' for m in ms)
            or '<option disabled selected>server nije vratio nijedan model</option>')

    @app.post("/llm/save", response_class=HTMLResponse)
    def llm_save(request: Request, pid: str = Form(...), key: str = Form(""),
                 base_url: str = Form(""), model: str = Form(""), gguf: str = Form("")):
        if pid not in llmmod.PROVIDERS:
            raise HTTPException(400)
        cfg, e = _form_entry(pid, key, base_url, model, gguf)
        # kljuc iz okruzenja se ne prepisuje u fajl
        env = llmmod.PROVIDERS[pid][4]
        if env and e.get("key") == os.environ.get(env) and not key:
            e.pop("key", None)
        cfg.setdefault("providers", {})[pid] = e
        cfg["active"] = pid
        cfg["dismissed"] = False
        llmmod.save_config(cfg)
        return talk(request)

    @app.post("/llm/skip", response_class=HTMLResponse)
    def llm_skip(request: Request):
        cfg = llmmod.load_config()
        cfg["dismissed"] = True
        llmmod.save_config(cfg)
        return talk(request)

    @app.post("/chat", response_class=HTMLResponse)
    def chat_post(request: Request, text: str = Form(...)):
        cfg = llmmod.load_config()
        pid = cfg.get("active")
        if not pid:
            raise HTTPException(400, "model nije podesen")
        text = text.strip()[:4000]
        reply = psel.respond(lib, pid, llmmod.entry(cfg, pid), text)
        out = ""
        for m in ({"role": "user", "text": text}, {"role": "assistant", "text": reply}):
            out += tpl.get_template("msg.html").render(m=m, render=psel.render)
        return HTMLResponse(out)

    @app.post("/chat/stream")
    def chat_stream(text: str = Form(...)):
        """Odgovor rec po rec (bez cekanja minutima na ceo). Svaki dogadjaj nosi
        CEO dosadasnji filtrirani tekst; poslednji nosi i gotov HTML sa dugmadima."""
        cfg = llmmod.load_config()
        pid = cfg.get("active")
        if not pid:
            raise HTTPException(400, "model nije podesen")
        text = text.strip()[:4000]
        e = llmmod.entry(cfg, pid)

        def gen():
            for kind, val in psel.respond_stream(lib, pid, e, text):
                if kind == "partial":
                    yield f"event: partial\ndata: {json.dumps({'text': val})}\n\n"
                else:
                    html_out = tpl.get_template("msg.html").render(
                        m={"role": "assistant", "text": val}, render=psel.render)
                    yield f"event: done\ndata: {json.dumps({'html': html_out})}\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/inbox", response_class=HTMLResponse)
    def inbox(request: Request):
        names = dict(lib.db.execute("SELECT model_hash, display_name FROM models").fetchall())
        items = [{**it, "model": names.get(it["model_hash"], it["model_hash"][:12])}
                 for it in lib.inbox()]
        deaths = []
        for rid, reason in lib.db.execute(
                "SELECT run_id, reason FROM deaths ORDER BY t DESC LIMIT 20").fetchall():
            if "—" in reason:
                txt = pg.refusal([f.strip() for f in reason.split("—")[-1].split(",")])
            else:
                txt = reason
            deaths.append((rid, txt))
        return tpl.TemplateResponse(request, "inbox.html",
                                    {"items": items, "active": lib.active_runs(), "deaths": deaths})

    @app.post("/inbox/{item_id}/seen")
    def inbox_seen(item_id: str):
        """Druga i poslednja mutacija u aplikaciji. Ne dira nijedan rezultat:
        stavka ostaje zapisana, samo prestaje da bude nova."""
        lib.mark_seen(item_id)
        return RedirectResponse("/?open=/inbox", status_code=303)

    # ------------------------------------------------------------------ artefakti
    @app.get("/artifact/{h}")
    def artifact(request: Request, h: str):
        """PECAT JE KES (§7.2): sadrzaj pod ovim hesom se nikad ne menja."""
        if request.headers.get("if-none-match") == f'"{h}"':
            return Response(status_code=304)
        try:
            data = lib.get_blob(h)
        except FileNotFoundError:
            raise HTTPException(404)
        return Response(data, media_type="application/json",
                        headers={"ETag": f'"{h}"',
                                 "Cache-Control": "public, max-age=31536000, immutable"})

    # ------------------------------------------------------------------ jedina mutacija
    @app.post("/run/{run_id}/note")
    def note(run_id: str, anchor: str = Form(...), text: str = Form(...),
             author: str = Form("andrej")):
        """AZ-6: beleska se dodaje, nikad menja. PUT/DELETE ne postoje."""
        text = text.strip()
        if not text:
            raise HTTPException(400, "prazna beleska")
        ri = run_info(run_id)
        try:
            lib.add_note(run_id, author, anchor, text[:4000])
        except ValueError as e:
            raise HTTPException(400, str(e))
        return RedirectResponse(f"/model/{ri['model_hash']}", status_code=303)

    # ------------------------------------------------------------------ SSE
    @app.get("/stream")
    async def stream(request: Request):
        """Jedan kanal. Samo id-jevi i skalari; heartbeat na 15 s (SSH tunel)."""
        async def gen():
            seen = {r[0] for r in lib.db.execute(
                "SELECT run_id FROM runs WHERE state='sealed'").fetchall()}
            last_hb = 0.0
            while not await request.is_disconnected():
                for a in lib.active_runs():
                    if a["age_s"] < 600:
                        payload = {"run_id": a["run_id"], "kind": a["kind"],
                                   "step": a["step"], "total": a["total"],
                                   "theta": round(a["theta"][-1], 2),
                                   "trace": [round(x, 2) for x in a["theta"][-60:]]}
                        yield f"event: unit.progress\ndata: {json.dumps(payload)}\n\n"
                n_unseen = len(lib.inbox(True))
                yield f"event: inbox.new\ndata: {json.dumps({'unseen': n_unseen})}\n\n"
                now_sealed = {r[0] for r in lib.db.execute(
                    "SELECT run_id FROM runs WHERE state='sealed'").fetchall()}
                for rid in now_sealed - seen:
                    yield f"event: run.sealed\ndata: {json.dumps({'run_id': rid})}\n\n"
                seen = now_sealed
                if time.time() - last_hb > 15:
                    yield ": heartbeat\n\n"
                    yield f"event: guardian.tier\ndata: {json.dumps(shield_state())}\n\n"
                    last_hb = time.time()
                await asyncio.sleep(2.0)
        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    return app


def _fmt(x, nd=2, pct=False):
    if x is None:
        return "—"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(x):
        return "—"
    return f"{100*x:+.{nd}f}%" if pct else f"{x:.{nd}f}"


def _isfin(x):
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False
