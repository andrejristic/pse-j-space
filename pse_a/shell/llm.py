"""Jezicki modeli za PSE-L razgovor -- mnogo provajdera, jedan interfejs.

Tri vrste pristupa:
  anthropic      -- Claude, preko ZVANICNOG SDK-a (pip anthropic)
  openai_compat  -- sve sto govori OpenAI /chat/completions protokol: OpenAI,
                    Gemini (Googlov kompatibilni endpoint), Groq, Mistral, DeepSeek,
                    xAI, OpenRouter (stotine modela iza jednog kljuca), Together,
                    Fireworks, Cerebras, DeepInfra, NVIDIA, Hugging Face, Cohere,
                    SambaNova, Moonshot, Perplexity... plus lokalni serveri (Ollama,
                    LM Studio, llama.cpp server, vLLM) i proizvoljna adresa
  local_gguf     -- GGUF direktno kroz nas runtime; radi bez interneta (minimum)

Nazivi modela se NE zakucavaju u kod (osim podrazumevanog Claude-a): posle
unosa kljuca model se bira iz liste koju provajder sam vrati (/models), pa
lista ne zastareva.

Kljucevi: ~/.config/pse-a/llm.json, prava 0600. Nikad se ne loguju i ne
salju nikome osim provajderu kome pripadaju. Spolja se salju samo SAZECI
rezultata (price i brojevi), nikad model, tezine ni aktivacije.
"""
from __future__ import annotations

import json
import os
import stat
import urllib.error
import urllib.request
from pathlib import Path

CLAUDE_DEFAULT = "claude-opus-5"

# id: (naziv, grupa, vrsta, osnovna adresa, promenljiva okruzenja, gde se dobija kljuc)
PROVIDERS: dict[str, tuple] = {
    "anthropic":   ("Claude (Anthropic)", "online", "anthropic", "", "ANTHROPIC_API_KEY",
                    "https://console.anthropic.com/settings/keys"),
    "openai":      ("ChatGPT / OpenAI", "online", "openai_compat", "https://api.openai.com/v1",
                    "OPENAI_API_KEY", "https://platform.openai.com/api-keys"),
    "gemini":      ("Gemini (Google)", "online", "openai_compat",
                    "https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY",
                    "https://aistudio.google.com/apikey"),
    "groq":        ("Groq", "online", "openai_compat", "https://api.groq.com/openai/v1",
                    "GROQ_API_KEY", "https://console.groq.com/keys"),
    "mistral":     ("Mistral", "online", "openai_compat", "https://api.mistral.ai/v1",
                    "MISTRAL_API_KEY", "https://console.mistral.ai/api-keys"),
    "deepseek":    ("DeepSeek", "online", "openai_compat", "https://api.deepseek.com/v1",
                    "DEEPSEEK_API_KEY", "https://platform.deepseek.com/api_keys"),
    "xai":         ("Grok (xAI)", "online", "openai_compat", "https://api.x.ai/v1",
                    "XAI_API_KEY", "https://console.x.ai"),
    "openrouter":  ("OpenRouter (stotine modela)", "online", "openai_compat",
                    "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
                    "https://openrouter.ai/keys"),
    "together":    ("Together AI", "online", "openai_compat", "https://api.together.xyz/v1",
                    "TOGETHER_API_KEY", "https://api.together.ai/settings/api-keys"),
    "fireworks":   ("Fireworks AI", "online", "openai_compat",
                    "https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY",
                    "https://fireworks.ai/account/api-keys"),
    "cerebras":    ("Cerebras", "online", "openai_compat", "https://api.cerebras.ai/v1",
                    "CEREBRAS_API_KEY", "https://cloud.cerebras.ai"),
    "deepinfra":   ("DeepInfra", "online", "openai_compat", "https://api.deepinfra.com/v1/openai",
                    "DEEPINFRA_API_KEY", "https://deepinfra.com/dash/api_keys"),
    "nvidia":      ("NVIDIA NIM", "online", "openai_compat", "https://integrate.api.nvidia.com/v1",
                    "NVIDIA_API_KEY", "https://build.nvidia.com"),
    "huggingface": ("Hugging Face", "online", "openai_compat", "https://router.huggingface.co/v1",
                    "HF_TOKEN", "https://huggingface.co/settings/tokens"),
    "cohere":      ("Cohere", "online", "openai_compat", "https://api.cohere.ai/compatibility/v1",
                    "COHERE_API_KEY", "https://dashboard.cohere.com/api-keys"),
    "sambanova":   ("SambaNova", "online", "openai_compat", "https://api.sambanova.ai/v1",
                    "SAMBANOVA_API_KEY", "https://cloud.sambanova.ai"),
    "moonshot":    ("Kimi (Moonshot)", "online", "openai_compat", "https://api.moonshot.ai/v1",
                    "MOONSHOT_API_KEY", "https://platform.moonshot.ai"),
    "perplexity":  ("Perplexity", "online", "openai_compat", "https://api.perplexity.ai",
                    "PERPLEXITY_API_KEY", "https://www.perplexity.ai/settings/api"),
    "ollama":      ("Ollama (lokalno)", "lokalno", "openai_compat", "http://localhost:11434/v1",
                    "", "https://ollama.com"),
    "lmstudio":    ("LM Studio (lokalno)", "lokalno", "openai_compat", "http://localhost:1234/v1",
                    "", "https://lmstudio.ai"),
    "llamacpp":    ("llama.cpp server (lokalno)", "lokalno", "openai_compat",
                    "http://localhost:8080/v1", "", "https://github.com/ggml-org/llama.cpp"),
    "vllm":        ("vLLM (lokalno)", "lokalno", "openai_compat", "http://localhost:8000/v1",
                    "", "https://docs.vllm.ai"),
    "local_gguf":  ("GGUF fajl, bez servera (ugrađeno)", "lokalno", "local_gguf", "", "", ""),
    "custom":      ("Druga adresa (OpenAI-kompatibilna)", "prilagođeno", "openai_compat", "",
                    "", ""),
}


class LLMError(Exception):
    pass


# ------------------------------------------------------------------ podesavanja
def config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or ""
    if not base or "/snap/" in base:
        base = str(Path.home() / ".config")
    return Path(base) / "pse-a" / "llm.json"


def load_config() -> dict:
    p = config_path()
    if not p.exists():
        return {"active": None, "providers": {}, "dismissed": False}
    try:
        return json.loads(p.read_text())
    except ValueError:
        return {"active": None, "providers": {}, "dismissed": False}


def save_config(cfg: dict) -> None:
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, "w") as f:
        json.dump(cfg, f, indent=1)
    os.replace(tmp, p)
    os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)


def entry(cfg: dict, pid: str) -> dict:
    e = dict(cfg.get("providers", {}).get(pid, {}))
    _, _, kind, base, env, _ = PROVIDERS[pid]
    e.setdefault("base_url", base)
    if not e.get("key") and env:
        e["key"] = os.environ.get(env, "")
    return e


def masked(key: str) -> str:
    return "" if not key else (key[:4] + "…" + key[-4:] if len(key) > 12 else "••••")


# ------------------------------------------------------------------ OpenAI-kompatibilno
def _http(url: str, key: str, body: dict | None = None, timeout: float = 60.0):
    headers = {"Content-Type": "application/json", "User-Agent": "pse-a"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method="POST" if body is not None else "GET")
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "ignore")[:300]
        raise LLMError(f"HTTP {e.code}: {detail}") from None
    except urllib.error.URLError as e:
        raise LLMError(f"nema veze sa {url.split('/')[2]}: {e.reason}") from None


def _oa_models(base: str, key: str) -> list[str]:
    with _http(base.rstrip("/") + "/models", key, timeout=20) as r:
        d = json.loads(r.read())
    items = d.get("data", d.get("models", d if isinstance(d, list) else []))
    out = []
    for m in items:
        mid = m.get("id") or m.get("name") if isinstance(m, dict) else m
        if mid:
            out.append(str(mid).removeprefix("models/"))
    return sorted(set(out))


def _oa_chat(base: str, key: str, model: str, system: str, messages: list[dict],
             max_tokens: int = 2048):
    local = any(h in base for h in ("localhost", "127.0.0.1"))
    if local:
        # lokalni model na CPU-u: 2048 tokena je pola sata, a server (Ollama)
        # obradjuje jedan zahtev u isto vreme -- dug odgovor blokira sledeci
        max_tokens = min(max_tokens, 600)
    body = {"model": model, "stream": True, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}] + messages}
    # lokalni model na slabom procesoru ume da cuti minutima pre prvog tokena
    tmo = 1200 if local else 300
    try:
        yield from _oa_stream(base, key, body, tmo)
    except (TimeoutError, OSError) as e:
        raise LLMError(f"model nije odgovorio na vreme ({type(e).__name__})") from None


def _oa_stream(base, key, body, tmo):
    with _http(base.rstrip("/") + "/chat/completions", key, body, timeout=tmo) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                return
            try:
                d = json.loads(chunk)
            except ValueError:
                continue
            for ch in d.get("choices", []):
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    yield piece


# ------------------------------------------------------------------ Claude (zvanicni SDK)
def _claude_client(key: str):
    try:
        import anthropic
    except ImportError:
        raise LLMError("zvanicni Claude SDK nije instaliran (pip install anthropic u venv-u pse-a)") from None
    return anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()


def _claude_models(key: str) -> list[str]:
    c = _claude_client(key)
    try:
        return [m.id for m in c.models.list()]
    except Exception as e:                       # noqa: BLE001 -- prevodi se za coveka
        raise LLMError(_claude_err(e)) from None


def _claude_err(e) -> str:
    import anthropic
    if isinstance(e, anthropic.AuthenticationError):
        return "kljuc nije prihvacen"
    if isinstance(e, anthropic.PermissionDeniedError):
        return "kljuc nema dozvolu za ovaj model"
    if isinstance(e, anthropic.NotFoundError):
        return "model ne postoji ili nije dostupan ovom kljucu"
    if isinstance(e, anthropic.RateLimitError):
        return "previse zahteva -- pokusaj za minut"
    if isinstance(e, anthropic.APIConnectionError):
        return "nema veze sa Anthropic API-jem"
    if isinstance(e, anthropic.APIStatusError):
        return f"greska servera ({e.status_code})"
    return str(e)


def _claude_chat(key: str, model: str, system: str, messages: list[dict],
                 max_tokens: int = 16000):
    """Streaming (duzi odgovori ne udaraju u timeout). Na claude-opus-5 i Fable 5.1
    ukljuceni su serverski fallback-ovi: ako bezbednosni klasifikator odbije zahtev,
    API ga sam ponovo izvrsi na preporucenom modelu umesto da vrati odbijanje."""
    import anthropic
    c = _claude_client(key)
    kw = dict(model=model, max_tokens=max_tokens, system=system, messages=messages)
    if model in ("claude-opus-5", "claude-fable-5-1"):
        stream_ctx = c.beta.messages.stream(betas=["server-side-fallback-2026-07-01"],
                                            extra_body={"fallbacks": "default"}, **kw)
    else:
        stream_ctx = c.messages.stream(**kw)
    try:
        with stream_ctx as stream:
            for text in stream.text_stream:
                yield text
            final = stream.get_final_message()
        if final.stop_reason == "refusal":
            yield "\n\n(Model je odbio da odgovori na ovo pitanje.)"
        elif final.stop_reason == "max_tokens":
            yield "\n\n(Odgovor je prekinut na granici dužine.)"
    except anthropic.APIError as e:
        raise LLMError(_claude_err(e)) from None


# ------------------------------------------------------------------ lokalni GGUF
_LOCAL: dict = {}


def _local_chat(path: str, system: str, messages: list[dict], max_new: int = 400):
    from ..core.runtime.qwen2 import Qwen2Runtime, generate, chat_prompt
    if not path or not Path(path).exists():
        raise LLMError("izaberi postojeci GGUF fajl (qwen2 instruct)")
    rt = _LOCAL.get(path)
    if rt is None:
        _LOCAL.clear()
        rt = _LOCAL[path] = Qwen2Runtime(path)
    turns = [(m["role"], m["content"]) for m in messages]
    ids = chat_prompt(rt, system, turns)
    stop = tuple(i for i in (rt.tok.token_to_id("<|im_end|>"),
                             rt.tok.token_to_id("<|endoftext|>")) if i is not None)
    buf = []
    for tok in generate(rt, ids, max_new=max_new, temperature=0.4, stop_ids=stop):
        buf.append(tok)
        text = rt.decode(buf)
        if not text.endswith("�"):          # ne salji polovinu UTF-8 znaka
            yield text
            buf = []


# ------------------------------------------------------------------ javni interfejs
def list_models(pid: str, e: dict) -> list[str]:
    kind = PROVIDERS[pid][2]
    if kind == "anthropic":
        return _claude_models(e.get("key", ""))
    if kind == "local_gguf":
        return []
    if not e.get("base_url"):
        raise LLMError("unesi adresu servera")
    return _oa_models(e["base_url"], e.get("key", ""))


def chat(pid: str, e: dict, system: str, messages: list[dict]):
    kind = PROVIDERS[pid][2]
    if kind == "anthropic":
        yield from _claude_chat(e.get("key", ""), e.get("model") or CLAUDE_DEFAULT,
                                system, messages)
    elif kind == "local_gguf":
        yield from _local_chat(e.get("gguf", ""), system, messages)
    else:
        if not e.get("model"):
            raise LLMError("izaberi model")
        yield from _oa_chat(e["base_url"], e.get("key", ""), e["model"], system, messages)
