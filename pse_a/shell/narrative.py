"""Ljudski registar -- AZ-7, [SPEC] 11.2.

"Every result tells two stories": matematicka (paneli) i ljudska (ovde).
Obe iz ISTIH artefakata; svaka tvrdnja nosi hes artefakta iz kog dolazi, pa
se dva registra ne mogu razici. Generator je DETERMINISTICKI: isti zapecaceni
artefakti -> isti tekst, pa je prica nepromenljiva koliko i pecat, i racuna se
bez jezickog modela (nikakav LLM poziv po otvaranju stranice, §7.4).

Pravila [PSE §5] vaze: domenski jezik, nesigurnost u samoj recenici,
bez imena formi i dijagnoza (AZ-8 test prolazi i kroz ovaj modul),
ograniceno trajanje po odeljku.
"""
from __future__ import annotations

import math


def _c(h: str) -> str:
    return f"[{h[:8]}]"


def _pct(x: float, nd: int = 1) -> str:
    return f"{100*x:.{nd}f} %"


def l1(rep: dict, h: str) -> list[str]:
    M = rep["matrices"]
    reg = {}
    for m in M:
        reg[m["regime"]] = reg.get(m["regime"], 0) + 1
    ht = reg.get("heavy_tailed", 0) + reg.get("mixed", 0)
    al = [m["alpha"] for m in M if m.get("alpha") is not None and math.isfinite(m["alpha"])]
    by_role = {}
    for m in M:
        if m.get("alpha") is not None and math.isfinite(m["alpha"]):
            by_role.setdefault(m["role"], []).append(m["alpha"])
    roles = sorted(((sum(v) / len(v), r) for r, v in by_role.items()))
    mp_ok = sum(1 for m in M if m.get("mp_fit_ok"))
    out = [f"Pregledano je {len(M)} matrica težina modela {rep['model']} {_c(h)}."]
    out.append(f"Kod {ht} od njih spektar ima težak rep, što znači da je trening u njih upisao "
               f"strukturu; preostale su bliže slučajnim {_c(h)}.")
    if roles:
        out.append(f"Najizraženiju strukturu, po proseku alfe, nose {roles[0][1]} (≈{roles[0][0]:.1f}), "
                   f"a najslabiju {roles[-1][1]} (≈{roles[-1][0]:.1f}); što je alfa manja, struktura "
                   f"je izraženija {_c(h)}.")
    if "mp_fit_ok" in M[0]:
        out.append(f"Kod {mp_ok} od {len(M)} matrica telo spektra zaista izgleda kao šum iste jačine; "
                   f"kod ostalih ta slika ne važi, pa se za njih oslanjamo samo na alfu {_c(h)}.")
    if not rep.get("kquant_clean", True):
        out.append("Napomena: dimenzije ovog modela nisu deljive sa 256, pa se njegove kvantizovane "
                   "verzije razlikuju po sastavu, ne samo po preciznosti.")
    return out


def qdiff(rep: dict, h: str) -> list[str]:
    S = rep["strata"]
    worst = min(S.items(), key=lambda kv: kv[1]["kept_quarter"])
    best = max(S.items(), key=lambda kv: kv[1]["kept_quarter"])
    conc = [k.split("|")[0] for k, v in S.items() if v["noise_ratio"] > 1.2]
    dE = sum(v["energy_change"] * v["n"] for v in S.values()) / max(sum(v["n"] for v in S.values()), 1)
    out = [f"Upoređene su dve preciznosti istog modela: {rep['ref']} i {rep['test']} {_c(h)}."]
    lab = lambda k: "{} ({})".format(k.split("|")[0], k.split("|")[1].replace("->", "→"))
    out.append(f"Glavni pravci matrica uglavnom su sačuvani: najviše je ostalo kod "
               f"{lab(best[0])}, {best[1]['kept_quarter']:.3f}, a najmanje kod "
               f"{lab(worst[0])}, {worst[1]['kept_quarter']:.3f} {_c(h)}.")
    out.append(f"Ukupna energija težina se u proseku menja za {_pct(dE, 2)} — kvantizacija je "
               f"ne premešta nego blago smanjuje {_c(h)}.")
    if conc:
        out.append(f"Kod {', '.join(sorted(set(conc)))} greška kvantizacije češće pada baš na naučene "
                   f"pravce nego na ostatak {_c(h)}.")
    out.append("Da li se ovo oseća u ponašanju modela, spektar sam ne kaže; za to je potrebno "
               "meriti i izlaz modela na tekstu.")
    return out


def lens(rep: dict, h: str) -> list[str]:
    it = rep["items"]
    out = [f"Za {len(it)} kratkih rečenica gledano je šta bi model rekao da mora da progovori na "
           f"svakom sloju {_c(h)}."]
    for x in it[:3]:
        out.append(f"„{x['prompt']}” — konačan izbor „{x['top'][-1][-1][0]}” se prvi put pojavljuje "
                   f"na sloju {x['commit_layer']} {_c(h)}.")
    out.append("Ovaj pogled vidi samo ono što je već izgovorljivo; računanje koje još nije u rečima "
               "ostaje mu skriveno.")
    return out


def jspace(m: dict, h: str) -> list[str]:
    pl = m["per_layer"]
    vs = [pl[str(L)]["var_frac"] for L in m["layers"]]
    out = [f"Mereno je koliki deo unutrašnjeg stanja model drži u obliku spremnom za izgovor, na "
           f"{len(m['layers'])} slojeva i {len(m['prompts'])} namerno dvosmislenih rečenica {_c(h)}."]
    out.append(f"Taj deo iznosi između {_pct(min(vs))} i {_pct(max(vs))} {_c(h)}.")
    r = m["regions"]
    if r.get("ignition") is not None:
        out.append(f"Na sloju {r['ignition']} model se oštro opredeljuje za jedno čitanje {_c(h)}.")
    elif r.get("resolution_ok") is False:
        out.append(f"Oštar trenutak opredeljenja nije uočen, ali mapa je gruba (svaki "
                   f"{r.get('stride')}. sloj), pa to nije dokaz da ga nema {_c(h)}.")
    else:
        out.append(f"Oštar trenutak opredeljenja nije uočen; za ovaj model podela na čulni deo, "
                   f"radni prostor i izlaz nije potvrđena {_c(h)}.")
    return out


def sae(rep: dict, h: str) -> list[str]:
    p = rep["superposition_profile"]
    out = [f"Iz {rep['n_tokens']} pozicija na sloju {rep['layer']} naučen je rečnik od "
           f"{rep['n_features']} pojmova {_c(h)}."]
    out.append(f"Rečnik rekonstruiše stanje sa preostalom greškom od {_pct(rep['fvu'])}, uz oko "
               f"{rep['l0']:.0f} aktivnih pojmova po poziciji {_c(h)}.")
    if "n_evaluable" in p:
        out.append(f"Za {p['n_evaluable']} od {p['n_alive']} živih pojmova ima dovoljno pojavljivanja "
                   f"za ocenu; od njih se {p['n_stable']} ne pojavljuje sa drugima češće nego što bi "
                   f"slučaj dao {_c(h)}.")
    else:
        out.append(f"Od {p['n_alive']} živih pojmova, {p['n_stable']} se ne pojavljuje sa drugima "
                   f"češće nego što bi slučaj dao {_c(h)}.")
    if not rep.get("accepted", True):
        out.append("Rečnik nije prošao prijem, pa nije ušao u znanje.")
    return out


def ppl(rep: dict, h: str) -> list[str]:
    R = rep["rows"]
    out = [f"Isti tekst od {rep['n_tokens']} tokena dat je svakoj od {len(R)} verzija modela {_c(h)}."]
    worst = max(R[1:], key=lambda r: r["ppl_rel"]) if len(R) > 1 else None
    if worst:
        out.append(f"Najviše trpi {worst['file']}: perplexity raste za {_pct(worst['ppl_rel'])} u "
                   f"odnosu na referencu {_c(h)}.")
    both = [r for r in R[1:] if "kept_quarter_mean" in r]
    if len(both) >= 2:
        both.sort(key=lambda r: 1 - r["kept_quarter_mean"])
        same = [r["ppl_rel"] for r in both] == sorted(r["ppl_rel"] for r in both)
        out.append("Redosled po gubitku glavnih pravaca " + ("poklapa se" if same else "ne poklapa se")
                   + f" sa redosledom po pogoršanju izlaza, na {len(both)} upoređenih verzija {_c(h)}.")
    if not rep.get("kquant_clean", True):
        out.append("Verzije ovog modela razlikuju se po sastavu, ne samo po preciznosti, pa je "
                   "ovo poređenje fajlova, a ne čista kriva po broju bita.")
    return out


GEN = {"l1_sweep": ("l1_report.json", l1), "qdiff": ("qdiff_report.json", qdiff),
       "l2_lens": ("lens_report.json", lens), "l4_jspace": ("jspace_map.json", jspace),
       "l3_sae": ("sae_report.json", sae), "ppl_ladder": ("ppl_ladder.json", ppl)}


def story(kind: str, artifacts: dict, load) -> list[str]:
    if kind not in GEN:
        return []
    name, fn = GEN[kind]
    h = artifacts.get(name)
    if not h:
        return []
    return fn(load(h), h)
