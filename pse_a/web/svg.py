"""SVG se crta na serveru iz zapecacenih artefakata -- nijedan grafik se ne
racuna u browseru, i nijedan nije potreban JS (§7.2). Boje su CSS promenljive
pa grafik prati svetlu i tamnu temu."""
from __future__ import annotations

import html
import math

import numpy as np

from ..core.spectral import mp


def esd_svg(counts, edges, sigma2: float, q: float, edge: float, tail_frac: float = 0.0,
            w: int = 600, h: int = 230) -> str:
    """ESD na LOGARITAMSKOJ x-osi, gustina po log-jedinici.

    MP kriva se mnozi sa (1 - udeo repa): MP opisuje samo bulk, a mesavina je
    F_mix = (1-p)·F_MP (§3.1). Bez toga kriva nadvisuje histogram i izgleda kao
    da bulk ne pripada MP-u, iako pripada.
    """
    counts = np.asarray(counts, float); edges = np.asarray(edges, float)
    N = counts.sum()
    if N == 0:
        return "<svg/>"
    e = edges.copy()
    if e[0] <= 0:
        e[0] = e[1] / 2 if e[1] > 0 else 1e-12
    le = np.log10(np.maximum(e, 1e-300))
    dens = counts / (N * np.diff(le))                      # gustina po dekadi
    a, b = mp.support(q, sigma2)
    xs = np.geomspace(max(a, e[0]) * 1.0001, b * 0.9999, 400)
    g = (1.0 - tail_frac) * xs * mp.pdf(xs, q, sigma2) * np.log(10)
    x0, x1 = float(le[0]), float(le[-1])
    ymax = float(max(dens.max(), g.max() if g.size else 0) * 1.08)
    pad = 30
    X = lambda x: pad + (w - 2 * pad) * (x - x0) / (x1 - x0)
    Y = lambda y: h - pad - (h - 2 * pad) * y / ymax
    bars = "".join(
        f'<rect x="{X(l0):.1f}" y="{Y(d):.1f}" width="{max(X(l1)-X(l0)-0.2,0.3):.1f}" '
        f'height="{h - pad - Y(d):.1f}" class="bar"/>'
        for l0, l1, d in zip(le[:-1], le[1:], dens) if d > 0)
    line = " ".join(f"{X(np.log10(x)):.1f},{Y(y):.1f}" for x, y in zip(xs, g))
    ticks = ""
    for dec in range(int(np.ceil(x0)), int(np.floor(x1)) + 1):
        tx = X(dec)
        ticks += (f'<line x1="{tx:.1f}" y1="{h-pad}" x2="{tx:.1f}" y2="{h-pad+4}" class="axis"/>'
                  f'<text x="{tx:.1f}" y="{h-pad+15}" class="lbl" text-anchor="middle">1e{dec}</text>')
    ex = X(np.log10(edge)) if edge > 0 else pad
    return (f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" '
            f'aria-label="Raspodela sopstvenih vrednosti, log skala, sa Marchenko-Pastur krivom bulk-a">'
            f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" class="axis"/>{ticks}'
            f'{bars}<polyline points="{line}" class="mp"/>'
            f'<line x1="{ex:.1f}" y1="{pad-6}" x2="{ex:.1f}" y2="{h-pad}" class="edgeline"/>'
            f'<text x="{ex+4:.1f}" y="{pad+2}" class="lbl">MP ivica</text>'
            f'<text x="{w-pad}" y="{pad+2}" class="lbl" text-anchor="end">rep →</text></svg>')


def alpha_color(a: float | None) -> str:
    """alfa 2..8 -> nijansa; van opsega i NaN posebno. Tekst uz boju je obavezan."""
    if a is None or not math.isfinite(a):
        return "var(--na)"
    t = min(max((a - 2.0) / 6.0, 0.0), 1.0)
    return f"color-mix(in oklab, var(--trained) {100*(1-t):.0f}%, var(--randomlike))"


def bands_svg(series: dict[str, list[tuple[float, float]]], w: int = 560, h: int = 220) -> str:
    """Zadrzana energija po udelu spektra, po jedna linija na ulogu."""
    pad = 34
    allv = [v for pts in series.values() for _, v in pts]
    if not allv:
        return "<svg/>"
    lo = min(min(allv), 0.95); hi = 1.0
    xmax = max(x for pts in series.values() for x, _ in pts) or 1.0
    X = lambda x: pad + (w - 2 * pad - 70) * x / xmax
    Y = lambda y: h - pad - (h - 2 * pad) * (y - lo) / (hi - lo)
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" '
           f'aria-label="Zadrzana energija podprostora po dubini spektra">',
           f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" class="axis"/>',
           f'<text x="{pad}" y="{h-10}" class="lbl">udeo dimenzija podprostora K/m →</text>',
           f'<text x="4" y="{pad}" class="lbl">{hi:.3f}</text>',
           f'<text x="4" y="{h-pad}" class="lbl">{lo:.3f}</text>']
    for i, (role, pts) in enumerate(sorted(series.items())):
        pts = sorted(pts)
        poly = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in pts)
        out.append(f'<polyline points="{poly}" class="s s{i % 8}"/>')
        out += [f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="2.2" class="d{i % 8}"/>' for x, y in pts]
        x, y = pts[-1]
        out.append(f'<text x="{X(x)+6:.1f}" y="{Y(y)+3:.1f}" class="lbl s{i % 8}t">'
                   f'{html.escape(role)}</text>')
    out.append("</svg>")
    return "".join(out)


def depth_svg(layers: list[int], var: list[float], ent: list[float], regions: dict,
              expected: tuple[float, float], n_layers: int, w: int = 600, h: int = 240) -> str:
    """Profil dubine: udeo varijanse u J-prostoru (puna linija, leva osa) i
    entropija dekompozicije (isprekidana, desna osa), obojeni regioni i
    ignition marker. Ocekivani opseg udela je osencen."""
    pad = 36
    if not layers:
        return "<svg/>"
    vmax = max(max(var) * 1.15, expected[1] * 1.3, 1e-6)
    emax = max(max(ent) * 1.15, 1e-6)
    X = lambda L: pad + (w - 2 * pad) * L / max(n_layers - 1, 1)
    Yv = lambda v: h - pad - (h - 2 * pad) * v / vmax
    Ye = lambda e: h - pad - (h - 2 * pad) * e / emax
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" '
           f'aria-label="Profil J-prostora po dubini">']
    for name, cls in (("sensory", "rs"), ("workspace", "rw"), ("motor", "rm")):
        Ls = regions.get(name) or []
        if Ls:
            x0, x1 = X(min(Ls)) - 6, X(max(Ls)) + 6
            out.append(f'<rect x="{x0:.1f}" y="{pad-10}" width="{x1-x0:.1f}" '
                       f'height="{h-2*pad+10}" class="{cls}"/>')
    out.append(f'<rect x="{pad}" y="{Yv(expected[1]):.1f}" width="{w-2*pad}" '
               f'height="{Yv(expected[0])-Yv(expected[1]):.1f}" class="band"/>')
    out.append(f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" class="axis"/>')
    pv = " ".join(f"{X(L):.1f},{Yv(v):.1f}" for L, v in zip(layers, var))
    pe = " ".join(f"{X(L):.1f},{Ye(e):.1f}" for L, e in zip(layers, ent))
    out.append(f'<polyline points="{pv}" class="s s0"/>')
    out.append(f'<polyline points="{pe}" class="s s2 dash"/>')
    out += [f'<circle cx="{X(L):.1f}" cy="{Yv(v):.1f}" r="2.5" class="d0"/>'
            for L, v in zip(layers, var)]
    ig = regions.get("ignition")
    if ig is not None:
        out.append(f'<line x1="{X(ig):.1f}" y1="{pad-10}" x2="{X(ig):.1f}" y2="{h-pad}" class="edgeline"/>'
                   f'<text x="{X(ig)+4:.1f}" y="{pad}" class="lbl">ignition</text>')
    for L in layers:
        out.append(f'<text x="{X(L):.1f}" y="{h-pad+14}" class="lbl" text-anchor="middle">{L}</text>')
    out.append(f'<text x="{pad}" y="{h-6}" class="lbl">sloj →</text>'
               f'<text x="4" y="{pad-14}" class="lbl s0t">udeo varijanse</text>'
               f'<text x="{w-4}" y="{pad-14}" class="lbl s2t" text-anchor="end">entropija ⋯</text>')
    out.append("</svg>")
    return "".join(out)


def scatter_svg(points: list[tuple[float, float, str]], xlabel: str, ylabel: str,
                w: int = 600, h: int = 260) -> str:
    """Tacke (x, y, oznaka). Koristi se za spektar-protiv-ponasanja (§4.5)."""
    pad = 44
    if not points:
        return "<svg/>"
    xs = [p[0] for p in points]; ys = [p[1] for p in points]
    x0, x1 = min(0.0, min(xs)), max(xs) * 1.1 or 1e-6
    y0, y1 = min(0.0, min(ys)), max(ys) * 1.1 or 1e-6
    X = lambda x: pad + (w - 2 * pad) * (x - x0) / (x1 - x0)
    Y = lambda y: h - pad - (h - 2 * pad) * (y - y0) / (y1 - y0)
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="{html.escape(ylabel)} prema {html.escape(xlabel)}">',
           f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" class="axis"/>',
           f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" class="axis"/>',
           f'<text x="{w-pad}" y="{h-10}" class="lbl" text-anchor="end">{html.escape(xlabel)} →</text>',
           f'<text x="6" y="{pad-10}" class="lbl">{html.escape(ylabel)} ↑</text>',
           f'<text x="{pad-4}" y="{h-pad+4}" class="lbl" text-anchor="end">{y0:.2g}</text>',
           f'<text x="{pad-4}" y="{pad+4}" class="lbl" text-anchor="end">{y1:.2g}</text>',
           f'<text x="{pad}" y="{h-pad+14}" class="lbl">{x0:.2g}</text>',
           f'<text x="{w-pad}" y="{h-pad+14}" class="lbl" text-anchor="end">{x1:.2g}</text>']
    for x, y, lab in points:
        out.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4" class="d0"/>'
                   f'<text x="{X(x)+6:.1f}" y="{Y(y)-6:.1f}" class="lbl">{html.escape(lab)}</text>')
    out.append("</svg>")
    return "".join(out)
