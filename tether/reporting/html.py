"""Shared HTML scaffolding for reports (self-contained page, light/dark aware)."""

from __future__ import annotations

import html
from typing import Iterable

import pandas as pd

PALETTE = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "muted": "#898781", "grid": "#e1e0d9"}

CSS = """
:root { --bg:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9; --line:#c3c2b7;
        --blue:#2a78d6; --orange:#eb6834; --aqua:#1baf7a; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7;
        --muted:#898781; --grid:#2c2c2a; --line:#383835; --blue:#3987e5; --orange:#d95926; --aqua:#199e70; } }
body { margin:0; padding:24px 16px; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
main { max-width:1100px; margin:0 auto; }
h1 { font-size:26px; margin:0 0 4px; } h2 { font-size:19px; margin:32px 0 8px; } h3 { font-size:16px; margin:20px 0 6px; }
.sub { color:var(--ink2); margin-bottom:16px; }
.tiles { display:flex; flex-wrap:wrap; gap:12px; margin:16px 0; }
.tile { background:var(--surface); border:1px solid var(--grid); border-radius:8px; padding:12px 16px; min-width:150px; flex:1; }
.tile .k { font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
.tile .v { font-size:26px; font-variant-numeric:tabular-nums; }
.tile .d { font-size:12px; color:var(--ink2); }
table { border-collapse:collapse; width:100%; background:var(--surface); font-size:13.5px; font-variant-numeric:tabular-nums; }
th, td { padding:6px 10px; border-bottom:1px solid var(--grid); text-align:left; vertical-align:top; }
th { color:var(--ink2); font-weight:600; } td.num, th.num { text-align:right; }
.wrap { overflow-x:auto; }
.chart { background:var(--surface); border:1px solid var(--grid); border-radius:8px; padding:12px; margin:8px 0 16px; }
.note { color:var(--ink2); font-size:13.5px; }
code { font-size:13px; background:var(--surface); padding:1px 4px; border-radius:3px; }
svg text { fill:var(--ink2); font-size:12px; } svg .grid { stroke:var(--grid); } svg .axis { stroke:var(--line); }
"""


def page(title: str, subtitle: str, body: str, head_extra: str = "") -> str:
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{CSS}</style>{head_extra}</head><body><main>"
            f"<h1>{html.escape(title)}</h1><div class='sub'>{html.escape(subtitle)}</div>{body}</main></body></html>")


def esc(v: object) -> str:
    return html.escape("" if v is None or (isinstance(v, float) and v != v) else str(v))


def table(df: pd.DataFrame, max_rows: int | None = None, floatfmt: str = "{:.4g}") -> str:
    if df is None or len(df) == 0:
        return "<p class='note'>none</p>"
    d = df.head(max_rows) if max_rows else df
    num = {c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])}
    head = "".join(f"<th class='{'num' if c in num else ''}'>{esc(c)}</th>" for c in d.columns)
    rows = []
    for _, r in d.iterrows():
        cells = []
        for c in d.columns:
            v = r[c]
            if c in num and isinstance(v, float) and v == v:
                v = floatfmt.format(v)
            cells.append(f"<td class='{'num' if c in num else ''}'>{esc(v)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    more = f"<p class='note'>showing {len(d)} of {len(df)} rows</p>" if max_rows and len(df) > max_rows else ""
    return f"<div class='wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>{more}"


def tiles(items: Iterable[tuple[str, str, str]]) -> str:
    return "<div class='tiles'>" + "".join(
        f"<div class='tile'><div class='k'>{esc(k)}</div><div class='v'>{esc(v)}</div><div class='d'>{esc(d)}</div></div>"
        for k, v, d in items) + "</div>"


def line_chart_svg(series: list[dict], x_label: str, y_label: str, width: int = 720, height: int = 400,
                   x_domain: tuple[float, float] = (0.0, 1.0), y_domain: tuple[float, float] = (0.0, 1.0),
                   marker_x: float | None = None, marker_label: str = "") -> str:
    """Inline SVG multi-line chart with legend, direct end labels, point tooltips, optional vertical marker.

    ``series`` items: ``{"name": str, "color": css color, "points": [(x, y, label), ...]}``.
    """
    ml, mr, mt, mb = 60, 120, 16, 44
    pw, ph = width - ml - mr, height - mt - mb
    x0, x1 = x_domain
    y0, y1 = y_domain
    sx = lambda x: ml + (x - x0) / (x1 - x0) * pw  # noqa: E731
    sy = lambda y: mt + (1 - (y - y0) / (y1 - y0)) * ph  # noqa: E731
    parts = [f"<svg viewBox='0 0 {width} {height}' width='100%' role='img' aria-label='{esc(y_label)} vs {esc(x_label)}'>"]
    for i in range(6):
        ty, tx = y0 + (y1 - y0) * i / 5, x0 + (x1 - x0) * i / 5
        parts.append(f"<line class='grid' x1='{sx(x0)}' x2='{sx(x1)}' y1='{sy(ty)}' y2='{sy(ty)}' stroke-width='1'/>")
        parts.append(f"<text x='{ml - 8}' y='{sy(ty) + 4}' text-anchor='end'>{ty:g}</text>")
        parts.append(f"<text x='{sx(tx)}' y='{mt + ph + 18}' text-anchor='middle'>{tx:g}</text>")
    parts.append(f"<line class='axis' x1='{sx(x0)}' x2='{sx(x1)}' y1='{sy(y0)}' y2='{sy(y0)}' stroke-width='1'/>")
    if marker_x is not None:
        parts.append(f"<line x1='{sx(marker_x):.1f}' x2='{sx(marker_x):.1f}' y1='{mt}' y2='{mt + ph}' stroke='var(--muted)' stroke-dasharray='4 3'/>"
                     f"<text x='{sx(marker_x) + 4:.1f}' y='{mt + 12}'>{esc(marker_label)}</text>")
    parts.append(f"<text x='{ml + pw / 2}' y='{height - 8}' text-anchor='middle'>{esc(x_label)}</text>")
    parts.append(f"<text transform='translate(14,{mt + ph / 2}) rotate(-90)' text-anchor='middle'>{esc(y_label)}</text>")
    for i, s in enumerate(series):
        pts = sorted(s["points"])
        path = " ".join(f"{'M' if j == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for j, (x, y, _) in enumerate(pts))
        parts.append(f"<path d='{path}' fill='none' stroke='{s['color']}' stroke-width='2' stroke-linejoin='round'/>")
        for x, y, label in pts:
            parts.append(f"<circle cx='{sx(x):.1f}' cy='{sy(y):.1f}' r='4' fill='{s['color']}' stroke='var(--surface)' stroke-width='2'>"
                         f"<title>{esc(s['name'])}: {esc(label)}</title></circle>")
        if pts:
            x, y, _ = pts[-1]
            parts.append(f"<text x='{sx(x) + 8:.1f}' y='{sy(y) + 4:.1f}'>{esc(s['name'])}</text>")
        ly = mt + 8 + i * 18
        parts.append(f"<rect x='{ml + pw - 150}' y='{ly - 9}' width='12' height='12' rx='2' fill='{s['color']}'/>"
                     f"<text x='{ml + pw - 132}' y='{ly + 2}'>{esc(s['name'])}</text>")
    parts.append("</svg>")
    return "".join(parts)
