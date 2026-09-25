#!/usr/bin/env python3
"""Linear against linearplus as bar charts, from ir-report.py's summary JSON.

One chart per machine and per range of IR lengths (cabinet-length, up to
8192 taps, and multi-second). Rows are callback sizes; inside each row, one
band per IR length, and inside each band one bar per implementation: p99,
with the mean drawn over it, both as a percentage of the callback's deadline. The x axis
runs from 0 to the largest value in that chart, a line marks the deadline, and
any bar past it is drawn red.

    Scripts/ir-charts.py --from-summary ir-study/data/linearplus-summary.json \\
        --html ir-study/linearplus-charts.html
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

LONG_FROM_TAPS = 8192

# Chart geometry, in SVG user units.
BAR = 10  # one bar's height
GAP = 5  # between bars, and above and below each row's bars
LEFT = 118  # frames label, then IR length label
RIGHT = 36
TOP = 46  # room for the axis labels
BOTTOM = 18
PLOT = 620  # width of the plotting area


def nice_ticks(maximum: float) -> list[float]:
    """Round tick values from 0 up to at most `maximum`, four to eight of them."""
    raw = maximum / 5
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    ticks = []
    value = 0.0
    while value <= maximum + 1e-9:
        ticks.append(value)
        value += step
    return ticks


def length_label(taps: int, rate: float) -> str:
    if taps <= LONG_FROM_TAPS:
        return f"{taps} taps"
    return f"{taps / rate:g} s"


def chart(machine: dict[str, Any], long: bool, variants: list[str], index: int, cap: float | None = None) -> str:
    """One chart; with `cap`, the axis stops there and longer bars are cut off, faded, and labelled."""
    rate = machine["runs"][0]["sampleRate"] if machine["runs"] else 48000.0
    cells = [c for c in machine["cells"] if (c["taps"] > LONG_FROM_TAPS) == long]
    if not cells:
        return ""
    frames_list = sorted({c["frames"] for c in cells})
    taps_list = sorted({c["taps"] for c in cells})
    by_key = {(c["frames"], c["taps"]): c for c in cells}
    # Only implementations measured somewhere in this chart get bars.
    order = [variants[0], *variants[2:], variants[1]]
    shown = [v for v in order if any(v in c["variants"] for c in cells)]

    values = [
        e[k]
        for c in cells
        for v in shown
        if (e := c["variants"].get(v, {})).get("succeeded")
        for k in ("meanPercent", "p99Percent")
        if k in e
    ]
    maximum = max(values) if values else 100.0
    capped = cap is not None and maximum > cap
    if capped:
        maximum = cap
    ticks = nice_ticks(maximum)
    scale = PLOT / maximum

    pair = BAR
    band_height = len(shown) * pair + (len(shown) - 1) * GAP
    row_height = GAP + len(taps_list) * band_height + (len(taps_list) - 1) * GAP + GAP
    height = TOP + len(frames_list) * row_height + BOTTOM
    width = LEFT + PLOT + RIGHT

    out: list[str] = []
    out.append(
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-labelledby="c{index}-t" preserveAspectRatio="xMinYMin meet">'
    )
    if capped:
        # Truncated bars fade out ahead of the value label at the cap.
        out.append(
            f'<defs><linearGradient id="c{index}-fade" gradientUnits="userSpaceOnUse" '
            f'x1="{LEFT + PLOT - 110}" x2="{LEFT + PLOT - 48}" y1="0" y2="0">'
            f'<stop offset="0" stop-color="#fff"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient>'
            f'<mask id="c{index}-mask" maskUnits="userSpaceOnUse" x="0" y="0" width="{width}" height="{height}">'
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="url(#c{index}-fade)"/></mask></defs>'
        )
    title = f"{machine['label']}, {'multi-second' if long else 'cabinet-length'} impulse responses"
    out.append(f'<title id="c{index}-t">{html.escape(title)}</title>')

    # Grid and ticks.
    for t in ticks:
        x = LEFT + t * scale
        out.append(f'<line class="grid" x1="{x:.1f}" y1="{TOP - 6}" x2="{x:.1f}" y2="{height - BOTTOM}"/>')
        label = f"{t:g}%"
        last = " last" if t == ticks[-1] else ""
        out.append(f'<text class="tick{last}" x="{x:.1f}" y="{TOP - 12}" text-anchor="middle">{label}</text>')

    y = TOP
    for r, frames in enumerate(frames_list):
        deadline_us = frames / rate * 1e6
        if r:
            out.append(f'<line class="row-rule" x1="0" y1="{y}" x2="{width - RIGHT}" y2="{y}"/>')
        out.append(
            f'<text class="frames" x="4" y="{y + row_height / 2 - 3:.1f}">{frames}</text>'
            f'<text class="deadline" x="4" y="{y + row_height / 2 + 11:.1f}">{deadline_us:.0f} µs</text>'
        )
        band_y = y + GAP
        for b, taps in enumerate(taps_list):
            cell = by_key.get((frames, taps))
            if b % 2 == 0:
                out.append(
                    f'<rect class="band" x="{LEFT - 50}" y="{band_y - GAP / 2:.1f}" '
                    f'width="{PLOT + 50}" height="{band_height + GAP:.1f}"/>'
                )
            out.append(
                f'<text class="length" x="{LEFT - 6}" y="{band_y + band_height / 2 + 3.5:.1f}" '
                f'text-anchor="end">{html.escape(length_label(taps, rate))}</text>'
            )
            by = band_y
            for name in shown:
                v = variants.index(name)
                entry = (cell or {}).get("variants", {}).get(name, {})
                for key, kind in (("p99Percent", "p99"), ("meanPercent", "mean")):
                    bar_y = by
                    if entry.get("succeeded") and key in entry:
                        value = entry[key]
                        over = value > 100.0
                        cls = f"bar v{v} {kind}{' over' if over else ''}"
                        tip = (
                            f"{name}, {frames} frames, {length_label(taps, rate)}: "
                            f"{kind} {value:.2f}% of deadline ({value / 100 * deadline_us:.1f} µs)"
                        )
                        cut = capped and value > maximum
                        mask = f' mask="url(#c{index}-mask)"' if cut else ""
                        out.append(
                            f'<rect class="{cls}" x="{LEFT}" y="{bar_y}" '
                            f'width="{max(min(value, maximum) * scale, 0.8):.1f}" '
                            f'height="{BAR - 1}"{mask}><title>{html.escape(tip)}</title></rect>'
                        )
                        if cut and kind == "p99":
                            out.append(
                                f'<text class="cut{" over" if over else ""}" x="{LEFT + PLOT - 4}" '
                                f'y="{bar_y + BAR - 2}" text-anchor="end">{value:.0f}%'
                                f'<title>{html.escape(tip)}</title></text>'
                            )
                    elif name in (cell or {}).get("variants", {}):
                        out.append(
                            f'<text class="missing" x="{LEFT + 3}" y="{bar_y + BAR - 1}">rejected</text>'
                        )
                        break
                by += pair + GAP
            band_y += band_height + GAP
        y += row_height

    # The deadline, drawn last so it sits over the bars.
    if maximum >= 100.0:
        x = LEFT + 100.0 * scale
        out.append(f'<line class="deadline-line" x1="{x:.1f}" y1="{TOP - 6}" x2="{x:.1f}" y2="{height - BOTTOM}"/>')
        out.append(
            f'<text class="deadline-label" x="{x + 4:.1f}" y="{height - 5}">deadline</text>'
        )
    out.append("</svg>")

    legend = "".join(
        f'<li><span class="key v{v} mean"></span><span class="key v{v} p99"></span>{html.escape(name)}</li>'
        for v, name in ((variants.index(n), n) for n in shown)
    )
    heading = "Up to 8192 taps" if not long else "Multi-second, 1 to 60 s"
    return f"""
<figure class="panel">
  <figcaption>
    <h3>{heading}</h3>
    <ul class="legend">{legend}<li class="legend-note">lower value of bar is mean, full bar is p99 · <span class="key over-key"></span>over the deadline</li></ul>
  </figcaption>
  {'<p class="chart-note"><strong>linear@4096</strong>: linear, but set to expect blocks of max 4096 frames</p>' if long else ''}
  <div class="scroll">{''.join(out)}</div>
  
</figure>"""


def render(summary: dict[str, Any]) -> str:
    # Colours follow this order; bars are drawn baseline, extra variants, then the candidate.
    variants = [summary["baseline"], summary["candidate"], *summary.get("also", [])]
    sections = []
    index = 0
    for machine in summary["machines"]:
        charts = []
        for long in (False, True):
            last = machine is summary["machines"][-1] and long
            charts.append(chart(machine, long, variants, index, cap=200.0 if last else None))
            index += 1
        sections.append(
            f'<section class="machine"><h2>{html.escape(machine["label"])}</h2>{"".join(charts)}</section>'
        )
    return f"""<title>Linear vs an optimised version, “linearplus”</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">
<style>
:root {{
  --page: #f4f6f5;
  --surface: #ffffff;
  --ink: #14181b;
  --muted: #69727a;
  --rule: #d9dfdc;
  --band: #eef1f0;
  --v0: #6d7780;
  --v0-light: #b9c0c6;
  --v1: #245f73;
  --v1-light: #8fbccb;
  --v2: #9a6a1f;
  --v2-light: #dcbd86;
  --over: #d23a3a;
  --over-light: #f19a9a;
  --sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --display: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --page: #101416; --surface: #182024; --ink: #eef2f1; --muted: #8e9896; --rule: #2a3438; --band: #1d272b;
    --v0: #a3adb4; --v0-light: #56606a; --v1: #6cb4cc; --v1-light: #2f6576; --v2: #e0a94e; --v2-light: #7a5a26;
    --over: #ff6b6b; --over-light: #8c3434;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --page: #101416; --surface: #182024; --ink: #eef2f1; --muted: #8e9896; --rule: #2a3438; --band: #1d272b;
  --v0: #a3adb4; --v0-light: #56606a; --v1: #6cb4cc; --v1-light: #2f6576; --v2: #e0a94e; --v2-light: #7a5a26;
  --over: #ff6b6b; --over-light: #8c3434;
}}
body {{ background: var(--page); color: var(--ink); font-family: var(--sans); font-size: 15px; line-height: 1.5; }}
main {{ max-width: 960px; margin: 0 auto; padding-inline: 16px; padding-block: 28px 48px; display: grid; gap: 28px; }}
h1 {{ font-family: var(--display); font-weight: 600; font-size: 30px; margin: 0; text-wrap: balance; }}
h2 {{ font-family: var(--display); font-weight: 600; font-size: 22px; margin: 0 0 12px; }}
h3 {{ font-family: var(--display); font-weight: 500; font-size: 17px; margin: 0; }}
.lede {{ max-width: 68ch; margin: 6px 0 0; color: var(--muted); }}
.machine {{ display: grid; gap: 18px; }}
.panel {{ margin: 0; background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 14px 14px 10px; display: grid; gap: 8px; }}
figcaption {{ display: flex; flex-wrap: wrap; gap: 6px 18px; align-items: baseline; justify-content: space-between; }}
.legend {{ list-style: none; margin: 0; padding: 0; display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 13px; }}
.legend li {{ display: flex; align-items: center; gap: 3px; }}
.chart-note {{ margin: 0; font-size: 14px; color: var(--muted); }}
.legend-note {{ color: var(--muted); }}
.key {{ display: inline-block; width: 14px; height: 8px; border-radius: 1px; }}
.key.v0.mean {{ background: var(--v0-light); }} .key.v0.p99 {{ background: var(--v0); margin-right: 4px; }}
.key.v1.mean {{ background: var(--v1-light); }} .key.v1.p99 {{ background: var(--v1); margin-right: 4px; }}
.key.v2.mean {{ background: var(--v2-light); }} .key.v2.p99 {{ background: var(--v2); margin-right: 4px; }}
.key.over-key {{ background: var(--over); margin-inline: 4px 3px; }}
.scroll {{ overflow-x: auto; }}
.chart {{ display: block; width: 100%; min-width: 640px; height: auto; font-family: var(--mono); }}
.chart .grid {{ stroke: var(--rule); stroke-width: 1; }}
.chart .row-rule {{ stroke: var(--muted); stroke-width: .6; opacity: .6; }}
.chart .band {{ fill: var(--band); }}
.chart .tick {{ fill: var(--muted); font-size: 20px; }}
.chart .tick.last {{ font-weight: 600; fill: var(--ink); }}
.chart .frames {{ fill: var(--ink); font-size: 13px; font-weight: 500; }}
.chart .deadline {{ fill: var(--muted); font-size: 9.5px; }}
.chart .length {{ fill: var(--muted); font-size: 9.5px; }}
.chart .cut {{ fill: var(--ink); font-family: "IBM Plex Mono", monospace; font-size: 9px; font-weight: 500; }}
.chart .cut.over {{ fill: var(--over); }}
.chart .missing {{ fill: var(--muted); font-size: 7px; }}
.chart .bar.v0.mean {{ fill: var(--v0-light); }} .chart .bar.v0.p99 {{ fill: var(--v0); }}
.chart .bar.v1.mean {{ fill: var(--v1-light); }} .chart .bar.v1.p99 {{ fill: var(--v1); }}
.chart .bar.v2.mean {{ fill: var(--v2-light); }} .chart .bar.v2.p99 {{ fill: var(--v2); }}
.chart .bar.over.mean {{ fill: var(--over-light); }} .chart .bar.over.p99 {{ fill: var(--over); }}
.chart .deadline-line {{ stroke: var(--over); stroke-width: 1.5; stroke-dasharray: 5 3; }}
.chart .deadline-label {{ fill: var(--over); font-size: 10px; }}
.method {{ max-width: 68ch; color: var(--muted); font-size: 14px; }}
.method p {{ margin: 0 0 8px; }}
</style>
<main>
  <header>
    <h1>Linear vs an optimised version, “linearplus”</h1>
    <p class="lede">Callback cost as % of its deadline at 48&nbsp;kHz.</p>
  </header>
  {''.join(sections)}
</main>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-summary", type=Path, required=True)
    parser.add_argument("--html", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.from_summary.read_text())
    args.html.write_text(render(summary))
    print(f"wrote {args.html}")


if __name__ == "__main__":
    main()
