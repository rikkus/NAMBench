#!/usr/bin/env python3
"""One IR implementation against another, as an HTML page.

Reads `nam_ir_benchmark` reports — one per block size, as run-benchmark.sh
writes them — for one or more machines, and draws one table per machine: block
sizes down, impulse-response lengths across. Each cell is the candidate's
99th-percentile callback minus the baseline's, both as a percentage of that
callback's deadline at the report's sample rate. Negative is the candidate
spending less of its deadline, which is what is wanted. Hovering or focusing a
cell shows the numbers it came from, and the only means on the page.

The p99, not the mean, because a partitioned FFT convolution does its transforms
in one callback: its average can look good while the callback that runs them is
late, and a late callback is a click. See BENCHMARKING.md.

Two steps, so the page can be rebuilt without the reports that made it:

    # reports -> summary + page
    Scripts/ir-report.py --baseline linear --candidate linearplus \\
        --machine "Apple M2" benchmark-results/M2-*-ir-block*.json \\
        --machine "Raspberry Pi 500" pi/*-ir-block*.json \\
        --summary ir-study/data/linearplus-summary.json \\
        --html ir-study/linearplus.html

    # summary -> page
    Scripts/ir-report.py --from-summary ir-study/data/linearplus-summary.json \\
        --html ir-study/linearplus.html

The summary keeps what the page shows plus enough about each run to say what
it measured: the machine, the protocol settings, the pinned commits and the
output parity. It is small enough to commit; the reports are not.
"""

from __future__ import annotations

import argparse
import datetime
import html
import json
import math
import sys
from pathlib import Path
from typing import Any

# The colour scale is fixed by what the page is for, not by the theme: white is
# no difference, and the scale saturates at 15 points either way.
SATURATION_POINTS = 15.0
WORSE_RGB = (0xFF, 0xAA, 0xAA)
SAME_RGB = (0xFF, 0xFF, 0xFF)
BETTER_RGB = (0xAA, 0xFF, 0xAA)


# Lengths past this go in a table of their own: a cabinet IR against a reverb.
LONG_FROM_TAPS = 8192


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def pins_of(repo_root: Path) -> dict[str, Any]:
    pins = repo_root / "vendor" / "pins.json"
    return load_json(pins) if pins.exists() else {}


def summarise_machine(
    label: str, paths: list[Path], baseline: str, candidate: str, also: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Pool one machine's reports into cells keyed by (frames, taps)."""
    cells: dict[tuple[int, int], dict[str, Any]] = {}
    environment: dict[str, Any] = {}
    runs: list[dict[str, Any]] = []

    for path in sorted(paths):
        report = load_json(path)
        if report.get("producer") != "nam_ir_benchmark":
            sys.exit(f"error: {path} is not an nam_ir_benchmark report")
        environment = environment or report.get("environment", {})
        rate = float(report.get("audio", {}).get("sampleRate", 48000.0))
        runs.append(
            {
                "file": path.name,
                "note": report.get("note", ""),
                "config": report.get("config", {}),
                "socTemperatureStartC": report.get("socTemperatureStartC"),
                "socTemperatureEndC": report.get("socTemperatureEndC"),
                "sampleRate": rate,
            }
        )

        parity: dict[int, float | None] = {}
        for p in report.get("parities", []):
            reference, _, taps = p.get("referenceVariant", "").partition(" @")
            comparison = p.get("comparisonVariant", "").partition(" @")[0]
            if reference == baseline and comparison == candidate and taps.isdigit():
                parity[int(taps)] = p.get("decibelsBelowSignal")

        for r in report.get("results", []):
            variant = r.get("variant")
            if variant not in (baseline, candidate, *also):
                continue
            frames = int(r["blockSize"])
            taps = int(r["requestedTaps"])
            deadline_us = frames / rate * 1.0e6
            cell = cells.setdefault(
                (frames, taps),
                {"frames": frames, "taps": taps, "deadlineUs": deadline_us, "variants": {}},
            )
            if taps in parity:
                cell["parityDbBelowSignal"] = parity[taps]
            entry: dict[str, Any] = {
                "succeeded": bool(r.get("succeeded")),
                "implementation": r.get("implementation"),
                "headTaps": r.get("headTaps"),
                "fftBlockSize": r.get("fftBlockSize"),
            }
            if entry["succeeded"]:
                entry.update(
                    {
                        "p99Percent": r["blockP99Percent"],
                        "p99Us": r["blockP99Percent"] / 100.0 * deadline_us,
                        # corePercent is the mean pass as a share of the audio's
                        # length, so it is the mean block's share of its own.
                        "meanUs": r["corePercent"] / 100.0 * deadline_us,
                        "meanPercent": r["corePercent"],
                        "maxPercent": r["blockMaxPercent"],
                        "blocks": r["blockSampleCount"],
                        # Older reports predate the count; None says unknown.
                        "overruns": r.get("blockOverruns"),
                        "missesPerMinute": (
                            r["blockOverruns"] / (r["blockSampleCount"] * frames / rate / 60.0)
                            if r.get("blockOverruns") is not None and r["blockSampleCount"]
                            else None
                        ),
                    }
                )
            else:
                entry["failureReason"] = r.get("failureReason", "rejected")
            cell["variants"][variant] = entry

    if not cells:
        sys.exit(f"error: no {baseline} or {candidate} results for {label}")
    return {
        "label": label,
        "environment": environment,
        "runs": runs,
        "cells": [cells[k] for k in sorted(cells)],
    }


def difference(cell: dict[str, Any], baseline: str, candidate: str, metric: str = "p99") -> float | None:
    """Candidate minus baseline, in points of the deadline, as displayed."""
    a = cell["variants"].get(baseline, {})
    b = cell["variants"].get(candidate, {})
    if not (a.get("succeeded") and b.get("succeeded")):
        return None
    key = "p99Percent" if metric == "p99" else "meanPercent"
    if key not in a or key not in b:
        return None
    value = round(b[key] - a[key], 1)
    return 0.0 if value == 0 else value


def colour(value: float) -> str:
    """#ffffff at 0, towards #ffaaaa above and #aaffaa below, full at 15."""
    k = min(abs(value), SATURATION_POINTS) / SATURATION_POINTS
    end = WORSE_RGB if value > 0 else BETTER_RGB
    rgb = [round(s + (e - s) * k) for s, e in zip(SAME_RGB, end)]
    return "#" + "".join(f"{c:02x}" for c in rgb)


def shown(value: float) -> str:
    return "0.0" if value == 0 else f"{value:+.1f}"


def fmt_us(x: float) -> str:
    return f"{x:.1f}" if x < 100 else f"{x:.0f}"


def missed(cell: dict[str, Any], name: str) -> bool:
    v = cell["variants"].get(name, {})
    return bool(v.get("succeeded")) and v.get("maxPercent", 0.0) > 100.0


def taps_label(taps: int, rate: float) -> str:
    if taps <= LONG_FROM_TAPS:
        return str(taps)
    return f"{taps / rate:g}&nbsp;s<span class=\"deadline\">{taps:,}</span>"


def tip_table(
    cell: dict[str, Any], baseline: str, candidate: str, long: bool = False, also: tuple[str, ...] = ()
) -> str:
    rows = []
    for name in (baseline, candidate, *also):
        v = cell["variants"].get(name)
        if v is None:
            continue
        if v.get("succeeded"):
            rows.append(
                f"<tr><th scope=\"row\">{html.escape(name)}</th>"
                f"<td>{fmt_us(v['p99Us'])}</td><td>{v['p99Percent']:.2f}%</td>"
                f"<td>{fmt_us(v['meanUs'])}</td>"
                + (f"<td>{v['meanPercent']:.2f}%</td>" if "meanPercent" in v else "<td>–</td>")
                + (
                    f"<td>{v['maxPercent']:.0f}%</td><td>"
                    + ("?" if v.get("missesPerMinute") is None else f"{v['missesPerMinute']:.0f}")
                    + "</td>"
                    if long
                    else ""
                )
                + "</tr>"
            )
        else:
            rows.append(
                f"<tr><th scope=\"row\">{html.escape(name)}</th>"
                f"<td colspan=\"{6 if long else 4}\">rejected: {html.escape(v.get('failureReason', ''))}</td></tr>"
            )
    return (
        f"<p class=\"tip-head\">{cell['taps']:,} taps, {cell['frames']}-frame callbacks "
        f"<span>deadline {fmt_us(cell['deadlineUs'])} µs</span></p>"
        "<table><thead><tr><th></th><th scope=\"col\">p99 µs</th>"
        "<th scope=\"col\">p99 % of deadline</th><th scope=\"col\">mean µs</th><th scope=\"col\">mean % of deadline</th>"
        + ("<th scope=\"col\">worst % of deadline</th><th scope=\"col\">missed per minute</th>" if long else "")
        + "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def machine_section(
    index: int, machine: dict[str, Any], baseline: str, candidate: str, also: tuple[str, ...] = ()
) -> str:
    cells = {(c["frames"], c["taps"]): c for c in machine["cells"]}
    frames_list = sorted({c["frames"] for c in machine["cells"]})
    taps_list = sorted({c["taps"] for c in machine["cells"]})
    env = machine.get("environment", {})

    rate = machine["runs"][0]["sampleRate"] if machine["runs"] else 48000.0

    def grid(subset: list[int], long: bool, metric: str) -> str:
        head = "".join(f"<th scope=\"col\">{taps_label(t, rate)}</th>" for t in subset)
        body = grid_rows(subset, long, metric)
        what = "p99 callback cost" if metric == "p99" else "mean callback cost"
        caption = (
            f"Multi-second impulse responses: change in {what}, points of the deadline. "
            "&#9888; marks a cell where a callback missed its deadline; the hover table says whose."
            if long
            else f"Change in {what}, points of the deadline ({html.escape(candidate)} minus {html.escape(baseline)})"
        )
        return f"""
  <div class="scroll">
    <table class="grid">
      <caption>{caption}</caption>
      <thead><tr><th scope="col" class="corner">frames <span>taps →</span></th>{head}</tr></thead>
      <tbody>{''.join(body)}</tbody>
    </table>
  </div>"""

    def grid_rows(subset: list[int], long: bool, metric: str) -> list[str]:
      body = []
      for f in frames_list:
          deadline = f / machine["runs"][0]["sampleRate"] * 1.0e6 if machine["runs"] else 0.0
          tds = []
          for t in subset:
              cell = cells.get((f, t))
              if cell is None:
                  tds.append("<td class=\"cell none\">–</td>")
                  continue
              value = difference(cell, baseline, candidate, metric)
              tip = tip_table(cell, baseline, candidate, long, also)
              if value is None:
                  tds.append(
                      f"<td class=\"cell none\" tabindex=\"0\">n/a<div class=\"tip\" hidden>{tip}</div></td>"
                  )
                  continue
              tds.append(
                  f"<td class=\"cell\" tabindex=\"0\" style=\"background:{colour(value)}\">"
                  f"{shown(value)}"
                  + (" &#9888;" if long and any(missed(cell, n) for n in (baseline, candidate, *also)) else "")
                  + f"<div class=\"tip\" hidden>{tip}</div></td>"
              )
          body.append(
              f"<tr><th scope=\"row\">{f}<span class=\"deadline\">{fmt_us(deadline)} µs</span></th>{''.join(tds)}</tr>"
          )
      return body

    # What each implementation chose, per length: the part of a cell that a
    # difference alone does not say.
    chose = []
    for t in taps_list:
        picks = []
        for name in (baseline, candidate):
            impls = {
                cells[(f, t)]["variants"].get(name, {}).get("implementation")
                for f in frames_list
                if (f, t) in cells
            }
            impls.discard(None)
            picks.append(f"{html.escape(name)} {'/'.join(sorted(impls)) or '?'}")
        chose.append(f"<li><span class=\"num\">{t}</span> taps: {', '.join(picks)}</li>")

    parities = [
        c.get("parityDbBelowSignal") for c in machine["cells"] if "parityDbBelowSignal" in c
    ]
    finite = [p for p in parities if p is not None]
    identical = sum(1 for p in parities if p is None)
    if finite:
        parity_text = f"Outputs agree to at least {min(finite):.0f} dB below the signal"
        if identical:
            parity_text += f", and are identical at {identical} of {len(parities)} points"
        parity_text += "."
    elif parities:
        parity_text = "Outputs are identical at every point."
    else:
        parity_text = ""

    # run-benchmark.sh writes "<host> cpus=<list> maxfreq=<kHz>" into the note.
    pinned = sorted(
        {
            word.partition("=")[2]
            for r in machine["runs"]
            for word in r.get("note", "").split()
            if word.startswith("cpus=")
        }
    )
    config = machine["runs"][0]["config"] if machine["runs"] else {}
    temps = [
        t
        for r in machine["runs"]
        for t in (r.get("socTemperatureStartC"), r.get("socTemperatureEndC"))
        if t is not None and t >= 0
    ]
    facts = [
        html.escape(env.get("cpu", "")),
        html.escape(f"{env.get('platform', '')} {env.get('osVersion', '')}".strip()),
        f"pinned to CPU {html.escape(', '.join(pinned))}" if pinned else "",
        f"{html.escape(env['cpuGovernor'])} governor" if env.get("cpuGovernor") else "",
        f"{config.get('warmupSeconds', 0):.0f}&nbsp;s warm-up and {config.get('timingWindowSeconds', 0):.0f}&nbsp;s timed per cell and implementation"
        if config
        else "",
        f"SoC {min(temps):.0f}–{max(temps):.0f}&nbsp;°C" if temps else "",
    ]
    facts_html = " · ".join(f for f in facts if f).replace("warm-up", "warm&#8209;up")

    return f"""
<section class="machine" aria-labelledby="m{index}">
  <h2 id="m{index}">{html.escape(machine['label'])}</h2>
  <p class="facts">{facts_html}</p>
{"".join(grid([t for t in taps_list if t <= LONG_FROM_TAPS], False, m) for m in ("p99", "mean")) if any(t <= LONG_FROM_TAPS for t in taps_list) else ""}
{"".join(grid([t for t in taps_list if t > LONG_FROM_TAPS], True, m) for m in ("p99", "mean")) if any(t > LONG_FROM_TAPS for t in taps_list) else ""}
  <details class="chose">
    <summary>Which convolution each chose</summary>
    <ul>{''.join(chose)}</ul>
  </details>
  <p class="small">{html.escape(parity_text)}</p>
</section>"""


def render(summary: dict[str, Any]) -> str:
    baseline = summary["baseline"]
    candidate = summary["candidate"]
    pins = summary.get("pins", {})

    def pin(key: str) -> str:
        p = pins.get(key, {})
        sha = p.get("sha", "")
        return f"<code>{html.escape(sha[:7])}</code>" if sha else "?"

    sections = "".join(
        machine_section(i, m, baseline, candidate, tuple(summary.get("also", [])))
        for i, m in enumerate(summary["machines"])
    )
    stops = ", ".join(
        f"{colour(v)} {100 * (v + SATURATION_POINTS) / (2 * SATURATION_POINTS):.0f}%"
        for v in (-15, -7.5, 0, 7.5, 15)
    )
    generated = html.escape(summary.get("generated", "")).replace("-", "&#8209;")

    return f"""<title>Linear vs linearplus</title>
<meta name="description" content="99th-percentile callback cost of NeuralAmpModelerCore's Linear against linearplus, by callback size and impulse-response length.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">
<style>
:root {{
  --page: #f4f6f5;
  --surface: #ffffff;
  --ink: #14181b;
  --ink-2: #454d53;
  --muted: #737c83;
  --rule: #d9dfdc;
  --accent: #245f73;
  --cell-ink: #101416;
  --cell-rule: #c9d0cc;
  --tip-bg: #ffffff;
  --tip-ink: #14181b;
  --tip-rule: #d9dfdc;
  --shadow: 0 6px 24px rgba(20, 30, 35, .16);
  --sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --display: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --page: #101416;
    --surface: #182024;
    --ink: #eef2f1;
    --ink-2: #b9c3c1;
    --muted: #87918f;
    --rule: #2a3438;
    --accent: #8cc3d4;
    --cell-rule: #0c1012;
    --tip-bg: #1d262b;
    --tip-ink: #eef2f1;
    --tip-rule: #33414a;
    --shadow: 0 6px 24px rgba(0, 0, 0, .5);
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --page: #101416;
  --surface: #182024;
  --ink: #eef2f1;
  --ink-2: #b9c3c1;
  --muted: #87918f;
  --rule: #2a3438;
  --accent: #8cc3d4;
  --cell-rule: #0c1012;
  --tip-bg: #1d262b;
  --tip-ink: #eef2f1;
  --tip-rule: #33414a;
  --shadow: 0 6px 24px rgba(0, 0, 0, .5);
}}
body {{ background: var(--page); color: var(--ink); font: 15px/1.55 var(--sans); }}
.wrap {{ max-width: 880px; margin: 0 auto; padding-inline: 20px; padding-block: 40px 64px; }}
h1 {{ font: 600 34px/1.1 var(--display); letter-spacing: -.01em; margin: 0 0 12px; text-wrap: balance; }}
h2 {{ font: 600 22px/1.2 var(--display); margin: 0 0 4px; text-wrap: balance; }}
h3 {{ font: 600 15px/1.3 var(--sans); margin: 0 0 6px; }}
p {{ margin: 0 0 12px; max-width: 68ch; color: var(--ink-2); }}
p strong {{ color: var(--ink); font-weight: 600; }}
code, .num {{ font-family: var(--mono); font-size: .92em; }}
.lede {{ font-size: 16.5px; }}
.meta {{ font-size: 12.5px; color: var(--muted); letter-spacing: .02em; margin: 0 0 24px; }}
.key {{ display: grid; gap: 6px; margin: 22px 0 8px; max-width: 420px; }}
.key .bar {{ height: 12px; border-radius: 2px; border: 1px solid var(--rule); background: linear-gradient(90deg, {stops}); }}
.key .ticks {{ display: flex; justify-content: space-between; font: 12px var(--mono); color: var(--muted); font-variant-numeric: tabular-nums; }}
.key .ends {{ display: flex; justify-content: space-between; font-size: 12.5px; color: var(--ink-2); }}
section {{ padding-block: 28px; border-top: 1px solid var(--rule); }}
section.intro {{ border-top: 0; padding-block: 0 8px; }}
.facts {{ font-size: 13px; color: var(--muted); margin-bottom: 14px; }}
.scroll {{ overflow-x: auto; }}
table.grid {{ border-collapse: collapse; font-variant-numeric: tabular-nums; min-width: 520px; }}
table.grid caption {{ caption-side: top; text-align: left; font-size: 12.5px; color: var(--muted); padding-bottom: 8px; letter-spacing: .02em; }}
table.grid th {{ font: 500 13px var(--mono); color: var(--ink-2); padding: 6px 12px; text-align: right; white-space: nowrap; }}
table.grid thead th {{ border-bottom: 1px solid var(--rule); }}
table.grid th.corner {{ text-align: left; font-family: var(--sans); font-weight: 500; color: var(--muted); }}
table.grid th.corner span {{ margin-left: 10px; }}
table.grid tbody th {{ text-align: left; }}
table.grid tbody th .deadline {{ display: block; font-size: 11px; color: var(--muted); font-weight: 400; }}
td.cell {{ position: relative; font: 500 16px var(--mono); color: var(--cell-ink); text-align: right; padding: 12px 16px; min-width: 64px; border: 2px solid var(--cell-rule); cursor: default; }}
td.cell:focus-visible {{ outline: 3px solid var(--accent); outline-offset: -3px; }}
td.cell.none {{ background: transparent; color: var(--muted); }}
.chose {{ margin: 14px 0 6px; font-size: 13.5px; color: var(--ink-2); }}
.chose summary {{ cursor: pointer; color: var(--accent); }}
.chose ul {{ margin: 8px 0 0; padding-left: 18px; }}
.small {{ font-size: 13px; color: var(--muted); }}
.method ul {{ margin: 0 0 12px; padding-left: 18px; color: var(--ink-2); max-width: 68ch; }}
.method li {{ margin-bottom: 6px; }}
#tip {{ position: fixed; z-index: 10; max-width: calc(100vw - 24px); background: var(--tip-bg); color: var(--tip-ink); border: 1px solid var(--tip-rule); border-radius: 6px; box-shadow: var(--shadow); padding: 10px 12px; pointer-events: none; }}
#tip .tip-head {{ margin: 0 0 6px; font-size: 12.5px; color: var(--tip-ink); font-weight: 600; }}
#tip .tip-head span {{ font-weight: 400; color: var(--muted); margin-left: 6px; }}
#tip table {{ border-collapse: collapse; font: 13px var(--mono); font-variant-numeric: tabular-nums; }}
#tip th, #tip td {{ padding: 3px 8px; text-align: right; white-space: nowrap; }}
#tip thead th {{ font: 500 11.5px var(--sans); color: var(--muted); border-bottom: 1px solid var(--tip-rule); }}
#tip tbody th {{ text-align: left; font-weight: 500; }}
@media (max-width: 480px) {{
  h1 {{ font-size: 28px; }}
  td.cell {{ padding: 10px 10px; font-size: 14.5px; }}
}}
</style>
<div class="wrap">
  <section class="intro">
    <h1>Linear vs linearplus</h1>
    <p class="meta">NeuralAmpModelerCore impulse-response convolution · 48 kHz · generated {generated}</p>
    <p class="lede">Each cell is how much of its <strong>48 kHz callback deadline</strong> the 99th-percentile callback of <strong>{html.escape(candidate)}</strong> uses, minus the same for <strong>{html.escape(baseline)}</strong>, in percentage points. <strong>Negative means {html.escape(candidate)} is faster.</strong> Hover over a cell, or tap it, to see the timings behind it.</p>
    <div class="key" aria-hidden="true">
      <div class="bar"></div>
      <div class="ticks"><span>-15</span><span>-7.5</span><span>0</span><span>+7.5</span><span>+15</span></div>
      <div class="ends"><span>{html.escape(candidate)} faster</span><span>{html.escape(candidate)} slower</span></div>
    </div>
    <p class="small">The colour runs from white at 0 to #aaffaa at 15 points faster and #ffaaaa at 15 points slower, and holds there beyond 15.</p>
  </section>
  {sections}
  <section class="method">
    <h2>How this was measured</h2>
    <ul>
      <li><strong>{html.escape(baseline)}</strong> is upstream NeuralAmpModelerCore at {pin('upstream')}. Its FFT path uses power-of-two tiers that grow along the impulse response, from a 128-tap direct head up to 2048-tap partitions at 8192 taps.</li>
      <li><strong>{html.escape(candidate)}</strong> is the same commit plus one change, {pin('linearplus')}. Its FFT path uses uniform partitions, 256 taps long up to 2048 taps and 512 up to 8192, with the partition multiplies spread across the callbacks between transforms.</li>
      <li>Both convolve directly up to 1024 taps, with the same code, so those columns measure the same thing twice. A difference of 0.1 there is measurement noise.</li>
      <li>Both run as each would in a plugin, choosing direct or FFT convolution by themselves. They were measured in one process by <code>nam_ir_benchmark</code>, taking turns at each point on the same generated mono IR and the same 10.9 s guitar DI.</li>
      <li>Every callback is timed. The p99 is taken over all callbacks from the passes the protocol accepted, which are the tightest 70% of passes. The mean is the mean of those passes, per callback.</li>
      <li>Before anything is timed, both outputs are rendered and compared, and a point where they disagree is not reported.</li>
      <li>The multi&#8209;second tables (48,000 to 2,880,000 taps, 1 to 60&nbsp;s) were run separately. Past 8192 taps {html.escape(baseline)} uses a 64&#8209;tap head with partitions up to 4096 (to 48,000 taps) and 8192 beyond. {html.escape(candidate)} uses 1024&#8209;tap partitions to 48,000 taps; past that, uniform partitions cover the first 2&nbsp;× T taps and a tail tier of T&#8209;tap partitions the rest, with T 8192 up to 240,000 taps and 16384 beyond. The tail's transforms are split into transforms of the uniform tier's own size and spread, with its multiplies, across the samples between its blocks, so no callback does a large transform.</li>
      <li>At those lengths {html.escape(baseline)}'s slowest callbacks are not its transforms. Every 32&nbsp;× the declared maximum block it copies its whole input history, the full length of the IR, back to the start of its buffer: 11.5&nbsp;MB at 60&nbsp;s. The tables use NAMBench's usual declaration, a maximum equal to the callback, so that copy lands in one callback in 32. A host may declare a larger maximum than it sends; the hover tables add {html.escape(baseline)}@4096, the same code told to expect up to 4096 frames, which makes the copy 4096&nbsp;/ callback times rarer. That lowers its mean and p99 but not the cost of the callback that makes the copy. {html.escape(candidate)} keeps only its head's history, so the declared maximum makes no difference to it. The hover tables also give the worst callback and the rate of missed deadlines, and &#9888; marks a cell where any of them missed one.</li>
      <li>On the M2 a few of {html.escape(candidate)}'s multi&#8209;second cells are marked too: at most 13 late callbacks in over three million, the worst taking 49&nbsp;ms against a p99 of 20&nbsp;µs. Nothing in the code takes that long, and the Pi, which runs pinned to one core with nothing else on it, has no such callbacks, so these are the machine stalling the process rather than the convolution.</li>
    </ul>
  </section>
</div>
<div id="tip" role="tooltip" hidden></div>
<script>
(function () {{
  var tip = document.getElementById("tip");
  var active = null;
  function place(cell) {{
    var r = cell.getBoundingClientRect();
    var w = tip.offsetWidth, h = tip.offsetHeight;
    var x = Math.min(Math.max(8, r.left + r.width / 2 - w / 2), window.innerWidth - w - 8);
    var y = r.bottom + 8;
    if (y + h > window.innerHeight - 8) y = Math.max(8, r.top - h - 8);
    tip.style.left = x + "px";
    tip.style.top = y + "px";
  }}
  function show(cell) {{
    var src = cell.querySelector(".tip");
    if (!src) return;
    active = cell;
    tip.innerHTML = src.innerHTML;
    tip.hidden = false;
    place(cell);
  }}
  function hide(cell) {{
    if (cell && cell !== active) return;
    active = null;
    tip.hidden = true;
  }}
  document.querySelectorAll("td.cell").forEach(function (cell) {{
    cell.addEventListener("mouseenter", function () {{ show(cell); }});
    cell.addEventListener("mouseleave", function () {{ hide(cell); }});
    cell.addEventListener("focus", function () {{ show(cell); }});
    cell.addEventListener("blur", function () {{ hide(cell); }});
    cell.addEventListener("click", function (e) {{ show(cell); e.stopPropagation(); }});
  }});
  document.addEventListener("click", function () {{ hide(); }});
  document.addEventListener("keydown", function (e) {{ if (e.key === "Escape") hide(); }});
  window.addEventListener("scroll", function () {{ if (active) place(active); }}, true);
  window.addEventListener("resize", function () {{ if (active) place(active); }});
}})();
</script>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", default="linear")
    parser.add_argument("--candidate", default="linearplus")
    parser.add_argument(
        "--also",
        action="append",
        default=[],
        help="another variant to show in the hover tables, e.g. linear@4096 (repeatable)",
    )
    parser.add_argument(
        "--machine",
        nargs="+",
        action="append",
        metavar=("LABEL", "REPORT"),
        help="a machine's label, then its nam_ir_benchmark reports",
    )
    parser.add_argument("--summary", type=Path, help="write the summary JSON here")
    parser.add_argument("--from-summary", type=Path, help="render from this summary instead of reports")
    parser.add_argument("--html", type=Path, required=True, help="write the page here")
    args = parser.parse_args()

    if args.from_summary:
        summary = load_json(args.from_summary)
    else:
        if not args.machine:
            parser.error("give --machine LABEL REPORT... at least once, or --from-summary")
        machines = []
        for group in args.machine:
            if len(group) < 2:
                parser.error(f"--machine {group[0]!r} has no reports")
            machines.append(
                summarise_machine(
                    group[0], [Path(p) for p in group[1:]], args.baseline, args.candidate, tuple(args.also)
                )
            )
        repo_root = Path(__file__).resolve().parent.parent
        summary = {
            "generated": datetime.date.today().isoformat(),
            "baseline": args.baseline,
            "candidate": args.candidate,
            "also": args.also,
            "pins": {k: v for k, v in pins_of(repo_root).items() if k in ("upstream", "linearplus", "eigen")},
            "machines": machines,
        }
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(json.dumps(summary, indent=1) + "\n")
            print(f"wrote {args.summary}")

    args.html.parent.mkdir(parents=True, exist_ok=True)
    args.html.write_text(render(summary))
    print(f"wrote {args.html}")


if __name__ == "__main__":
    main()
