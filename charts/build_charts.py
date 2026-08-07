#!/usr/bin/env python3
"""Render the two presentation charts to PNG, light and dark.

One generator rather than four hand-written files, so a number can only be
wrong in one place. Every value here is copied from the benchmark reports in
benchmark-results/ — see SLIMMED-PATH.md for the protocol behind them.

    python3 charts/build_charts.py

Needs Google Chrome for the PNG render; the intermediate HTML is left in place
so the charts can be inspected in a browser.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# --- Palette -----------------------------------------------------------------
#
# From the data-viz reference palette, validated with its own checker:
# the blue ordinal ramps pass monotone-lightness / step-gap / light-end-contrast
# in both modes, and the blue-vs-red diverging pair passes CVD separation
# (worst adjacent dE 21.6 protan light, 19.2 dark) and 3:1 contrast in both.

THEMES = {
    "light": dict(
        surface="#fcfcfb", page="#f9f9f7",
        ink="#0b0b0b", ink2="#52514e", muted="#898781",
        grid="#e1e0d9", axis="#c3c2b7",
        # before -> after pair (1 hue, 2 shades): recessive, prominent
        before="#86b6ef", after="#1c5cab",
        # ordinal ramp for the build-up ladder, stage 1 -> stage 5
        ramp=["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"],
        faster="#2a78d6", slower="#e34948",
        good="#006300",
    ),
    "dark": dict(
        surface="#1a1a19", page="#0d0d0d",
        ink="#ffffff", ink2="#c3c2b7", muted="#898781",
        grid="#2c2c2a", axis="#383835",
        before="#184f95", after="#6da7ec",
        ramp=["#184f95", "#256abf", "#3987e5", "#6da7ec", "#b7d3f6"],
        faster="#3987e5", slower="#e66767",
        good="#0ca30c",
    ),
}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# --- Data --------------------------------------------------------------------
# benchmark-results/, Apple M2, 48 kHz, 64-frame blocks, Release.

HEADLINE = [
    dict(
        title="A2 full", channels="8 channels",
        base_ms=428.34, opt_ms=225.67, speedup=1.898,
        base_label="a2_fast", opt_label="fused NEON",
        note="already in the Core PR", parity="132.6 dB below signal",
        base_rtf="25.5×", opt_rtf="48.4×",
    ),
    dict(
        title="A2 nano", channels="3 channels",
        base_ms=60.11, opt_ms=28.65, speedup=2.098,
        base_label="a2_fast", opt_label="planar NEON",
        note="new — not yet proposed", parity="bit-identical",
        base_rtf="181.6×", opt_rtf="380.9×",
    ),
]

BUILDUP = [
    ("a2_fast today", "scalar 3×3 GEMV, one channel at a time", 60.11, 1.000),
    ("Planar layout", "vectorise across frames, not channels", 50.97, 1.179),
    ("32-frame tile", "amortise weight loads; more FMA chains", 34.30, 1.752),
    ("Ring-direct + skip-last", "stop copying residuals; drop dead work", 30.77, 1.954),
    ("Linear ring", "no per-block mirror memcpy", 28.65, 2.098),
]

# name, ms, speedup, annotation (empty for most - label selectively)
CANDIDATES = [
    ("stacked_linear", 28.65, 2.098, "winner"),
    ("stacked32", 30.77, 1.954, ""),
    ("widetile32", 34.30, 1.752, ""),
    ("stacked16", 34.71, 1.732, ""),
    ("widetile16", 38.52, 1.561, ""),
    ("widetile64", 41.24, 1.458, "tile too wide: accumulators spill"),
    ("widetile8", 43.12, 1.394, ""),
    ("ringdirect", 47.21, 1.273, ""),
    ("planar_linear", 48.58, 1.237, ""),
    ("skiplast", 50.02, 1.202, ""),
    ("vext_taps", 50.10, 1.200, ""),
    ("planar", 50.97, 1.179, "the layout change on its own"),
    ("prefetch", 53.60, 1.121, ""),
    ("ld3", 56.44, 1.065, ""),
    ("planar_ring", 57.45, 1.046, "exact-size rings: smaller, but more copying"),
    ("restrict", 58.72, 1.024, ""),
    ("baseline", 59.57, 1.009, "control: verbatim port of a2_fast"),
    ("pad4", 71.99, 0.835, "just pad 3→4 channels — slower than scalar"),
    ("framemajor", 99.14, 0.606, "9 scalar accumulators — spills badly"),
]

BASELINE_MS = 60.11


def css(t):
    return f"""
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; background: {t['page']}; }}
  body {{ font-family: {FONT}; color: {t['ink']}; -webkit-font-smoothing: antialiased; }}
  .card {{ background: {t['surface']}; padding: 28px 32px 24px; }}
  h1 {{ font-size: 19px; font-weight: 600; margin: 0 0 5px; letter-spacing: -0.01em; }}
  .sub {{ font-size: 12.5px; color: {t['ink2']}; margin: 0 0 22px; line-height: 1.5; }}
  .foot {{ font-size: 11.5px; color: {t['muted']}; margin-top: 20px; line-height: 1.6;
           border-top: 1px solid {t['grid']}; padding-top: 12px; }}
  .foot b {{ color: {t['ink2']}; font-weight: 600; }}
  .legend {{ display: flex; gap: 20px; align-items: center; margin: 0 0 20px;
             font-size: 12px; color: {t['ink2']}; }}
  .legend span {{ display: inline-flex; align-items: center; gap: 7px; }}
  .swatch {{ width: 22px; height: 10px; border-radius: 0 3px 3px 0; display: inline-block; }}
  .mono {{ font-variant-numeric: tabular-nums; }}
"""


# --- Chart 1: the headline ---------------------------------------------------

def headline_html(t):
    # Bars are indexed to each row's own a2_fast = 100%, which is the sanctioned
    # way to put two measures of very different scale on one axis. Absolute
    # milliseconds ride the bars so nothing is hidden by the normalisation.
    PLOT = 470          # px for 100%
    LABEL = 96          # left gutter for the series name
    rows = []
    for g in HEADLINE:
        bars = []
        for kind, ms, colour, name, rtf in (
            ("before", g["base_ms"], t["before"], g["base_label"], g["base_rtf"]),
            ("after", g["opt_ms"], t["after"], g["opt_label"], g["opt_rtf"]),
        ):
            w = PLOT * ms / g["base_ms"]
            bars.append(f"""
        <div class="barrow">
          <div class="bname">{name}</div>
          <div class="btrack"><div class="bar" style="width:{w:.1f}px;background:{colour}"></div>
            <div class="bval mono">{ms:.1f} ms</div>
            <div class="brtf mono">{rtf} real time</div>
          </div>
        </div>""")
        rows.append(f"""
      <div class="group">
        <div class="ghead">
          <div>
            <div class="gtitle">{g['title']}</div>
            <div class="gsub">{g['channels']} &middot; {g['note']}</div>
          </div>
          <div class="gspeed">
            <div class="gnum">{g['speedup']:.2f}&times;</div>
            <div class="glab">faster &middot; {g['parity']}</div>
          </div>
        </div>
        {''.join(bars)}
      </div>""")

    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{css(t)}
  .group {{ margin-bottom: 26px; }}
  .group:last-of-type {{ margin-bottom: 0; }}
  .ghead {{ display: flex; justify-content: space-between; align-items: flex-end;
            border-bottom: 1px solid {t['grid']}; padding-bottom: 7px; margin-bottom: 13px; }}
  .gtitle {{ font-size: 15px; font-weight: 600; }}
  .gsub {{ font-size: 11.5px; color: {t['muted']}; margin-top: 2px; }}
  .gspeed {{ text-align: right; }}
  .gnum {{ font-size: 30px; font-weight: 600; line-height: 1; letter-spacing: -0.02em; }}
  .glab {{ font-size: 11px; color: {t['muted']}; margin-top: 4px; }}
  .barrow {{ display: flex; align-items: center; margin-bottom: 9px; }}
  .barrow:last-child {{ margin-bottom: 0; }}
  .bname {{ width: {LABEL}px; font-size: 12.5px; color: {t['ink2']}; text-align: right;
            padding-right: 14px; flex: none; }}
  .btrack {{ position: relative; flex: 1; display: flex; align-items: center; height: 20px; }}
  .bar {{ height: 18px; border-radius: 0 4px 4px 0; flex: none; }}
  .bval {{ font-size: 12.5px; font-weight: 600; color: {t['ink']}; margin-left: 11px; }}
  .brtf {{ font-size: 11px; color: {t['muted']}; margin-left: 9px; }}
</style></head><body>
  <div class="card">
    <h1>A hand-written NEON WaveNet, against <span class="mono">a2_fast</span></h1>
    <p class="sub">Time to process 10.9&nbsp;s of guitar DI &middot; Apple M2, macOS 27, Release &middot; 48&nbsp;kHz, 64-frame blocks &middot; <b>lower is better</b><br>
      Each pair is scaled to its own <span class="mono">a2_fast</span> bar, so the two models can be compared despite a 7&times; difference in absolute cost.</p>
    <div class="legend">
      <span><i class="swatch" style="background:{t['before']}"></i>a2_fast &mdash; what ships today</span>
      <span><i class="swatch" style="background:{t['after']}"></i>optimised NEON engine</span>
    </div>
    {''.join(rows)}
    <p class="foot"><b>Same audio out.</b> A2 nano is <b>bit-identical</b> to <span class="mono">a2_fast</span> over all 523,808 frames; A2 full differs by 132.6&nbsp;dB below signal (fp32 reassociation only). Both are fp32 throughout &mdash; no quantisation, no listening test needed.</p>
  </div>
</body></html>"""


# --- Chart 2: where the nano speedup comes from, and what else was tried -----

def detail_html(t):
    LADDER_PLOT = 380
    LADDER_MAX = 62.0
    ladder = []
    for i, (name, why, ms, sp) in enumerate(BUILDUP):
        w = LADDER_PLOT * ms / LADDER_MAX
        colour = t["ramp"][i]
        ladder.append(f"""
      <div class="lrow">
        <div class="lname"><b>{name}</b><span>{why}</span></div>
        <div class="ltrack"><div class="lbar" style="width:{w:.1f}px;background:{colour}"></div>
          <div class="lval mono">{ms:.1f} ms</div>
          <div class="lsp mono">{sp:.2f}&times;</div>
        </div>
      </div>""")

    # Bar length is the speedup itself, so longer is unambiguously better and
    # the losers visibly fall short of the 1.00x rule. Annotations live in
    # their own column past the plot, so nothing can collide with that rule.
    CAND_NAME = 122
    CAND_PLOT = 400
    CAND_MAX = 2.2
    base_x = CAND_NAME + CAND_PLOT / CAND_MAX
    cands = []
    for name, ms, sp, note in CANDIDATES:
        w = CAND_PLOT * sp / CAND_MAX
        colour = t["faster"] if sp >= 1.0 else t["slower"]
        weight = "600" if note == "winner" else "400"
        cands.append(f"""
      <div class="crow">
        <div class="cname mono" style="font-weight:{weight}">{name}</div>
        <div class="cplot"><div class="cbar" style="width:{w:.1f}px;background:{colour}"></div></div>
        <div class="csp mono" style="font-weight:{weight}">{sp:.2f}&times;</div>
        <div class="cnote">{note}</div>
      </div>""")

    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{css(t)}
  h2 {{ font-size: 14px; font-weight: 600; margin: 0 0 3px; }}
  .h2sub {{ font-size: 11.5px; color: {t['muted']}; margin: 0 0 15px; }}
  .panel {{ margin-bottom: 26px; }}
  .lrow {{ display: flex; align-items: center; margin-bottom: 10px; }}
  .lname {{ width: 252px; flex: none; padding-right: 16px; }}
  .lname b {{ display: block; font-size: 12.5px; font-weight: 600; }}
  .lname span {{ display: block; font-size: 10.5px; color: {t['muted']}; margin-top: 1px; }}
  .ltrack {{ position: relative; flex: 1; display: flex; align-items: center; height: 22px; }}
  .lbar {{ height: 18px; border-radius: 0 4px 4px 0; flex: none; }}
  .lval {{ font-size: 12px; font-weight: 600; margin-left: 11px; }}
  .lsp {{ font-size: 12px; color: {t['ink2']}; margin-left: 10px; }}

  .crow {{ display: flex; align-items: center; height: 16px; margin-bottom: 5px; }}
  .cname {{ width: {CAND_NAME}px; flex: none; font-size: 11px; color: {t['ink2']};
            text-align: right; padding-right: 12px; }}
  .cplot {{ width: {CAND_PLOT}px; flex: none; display: flex; align-items: center; }}
  .cbar {{ height: 11px; border-radius: 0 3px 3px 0; flex: none; }}
  .csp {{ width: 46px; flex: none; font-size: 11px; margin-left: 10px; color: {t['ink']}; }}
  .cnote {{ font-size: 10.5px; color: {t['muted']}; }}
  .baseline {{ position: absolute; left: {base_x:.1f}px; top: 16px; bottom: 0;
               width: 1px; background: {t['axis']}; }}
  .blabel {{ position: absolute; left: {base_x - 4:.1f}px; top: 0;
             font-size: 10.5px; color: {t['muted']}; transform: translateX(-100%); }}
  .candwrap {{ position: relative; padding-top: 18px; }}
</style></head><body>
  <div class="card">
    <div class="panel">
      <h2>Where the A2 nano speedup comes from</h2>
      <p class="h2sub">Cumulative &mdash; each row adds one change to the row above. Time per pass, lower is better.</p>
      {''.join(ladder)}
    </div>

    <div class="panel" style="margin-bottom:0">
      <h2>Everything that was measured</h2>
      <p class="h2sub">Nineteen kernels, same protocol, same parity gate. Bar length is the speedup, so past the rule is faster. Sixteen are bit-identical to <span class="mono">a2_fast</span>; the three that are not are the three that lost.</p>
      <div class="candwrap">
        <div class="baseline"></div>
        <div class="blabel">1.00&times; &mdash; a2_fast today, 60.1 ms&nbsp;&nbsp;</div>
        {''.join(cands)}
      </div>
    </div>
    <p class="foot">Measured by the project's own benchmark harness, not a micro-benchmark: 5&nbsp;s warm-up discarded, 30&nbsp;s of full-file passes, mean of the fastest 70&percnt; required to agree within 3&percnt;, denormals flushed, model reset outside the timed region, engine asserted rather than assumed. <b>Every kernel above was accepted on its first attempt.</b></p>
  </div>
</body></html>"""


# --- Render ------------------------------------------------------------------

CHARTS = [
    ("nam-speedup-headline", headline_html, 880, 500),
    ("nam-nano-detail", detail_html, 880, 872),
]


def main():
    if not os.path.exists(CHROME):
        sys.exit(f"Google Chrome not found at {CHROME}")

    for name, builder, width, height in CHARTS:
        for mode, tokens in THEMES.items():
            suffix = "" if mode == "light" else "-dark"
            html_path = os.path.join(HERE, f"{name}{suffix}.html")
            png_path = os.path.join(HERE, f"{name}{suffix}.png")
            with open(html_path, "w") as f:
                f.write(builder(tokens))
            subprocess.run(
                [CHROME, "--headless", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
                 f"--screenshot={png_path}", f"--window-size={width},{height}",
                 "--force-device-scale-factor=2", "--virtual-time-budget=1500",
                 f"file://{html_path}"],
                check=True, capture_output=True,
            )
            print(f"  {os.path.relpath(png_path, os.path.dirname(HERE))}")


if __name__ == "__main__":
    main()
