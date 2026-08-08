# Charts

Presentation graphics for the speedup results. Two charts, each rendered light
and dark so a `<picture>` element can serve the right one to GitHub's theme.

| File | What it shows |
|---|---|
| `nam-speedup-headline.png` | A2 full and A2 nano, `a2_fast` vs NEON, one image |
| `nam-nano-detail.png` | Where the nano speedup comes from, and all 19 kernels measured |
| `nam-planar-headline.png` | The bit-identical pair as proposed to Core: A2 full 2.43×, A2 nano 2.01× |

All three also exist as `*-dark.png`.

`nam-planar-headline` is the headline chart one round later. The earlier one had
to qualify A2 full with "132.6 dB below signal", because the engine that won
there reassociated; this one says **bit-identical** on both rows. Its numbers
come from the Core PR's own `tools/bench_a2_planar` rather than from this
repository's harness — same machine, same protocol, but measuring the code that
was actually proposed rather than the lab kernels it came from.

## Regenerating

```bash
python3 charts/build_charts.py
```

One chart at a time, so a rebuild of one cannot silently rewrite the others:

```bash
python3 charts/build_charts.py nam-planar-headline
```

Every number lives in the `HEADLINE`, `BUILDUP` and `CANDIDATES` tables at the
top of that script and nowhere else, so a figure can only be wrong in one place.
They are transcribed from the reports in `benchmark-results/`.

The renderer drives headless Google Chrome at `--force-device-scale-factor=2`,
so the PNGs are 2× and stay crisp on a Retina display or when GitHub scales them
down. The intermediate `.html` files are left in place — open them in a browser
to check a change before re-rendering.

## Design notes

Colours come from the data-viz reference palette and were checked with its
validator rather than by eye:

- the before/after pair and the build-up ladder are **ordinal** blue ramps —
  single hue, monotone lightness, visible step gaps, light end clearing the
  surface, in both modes;
- the faster/slower pair on the candidate chart is the **diverging** blue↔red,
  which passes CVD separation (worst adjacent ΔE 21.6 protan light, 19.2 dark)
  and 3:1 contrast against both surfaces.

Two things worth preserving if you edit these:

- **The headline bars are indexed to each row's own `a2_fast` = 100%.** A2 full
  and A2 nano differ 7× in absolute cost, so a shared linear axis would flatten
  nano into nothing. Indexing to a common base is the correct fix for that (the
  wrong fix is two y-scales on one plot). The absolute milliseconds ride the bars
  so the normalisation hides nothing.
- **On the candidate chart, bar length is the speedup, not the time.** An earlier
  draft plotted time, which meant the best kernel had the shortest bar while its
  label read `2.10×` — it read backwards. Length-is-goodness with a rule at 1.00×
  is the version that survives a glance.
