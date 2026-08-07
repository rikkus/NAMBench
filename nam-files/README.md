# Captures

**The `.nam` files are deliberately not committed.** They are captures made by
other people, and this repository has no right to redistribute them. Drop them
into this directory yourself and everything works; the `.gitignore` keeps them
out of git.

The input DI (`audio-input/input.wav`) *is* committed, so the only thing you
have to supply is the model.

## What the published results were measured on

Every number in [`SLIMMED-PATH.md`](../SLIMMED-PATH.md), in `benchmark-results/`
and in the charts came from the first capture below. It is the default the CLI
picks (`defaultModelURL` prefers a file starting `Ampeg SVT - Gain 10`), and the
one the slimmed-path work is specified around: an A2 `SlimmableContainer` whose
submodel 0 is `channels=3, bottleneck=3, max_value=0.5`, 1871 weights, and whose
submodel 1 is the 8-channel full model.

| Capture | Modeled by | SHA-1 | Source |
|---|---|---|---|
| `Ampeg SVT - Gain 10 Ultra Lo and Hi MD 421.nam` | tone3000 | `56f251f9a603e342bfe4ea45a568341207407d8f` | <!-- FILL: tone3000.com URL --> |
| `Ampeg SVT - Ultra Hi SM57.nam` | tone3000 | `5c88f2fcdab2dd882e3ee2466dd2eb92ea7c57d6` | <!-- FILL: tone3000.com URL --> |
| `ORNG-V30-e609-Center.nam` | sunburst1313 | `eb1546a297fcd5ab065cf525a79e99f448bd8d58` | <!-- FILL: tone3000.com URL --> |
| `RIFF TAPE 001_Green Russian + OR120 + PPC412.nam` | hafishmaulana | `9840014fdba4cef6088389db163097b34a262fca` | <!-- FILL: tone3000.com URL --> |

Only the first is needed to reproduce anything published here; the other three
are what happened to be in the directory and are listed so a checksum mismatch
is diagnosable rather than mysterious.

Verify a download with:

```bash
shasum -a 1 nam-files/*.nam
```

## Using a different capture

Any A2-shape capture works — pass it with `--model`:

```bash
nambench --model /path/to/your.nam --submodel narrowest --slim all
```

The harness asserts the shape rather than assuming it, so a capture that is not
A2 fails with a readable message instead of quietly measuring the generic
WaveNet. "A2 shape" means what `a2_fast::is_a2_shape` accepts: one layer array
of 23 layers, `channels == bottleneck` of 3 or 8, the fixed kernel-size and
dilation pattern, LeakyReLU(0.01) throughout, `layer1x1` active with `groups=1`,
and a head rechannel of `kernel_size=16` with bias.

Absolute timings will differ from the published ones on a different capture only
through the machine, not the model — every A2 capture of the same channel count
does identical arithmetic. What must not differ is the *ratio* between engines.
