# Captures

**The `.nam` files are deliberately not committed.** They are captures made by
other people, and this repository has no right to redistribute them. Drop them
into this directory yourself and everything works; the `.gitignore` keeps them
out of git.

The input DI (`audio-input/input.wav`) *is* committed, so the only thing you
have to supply is the model.

## What the published results were measured on

Every number in [`SLIMMED-PATH.md`](../SLIMMED-PATH.md), in `benchmark-results/` and in the charts came from the capture *Ampeg SVT - Gain 10 Ultra Lo and Hi MD 421.nam* - which can be found [here](https://www.tone3000.com/tones/ampeg-svt-classic-with-6x10-28202) at Tone3000. (SHA-1: `56f251f9a603e342bfe4ea45a568341207407d8f`)

The slimmed-path work is specified around: A2 `SlimmableContainer` whose submodel 0 is `channels=3, bottleneck=3, max_value=0.5`, 1871 weights, and whose submodel 1 is the 8-channel full model.

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
