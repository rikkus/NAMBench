# Optimising the A2 full (8-channel) path

The slimmed-path work ([SLIMMED-PATH.md](SLIMMED-PATH.md)) beat `a2_fast` by
2.1× on the 3-channel submodel while staying **bit-identical** to it. This is
the third round: taking what the nano lab learned and pointing it at the
8-channel path.

One question: **can `a2_fast` at C=8 be beaten while staying bit-identical to
it** — the way the nano winner was — so that the result needs no listening
test at all?

The answer is yes, and answering it required first establishing something
about `a2_fast` that was not obvious.

*(Historical note: this lab originally also carried a second family, `fu*`,
that reproduced the arithmetic of a NEON engine called `fused` from a fork —
and a second question, whether `fused` itself could be beaten bit-identically.
`fused` has since been retired from this repository entirely, superseded by
the planar kernels that shipped as Core PR #313, and the `fu*` family was
removed along with it. Where the analysis below explains something about
`a2_fast` *by contrast* with how `fused` used to behave, that contrast is kept
— it is real and it explains real numbers — but clearly marked as history, not
as a currently buildable comparison.)*

*(Result tables follow the analysis.)*

## `a2_fast` at C=8 is Eigen, and its arithmetic is reproducible

`a2_fast` branches on channel count. The C=3 branch is a hand-written scalar
GEMV, which is why the nano lab could match it bit-for-bit just by writing the
same chain. **The C=8 branch is Eigen**
([a2_fast.cpp:558-606](vendor/upstream/NAM/wavenet/a2_fast.cpp)):

```cpp
ztile.setZero();
for (int k = 0; k < K; k++)              // one 8x8 × 8xN GEMM per tap
  ztile.noalias() += W_k * input_block_k;
ztile.colwise() += conv_b_vec;           // bias last, not seeded
ztile.noalias() += mixin_vec * cond_row; // product then add — two roundings
ztile = (ztile.array() < 0).select(ztile.array() * kLeakySlope, ztile.array());
hsum_block += ztile;
lin_block.noalias() += l1x1_mat * ztile;
lin_block.colwise() += l1x1_b_vec;       // bias after the residual add
```

A hand-written NEON replacement can only be bit-identical to that if Eigen's
reduction order is knowable. Rather than assume, `Scripts/eigen-order-probe/`
replicates those exact expressions on those exact shapes and compares them
bit-for-bit against candidate orderings, across 8 random trials × K ∈ {6, 15} ×
N ∈ {1, 2, 3, 4, 7, 8, 16, 31, 32, 33, 64, 128, 256, 512} — the block sizes the
benchmark uses, plus the 32-frame tail block a 523,808-frame file actually
produces at `--block-size 64`, because Eigen's blocking can depend on N.

| Candidate ordering | Matches |
|---|---|
| per tap, inner sum over j from zero in increasing order, **FMA** | **224 / 224** |
| per tap, increasing j, multiply and add rounded apart | 0 / 224 |
| one chain across every tap and channel, FMA (*the retired `fused` engine's shape*) | 0 / 224 |
| per tap, depth split into two halves summed at the end | 0 / 224 |
| per tap, decreasing j | 0 / 224 |

So the order is:

```
per tap k:  t_i = 0;  for j = 0..7:  t_i = fma(W_k(i,j), x_k(j), t_i)
then:       z_i = 0;  for k:         z_i = z_i + t_i^(k)
then:       z_i = z_i + bias_i
            z_i = z_i + (mixin_i * cond)     <- two roundings, not an FMA
            z_i = LeakyReLU(z_i)
            head_sum_i = head_sum_i + z_i
            u_i = 0;  for j:  u_i = fma(L(i,j), z_j, u_i)
            lin_i = (lin_i + u_i) + l1x1_b_i
```

and the whole layer body written that way is bit-identical too, 224/224. The
head is plainly scalar — 16 taps × 8 channels of sequential FMA from the bias —
and the probe confirms both that the same chain run four frames at a time in a
vector preserves the bits, and that its FMA contraction is load-bearing: the
same order with the multiply and add rounded separately matches only 6 times in
224.

**Historical: this also explains why the retired `fused` engine landed 132.6 dB
away from `a2_fast`.** It seeded its accumulator with the *bias* and folded
every tap and every input channel into a single 48-long chain
(`fused.cpp:508-528`); it used a fused multiply-add for the mixin where Eigen
rounds twice (`fused.cpp:534`); and it finished the head with a pairwise
`vaddvq_f32` (`fused.cpp:663`). Three specific, identifiable divergences — not
accumulated drift.

And it had a consequence that runs against intuition: **`a2_fast`'s order is
friendlier to the machine than `fused`'s was.** Per output element it offers K
independent 8-long chains plus a K-long reduction, where `fused` offered one
48-long chain. The bit-identical constraint is not a handicap here.

## Historical: why the old comparison was not a FLOP problem

*(This section is entirely about the retired `fused` engine's performance
characteristics, kept because it explains real, previously-published numbers —
not because it describes anything still buildable in this repo.)*

| | |
|---|---:|
| Real arithmetic in A2 full | **≈2,928 vector FMA per frame** |
| `fused` per frame (as measured) | ≈430 ns ≈ 1,500 cycles on an M2 P-core |
| → sustained | **≈1.95 vector FMA/cycle, against four FP pipes** |
| Loads+stores per frame per layer | ≈62, against 112 FMAs |
| µops per frame | ≈4,800, at IPC ≈ 3.2 of a possible 8 |
| Ring footprint | **≈460 KB** (M2 P-core L1D is 128 KB) |
| Weight working set | ≈46 KB, stays hot |

`fused` ran at about half the FMA issue rate. It was not load-bound and not
decode-bound; the gap was dependency stalls. And the code said where they were.

**`conv_block` picked its frame tile as `(Q <= 4) ? 4 : 2`** (`fused.cpp:542`).
At C=8, Q is 2, so the tile was 4 frames and `acc[Q][T]` was **8 independent
FMA chains** — against four pipes at 4-cycle latency, which want about 16. That
heuristic was right for C=16 and under-served C=8 exactly. **`tail_block`
picked `(Q <= 4) ? 2 : 1`** (`fused.cpp:611`), giving *four* chains and
reloading all sixteen `layer1x1` weight vectors every two frames.

That is the same lever the nano lab found dominated everything else, wearing a
different hat, and it is the single largest effect in this lab too.

## The `a2*` family

The full lab (`NAMEngineFull`, built from `vendor/upstream` through the same
target template as everything else) holds one family now: `a2*`, planar
kernels that vectorise across **frames**, reproduce `a2_fast`'s arithmetic
exactly, and are validated against `a2_baseline` — a verbatim port of
`a2_fast`'s `Channels == 8` branch that has to land on `a2_fast`'s own number.

*(Historical: a second family, `fu*`, vectorised across channels and
reproduced the retired `fused` engine's arithmetic, with its own control
`fu_baseline`. It has been removed along with `fused`. Where results below
compare against that family, they are kept as history.)*

### The planar correction

The obvious objection to planar at C=8 is that a channel-major vector wastes no
lanes there, so the layout change that won at C=3 has nothing to win. That was
my first judgement and it was wrong, for two reasons.

**It is what makes bit-identity practical.** A planar register holds four
consecutive *frames* of one channel, so every lane independently executes
`a2_fast`'s per-frame scalar chain. Nothing is reassociated. The retired
`fused` engine could not do this: it vectorised across channels, so its
reduction ran across lanes and landed on a different association by
construction.

**And it costs nothing.** Per tap per frame both layouts are 2 input loads, 4
weight loads and 16 FMAs. What differs is register pressure: the `a2*` conv
needs `z` 2T + `t` 2T + input T/4 + weights 2 — because `a2_fast`'s association
requires the running total and the current tap's partial to be live at the
same time. (The channel-major layout used by the retired `fu*` family needed
`acc` 2T + `iv` T + weights 2 instead — a different trade, in the other
direction.) That is the price of exactness, and it shows up as a lower tile
ceiling.

## What each candidate did

### The lever that dominated: the frame tile

The `a2*` family's tile sweep is the same story as the nano lab's `widetile`.
It peaks at tile 8, needing 36 registers and already spilling there — and
still wins, because the extra independent chains and the halved weight-load
count outweigh the spill traffic. It falls off at 16.

*(Historical: in the retired `fu*` family the mechanism was unusually legible,
because the register budget predicted the shape of the curve in advance —
peaking at tile 8 out of 26 registers needed, and collapsing at tile 12 where
32 vector registers ran out.)*

### The switches that paid

**`ringdirect`** has each layer's tail store its residual straight into the next
layer's ring instead of into a scratch buffer that the next layer then copies
in. That removes 22 block-sized copies and a whole extra pass over a working
buffer — about 46 KB of memcpy per 64-frame block at C=8, where the nano lab was
removing 18 KB. `head_sum` gets the same treatment, accumulating straight into
the head ring's write window. The block being written can straddle the ring
wrap; rather than splitting the frame loop, the write runs off the end into the
mirror region and the overhang is folded back afterwards, so the inner loop
stays straight-line — the same device the nano lab used.

**The head tile.** `a2_fast` runs 128 sequential FMAs per frame in one chain
(the retired `fused` engine ran 32 vector FMAs per frame in one chain). Either
way it is latency, not throughput: about 1% of the arithmetic on a chain
hundreds of cycles deep. Running several of those chains at once is free in
registers and costs nothing in exactness, because each chain still covers its
own frames in the reference's own order.

**`l1x1lane`** stores the `layer1x1` weights transposed, so the eight weights
feeding one output channel are contiguous and reach the FMA as two vector loads
addressed by lane, instead of eight separate scalar broadcasts. Identical FMAs
in an identical order — only the route the weight takes to the instruction
changes. It is worth about 6% on its own and rather more inside the composition,
which is the pattern the whole `a2*` family shows: it sits close enough to the
register ceiling that anything freeing a register or a load helps the next thing
along.

**`skiplast`** is two pieces of work the model never reads. The final layer's
`layer1x1` computes a residual that nothing consumes — there is no layer 24, and
the head reads `head_sum`, not the residual. (The retired `fused` engine already
skipped it, via `skip_l1x1_output`; `a2_fast` does not.) And the first layer's
`head_sum` accumulate has nothing to accumulate onto, so the per-block `memset` is
unnecessary. Note that the fold is kept as an add against `+0.0` rather than a
plain store: `0.0f + (-0.0f)` is `+0.0f`, so storing would differ from `a2_fast`
on a signed zero. It costs one vector add per four frames in one layer, and buys
provable exactness instead of an argument about whether that case can arise.

### The ring sweep

`NAM_A2_RING_MODE` is a compile-time `#define` inside `a2_fast`; here the
strategy is a template parameter, so the sweep needs no rebuild and the kernels
are otherwise identical.

| Strategy | Who ships it | Footprint | Copy traffic |
|---|---|---:|---|
| pow2 + eager mirror | `a2_fast` (mode 1) | 460 KB | a mirror memcpy every block |
| pow2 + lazy mirror | *the retired `fused` engine* | 460 KB | only on blocks whose read wraps |
| exact + lazy mirror | — | 295 KB | smallest, but wraps almost every block |
| linear + rewind | `a2_fast` at mode 0 | 445 KB | occasional large memmove |

The nano lab's finding was that `exact + lazy` — the smallest footprint — was
the worst, because an exactly-sized ring wraps nearly every block and then the
lazy mirror fires once per *tap* rather than once per layer. That reproduces at
C=8: it is the worst of the four.

### The ones that were meant to lose, and did

**`prefetch`** on the far-dilation layers. It lost at C=3, and the premise here
is different — those layers reach about 38 KB back at C=8 against 14 KB at C=3,
and planar layout means 8 streams per tap rather than 3 — so it was worth one
measurement rather than an assumption. It lost again. This is the usual outcome
when the access pattern is a fixed repeating offset the hardware prefetchers
already handle.

**The output-channel split.** The `a2*` conv spills at tile 8 because it must
hold both `z` and `t` for all eight output channels. Splitting the conv into two
passes over the output channels halves what the tap loop holds, at the cost of
reading each input plane twice per tap. It was a clean, falsifiable
register-pressure hypothesis, and it is false: the extra loads cost more than
the spills they save, at every tile width tried.

*(Historical: the retired `fu*` family also tried `fusez` — running the
channel-major conv, activation and tail as one pass with `z` in registers,
instead of three passes over a `_z` buffer. It was expected to win and lost:
holding the `layer1x1` accumulators in the same tile as the conv accumulators
added 16 more registers on top of a conv that already wanted 26 at tile 8, and
the spills cost more than the round-trips they saved. The `a2*` family gets the
same fusion for free only because its `post` stage reloads the residual input
rather than carrying it.)*

## The result

Apple M2 (4P + 4E), macOS 27.0, Release, 48 kHz, 64-frame blocks, full submodel
of `Ampeg SVT - Gain 10 Ultra Lo and Hi MD 421.nam` (8 channels, 12,146
weights), 523,808 frames per pass. Full protocol: 5 s warm-up discarded, 30 s of
timed passes, mean of the fastest 70% required to agree within 3%.

| Kernel | Mean / pass | Fastest | vs `a2_fast` | Bit-identical to |
|---|---:|---:|---:|---|
| `upstream` (`a2_fast`) | 414.71 ms | 410.76 ms | — | — |
| `a2_baseline` — **control** | 415.13 ms | 411.32 ms | 0.999× | **`a2_fast`** |
| **`a2s8_h8_lane`** | **166.56 ms** | 164.49 ms | **2.490×** | **`a2_fast`** |

*(Historical, for context: the retired `fused` engine measured 219.71 ms
(1.888× over `a2_fast`) in the same run, and its own best composed candidate,
`fu_s8_head_lazy`, reached 147.50 ms (2.812× over `a2_fast`, 1.489× over
`fused`) while staying bit-identical to `fused`.)*

**`a2_fast` can be beaten by 2.490× by a kernel bit-identical to it.** That is
the result this lab now exists to produce: no listening test, no tolerance, and
no argument about audibility.

Nothing here trades accuracy for speed. Every candidate in this document is
bit-identical to `a2_fast`.

The control holds. `a2_baseline` is bit-identical to `upstream` and lands
within 0.1% of it. Run-to-run spread on this machine is around ±2%, so
differences smaller than that are not differences — which is exactly why the
control is there.

## The ablation

Each row is one switch away from `a2_baseline`, over 2 s warm-up and 8 s of
timed passes at 4% agreement — enough to rank, where the headline table above
is the full protocol. Ratios are against the `upstream` number measured in the
same run, so they are not affected by drift between runs.

### `a2*` — planar, bit-identical to `a2_fast`

*Run 1 — `upstream` 406.99 ms:*

| Kernel | Mean | vs `a2_fast` | |
|---|---:|---:|---|
| `a2p4` | 211.13 ms | 1.93× | the base for this family |
| **`a2p8`** | **192.66 ms** | **2.11×** | |
| `a2p12` | 196.01 ms | 2.08× | |
| `a2p16` | 229.27 ms | 1.78× | |
| `a2p_split8` | 197.59 ms | 2.06× | **lost** to plain `a2p8` |
| `a2p_split16` | 226.82 ms | 1.79× | **lost** to plain `a2p16` |
| `a2p_head2` | 211.22 ms | 1.93× | |
| `a2p_headtile` (head 4) | 205.23 ms | 1.98× | |
| `a2p_head8` | 207.86 ms | 1.96× | |
| `a2p_ringdirect` | 207.02 ms | 1.97× | |
| `a2p_skiplast` | 207.10 ms | 1.97× | |
| `a2p_l1x1lane` | 199.15 ms | 2.04× | |

*Run 2 — `upstream` 408.53 ms:*

| Kernel | Mean | vs `a2_fast` | |
|---|---:|---:|---|
| `a2p4` | 208.85 ms | 1.96× | the base for this run |
| `a2p_ringlazy` | 213.91 ms | 1.91× | |
| `a2p_ringexact` | 224.68 ms | 1.82× | worst of the four again |
| `a2p_ringlinear` | 214.68 ms | 1.90× | |
| `a2p_prefetch` | 218.38 ms | 1.87× | **lost**, as at C=3 |
| `a2s4` | 198.47 ms | 2.06× | |
| `a2s8` | 182.51 ms | 2.24× | |
| `a2s12` | 186.90 ms | 2.19× | |
| `a2s8_linear` | 174.86 ms | 2.34× | |
| `a2s12_linear` | 175.90 ms | 2.32× | |
| `a2s8_h8` | 172.74 ms | 2.37× | |
| **`a2s8_h8_lane`** | **165.11 ms** | **2.47×** | the composition |
| `a2s8_h8_split` | 176.02 ms | 2.32× | the split loses inside the stack too |
| `a2s16_split` | 207.35 ms | 1.97× | |

The `a2*` family is flatter: no single switch dominates the way the tile does,
and the composition is worth more than the sum of the parts looks like it
should be. That is consistent with it being closer to the register ceiling
throughout — every switch that frees a register or removes a buffer pass helps
the next one.

*(Historical: the retired `fu*` family's own ablation showed the tile as
nearly the whole story — `fu_t8` alone captured 1.40× of the 1.48× its full
composition reached, against `fused`'s own 218–220 ms baseline in the same
runs.)*

## The two sweeps

### Block size

Frames per `process()` call, 2 s warm-up / 8 s timing:

| Block | `a2_fast` | `a2s8_h8_lane` | speedup |
|---:|---:|---:|---:|
| 32 | 466.76 ms | 173.09 ms | **2.697×** |
| 64 | 410.40 ms | 167.75 ms | 2.446× |
| 128 | 387.02 ms | 162.04 ms | 2.388× |
| 256 | 372.88 ms | 159.30 ms | 2.341× |
| 512 | 361.43 ms | 156.41 ms | 2.311× |

**The 32-frame column is the one that matters for a plugin**, and the advantage
is fully intact there — 2.697× over `a2_fast`, its *best* relative showing of
the five block sizes measured, because `a2_fast` degrades sharply at small
blocks (466.76 ms at 32 against 361.43 ms at 512) while the optimised kernel
barely moves. This is not a win that exists only at unrealistic buffer sizes.

### Ring strategy

Four strategies, measured as a template parameter so no rebuild is needed and
the arithmetic is identical across them:

| Strategy | `a2*` (run 2) |
|---|---:|
| pow2 + eager mirror — *`a2_fast`'s default* | **208.85 ms** *(`a2p4`)* |
| pow2 + lazy mirror — *the retired `fused` engine's* | 213.91 ms |
| exact + lazy mirror | 224.68 ms |
| linear + rewind — *`a2_fast` at mode 0* | 214.68 ms |

Two things worth recording. First, **`exact + lazy` is the worst of the four**,
reproducing the nano lab's most counter-intuitive finding: the smallest
footprint loses, because an exactly-sized ring wraps nearly every block and
the lazy mirror then fires once per *tap* rather than once per layer. Second,
the spread here is only about 5%, where at C=3 the nano lab measured 14% — the
ring matters less at C=8 because there is far more arithmetic per byte moved.
Inside the composition the ranking shifts again (`a2s8_linear` beats `a2s8`),
which is why `a2s8_h8_lane` carries the linear ring forward rather than the
default one.

## Verification

1. **The lab is honest.** `a2_baseline` is bit-identical to `upstream` and
   times within 0.4% of it.

2. **Parity gates every candidate.** Every full-lab kernel is compared over the
   full 523,808-frame file against `upstream`, and the number is reported next
   to the speed rather than in a footnote. Every kernel in this document is
   bit-identical to `a2_fast`.

3. **The reduction order was established, not assumed.**
   `Scripts/eigen-order-probe/` compares Eigen's own output bit-for-bit against
   candidate orderings across 224 combinations before any kernel was written. It
   is a standalone program built with the same flags, so it can be re-run against
   a different Eigen without touching the benchmark.

4. **The engine is asserted, not assumed.** The full-lab framework reports
   `NbEngineFull` only once a kernel has actually been selected; until then it
   routes and reports exactly as `upstream` (`a2_fast`), which is the engine it
   is built alongside. A run that forgot to select one fails the existing
   engine check rather than quietly measuring `a2_fast` forty-eight times.

5. **The lab is dropped by shape, not by flag.** Its kernels are specialised for
   8 channels and refuse anything else, so a `--full` run on the nano submodel
   drops them, from the channel count read out of the file. Checked, not
   assumed:

   ```
   full lab excluded: its kernels are specialised for 8 channels and this submodel has 3
   ```

6. **Nothing else moved.** Re-run after the harness changes described above
   (measured while `fused` was still part of the line-up, and kept here as the
   historical check it was):

   | | before | after |
   |---|---:|---:|
   | A2 full, `fused` vs `a2_fast` | 1.898× | 1.888× |
   | A2 nano, `slim:baseline` (control) | 1.009× | 1.008×, still bit-identical |
   | A2 nano, `slim:planar` | 1.179× | 1.162× |
   | A2 nano, `slim:stacked32` | 1.752×* | 1.925× |
   | A2 nano, `slim:stacked_linear` | 2.098× | 2.016× |

   All inside the run-to-run band on a shorter protocol, and every slim kernel
   still bit-identical to `upstream`. (*`stacked32` is `widetile32` in the
   SLIMMED-PATH table; the composed `stacked32` was 1.954× there.)

7. **The winner holds at the block size that matters.** At `--block-size 32`,
   `a2s8_h8_lane` is 2.697× `a2_fast` — see the block sweep above. A win that
   only existed at large buffers would not be a win for a plugin.

## Considered and rejected without coding

- **fp16 / bf16 storage**, which would halve the ring footprint. Out of
  scope under the same fp32-only constraint the nano work held itself to: the
  point of both exercises is a drop-in replacement that needs no listening test,
  and a format change forfeits that by construction.
- **int8 / SDOT.** Same reason, more so.
- **Accelerate / BNNS / AMX / SME.** Call and setup overhead against an 8×8.
- **Threading.** The layer stack is a serial dependency chain.
- **Folding `layer1x1` into the next layer's conv weights.** Blocked by the
  residual add, exactly as at C=3: history stores `layer_in + L·a`, not `L·a`.

## Promotion

**To `a2_fast`, worth about 2.49× on this submodel, bit-identically:** add the
planar C=8 kernel as the `Channels == 8` branch. This is exact: the output does
not move by one bit over 523,808 frames, so it needs no listening test and no
tolerance argument. The structure is the same one the nano lab promoted for
`Channels == 3` — vectorise across frames, keep `z` in registers, wide tile,
residuals into the next ring — so the two branches end up sharing a shape
rather than diverging further. This is the change that shipped as Core PR
#313.

Independently: **the ring strategy is worth choosing per channel count.** At
C=3 the nano lab measured 14% between `a2_fast`'s shipped mode and its other
one. Here the spread across four strategies is smaller but real, and the
ranking is not the same as at C=3. It is a `#define` in `a2_fast`; it deserves
to be a parameter.

*(Historical: a parallel set of changes to `fused.cpp` — fixing its frame tile,
writing residuals straight into the next ring, tiling its head across frames,
and reconsidering its ring strategy — was worth about 1.5× on the engine as it
existed at the time. `fused` has since been retired rather than carried
forward, so this promotion path no longer applies to anything buildable here.)*

## Reproducing

```bash
./Scripts/fetch-vendor.sh && xcodegen generate && xcodebuild -project NAMBench.xcodeproj -scheme nambench-cli -configuration Release build
```

```bash
"$(xcodebuild -project NAMBench.xcodeproj -scheme nambench-cli -configuration Release -showBuildSettings | awk -F' = ' '/ BUILT_PRODUCTS_DIR =/{print $2; exit}')/nambench" --submodel widest --full all
```

`nambench --list-full` prints the kernel table; `--full a2s8_h8_lane`
selects a subset by name or index. The Eigen reduction-order probe is standalone
and needs no build of the benchmark:

```bash
./Scripts/eigen-order-probe/run.sh
```
