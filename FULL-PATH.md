# Optimising the A2 full (8-channel) path

`fused` beats `a2_fast` by 1.9× on this submodel, and the slimmed-path work
([SLIMMED-PATH.md](SLIMMED-PATH.md)) beat `a2_fast` by 2.1× on the 3-channel one
while staying **bit-identical** to it. This is the third round: taking what the
nano lab learned and pointing it at the 8-channel path, where the thing to beat
is `fused` rather than `a2_fast`.

Two questions, not one:

1. Can `fused` be beaten while staying bit-identical **to `fused`**?
2. Can it be beaten by something bit-identical **to `a2_fast`** — the way the
   nano winner was — so that the result needs no listening test at all?

The answer to both is yes, and the second one turned out to be the more
interesting question, because answering it required first establishing something
about `a2_fast` that was not obvious.

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
| one chain across every tap and channel, FMA (*`fused`'s shape*) | 0 / 224 |
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

**This also explains `fused`'s 132.6 dB.** It seeds its accumulator with the
*bias* and folds every tap and every input channel into a single 48-long chain
([fused.cpp:508-528](vendor/fused/NAM/wavenet/fused.cpp)); it uses a fused
multiply-add for the mixin where Eigen rounds twice
([fused.cpp:534](vendor/fused/NAM/wavenet/fused.cpp)); and it finishes the head
with a pairwise `vaddvq_f32` ([fused.cpp:663](vendor/fused/NAM/wavenet/fused.cpp)).
Three specific, identifiable divergences — not accumulated drift.

And it has a consequence that runs against intuition: **`a2_fast`'s order is
friendlier to the machine than `fused`'s.** Per output element it offers K
independent 8-long chains plus a K-long reduction, where `fused` offers one
48-long chain. The bit-identical constraint is not a handicap here.

## Why it is not a FLOP problem

| | |
|---|---:|
| Real arithmetic in A2 full | **≈2,928 vector FMA per frame** |
| `fused` per frame | ≈430 ns ≈ 1,500 cycles on an M2 P-core |
| → sustained | **≈1.95 vector FMA/cycle, against four FP pipes** |
| Loads+stores per frame per layer | ≈62, against 112 FMAs |
| µops per frame | ≈4,800, at IPC ≈ 3.2 of a possible 8 |
| Ring footprint | **≈460 KB** (M2 P-core L1D is 128 KB) |
| Weight working set | ≈46 KB, stays hot |

`fused` is at about half the FMA issue rate. It is not load-bound and not
decode-bound; the gap is dependency stalls. And the code says where they are.

**`conv_block` picks its frame tile as `(Q <= 4) ? 4 : 2`**
([fused.cpp:542](vendor/fused/NAM/wavenet/fused.cpp)). At C=8, Q is 2, so the
tile is 4 frames and `acc[Q][T]` is **8 independent FMA chains** — against four
pipes at 4-cycle latency, which want about 16. That heuristic is right for C=16
and under-serves C=8 exactly. **`tail_block` picks `(Q <= 4) ? 2 : 1`**
([fused.cpp:611](vendor/fused/NAM/wavenet/fused.cpp)), giving *four* chains and
reloading all sixteen `layer1x1` weight vectors every two frames.

That is the same lever the nano lab found dominated everything else, wearing a
different hat, and it is the single largest effect here too.

## The two families

`vendor/fused` is a superset of upstream: it carries a byte-identical `a2_fast`
as well as `fused`. So one lab framework (`NAMEngineFull`, built from that tree
through the same target template as everything else) can hold both families,
each with its own verbatim-port control:

| | Family | Vectorises across | Control | Bit-identical to |
|---|---|---|---|---|
| `a2*` | planar | **frames** | `a2_baseline` | `a2_fast` (`upstream`) |
| `fu*` | channel-major | **channels** | `fu_baseline` | `fused` |

Every full-lab kernel is measured against **both** shipping engines, and the
reports carry two parity columns, because "bit-identical" only means something
once it says to what.

### The planar correction

The obvious objection to planar at C=8 is that a channel-major vector wastes no
lanes there, so the layout change that won at C=3 has nothing to win. That was
my first judgement and it was wrong, for two reasons.

**It is what makes bit-identity practical.** A planar register holds four
consecutive *frames* of one channel, so every lane independently executes
`a2_fast`'s per-frame scalar chain. Nothing is reassociated. `fused` cannot do
this: it vectorises across channels, so its reduction runs across lanes and
lands on a different association by construction.

**And it costs nothing.** Per tap per frame both layouts are 2 input loads, 4
weight loads and 16 FMAs. What differs is register pressure, and it differs in
both directions: the `fu*` conv needs `acc` 2T + `iv` T + weights 2, while the
`a2*` conv needs `z` 2T + `t` 2T + input T/4 + weights 2 — because `a2_fast`'s
association requires the running total and the current tap's partial to be live
at the same time. That is the price of exactness, and it shows up as a lower
tile ceiling.

## What each candidate did

### The lever that dominated: the frame tile

Both families' tile sweeps are the same story as the nano lab's `widetile`, and
in the `fu*` family the mechanism is unusually legible, because the register
budget predicts the shape of the curve in advance:

| `fu` conv tile | Accumulator chains | Vector registers needed | |
|---:|---:|---:|---|
| 4 (*`fused`'s own*) | 8 | 8 + 4 + 2 = 14 | half the chains four 4-cycle pipes want |
| 6 | 12 | 12 + 6 + 2 = 20 | |
| **8** | **16** | 16 + 8 + 2 = **26** | the sweet spot: enough chains, still fits |
| 10 | 20 | 20 + 10 + 2 = 32 | exactly at the limit |
| 12 | 24 | 24 + 12 + 2 = 38 | spills |
| 16 | 32 | 32 + 16 + 2 = 50 | spills badly |

The measured curve turns at exactly 8 and collapses at exactly 12, which is
where the arithmetic says the 32 vector registers run out.

The `a2*` family peaks at tile 8 as well, but for a different reason: at tile 8
it already needs 36 registers and *is* spilling, and still wins, because the
extra independent chains and the halved weight-load count outweigh the spill
traffic. It falls off at 16.

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

**The head tile.** `a2_fast` runs 128 sequential FMAs per frame in one chain;
`fused` runs 32 vector FMAs per frame in one chain. Either way it is latency,
not throughput: about 1% of the arithmetic on a chain hundreds of cycles deep.
Running several of those chains at once is free in registers and costs nothing
in exactness, because each chain still covers its own frames in the reference's
own order.

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
the head reads `head_sum`, not the residual. `fused` already skips it
(`skip_l1x1_output`); `a2_fast` does not. And the first layer's `head_sum`
accumulate has nothing to accumulate onto, so the per-block `memset` is
unnecessary. Note that the fold is kept as an add against `+0.0` rather than a
plain store: `0.0f + (-0.0f)` is `+0.0f`, so storing would differ from `a2_fast`
on a signed zero. It costs one vector add per four frames in one layer, and buys
provable exactness instead of an argument about whether that case can arise.

### The ring sweep, and where it disagrees with both engines

`NAM_A2_RING_MODE` is a compile-time `#define` inside `a2_fast`; here the
strategy is a template parameter, so the sweep needs no rebuild and the kernels
are otherwise identical. Four strategies, not the nano lab's three, because
`fused` uses a fourth:

| Strategy | Who ships it | Footprint | Copy traffic |
|---|---|---:|---|
| pow2 + eager mirror | `a2_fast` (mode 1) | 460 KB | a mirror memcpy every block |
| pow2 + lazy mirror | **`fused`** | 460 KB | only on blocks whose read wraps |
| exact + lazy mirror | — | 295 KB | smallest, but wraps almost every block |
| linear + rewind | `a2_fast` at mode 0 | 445 KB | occasional large memmove |

The nano lab's finding was that `exact + lazy` — the smallest footprint — was
the worst, because an exactly-sized ring wraps nearly every block and then the
lazy mirror fires once per *tap* rather than once per layer. That reproduces at
C=8: it is the worst of the four in both families.

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

**`fusez`** — running `fused`'s conv, activation and tail as one pass with `z`
in registers, instead of three passes over the `_z` buffer. This was expected to
win: it removes 8 vector memory operations per frame per layer across 23 layers,
and it is exactly what the planar family does by construction. It lost. Holding
the `layer1x1` accumulators in the same tile as the conv accumulators is 16 more
registers on top of a conv that already wants 26 at tile 8, and the spills cost
more than the round-trips they save. The planar family gets the same fusion for
free only because its `post` stage reloads the residual input rather than
carrying it.

## The result

Apple M2 (4P + 4E), macOS 27.0, Release, 48 kHz, 64-frame blocks, full submodel
of `Ampeg SVT - Gain 10 Ultra Lo and Hi MD 421.nam` (8 channels, 12,146
weights), 523,808 frames per pass. Full protocol: 5 s warm-up discarded, 30 s of
timed passes, mean of the fastest 70% required to agree within 3%.

| Kernel | Mean / pass | Fastest | vs `a2_fast` | vs `fused` | Bit-identical to |
|---|---:|---:|---:|---:|---|
| `upstream` (`a2_fast`) | 414.71 ms | 410.76 ms | — | 0.53× | — |
| `fused` | 219.71 ms | 215.91 ms | 1.888× | — | — *(132.6 dB from `a2_fast`)* |
| `a2_baseline` — **control** | 415.13 ms | 411.32 ms | 0.999× | — | **`a2_fast`** |
| `fu_baseline` — **control** | 217.88 ms | 216.86 ms | 1.903× | 1.008× | **`fused`** |
| **`a2s8_h8_lane`** | **166.56 ms** | 164.49 ms | **2.490×** | **1.319×** | **`a2_fast`** |
| **`fu_s8_head_lazy`** | **147.50 ms** | 146.22 ms | **2.812×** | **1.489×** | **`fused`** |

Both questions come out yes:

- **`fused` can be beaten by 1.489×**, by a kernel that is bit-identical to
  `fused` — a drop-in replacement with nothing to argue about.
- **`fused` can also be beaten by 1.319× by a kernel bit-identical to
  `a2_fast`.** That is the more interesting result even though it is the slower
  number: it is 2.490× the engine it is exact against, and it needs no listening
  test, no tolerance, and no argument about audibility.

Nothing here trades accuracy for speed. Every candidate in this document is
bit-identical to one of the two shipping engines, and the reports carry both
comparisons for every row.

Both controls hold. `a2_baseline` is bit-identical to `upstream` and lands
within 0.1% of it; `fu_baseline` is bit-identical to `fused` and lands within
0.8% of it. Run-to-run spread on this machine is around ±2%, so differences
smaller than that are not differences — which is exactly why the controls are
there.

**The single sharpest finding is `fu_t8`: 1.37× over `fused` from changing one
constant.** It is `fused`'s own kernel with `conv_block`'s frame tile changed
from 4 to 8 and nothing else — same layout, same association, bit-identical
output. `(Q <= 4) ? 4 : 2` is the right choice at C=16 and leaves C=8 with half
the accumulator chains the machine wants.

## The ablation

Each row is one switch away from its family's control, over 2 s warm-up and 8 s
of timed passes at 4% agreement — enough to rank, where the headline table above
is the full protocol. Ratios are against the `upstream` and `fused` numbers
measured in the same run, so they are not affected by drift between runs.

Each family was measured in two runs, and **ratios are always against the
`upstream`/`fused` numbers from the same run** — the run boundary is marked, and
no number below is compared against a base from a different run.

### `fu*` — channel-major, bit-identical to `fused`

*Run 1 — `upstream` 409.31 ms, `fused` 218.66 ms:*

| Kernel | Mean | vs `fused` | |
|---|---:|---:|---|
| `fu_t4` — *fused's own settings* | 216.18 ms | 1.01× | lands on `fused`, as it must |
| `fu_t6` | 200.64 ms | 1.09× | |
| **`fu_t8`** | **156.56 ms** | **1.40×** | the sweet spot |
| `fu_t10` | 176.52 ms | 1.24× | at the register limit |
| `fu_t12` | 237.39 ms | 0.92× | spills |
| `fu_t16` | 259.39 ms | 0.84× | spills badly |
| `fu_tail4` | 215.24 ms | 1.02× | inside the noise band |
| `fu_tail8` | 215.02 ms | 1.02× | inside the noise band |
| `fu_fusez` | 232.11 ms | 0.94× | **lost** — see above |
| `fu_ringdirect` | 210.41 ms | 1.04× | |
| `fu_headtile` | 210.04 ms | 1.04× | |
| `fu_storehead` | 212.88 ms | 1.03× | |

*Run 2 — `upstream` 408.81 ms, `fused` 214.45 ms:*

| Kernel | Mean | vs `fused` | |
|---|---:|---:|---|
| `fu_t4` — *fused's own settings* | 210.38 ms | 1.02× | |
| `fu_ringeager` | 218.33 ms | 0.98× | |
| `fu_ringexact` | 220.25 ms | 0.97× | worst of the four, as at C=3 |
| `fu_ringlinear` | 219.26 ms | 0.98× | |
| `fu_s6` | 195.20 ms | 1.10× | |
| `fu_s8` | 148.51 ms | 1.44× | |
| `fu_s8_eager` | 151.72 ms | 1.41× | |
| `fu_s8_lazy` | 148.21 ms | 1.45× | |
| `fu_s8_head` | 145.76 ms | 1.47× | |
| **`fu_s8_head_lazy`** | **145.35 ms** | **1.48×** | the composition |

The tile is nearly the whole story here: `fu_t8` alone captures 1.40× of the
1.48× the full composition reaches. Everything else is worth a few percent each.

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

The `a2*` family is flatter: no single switch dominates the way the tile does in
`fu*`, and the composition is worth more than the sum of the parts looks like it
should be. That is consistent with it being closer to the register ceiling
throughout — every switch that frees a register or removes a buffer pass helps
the next one.

## The two sweeps

### Block size

Frames per `process()` call, 2 s warm-up / 8 s timing:

| Block | `a2_fast` | `fused` | `a2s8_h8_lane` | `fu_s8_head_lazy` | best vs `fused` |
|---:|---:|---:|---:|---:|---:|
| 32 | 466.76 ms | 220.40 ms | 173.09 ms | 148.22 ms | **1.487×** |
| 64 | 410.40 ms | 218.31 ms | 167.75 ms | 144.43 ms | 1.511× |
| 128 | 387.02 ms | 214.79 ms | 162.04 ms | 142.71 ms | 1.505× |
| 256 | 372.88 ms | 214.16 ms | 159.30 ms | 142.86 ms | 1.499× |
| 512 | 361.43 ms | 209.25 ms | 156.41 ms | 143.52 ms | 1.458× |

**The 32-frame column is the one that matters for a plugin**, and the advantage
is fully intact there — 1.487× over `fused`, and 3.15× over `a2_fast`, which is
its *best* relative showing of the five because `a2_fast` degrades sharply at
small blocks (466.76 ms at 32 against 361.43 ms at 512) while the optimised
kernels barely move. This is not a win that exists only at unrealistic buffer
sizes.

### Ring strategy

Four strategies, measured as a template parameter so no rebuild is needed and
the arithmetic is identical across them:

| Strategy | `a2*` (run 2) | `fu*` (run 2) |
|---|---:|---:|
| pow2 + eager mirror — *`a2_fast`'s default* | **208.85 ms** *(`a2p4`)* | 218.33 ms |
| pow2 + lazy mirror — *`fused`'s* | 213.91 ms | **210.38 ms** *(`fu_t4`)* |
| exact + lazy mirror | 224.68 ms | 220.25 ms |
| linear + rewind — *`a2_fast` at mode 0* | 214.68 ms | 219.26 ms |

Two things worth recording. First, **`exact + lazy` is the worst of the four in
both families**, reproducing the nano lab's most counter-intuitive finding: the
smallest footprint loses, because an exactly-sized ring wraps nearly every block
and the lazy mirror then fires once per *tap* rather than once per layer.
Second, **each engine's shipped choice is the best one for its own family and
not for the other's** — and the spread is only about 5%, where at C=3 the nano
lab measured 14%. The ring matters less at C=8 because there is far more
arithmetic per byte moved. Inside the compositions the ranking shifts again
(`fu_s8_head_lazy` beats `fu_s8_head`, and `a2s8_linear` beats `a2s8`), which is
why both survivors were carried forward rather than one.

## Verification

1. **The lab is honest, twice over.** `a2_baseline` is bit-identical to
   `upstream` and times within 0.4% of it; `fu_baseline` is bit-identical to
   `fused` and times within 1.0% of it. Two controls because there are two
   references — without both, a gain in one family could not be attributed.
   `fu_t4`, which instantiates the `fu*` template with exactly `fused`'s own
   settings, is a third check on the template rather than the lab, and lands in
   the same place.

2. **Parity gates every candidate, against both engines.** Every full-lab kernel
   is compared over the full 523,808-frame file against `upstream` *and* against
   `fused`, and both numbers are reported next to the speed rather than in a
   footnote. Every kernel in this document is bit-identical to one of the two.

3. **The reduction order was established, not assumed.**
   `Scripts/eigen-order-probe/` compares Eigen's own output bit-for-bit against
   candidate orderings across 224 combinations before any kernel was written. It
   is a standalone program built with the same flags, so it can be re-run against
   a different Eigen without touching the benchmark.

4. **The engine is asserted, not assumed.** The full-lab framework reports
   `NbEngineFull` only once a kernel has actually been selected; until then it
   routes and reports exactly as `fused`, which is the engine it is built
   alongside. A run that forgot to select one fails the existing engine check
   rather than quietly measuring `fused` forty-eight times.

5. **The lab is dropped by shape, not by flag.** Its kernels are specialised for
   8 channels and refuse anything else, so a `--full` run on the nano submodel
   drops them the same way `fused` is dropped there — from the channel count read
   out of the file. Checked, not assumed:

   ```
   fused excluded: 3 channels is not a multiple of 4, ...
   full lab excluded: its kernels are specialised for 8 channels and this submodel has 3
   ```

6. **Nothing else moved.** Re-run after all the harness changes:

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

7. **The winners hold at the block size that matters.** At `--block-size 32`,
   `fu_s8_head_lazy` is 1.487× `fused` and `a2s8_h8_lane` is 1.273× — see the
   block sweep above. A win that only existed at large buffers would not be a
   win for a plugin.

## Considered and rejected without coding

- **fp16 / bf16 storage**, which would halve the 460 KB ring footprint. Out of
  scope under the same fp32-only constraint the nano work held itself to: the
  point of both exercises is a drop-in replacement that needs no listening test,
  and a format change forfeits that by construction.
- **int8 / SDOT.** Same reason, more so.
- **Accelerate / BNNS / AMX / SME.** Call and setup overhead against an 8×8.
- **Threading.** The layer stack is a serial dependency chain.
- **Folding `layer1x1` into the next layer's conv weights.** Blocked by the
  residual add, exactly as at C=3: history stores `layer_in + L·a`, not `L·a`.

## Promotion

Two independent changes, and they do not compete — one improves `fused`, the
other improves `a2_fast`, and each is bit-identical to the thing it replaces.

**To `fused.cpp`, worth about 1.5×:**

1. **Fix the frame tile.** `conv_block`'s `T = (Q <= 4) ? 4 : 2` and
   `tail_block`'s `T = (Q <= 4) ? 2 : 1` are the whole of the largest effect
   here. The budget that predicts the curve is
   `accumulators (Q·T) + inputs (T) + weights (Q) ≲ 30` vector registers, which
   at Q=2 says T=8, and 8 is exactly where the measured curve turns. This is a
   one-line change with bit-identical output. **It should be swept per channel
   count rather than generalised from C=8** — the existing heuristic is right at
   C=16, and the only claim this document supports is the C=8 one.
2. **Write residuals straight into the next layer's ring.** Removes 22
   block-sized copies and a pass over a scratch buffer, about 46 KB of memcpy
   per 64-frame block at C=8.
3. **Tile the head across frames.** `head_conv_block` runs one 32-deep FMA chain
   per frame and reloads all sixteen weight vectors each time; several chains at
   once is free in registers and exact.
4. **Re-examine the ring strategy.** `fused` uses pow2 + lazy mirror. That is not
   the best of the four at C=8, and the nano lab found the shipped choice wrong
   at C=3 too. This is cheap to make a template parameter, as it is here, and
   then it is a measurement rather than a guess.

**To `a2_fast`, worth about 2.49× on this submodel, bit-identically:** add the
planar C=8 kernel as the `Channels == 8` branch. This is the more interesting
change, because it is exact: the output does not move by one bit over 523,808
frames, so it needs no listening test and no tolerance argument. The structure
is the same one the nano lab promoted for `Channels == 3` — vectorise across
frames, keep `z` in registers, wide tile, residuals into the next ring — so the
two branches would end up sharing a shape rather than diverging further.

Independently of either: **the ring strategy is worth choosing per channel
count.** At C=3 the nano lab measured 14% between `a2_fast`'s shipped mode and
its other one. Here the spread across four strategies is smaller but real, and
the ranking is not the same as at C=3. It is a `#define` in `a2_fast` and a
hardcoded structure in `fused`; in both it deserves to be a parameter.

## Reproducing

```bash
./Scripts/fetch-vendor.sh && xcodegen generate && xcodebuild -project NAMBench.xcodeproj -scheme nambench-cli -configuration Release build
```

```bash
"$(xcodebuild -project NAMBench.xcodeproj -scheme nambench-cli -configuration Release -showBuildSettings | awk -F' = ' '/ BUILT_PRODUCTS_DIR =/{print $2; exit}')/nambench" --submodel widest --full all
```

`nambench --list-full` prints the kernel table; `--full a2s8_h8_lane,fu_s8_head_lazy`
selects a subset by name or index. The Eigen reduction-order probe is standalone
and needs no build of the benchmark:

```bash
./Scripts/eigen-order-probe/run.sh
```
