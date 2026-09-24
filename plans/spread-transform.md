# Spread the FFT path's transform callback

## Context

At 8192 taps and 64-frame callbacks, the partitioned FFT path does all of its
transform work in one callback in eight. That callback does the forward FFT, the
15 partition multiplies, the inverse FFT and the overlap-add. It is what the
published p99 measures (branch at `a09e360`):

| 8192 taps, 64 frames | transform callback (p99) | other seven (derived) |
|---|---:|---:|
| M2 | 1.61% of deadline | ~0.20% |
| Pi 500 | 3.22% | ~0.37% |
| Tinker Board, 1.416 GHz | 21.74% | ~2.95% |

IR-PATH.md recorded spreading the multiplies as "Not done: more code than it is
currently worth". It is worth another look because a callback meets or misses
its deadline on its own cost, not on the average. The case with the least
headroom is a pedal's 32-frame blocks on the Tinker Board. There the same
transform work falls in a deadline half as long, which is about 40% of it
(extrapolated from the 64-frame p99; Step 4 measures it). The model already
averages about 55% of a core on that board.

What this should deliver:

- a smaller worst callback;
- zero latency kept;
- no extra arithmetic;
- output that does not depend on callback size;
- measurements on all three machines before anything is pinned or published.

## Is it worth doing? Yes, down to a known floor

**What can move.** These are the parts whose inputs exist before the callback
that completes the block:

- **14 of the 15 partition multiplies.** Partitions 1..P-1 multiply spectra of
  blocks that arrived earlier. Only partition 0 needs the block that has just
  completed.
- **The overlap-add.** The inverse transform's 1,023 outputs are used one per
  output sample over the next two blocks. Each one can be added when it is
  output instead of all of them in one burst.
  - This also removes the ring's 64-bit `%`, which today runs once per output
    sample and 1,023 times per transform. On 32-bit ARM each one is a library
    call: an ARMv7 listing of the file, built here, shows two `___moddi3` call
    sites, one in `_ProcessFft` and one in `_RunFftBlock`. GCC names it
    `__aeabi_ldivmod`.
  - Evidence of its cost: the forced-FFT 256-tap row runs no transform but
    still reads the ring on every sample. It costs 27% more than direct on the
    Tinker Board and 3% more on the M2. On the board that is about 80 ns a
    sample. At that rate the overlap-add alone would be about 80 µs of the
    ~290 µs transform callback.

**What cannot move.** Three parts stay in the callback that completes the block:

- the forward transform of that block;
- its multiply by partition 0;
- the inverse transform.

The direct head is exactly one block long, which is what keeps latency at zero.
That means the inverse's first output is needed at the very next sample. Only a
longer direct head (more work in every callback) or added latency would make
room. Those three parts are the floor.

**Expected effect.** These are estimates; Step 1 replaces them with
measurements.

- The half-spectrum change halved the multiplies and cut the transform callback
  by 28-29% on the M2 and Pi 500. So the multiplies are still about 40% of it
  there, and about 31% on the Tinker Board.
- Spreading 14/15 of them, plus the overlap-add, should cut the transform
  callback by about 40% on the M2 and Pi 500, and by about half or more on the
  Tinker Board.
- Each of the seven quiet callbacks takes on about two partitions' multiplies.
  So the median goes up, p99 goes down, and core% stays flat or falls a little.
- A falsifiable side prediction: the forced-FFT 256-tap row should close most of
  its gap to direct, because that gap is the ring's modulo.

## Where the work happens

- **AudioDSPTools.** Use a fresh clone of rikkus/AudioDSPTools on a new branch,
  `spread-transform`, cut from `e2dc6bc`. Get Eigen with `git submodule update
  --init`. Do not work in `vendor/adt-partitioned`, which `fetch-vendor.sh`
  owns. No local clone has the branch today, and the AudioDSPTools paths in
  `fetch-vendor.sh`'s `LOCAL_HINTS` no longer exist.
- **What changes.** Only `dsp/PartitionedConvolution.cpp`. `FftState` is
  private to that file, so the header, `ImpulseResponse` and the NAMBench shim
  are untouched.
- **NAMBench.** Two new `ir-study/` tools, a temporary pin for the dry runs, and
  the write-up.

## Step 1: Measure what the transform callback is made of

The two tools follow `make_storage_variants.py` and `spectrum_storage_ab.cpp`.

**`ir-study/make_spread_variants.py <adt tree> OUTDIR <rev>...`**

- Writes each revision's convolver as `PC<n>`, reusing the renaming and the
  `common_enum.h` trick, so that all of them link into one binary.
- Also writes `PCt`, a copy of `e2dc6bc` with a clock read between the phases of
  `_RunFftBlock`.

**`ir-study/spread_ab.cpp`**

- Setup:
  - The variants take turns in an order that rotates each round, as in
    `storage_ab`.
  - Same noise input and synthetic IR for all of them.
  - FFT path forced.
  - Taps: 1024, 2048, 4096 and 8192.
  - Callback size passed as an argument: 32, 48 (does not divide the FFT
    block), 64 or 128.
  - On Linux, pinned to one core.
  - Build flags match the benchmark's `nb_conformance_flags`, including the
    ARMv7 block.
- Reports for each variant:
  - the median cost at each position in the FFT block's cycle of callbacks.
    This is the profile the change is meant to flatten. For 48 frames,
    callbacks are classed by whether they contain a block boundary;
  - the transform callback's median;
  - p99 and max;
  - the mean.
- Output checks, each described in the step it belongs to.
- **Attribution mode (`PCt`).** Reports each phase in µs and as a share of the
  transform callback:
  - forward FFT;
  - multiplies;
  - inverse FFT;
  - overlap-add;
  - the rest.

  It also reports the median quiet callback, and checks that the phases add up
  to the uninstrumented callback.

**Machines:**

- M2.
- Pi 500, built natively.
- Tinker Board, cross-built the same way `storage_ab` was, and run at the
  1.416 GHz cap.

**Checkpoint.** Share the attribution table. If the multiplies and the
overlap-add together are less than a quarter of the transform callback on every
machine, stop and write that up instead of implementing.

## Step 2: Overlap-add per output sample (byte-identical)

### Commit A: the test, before any change

`tools/test_ir_convolution.cpp` gets a check that the FFT path's output does not
depend on callback size.

- Tap counts: 1000, 2049, 4096 and 8192.
- Callback patterns: {1}, {23}, {32}, {48}, {64}, {512}, {1024},
  {1,64,7,512,33}, and one seeded random pattern.
- Every output must be `==` to the {64} output.

It should already pass at `e2dc6bc`. It fixes that property so both later
commits have to keep it. Callbacks of 512 or more do all the multiplies at the
transform. So once Step 3 lands, this same check also compares the spread
schedule with the unspread one.

### Commit B: `dsp/PartitionedConvolution.cpp`

- Replace `output_ring` (4B floats) and `ifft_time` (2B) with two 2B buffers,
  current and previous.
  - Each transform swaps the two buffers and writes the inverse into the new
    current one.
  - It then zeroes the last sample of that buffer. The ring loop never added
    that sample.
- Per output sample, compute `tail = (0.0f + previous[B + input_pos]) +
  current[input_pos]`.
  - These are the ring's own two additions onto +0.0f, in the same order, so the
    output stays byte-identical, down to the sign of zero.
  - With no partitions, keep `dot + 0.0f`, as today.
- `sample_index`, the ring and its `%` all go. `Reset()` zeroes both buffers.

### Verify and measure

- The tests pass.
- `spread_ab`:
  - B matches `e2dc6bc` byte for byte at every tap count and callback size.
  - Measure `e2dc6bc` against B.
- `ir-study/ab_render.cpp` produces byte-identical WAVs from the four real IRs,
  the same check `a09e360` passed.

## Step 3: Spread the partition multiplies

### Commit C

- **The partial sum.** The accumulator now holds the next transform's partial
  sum between callbacks. It is zeroed after each inverse and in `Reset()`.
- **Work units.** The work is W = (P-1) × `num_bins` units.
  - A `pending_done` cursor tracks how far through it we are.
  - The units are ordered by partition (1..P-1), then by bin. So every bin
    still gets partitions 1..P-1 added in ascending order, however the work is
    sliced.
- **One helper.** A single function multiplies and adds a (partition, bin
  range). It uses today's `acc += x * h` and is called from both places below.
- **At a transform:**
  1. forward FFT;
  2. finish any pending units (there are only any if callbacks were irregular);
  3. add partition 0 last;
  4. inverse;
  5. zero the accumulator and reset the cursor.
- **In a callback that ran no transform.** After the sample loop, do
  `ceil(remaining / ceil((B - input_pos) / numFrames))` units. That is the work
  left, divided among the callbacks still to come before the one that runs the
  transform.
  - At 64 frames with B = 512, that is exactly two partitions in each of the
    seven quiet callbacks and none in the eighth.
  - Irregular callback sizes stay correct because of the catch-up at the
    transform.
  - Callbacks of B frames or more behave exactly as today.
- **Integer width.** Keep the scheduling arithmetic in `int`. On ARMv7 a 64-bit
  division is a library call, which is exactly what Step 2 removed.

The output changes at rounding level, because partition 0 is now added last. This
cannot be avoided: its input spectrum does not exist until the transform. When
P ≤ 2, nothing changes.

### Verify and measure

- The callback-size check still passes byte-exact.
- The reference comparisons still pass (-100 dB relative RMS, and at most 1e-5
  of peak on any sample).
- `Process` still allocates nothing after `Reset(maxFrames)`.
- `TestResetClearsState` passes. It dirties the state partway through a block,
  so it catches a partial sum that is not cleared.
- `spread_ab`:
  - Report C against B in dB below signal, plus max|diff|; expect around
    -140 dB.
  - Measure `e2dc6bc`, B and C against each other.
- `ab_render`: report the difference from `e2dc6bc` for the four IRs, in the
  same form as IR-PATH's Listening table.

## Step 4: Full protocol on all three machines, nothing uploaded

- **Point NAMBench at the branch, temporarily.** Add the dev clone to
  `LOCAL_HINTS`, set `ADT_PARTITIONED_SHA` to C (or to B if C was rejected), and
  run `Scripts/fetch-vendor.sh`. Nothing is committed yet.
- **M2 and Pi 500.** Configure and build as in BENCHMARKING.md, then run
  `Scripts/run-benchmark.sh --ir --blocks 64 --build-dir build-benchmark`. The
  Pi runs from its usual rsync'd copy.
- **Tinker Board.** Run `Scripts/a32-deploy.sh -- --ir --blocks 32,64 --max-freq
  1416000`. Run it from the machine that has the ARMv7 cross toolchain; this Mac
  has none.
- **Baselines.**
  - At 64 frames, compare with the published `a09e360` reports in
    `benchmark-results/`. `e2dc6bc` changed no FFT-path number.
  - At 32 frames there is no published baseline, so also run the board at
    `e2dc6bc`.
- **Accuracy.** The parity check must still put the FFT path 136-138 dB below
  upstream.

## Step 5: Decide, then record

**Criteria.**

- Adopt B if it is byte-identical and slower nowhere.
- Adopt C only if all of these hold:
  - callback-size independence is exact;
  - accuracy is unchanged;
  - the 8192-tap p99 is clearly lower on all three machines;
  - no tap count's p99 or core% is worse than the run's spread.

**If adopted:**

- Fast-forward `partitioned-ir` to the adopted commit and push it (after asking).
- Pin it in `fetch-vendor.sh` and `pins.json`.
- Measure and upload all three machines at the pin, as before.
- Regenerate the README and BENCHMARKING.md tables with
  `ir-study/docs_tables.py`.

**Either way:**

- IR-PATH.md gets:
  - the updated Decisions row;
  - the attribution table;
  - the per-position profiles before and after;
  - the results;
  - what is left in the spike.
- `ir-study/README.md` lists the new tools.
- Optionally, add a page in the style of `burstiness.html` showing the profiles.
- Commit this plan to `plans/`.

## Not in this plan (Step 1 sizes them)

- **The multiplies are scalar.** `std::complex<float>` keeps C99 Annex G NaN
  handling, so the arm64 listing has a `__mulsc3` fallback in `_RunFftBlock`,
  and that stops the loop from vectorising. Cheaper multiplies would help
  wherever they end up running.
- **The floor itself.** Two separate experiments would lower it, each with its
  own trade-off:
  - Eigen's default FFT backend here is kissfft; a faster backend would shorten
    both transforms. kissfft's butterflies go through the same `std::complex`
    multiply: the ARMv7 listing has `___mulsc3` calls in `bfly3`, `bfly4` and
    `bfly5`.
  - A 256-sample FFT block at 4096 and 8192 taps would halve the transforms, at
    the cost of twice as many (by then spread) multiplies.

## Verification commands

```bash
# AudioDSPTools dev clone
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target test_ir_convolution
build/tools/test_ir_convolution

# NAMBench, same-machine A/B (header of spread_ab.cpp has the ARMv7 variant)
python3 ir-study/make_spread_variants.py <dev clone> <out> e2dc6bc <B> <C>
c++ -std=c++17 -O3 -I<out> -isystem vendor/eigen-adt ir-study/spread_ab.cpp <out>/PC*.cpp -o spread_ab
./spread_ab [cpu] [rounds] [frames]

# Full protocol, no upload
Scripts/run-benchmark.sh --ir --blocks 64 --build-dir build-benchmark
Scripts/a32-deploy.sh -- --ir --blocks 32,64 --max-freq 1416000
```
