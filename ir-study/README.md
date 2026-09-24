# Impulse-response study

The tools, pages and reduced data behind [IR-PATH.md](../IR-PATH.md). None of
this is part of the benchmark itself: `nam_ir_benchmark` and the scripts in
`Scripts/` produce every published number. These are the side investigations
around them, kept so each can be rerun.

Commands are run from the repository root, after `Scripts/fetch-vendor.sh` and a
`cmake --build build-benchmark --target nam_ir_benchmark`.

| File | What it is |
|---|---|
| `burstiness.html` | Per-callback CPU traces: where the FFT path spends its time, and what does and does not change it |
| `half-spectrum.html` | Why bins 513-1023 of a real signal's FFT mirror bins 1-511, and what skipping them saved |
| `ir_trace.cpp` | Times every callback through the benchmark's shim libraries (upstream against the FFT path) and through `PartitionedConvolution` directly with chosen IRs |
| `trace_reduce.py` | Reduces `ir_trace`'s CSV to the JSON `burstiness.html` charts |
| `data/m2-trace-summary.json` | That JSON for the M2, as embedded in the page |
| `ab_render.cpp` | Renders a WAV through `dsp::ImpulseResponse` from an IR file, as the plugin does; built once per AudioDSPTools tree |
| `ab_mix.py` | Level-matched A/B WAVs and a boosted difference from two renders |
| `data/ab-samples-README.txt` | The note that went with the A/B samples |
| `make_storage_variants.py` | Generates the three convolvers `spectrum_storage_ab.cpp` compares |
| `spectrum_storage_ab.cpp` | Separates the half-spectrum change's storage saving from its arithmetic saving |
| `make_spread_variants.py` | Generates the convolvers `spread_ab.cpp` compares, one per AudioDSPTools revision, plus `e2dc6bc` with the transform's phases timed |
| `spread_ab.cpp` | Per-position callback profile, transform p99 and output comparison for each revision, and the transform callback's attribution; build commands in its header |
| `heap_footprint.cpp` | Heap held by one `ImpulseResponse`, per version and path (macOS) |
| `ir_survey.py` | Inventory of an IR folder, and how much energy a shorter cut would discard |
| `docs_tables.py` | Prints the README's IR tables from `nam_ir_benchmark` reports |

Each source file's header has its build and run commands.

## The pages

Both are self-contained HTML with their data embedded; open them in a browser.
They were written for this study and updated once the half-spectrum change
landed, so their prose describes the state before and after it.

## What is not here

- **The A/B WAVs.** They are renders of commercial cabinet IRs, and about
  26 MB. `ab_render` and `ab_mix.py` regenerate them from any IR files.
- **The raw trace CSV.** Roughly 770,000 rows; `ir_trace` regenerates it in
  about a minute on an M2, and the page needs only the reduced JSON.
- **The IR library.** `ir_survey.py` takes a path to your own.
