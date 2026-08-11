#!/usr/bin/env python3
"""Convert a NAMBench report into Bencher Metric Format (BMF) JSON.

Reads either producer's report — the Swift `nambench` CLI or the portable
`nam_benchmark` — because the two share key names wherever they overlap, and
emits the BMF that `bencher run --adapter json --file ...` expects.

One measure:

  core_percent  what one NAM instance costs, as a percentage of one CPU core,
               while keeping up with real-time audio.

                   (meanMs / 1000) / audio seconds x 100

Real-time audio is constrained by how much work the CPU can do inside each
callback, so that fraction is the number worth tracking. The whole-file render
time this is derived from is not: the file's length is arbitrary, so a duration
in milliseconds says as much about the test signal as about the engine. Dividing
it out leaves a figure that is invariant to the length of the input and to the
sample rate, and that can be compared across machines and across changes of test
signal.

Of one **core**, deliberately, not of the whole CPU package. process() is
single-threaded, so an instance can never use more than one core; dividing by
the core count would give a smaller number that hides how close the audio thread
is to its deadline, and on a heterogeneous part like an M2 it would pretend four
efficiency cores are interchangeable with four performance cores. It also keeps
the useful inverse honest: 100 / core_percent is the number of instances that fit
on one core, under ideal conditions with no contention, no thermal limit and no
headroom.

What it does not measure: whether a block misses its deadline. That depends on
the worst case, and the protocol deliberately discards slow outliers as
interference. core_percent is a cost and capacity measure, not a real-time-safety
guarantee.

It is also specific to the block size the run used — smaller blocks cost more
per sample — so a run at a size other than the default is warned about below.

A variant that failed its agreement threshold is **left out entirely**, not
reported as zero. The protocol rejected it because the machine was too noisy to
measure, and a zero — or worse, a plausible-looking number — entering the
history would be a fabricated data point that every future comparison is drawn
against.

Several reports can be merged into one upload, which is how a run that measured
both A2 submodels becomes a single Bencher report.

A benchmark name is `<model>/<kernel>` — `a2_standard/a2_planar`, `a2_nano/a2_fast`
— because Bencher has exactly one free-text dimension per series and two things
to say with it. The machine is not in the name: that is the *testbed* dimension,
which is what keeps an M2's history from being averaged with a Pi's. So the four
names, across three testbeds, are twelve independent series, and a threshold on
one of them alerts on that kernel, on that model, on that machine alone.

Both halves are spelled the way the Core code path spells them, underscores and
all, so that a label on a dashboard and a symbol in the source are the same word.

Usage:
    bencher-report.py <report.json> [more.json ...] [--output bmf.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def submodel_name(report: dict[str, Any]) -> str:
    """What to call this submodel in Bencher.

    Named by channel count, and by the names the A2 models are actually known
    by rather than the repository's internal vocabulary. `widest` and
    `narrowest` exist in the code because selection is by max_value rather than
    by position — a deliberate choice, and the wrong words for a dashboard.
    Core PR #313, and everyone discussing these models, says A2 standard and
    A2 nano.

    Underscores, not hyphens, so that the model half of a benchmark name is
    punctuated like the kernel half — `a2_standard/a2_fast`, not the
    `a2-standard/a2_fast` this once emitted, where the two halves of one name
    disagreed about their own spelling and neither matched the source.

    Channel count first because it is unambiguous and present in every report;
    the submodel selector is the fallback, and it is spelled differently by the
    two producers — the portable driver writes a plain string, while Swift's
    `SubmodelSelection` is an enum with an associated value, so Codable renders
    it as `{"widest": {}}` or `{"index": {"_0": 2}}`.
    """
    channels = report.get("model", {}).get("channels")
    if channels == 8:
        return "a2_standard"
    if channels == 3:
        return "a2_nano"

    value = report.get("config", {}).get("submodel")

    if isinstance(value, str):
        return value

    if isinstance(value, dict) and value:
        key = next(iter(value))
        if key == "index":
            inner = value[key]
            if isinstance(inner, dict) and inner:
                return f"index{next(iter(inner.values()))}"
            return "index"
        return key

    if channels:
        return f"{channels}ch"
    return "unknown"


# The block size every published number was measured at, and the one the
# benchmark defaults to.
DEFAULT_BLOCK_SIZE = 64


def convert(report: dict[str, Any], prefix: str) -> tuple[dict[str, Any], list[str]]:
    submodel = submodel_name(report)

    # Block size changes the answer — Core PR #313 measures a2_fast at 418 ms on
    # 64-frame blocks and 470 ms on 32-frame ones — but it is not in the
    # benchmark name, so a run at another size would land silently on top of the
    # 64-frame history and corrupt it. Say so, and point at the flag that keeps
    # them apart.
    block_size = report.get("config", {}).get("blockSize")
    if block_size and block_size != DEFAULT_BLOCK_SIZE and not prefix:
        print(
            f"warning: this run used {block_size}-frame blocks, not "
            f"{DEFAULT_BLOCK_SIZE}. Block size is not part of the benchmark name, so "
            f"these results would be recorded on top of the {DEFAULT_BLOCK_SIZE}-frame "
            f"series as though they were comparable.\n"
            f"  Separate them with:  --prefix 'block{block_size}/'",
            file=sys.stderr,
        )
    bmf: dict[str, Any] = {}
    skipped: list[str] = []

    for result in report.get("results", []):
        variant = result.get("variant", "?")
        name = f"{prefix}{submodel}/{variant}"

        if not result.get("succeeded", False):
            skipped.append(f"{name} ({result.get('failureReason', 'no reason given')})")
            continue

        mean = result.get("meanMs")
        if not mean or mean <= 0:
            skipped.append(f"{name} (no usable mean)")
            continue

        # The bounds are the min and max of the accepted set, so the error bars
        # describe what was actually kept rather than a symmetric guess.
        accepted = result.get("acceptedMs") or []
        low = min(accepted) if accepted else result.get("minMs")
        high = max(accepted) if accepted else result.get("maxMs")

        audio_seconds = report.get("audio", {}).get("durationSeconds")
        if not audio_seconds or audio_seconds <= 0:
            skipped.append(f"{name} (report has no audio duration to normalise by)")
            continue

        def core_percent(milliseconds: float) -> float:
            return (milliseconds / 1000.0) / audio_seconds * 100.0

        measure: dict[str, float] = {"value": core_percent(float(mean))}
        # Faster is a smaller percentage, so the bounds keep their order.
        if low and high and low > 0 and high >= low:
            measure["lower_value"] = core_percent(float(low))
            measure["upper_value"] = core_percent(float(high))

        measures: dict[str, Any] = {"core_percent": measure}

        bmf[name] = measures

    return bmf, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reports",
        type=Path,
        nargs="+",
        help="one or more nambench / nam_benchmark JSON reports, merged into one BMF",
    )
    parser.add_argument("--output", "-o", type=Path, help="where to write BMF (default: stdout)")
    parser.add_argument(
        "--prefix",
        default="",
        help="prepended to every benchmark name, e.g. 'blocksize64/'",
    )
    args = parser.parse_args()

    bmf: dict[str, Any] = {}
    skipped: list[str] = []

    for path in args.reports:
        try:
            report = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            sys.exit(f"error: could not read {path}: {error}")

        converted, missing = convert(report, args.prefix)

        # Two reports of the same submodel in one upload is a mistake worth
        # stopping for: Bencher would take the last one silently, and the run
        # would look complete while half of it had been discarded.
        collisions = sorted(set(converted) & set(bmf))
        if collisions:
            sys.exit(
                f"error: {path} repeats benchmarks already present: "
                f"{', '.join(collisions[:4])}"
                + (" ..." if len(collisions) > 4 else "")
            )

        bmf.update(converted)
        skipped.extend(missing)

    if skipped:
        # stderr, so it is visible in the CI log without contaminating the BMF.
        print(f"omitted {len(skipped)} variant(s) that produced no trustworthy result:",
              file=sys.stderr)
        for entry in skipped:
            print(f"  - {entry}", file=sys.stderr)

    if not bmf:
        # Publishing an empty result set would look like "nothing regressed".
        sys.exit("error: no variant produced a usable result; refusing to report an empty run")

    text = json.dumps(bmf, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
        print(f"wrote {args.output} ({len(bmf)} benchmarks)", file=sys.stderr)
    else:
        sys.stdout.write(text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
