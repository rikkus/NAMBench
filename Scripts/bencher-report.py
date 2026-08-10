#!/usr/bin/env python3
"""Convert a NAMBench report into Bencher Metric Format (BMF) JSON.

Reads either producer's report — the Swift `nambench` CLI or the portable
`nam_benchmark` — because the two share key names wherever they overlap, and
emits the BMF that `bencher run --adapter json --file ...` expects.

Two measures per variant:

  latency     milliseconds per pass over the whole input file. value is the mean
              of the accepted samples, with lower_value and upper_value set to
              the min and max of that same accepted set — so the error bars
              Bencher draws are the real spread of what was kept, not a
              symmetric guess around the mean.

  throughput  the real-time factor: seconds of audio processed per second of
              wall clock. Bigger is better, which is the direction Bencher
              already assumes for this measure.

A variant that failed its agreement threshold is **left out entirely**, not
reported as zero. The protocol rejected it because the machine was too noisy to
measure, and a zero — or worse, a plausible-looking number — entering the
history would be a fabricated data point that every future comparison is drawn
against.

Several reports can be merged into one upload, which is how a run that measured
both A2 submodels becomes a single Bencher report. Names carry the submodel, so
`a2-standard/planar` and `a2-nano/planar` are distinct series that never collide.

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

    Channel count first because it is unambiguous and present in every report;
    the submodel selector is the fallback, and it is spelled differently by the
    two producers — the portable driver writes a plain string, while Swift's
    `SubmodelSelection` is an enum with an associated value, so Codable renders
    it as `{"widest": {}}` or `{"index": {"_0": 2}}`.
    """
    channels = report.get("model", {}).get("channels")
    if channels == 8:
        return "a2-standard"
    if channels == 3:
        return "a2-nano"

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


def convert(report: dict[str, Any], prefix: str) -> tuple[dict[str, Any], list[str]]:
    submodel = submodel_name(report)
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

        latency: dict[str, float] = {"value": float(mean)}

        # Bounds come from the accepted set, so they describe what was actually
        # kept. Fall back to the reported min/max, then to nothing at all.
        accepted = result.get("acceptedMs") or []
        low = min(accepted) if accepted else result.get("minMs")
        high = max(accepted) if accepted else result.get("maxMs")
        if low and high and low > 0 and high >= low:
            latency["lower_value"] = float(low)
            latency["upper_value"] = float(high)

        measures: dict[str, Any] = {"latency": latency}

        rtf = result.get("realTimeFactor")
        if rtf and rtf > 0:
            measures["throughput"] = {"value": float(rtf)}

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
