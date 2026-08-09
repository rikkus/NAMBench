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

Usage:
    bencher-report.py <report.json> [--output bmf.json] [--prefix <text>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def submodel_name(report: dict[str, Any]) -> str:
    """Which submodel the run measured.

    The two producers spell this differently. The portable driver writes a plain
    string. Swift's `SubmodelSelection` is an enum with an associated value, so
    Codable renders it as an object — `{"widest": {}}`, or `{"index": {"_0": 2}}`.
    Both are handled rather than assumed, because guessing wrong would silently
    merge the 3-channel and 8-channel histories into one series.
    """
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

    # Fall back to the channel count, which distinguishes the two A2 submodels
    # just as well and is present in every report.
    channels = report.get("model", {}).get("channels")
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
    parser.add_argument("report", type=Path, help="a nambench or nam_benchmark JSON report")
    parser.add_argument("--output", "-o", type=Path, help="where to write BMF (default: stdout)")
    parser.add_argument(
        "--prefix",
        default="",
        help="prepended to every benchmark name, e.g. 'blocksize64/'",
    )
    args = parser.parse_args()

    try:
        report = json.loads(args.report.read_text())
    except (OSError, json.JSONDecodeError) as error:
        sys.exit(f"error: could not read {args.report}: {error}")

    bmf, skipped = convert(report, args.prefix)

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
