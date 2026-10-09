#!/usr/bin/env python3
"""Summarise NAMBenchAUHost --contention JSON into the tables CONTENTION.md carries.

    Scripts/contention-summary.py benchmark-results/contention_*.json

Per machine, submodel and block size: the timed instance's per-block median
and p99 against running alone, and its core %, for each scenario (serial N
instances on one thread, parallel N threads, a streaming neighbour of a given
size), plus the memory one instance holds and whether its output stayed
bit-identical.
"""

import json
import sys


def label(s):
    if s["scenario"] == "solo":
        return "solo"
    if s["scenario"] == "thrash":
        kib = s["thrashKiB"]
        return f"thrash {kib // 1024} MiB" if kib >= 1024 else f"thrash {kib} KiB"
    return f"{s['scenario']} ×{s['instances']}"


def main(paths):
    for path in paths:
        with open(path) as f:
            doc = json.load(f)
        env = doc["environment"]
        rows = doc.get("contention", [])
        print(f"## {env['deviceModel']} ({env['cpu']}), OS {env['os']} — `{path}`\n")
        for s in rows:
            if not s["ok"]:
                print(f"- FAILED {label(s)}/{s['submodel']}/{s['blockSize']}: {s['failure']}")
            elif s["parity"] and s["mismatchedSamples"]:
                print(f"- OUTPUT DIFFERS {label(s)}/{s['submodel']}/{s['blockSize']}: "
                      f"{s['mismatchedSamples']} samples")
        for submodel in ["nano", "standard"]:
            for block in sorted({s["blockSize"] for s in rows}):
                sub = [s for s in rows if s["submodel"] == submodel and s["blockSize"] == block and s["ok"]]
                if not sub:
                    continue
                kib = sub[0]["instanceBytes"] / 1024
                print(f"### {submodel}, {block} frames (one instance holds {kib:.0f} KiB)\n")
                print("| Scenario | block median (µs) | × solo | block p99 (µs) | p99 × solo p99 | core % | spread |")
                print("|---|---:|---:|---:|---:|---:|---:|")
                solo = next((s for s in sub if s["scenario"] == "solo"), None)
                for s in sub:
                    p99x = s["blockP99Ns"] / solo["blockP99Ns"] if solo and solo["blockP99Ns"] else 0
                    print(f"| {label(s)} | {s['blockMedianNs'] / 1e3:.2f} | {s['slowdownMedian']:.2f} | "
                          f"{s['blockP99Ns'] / 1e3:.2f} | {p99x:.2f} | {s['corePercent']:.3f} | "
                          f"{s['spread'] * 100:.1f}% |")
                print()


if __name__ == "__main__":
    main(sys.argv[1:])
