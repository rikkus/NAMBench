#!/usr/bin/env python3
"""Summarise NAMBenchAUHost JSON into the tables AUV3-PATH.md carries.

    Scripts/auv3-summary.py benchmark-results/auv3_*.json

Per machine and submodel: each arm's core%, and per-call wrapper overhead
(host-side block time minus kernel time, same block) as median and p99, in ns
and as a share of the callback period. Then, per arm, a least-squares fit of
median overhead against block size, overhead_ns = a + b * frames, which is the
test of "fixed per call, not per frame".
"""

import json
import sys

ARMS = ["A", "A1", "B", "C", "D"]
LABEL = {
    "A": "A: shim whole-file",
    "A1": "A1: bare per-block",
    "B": "B: AU in host",
    "C": "C: extension in-process",
    "D": "D: extension out-of-process",
}


def fit(points):
    n = len(points)
    if n < 2:
        return None
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * y for x, y in points)
    den = n * sxx - sx * sx
    if den == 0:
        return None
    b = (n * sxy - sx * sy) / den
    a = (sy - b * sx) / n
    return a, b


def main(paths):
    for path in paths:
        with open(path) as f:
            doc = json.load(f)
        env = doc["environment"]
        print(f"## {env['deviceModel']} ({env['cpu']}), OS {env['os']} — `{path}`\n")
        subjects = [s for s in doc["subjects"]]
        failed = [s for s in subjects if not s["ok"]]
        for s in failed:
            print(f"- FAILED {s['arm']}/{s['submodel']}/{s['blockSize']}: {s['failure']}")
        if failed:
            print()

        for submodel in ["nano", "standard"]:
            rows = [s for s in subjects if s["submodel"] == submodel and s["ok"]]
            if not rows:
                continue
            blocks = sorted({s["blockSize"] for s in rows})
            arms = [a for a in ARMS if any(s["arm"] == a for s in rows)]
            get = {(s["arm"], s["blockSize"]): s for s in rows}

            print(f"### {submodel}: core %\n")
            print("| Arm | " + " | ".join(f"{b}" for b in blocks) + " |")
            print("|---|" + "---:|" * len(blocks))
            for a in arms:
                cells = []
                for b in blocks:
                    s = get.get((a, b))
                    cells.append(f"{s['corePercent']:.3f}" if s else "—")
                print(f"| {LABEL[a]} | " + " | ".join(cells) + " |")
            print()

            print(f"### {submodel}: wrapper overhead per call, median / p99 (ns)\n")
            print("| Arm | " + " | ".join(f"{b}" for b in blocks) + " |")
            print("|---|" + "---:|" * len(blocks))
            for a in arms:
                if a == "A":
                    continue
                cells = []
                for b in blocks:
                    s = get.get((a, b))
                    if not s:
                        cells.append("—")
                        continue
                    o = s["overheadNs"]
                    cells.append(f"{o['median']:.0f} / {o['p99']:.0f}")
                print(f"| {LABEL[a]} | " + " | ".join(cells) + " |")
            print()

            print(f"### {submodel}: out-of-process call, decomposed (median ns)\n")
            print("| Block | hop in | wrapper in | kernel | wrapper out | hop out | kernel in A1 | parity |")
            print("|---:|---:|---:|---:|---:|---:|---:|---|")
            for b in blocks:
                s = get.get(("D", b))
                ref = get.get(("A1", b))
                if not s:
                    continue
                parity = "bit-identical" if s["parity"] and s["mismatchedSamples"] == 0 else (
                    f"{s['mismatchedSamples']} differ" if s["parity"] else "—")
                print(
                    f"| {b} | {s['hopInNs']['median']:.0f} | {s['wrapInNs']['median']:.0f} | "
                    f"{s['kernelNs']['median']:.0f} | {s['wrapOutNs']['median']:.0f} | "
                    f"{s['hopOutNs']['median']:.0f} | {ref['kernelNs']['median'] if ref else 0:.0f} | {parity} |"
                )
            print()

            print(f"### {submodel}: overhead_ns = a + b·frames (median overhead)\n")
            print("| Arm | a (ns per call) | b (ns per frame) | at 64 frames, % of period (median / p99) |")
            print("|---|---:|---:|---:|")
            for a in arms:
                if a == "A":
                    continue
                pts = [(b, get[(a, b)]["overheadNs"]["median"]) for b in blocks if (a, b) in get]
                r = fit(pts)
                s64 = get.get((a, 64))
                period = 64 / 48000 * 1e9
                share = (
                    f"{s64['overheadNs']['median'] / period * 100:.2f}% / {s64['overheadNs']['p99'] / period * 100:.2f}%"
                    if s64 else "—")
                if r:
                    print(f"| {LABEL[a]} | {r[0]:.0f} | {r[1]:.2f} | {share} |")
            print()


if __name__ == "__main__":
    main(sys.argv[1:])
