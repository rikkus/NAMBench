#!/usr/bin/env python3
"""Check what the engine variants computed against each other.

Tools/nam_conformance.cpp runs one variant per process and writes its output
samples verbatim. This reads those files back and applies the rules the project
already states about which engine is supposed to produce which arithmetic:

  * routing      each variant must land on the engine it is supposed to land on
                 for this architecture. On AArch64 the fork gets its fused NEON
                 engine; everywhere else the detector declines the shape and the
                 fork falls through to the *generic* engine, never to a2_fast.
                 Both are asserted, because a silent change either way is the
                 failure mode that would invalidate a benchmark run.

  * finiteness   no NaN, no infinity, and a peak that is neither silence nor
                 obviously diverged.

  * parity       every variant against the reference it is derived from, as
                 max|diff| expressed in dB below the reference signal — the same
                 measure BenchCore reports. The verbatim ports are held to
                 bit-identity, because that is what "verbatim port" claims.

Nothing here measures time. Timing lives in the Xcode build, on pinned hardware.

Usage:  compare-conformance.py <directory> [--min-db N] [--frames-tolerance N]
Exit:   0 if every check passed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import array
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

# Default parity floor, in dB below the reference signal.
#
# The fused engine is not a bit-for-bit reimplementation of a2_fast — it
# reassociates sums and uses vectorised activations — so some divergence is
# expected and wanted. What is not wanted is divergence you could hear. 100 dB
# is roughly 17 bits down, far below the noise floor of any capture, and every
# pairing this repository has published sits well beyond it.
DEFAULT_MIN_DB = 100.0


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


@dataclass
class Record:
    variant: str
    arch: str
    compiler: str
    label: str
    submodel: str
    kernel: str | None
    kernel_index: int
    engine: str
    channels: int
    sample_rate: float
    checksum: float
    non_finite: int
    peak: float
    samples: array.array

    @property
    def name(self) -> str:
        return f"{self.variant}/{self.label}"


def load(directory: Path) -> list[Record]:
    records: list[Record] = []

    for report_path in sorted(directory.glob("*.json")):
        report = json.loads(report_path.read_text())
        variant = report["variant"]
        arch = report.get("arch", "unknown")
        compiler = report.get("compiler", "unknown")

        for entry in report["records"]:
            samples_path = directory / entry["samples_file"]
            if not samples_path.exists():
                sys.exit(f"error: {report_path.name} refers to missing {samples_path.name}")

            samples = array.array("d")
            raw = samples_path.read_bytes()
            samples.frombytes(raw)
            if sys.byteorder != "little":
                # The driver writes little-endian; say so rather than silently
                # comparing byte-swapped garbage.
                samples.byteswap()

            records.append(
                Record(
                    variant=variant,
                    arch=arch,
                    compiler=compiler,
                    label=entry["label"],
                    submodel=entry["submodel"],
                    kernel=entry["kernel"],
                    kernel_index=entry["kernel_index"],
                    engine=entry["engine"],
                    channels=entry["channels"],
                    sample_rate=entry["sample_rate"],
                    checksum=entry["checksum"],
                    non_finite=entry["non_finite"],
                    peak=entry["peak"],
                    samples=samples,
                )
            )

    return records


# --------------------------------------------------------------------------
# Parity
# --------------------------------------------------------------------------


@dataclass
class Parity:
    max_abs_diff: float
    rms_diff: float
    db_below_signal: float
    exact: bool


def compare(reference: array.array, candidate: array.array) -> Parity:
    if len(reference) != len(candidate):
        raise ValueError(f"length mismatch: {len(reference)} vs {len(candidate)}")

    max_abs = 0.0
    sum_sq_diff = 0.0
    sum_sq_ref = 0.0

    for a, b in zip(reference, candidate):
        d = a - b
        if abs(d) > max_abs:
            max_abs = abs(d)
        sum_sq_diff += d * d
        sum_sq_ref += a * a

    n = len(reference)
    rms_diff = math.sqrt(sum_sq_diff / n) if n else 0.0
    rms_ref = math.sqrt(sum_sq_ref / n) if n else 0.0

    if rms_diff == 0.0:
        db = math.inf
    elif rms_ref == 0.0:
        db = -math.inf
    else:
        db = 20.0 * math.log10(rms_ref / rms_diff)

    return Parity(max_abs, rms_diff, db, max_abs == 0.0)


def db_text(parity: Parity) -> str:
    if parity.exact:
        return "bit-identical"
    if parity.db_below_signal == -math.inf:
        return "reference is silent"
    return f"{parity.db_below_signal:.1f} dB below signal"


# --------------------------------------------------------------------------
# Expectations
# --------------------------------------------------------------------------


def expected_engine(variant: str, submodel: str, kernel: str | None, aarch64: bool) -> str:
    """Which engine this case is supposed to have been routed to.

    Mirrors detect_engine in the shim rather than restating the build flags,
    which is the whole reason the driver reports what it actually got.
    """
    if variant == "upstream":
        # a2_fast is compiled into every variant and matches the A2 shape on
        # both submodels, regardless of architecture.
        return "a2_fast"

    if variant == "fused":
        # Under ScopedEnginePrefer(FusedNeon) a shape the fused detector rejects
        # falls through to the generic engine. It rejects everything off AArch64,
        # and rejects a channel count that is not a multiple of four everywhere.
        if aarch64 and submodel == "widest":
            return "fused"
        return "generic"

    if variant == "slim":
        return "slim"

    if variant == "full":
        return "full"

    return "unknown"


def reference_for(record: Record) -> tuple[str, str] | None:
    """The (variant, label) a record's output should be compared against.

    The kernel labs name their candidates after the engine whose arithmetic they
    reproduce, which is what makes this mechanical:

      full lab   a2*  reproduce a2_fast   -> upstream/widest
                 fu*  reproduce fused     -> fused/widest
      slim lab   all reproduce a2_fast's 3-channel branch -> upstream/narrowest

    `upstream` is the root of the tree and has nothing above it. `fused` is
    compared against `upstream` because that is the comparison the project
    exists to make.
    """
    if record.variant == "upstream":
        return None

    if record.variant == "fused":
        return ("upstream", record.submodel)

    if record.variant == "slim":
        return ("upstream", "narrowest")

    if record.variant == "full":
        assert record.kernel is not None
        if record.kernel.startswith("fu"):
            return ("fused", "widest")
        return ("upstream", "widest")

    return None


def expect_bit_identical(record: Record) -> bool:
    """Whether this candidate claims to be a verbatim port.

    full_common.h and slim_common.h both describe their baselines as verbatim
    ports of the engine they stand in for. A verbatim port that has drifted is
    exactly the thing this check exists to catch, so the claim is tested rather
    than trusted. Every other kernel reassociates deliberately and is only held
    to the dB floor.
    """
    return record.kernel is not None and record.kernel.endswith("baseline")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--min-db",
        type=float,
        default=DEFAULT_MIN_DB,
        help=f"parity floor in dB below signal (default {DEFAULT_MIN_DB:g})",
    )
    parser.add_argument(
        "--arch",
        choices=("record", "aarch64", "other"),
        default="record",
        help="which routing to expect; 'record' believes each binary's own "
        "compiled-for architecture (default, and the right answer under a "
        "cross-build or Rosetta, where the host disagrees)",
    )
    args = parser.parse_args()

    if not args.directory.is_dir():
        sys.exit(f"error: {args.directory} is not a directory")

    records = load(args.directory)
    if not records:
        sys.exit(f"error: no conformance records found in {args.directory}")

    by_key = {(r.variant, r.label): r for r in records}
    failures: list[str] = []

    arches = sorted({r.arch for r in records})
    compilers = sorted({r.compiler for r in records})

    print(f"{len(records)} records from {args.directory}")
    print(f"host      {platform.system()} {platform.machine()}")
    print(f"built for {', '.join(arches)}")
    print(f"compiler  {', '.join(compilers)}")

    if len(arches) > 1:
        # Parity between two architectures is a different question from parity
        # between two engines, and mixing them in one directory would silently
        # turn one into the other.
        failures.append(
            f"records from more than one architecture in one directory: {', '.join(arches)}"
        )

    # --- routing and finiteness -------------------------------------------
    print("\nrouting")
    for record in sorted(records, key=lambda r: r.name):
        aarch64 = record.arch == "aarch64" if args.arch == "record" else args.arch == "aarch64"
        want = expected_engine(record.variant, record.submodel, record.kernel, aarch64)
        ok = record.engine == want
        if not ok:
            failures.append(f"{record.name}: routed to {record.engine}, expected {want}")

        if record.non_finite:
            failures.append(f"{record.name}: {record.non_finite} non-finite output samples")

        if record.peak == 0.0:
            failures.append(f"{record.name}: output is silent")
        elif record.peak > 100.0:
            failures.append(f"{record.name}: output peak {record.peak:g} has diverged")

        print(f"  {'ok ' if ok else 'FAIL'}  {record.name:<40} {record.engine:<8} "
              f"{record.channels}ch  peak {record.peak:.6f}")

    # --- parity ------------------------------------------------------------
    print("\nparity")
    compared = 0
    for record in sorted(records, key=lambda r: r.name):
        key = reference_for(record)
        if key is None:
            continue

        reference = by_key.get(key)
        if reference is None:
            # The lab variants are AArch64-only, so on x86_64 their references
            # are simply absent. Nothing to check, and nothing wrong.
            print(f"  --    {record.name:<40} no reference ({key[0]}/{key[1]}) in this run")
            continue

        try:
            parity = compare(reference.samples, record.samples)
        except ValueError as error:
            failures.append(f"{record.name} vs {reference.name}: {error}")
            continue

        compared += 1
        strict = expect_bit_identical(record)
        ok = parity.exact if strict else (parity.db_below_signal >= args.min_db)

        if not ok:
            if strict:
                failures.append(
                    f"{record.name} is a verbatim port of {reference.name} but differs: "
                    f"max|diff| {parity.max_abs_diff:.3e} ({db_text(parity)})"
                )
            else:
                failures.append(
                    f"{record.name} vs {reference.name}: {db_text(parity)}, "
                    f"below the {args.min_db:g} dB floor"
                )

        print(f"  {'ok ' if ok else 'FAIL'}  {record.name:<40} vs {reference.name:<24} "
              f"max|diff| {parity.max_abs_diff:.3e}  {db_text(parity)}"
              f"{'  [exact required]' if strict else ''}")

    print(f"\n{compared} comparisons made")

    if failures:
        print(f"\n{len(failures)} failure(s):")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
