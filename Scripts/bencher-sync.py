#!/usr/bin/env python3
"""Keep the Bencher project's thresholds and perf plots matching the data.

`bencher run` uploads metrics. It does not, on its own, leave behind a project
anyone can read: a plot has to be pinned before it appears, and a threshold has
to exist before a regression can raise an alert. Both are per-dimension, and the
dimensions grow — a new kernel, a new machine — so doing it by hand means the
newest thing being measured is the one thing nobody is watching.

This script derives both from what is actually in the project, and is safe to
run after every upload.

  Thresholds  one per (branch, testbed, core_percent).

              That is the whole of Bencher's threshold scope; it is deliberately
              *not* per benchmark. One model per machine is evaluated against
              every benchmark on that machine separately, so a t-test installed
              once still alerts on `a2_nano/a2_planar` regressing while
              `a2_standard/a2_fast` holds — per kernel, per model, per testbed,
              which is what was wanted, without twelve models to maintain.

              Installed for every testbed the moment it exists, rather than
              waiting for a run to bring one along, because `--threshold-*` flags
              on `bencher run` have to be repeated identically by every upload
              path or `--thresholds-reset` lets the last one silently redefine
              the model. Owning them here means there is one definition.

  Plots       one per (testbed, model), with every kernel of that model on it.

              A plot is pinned to fixed UUIDs, so it cannot follow new data by
              itself. Grouping this way puts a2_fast and a2_planar on one axis
              at one scale, which is the comparison the project exists to make;
              splitting by machine keeps an M2 and a Pi off the same axis, where
              the faster one would flatten the other against the baseline. A
              model is not split across machines for the same reason, and
              a2_standard is not put beside a2_nano because they differ by
              roughly seven times.

Benchmark names are `<model>/<kernel>`, per Scripts/bencher-report.py. A name
without a `/` is left out of the plots — it is not one of ours, and guessing
which axis it belongs on would be worse than omitting it.

Nothing here deletes anything. A plot this script does not recognise is left
alone, and so is a testbed or benchmark that has been archived.

Usage:
    BENCHER_PROJECT=nambench BENCHER_API_KEY=bencher_user_... \\
        Scripts/bencher-sync.py [--branch main] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn

MEASURE = "core_percent"

# A t-test on the last 64 runs, with nothing alerting until there are ten of
# them: below that the test is arithmetic on nothing, and every early run would
# alert on itself.
#
# Both boundaries, not just the upper one. Slower is the regression anyone
# expects, but a result that is suddenly and impossibly *fast* is the more
# dangerous of the two, because it reads as a win: it usually means the kernel
# stopped doing the work — a model that failed to load, a routing change that
# quietly fell through to a smaller path, a compiler that found the whole loop
# dead. That is a broken measurement, and it should be reported as one rather
# than recorded as an improvement and then defended.
THRESHOLD_MODEL = {
    "test": "t_test",
    "min_sample_size": 10,
    "max_sample_size": 64,
    "lower_boundary": 0.98,
    "upper_boundary": 0.98,
}

# 84 days. Long enough that a plot still has a shape after a quiet fortnight,
# short enough that a rewrite six months ago is not still setting the y-axis.
PLOT_WINDOW_SECONDS = 84 * 24 * 60 * 60

PLOT_STYLE = {
    "x_axis": "date_time",
    "y_axis": "linear",
    # The error bars are the min and max of the accepted set, so they show how
    # noisy the machine was, which is the first thing to check before believing
    # a step in the line.
    "lower_value": True,
    "upper_value": True,
    # And the boundary limits, so a point can be read against the threshold that
    # would have alerted on it.
    "lower_boundary": True,
    "upper_boundary": True,
}


def die(message: str) -> NoReturn:
    sys.exit(f"error: {message}")


def load_dotenv(path: Path) -> str | None:
    """Take BENCHER_* out of a .env, the way Scripts/track-benchmark.sh does.

    Parsed rather than executed, and only BENCHER_*, so a line in a file nobody
    reads any more cannot reach into the environment of anything else this runs.
    Anything already exported wins. Values are never printed.
    """
    try:
        text = path.read_text()
    except OSError:
        return None

    found = None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ")
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or not name.startswith("BENCHER_") or os.environ.get(name):
            continue
        value = value.strip()
        if len(value) > 1 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ[name] = value
        found = str(path)
    return found


def bencher(*args: str) -> Any:
    """Run the Bencher CLI and parse its JSON.

    The API key reaches it through the environment, never argv: process
    arguments are readable by anything else on the machine, and these runners
    are somebody's laptop.
    """
    command = ["bencher", *args]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        die(
            "the Bencher CLI is not on PATH.\n"
            "  curl --proto '=https' --tlsv1.2 -sSfL "
            "https://bencher.dev/download/install-cli.sh | sh"
        )
    except subprocess.CalledProcessError as error:
        die(f"`bencher {' '.join(args[:2])}` failed:\n{error.stderr.strip()}")

    if not completed.stdout.strip():
        return None
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        die(f"`bencher {' '.join(args[:2])}` did not return JSON:\n{completed.stdout[:400]}")


def listing(project: str, dimension: str, *extra: str) -> list[dict[str, Any]]:
    """Every unarchived member of a dimension.

    255 is the CLI's maximum page size and comfortably more than this project
    will ever have; the assertion below is what catches the day that stops being
    true, rather than a plot quietly losing half its series.
    """
    items = bencher(dimension, "list", "--per-page", "255", *extra, project) or []
    if len(items) == 255:
        die(f"{dimension} list hit the 255-item page limit; this script needs paging")
    return items


def split_name(name: str) -> tuple[str, str] | None:
    """`a2_standard/a2_fast` -> (model, kernel).

    Split on the *last* slash so that a prefixed run — `bencher-report.py
    --prefix 'block32/'` — groups under `block32/a2_standard` rather than
    landing on top of the 64-frame series it is deliberately kept apart from.
    """
    model, separator, kernel = name.rpartition("/")
    if not separator or not model or not kernel:
        return None
    return model, kernel


def sync_thresholds(
    project: str, branch: str, testbeds: list[dict[str, Any]], dry_run: bool
) -> None:
    existing = {
        (t["branch"]["slug"], t["testbed"]["slug"]): t
        for t in listing(project, "threshold")
        if t["measure"]["name"] == MEASURE
    }

    for testbed in testbeds:
        slug = testbed["slug"]
        current = existing.get((branch, slug))
        model = (current or {}).get("model") or {}

        if current and all(model.get(k) == v for k, v in THRESHOLD_MODEL.items()):
            print(f"threshold {branch}/{slug}: up to date")
            continue

        flags = [
            "--test", THRESHOLD_MODEL["test"],
            "--min-sample-size", str(THRESHOLD_MODEL["min_sample_size"]),
            "--max-sample-size", str(THRESHOLD_MODEL["max_sample_size"]),
            "--lower-boundary", str(THRESHOLD_MODEL["lower_boundary"]),
            "--upper-boundary", str(THRESHOLD_MODEL["upper_boundary"]),
        ]

        if current:
            # Updating replaces the model rather than the threshold, so the
            # threshold's identity — and the alerts already filed against it —
            # survive a change of parameters.
            verb, action = "updating", ["threshold", "update", project, current["uuid"], *flags]
        else:
            verb, action = "creating", [
                "threshold", "create", project,
                "--branch", branch, "--testbed", slug, "--measure", MEASURE, *flags,
            ]

        print(f"threshold {branch}/{slug}: {verb}")
        if not dry_run:
            bencher(*action)


def coverage(
    project: str,
    branch: dict[str, Any],
    testbeds: list[dict[str, Any]],
    benchmarks: list[dict[str, Any]],
    measure_uuid: str,
) -> set[tuple[str, str]]:
    """Which (testbed, benchmark name) pairs actually have metrics.

    Without this, every testbed would get a plot for every model, including the
    ones it has never measured — an empty chart that looks like a machine whose
    results stopped rather than one that was never asked.

    One query for the whole matrix: `bencher perf` returns a result per
    combination, and the ones that were never measured come back with no
    metrics.
    """
    if not testbeds or not benchmarks:
        return set()

    # Every dimension here is by UUID: `bencher perf` is the one subcommand that
    # will not take a slug.
    args = ["perf", project, "--branches", branch["uuid"], "--measures", measure_uuid]
    for testbed in testbeds:
        args += ["--testbeds", testbed["uuid"]]
    for benchmark in benchmarks:
        args += ["--benchmarks", benchmark["uuid"]]

    response = bencher(*args) or {}
    return {
        (result["testbed"]["slug"], result["benchmark"]["name"])
        for result in response.get("results", [])
        if result.get("metrics")
    }


def sync_plots(
    project: str,
    branch: dict[str, Any],
    testbeds: list[dict[str, Any]],
    benchmarks: list[dict[str, Any]],
    measure_uuid: str,
    dry_run: bool,
) -> None:
    measured = coverage(project, branch, testbeds, benchmarks, measure_uuid)

    # (testbed slug, model) -> the benchmarks to draw on it, in kernel order so
    # the legend does not reshuffle between runs.
    wanted: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for testbed in testbeds:
        for benchmark in benchmarks:
            parts = split_name(benchmark["name"])
            if not parts or (testbed["slug"], benchmark["name"]) not in measured:
                continue
            wanted.setdefault((testbed["slug"], parts[0]), []).append(benchmark)

    for series in wanted.values():
        series.sort(key=lambda b: b["name"])

    # Sorted by index, because the index is the one property the API does not
    # report back and position is the only way to see it. Without that, adding a
    # machine whose name sorts before an existing one — an `m1-air` arriving
    # after `m2-air` — would give the new plots indices already in use, and
    # every later run would call the pair up to date while the dashboard stayed
    # in whatever order the collision happened to resolve to.
    plots = listing(project, "plot", "--sort", "index", "--direction", "asc")
    existing = {plot["title"]: plot for plot in plots if plot.get("title")}

    # Where the plots this script manages sit relative to each other. Plots it
    # does not manage are left out rather than counted, so somebody's own chart
    # pinned in the middle does not make every managed plot look misplaced and
    # get rewritten on every run.
    titles = {f"{testbed} : {model}" for testbed, model in wanted}
    placed = [plot["title"] for plot in plots if plot.get("title") in titles]

    # Bencher indexes plots 0..64. Reaching that would mean roughly thirty
    # machines, but a run that silently stopped charting the newest one is the
    # exact failure this script exists to prevent, so say so instead.
    if len(wanted) > 64:
        die(f"{len(wanted)} testbed/model pairs, and Bencher pins at most 64 plots")

    for index, (testbed_slug, model) in enumerate(sorted(wanted)):
        series = wanted[(testbed_slug, model)]
        # Testbed first, so the list sorts into one block per machine — which is
        # how anyone reads it, having usually come to look at one machine.
        title = f"{testbed_slug} : {model}"
        if len(title) > 64:
            print(f"skipping plot {title!r}: Bencher titles are limited to 64 characters")
            continue

        testbed_uuid = next(t["uuid"] for t in testbeds if t["slug"] == testbed_slug)
        desired = {
            "title": title,
            "index": index,
            "window": PLOT_WINDOW_SECONDS,
            "branches": [branch["uuid"]],
            "testbeds": [testbed_uuid],
            "measures": [measure_uuid],
            "benchmarks": [b["uuid"] for b in series],
            **PLOT_STYLE,
        }

        current = existing.get(title)
        if current and index < len(placed) and placed[index] == title and all(
            sorted(current.get(key, [])) == sorted(value) if isinstance(value, list)
            else current.get(key) == value
            for key, value in desired.items()
            if key != "index"  # not reported by the API; checked by position above
        ):
            print(f"plot {title!r}: up to date")
            continue

        flags: list[str] = [
            "--title", title,
            "--index", str(index),
            "--window", str(PLOT_WINDOW_SECONDS),
            "--x-axis", PLOT_STYLE["x_axis"],
            "--y-axis", PLOT_STYLE["y_axis"],
        ]
        for key in ("lower_value", "upper_value", "lower_boundary", "upper_boundary"):
            # `create` takes these as bare switches, `update` as explicit
            # booleans, because update has to be able to turn one back off.
            flag = "--" + key.replace("_", "-")
            flags += [flag, "true"] if current else [flag]
        flags += ["--branches", branch["uuid"], "--testbeds", testbed_uuid, "--measures", measure_uuid]
        for benchmark in series:
            flags += ["--benchmarks", benchmark["uuid"]]

        kernels = ", ".join(split_name(b["name"])[1] for b in series)  # type: ignore[index]
        if current:
            print(f"plot {title!r}: updating ({kernels})")
            action = ["plot", "update", project, current["uuid"], *flags]
        else:
            print(f"plot {title!r}: creating ({kernels})")
            action = ["plot", "create", project, *flags]

        if not dry_run:
            bencher(*action)

    if not wanted:
        print("no plots: no benchmark has metrics on any testbed yet")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=os.environ.get("BENCHER_PROJECT", ""))
    parser.add_argument(
        "--branch",
        default="main",
        help="the branch to plot and to install thresholds on (default: main)",
    )
    parser.add_argument("--dry-run", action="store_true", help="say what would change, change nothing")
    parser.add_argument("--skip-thresholds", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    # The working directory, then the checkout — the same two places
    # Scripts/track-benchmark.sh looks, so running this by hand after a run
    # needs no more setup than the run did.
    for candidate in (Path.cwd(), Path(__file__).resolve().parent.parent):
        load_dotenv(candidate / ".env")

    if not args.project:
        args.project = os.environ.get("BENCHER_PROJECT", "")

    if not args.project:
        die("no Bencher project.\n  Pass --project <slug>, or: export BENCHER_PROJECT=<slug>")

    # Same translation Scripts/track-benchmark.sh does, and for the same reason:
    # the CLI reserves --token for JWTs and rejects an API key handed to it, and
    # it reads BENCHER_API_TOKEN itself, so a stale one still exported would
    # sink the call whatever this script passes.
    if not os.environ.get("BENCHER_API_KEY") and os.environ.get("BENCHER_API_TOKEN"):
        os.environ["BENCHER_API_KEY"] = os.environ["BENCHER_API_TOKEN"]
    os.environ.pop("BENCHER_API_TOKEN", None)

    if not os.environ.get("BENCHER_API_KEY"):
        die("no API key.\n  export BENCHER_API_KEY=\"$(op read 'op://...')\"")

    testbeds = listing(args.project, "testbed")
    if not testbeds:
        die(f"{args.project} has no testbeds; upload a run before syncing")

    branches = listing(args.project, "branch")
    branch = next((b for b in branches if b["slug"] == args.branch), None)
    if branch is None:
        die(f"{args.project} has no branch {args.branch!r}")

    if not args.skip_thresholds:
        sync_thresholds(args.project, args.branch, testbeds, args.dry_run)

    if not args.skip_plots:
        measures = listing(args.project, "measure")
        measure = next((m for m in measures if m["name"] == MEASURE), None)
        if measure is None:
            die(f"{args.project} has no {MEASURE} measure; upload a run before syncing plots")
        sync_plots(
            args.project, branch, testbeds, listing(args.project, "benchmark"),
            measure["uuid"], args.dry_run,
        )

    if args.dry_run:
        print("dry run: nothing was changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
