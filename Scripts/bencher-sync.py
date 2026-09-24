#!/usr/bin/env python3
"""Keep the Bencher project's thresholds and perf plots matching the data.

`bencher run` uploads metrics. It does not, on its own, leave behind a project
anyone can read: a plot has to be pinned before it appears, and a threshold has
to exist before a regression can raise an alert. Both are per-dimension, and the
dimensions grow — a new kernel, a new machine — so doing it by hand means the
newest thing being measured is the one thing nobody is watching.

This script derives both from what is actually in the project, and is safe to
run after every upload.

  Thresholds  one per (branch, testbed, measure).

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

  Plots       one per (measure, testbed, model), with every kernel of that model
              on it.

              A plot is pinned to fixed UUIDs, so it cannot follow new data by
              itself. Grouping this way puts a2_fast and a2_planar on one axis
              at one scale, which is the comparison the project exists to make;
              splitting by machine keeps an M2 and a Pi off the same axis, where
              the faster one would flatten the other against the baseline. A
              model is not split across machines for the same reason, and
              a2_standard is not put beside a2_nano because they differ by
              roughly seven times.

There are two measures. `core_percent` is what a variant costs on average;
`block_p99_percent` is its 99th-percentile block as a fraction of that block's
deadline, which only the impulse-response benchmark produces. They get separate
plots because they are separate questions — one is capacity, the other is
whether the audio thread makes it — and separate thresholds because a p99 is
noisier than a mean and a boundary tight enough for one would cry wolf on the
other.

Benchmark names are `<model>/<kernel>`, per Scripts/bencher-report.py. A name
without a `/` is left out of the plots — it is not one of ours, and guessing
which axis it belongs on would be worse than omitting it.

Impulse responses live in a project of their own and get `--ir`: one plot per
testbed, titled by architecture, with the mean and the p99 of upstream and the
FFT path at 8192 taps on it. Split per IR length and per measure like the
WaveNet plots they were unreadable. For the same reason the ordinary sync
leaves any `ir_*` benchmark out of its plots, which keeps a project that still
holds old impulse-response data from growing those plots back.

Nothing here deletes anything. A plot this script does not recognise is left
alone, and so is a testbed or benchmark that has been archived. A plot is
recognised by its measure, testbed and model, not its title, so renaming one on
the dashboard is safe.

Usage:
    BENCHER_PROJECT=nambench BENCHER_API_KEY=bencher_user_... \\
        Scripts/bencher-sync.py [--branch main] [--dry-run]
    BENCHER_API_KEY=... Scripts/bencher-sync.py --project nam-ir --ir
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn

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

# The same test, one boundary wider. A p99 is a tail statistic drawn from
# millions of blocks: it moves with a scheduling decision in a way a mean over
# the same passes does not, and the protocol's window selection — which throws
# out a noisy *pass* — cannot throw out a noisy block within an accepted one. A
# boundary tight enough for the mean would alert on the machine rather than on
# the code, and a threshold that cries wolf gets muted, which is worse than not
# having it.
P99_THRESHOLD_MODEL = {**THRESHOLD_MODEL, "lower_boundary": 0.99, "upper_boundary": 0.99}

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
    # Not the boundary limits. Bencher draws them as warning triangles at every
    # point, which reads as a problem when it is only where the threshold sits,
    # and the alert itself is what matters.
    "lower_boundary": False,
    "upper_boundary": False,
}

# No error bars on the p99 plot. A percentile has no interval around it to draw:
# the median below it and the maximum above it are different statistics, not
# uncertainty in this one, and bencher-report.py deliberately uploads no bounds
# for it. Asking for bars there would draw nothing, or worse, draw something.
P99_PLOT_STYLE = {**PLOT_STYLE, "lower_value": False, "upper_value": False}

# Every measure this script manages, in the order their plots are pinned.
#
# core_percent first, and the order fixed here rather than sorted, because plot
# indices are positions on the dashboard: sorting by name would put
# block_p99_percent ahead of it, shift every existing plot by one, and rewrite
# the whole dashboard the first time this ran.
MEASURES: dict[str, dict[str, Any]] = {
    "core_percent": {"threshold": THRESHOLD_MODEL, "style": PLOT_STYLE, "suffix": ""},
    "block_p99_percent": {
        "threshold": P99_THRESHOLD_MODEL,
        "style": P99_PLOT_STYLE,
        "suffix": " p99 block",
    },
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
    project: str,
    branch: str,
    testbeds: list[dict[str, Any]],
    measures: dict[str, str],
    dry_run: bool,
) -> None:
    """One threshold per (branch, testbed, measure), for each measure present.

    Measures the project does not have yet are skipped rather than created: a
    measure comes into existence when a run uploads one, and installing a
    threshold on a measure with no data would watch a series that does not
    exist.
    """
    listed = listing(project, "threshold")

    for measure_name, spec in MEASURES.items():
        if measure_name not in measures:
            print(f"threshold {measure_name}: no such measure in {project} yet; skipping")
            continue

        wanted_model = spec["threshold"]
        existing = {
            (t["branch"]["slug"], t["testbed"]["slug"]): t
            for t in listed
            if t["measure"]["name"] == measure_name
        }

        for testbed in testbeds:
            slug = testbed["slug"]
            label = f"threshold {branch}/{slug}/{measure_name}"
            current = existing.get((branch, slug))
            model = (current or {}).get("model") or {}

            if current and all(model.get(k) == v for k, v in wanted_model.items()):
                print(f"{label}: up to date")
                continue

            flags = [
                "--test", wanted_model["test"],
                "--min-sample-size", str(wanted_model["min_sample_size"]),
                "--max-sample-size", str(wanted_model["max_sample_size"]),
                "--lower-boundary", str(wanted_model["lower_boundary"]),
                "--upper-boundary", str(wanted_model["upper_boundary"]),
            ]

            if current:
                # Updating replaces the model rather than the threshold, so the
                # threshold's identity — and the alerts already filed against it
                # — survive a change of parameters.
                verb, action = "updating", ["threshold", "update", project, current["uuid"], *flags]
            else:
                verb, action = "creating", [
                    "threshold", "create", project,
                    "--branch", branch, "--testbed", slug, "--measure", measure_name, *flags,
                ]

            print(f"{label}: {verb}")
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
    measures: dict[str, str],
    dry_run: bool,
) -> None:
    """Pin one plot per (measure, testbed, model).

    All measures are handled in one pass rather than one call each, because the
    plot index is a single 0..64 space shared by every plot in the project.
    Assigning indices per measure would have each pass renumber the other's.
    """
    uuid_of_measure = {name: measures[name] for name in MEASURES if name in measures}
    if not uuid_of_measure:
        print("no plots: the project has none of the measures this script manages")
        return
    measure_of_uuid = {uuid: name for name, uuid in uuid_of_measure.items()}

    # (measure, testbed slug, model) -> the benchmarks to draw on it, in kernel
    # order so the legend does not reshuffle between runs.
    wanted: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for measure_name, measure_uuid in uuid_of_measure.items():
        measured = coverage(project, branch, testbeds, benchmarks, measure_uuid)
        for testbed in testbeds:
            for benchmark in benchmarks:
                parts = split_name(benchmark["name"])
                if not parts or parts[0].startswith("ir_"):
                    continue
                if (testbed["slug"], benchmark["name"]) not in measured:
                    continue
                wanted.setdefault((measure_name, testbed["slug"], parts[0]), []).append(benchmark)

    for series in wanted.values():
        series.sort(key=lambda b: b["name"])

    # Sorted by index, because the index is the one property the API does not
    # report back and position is the only way to see it. Without that, adding a
    # machine whose name sorts before an existing one — an `m1-air` arriving
    # after `m2-air` — would give the new plots indices already in use, and
    # every later run would call the pair up to date while the dashboard stayed
    # in whatever order the collision happened to resolve to.
    plots = listing(project, "plot", "--sort", "index", "--direction", "asc")

    # A plot is recognised by what it draws, not by its title, so one renamed on
    # the dashboard keeps its name and is still updated in place rather than
    # joined by a duplicate under the name this script would have picked.
    slug_of = {t["uuid"]: t["slug"] for t in testbeds}
    name_of = {b["uuid"]: b["name"] for b in benchmarks}

    def key_of(plot: dict[str, Any]) -> tuple[str, str, str] | None:
        """(measure, testbed slug, model) for a plot of one testbed and one
        model's kernels in one managed measure, or None for any other plot."""
        drawn = plot.get("measures") or []
        if len(drawn) != 1 or drawn[0] not in measure_of_uuid:
            return None
        if len(plot.get("testbeds", [])) != 1:
            return None
        slug = slug_of.get(plot["testbeds"][0])
        models = {
            (split_name(name_of[uuid]) or (None,))[0] if uuid in name_of else None
            for uuid in plot.get("benchmarks", [])
        }
        if slug is None or len(models) != 1 or None in models:
            return None
        return (measure_of_uuid[drawn[0]], slug, models.pop())

    existing: dict[tuple[str, str, str], dict[str, Any]] = {}
    for plot in plots:
        key = key_of(plot)
        if key is not None:
            existing.setdefault(key, plot)

    # Where the plots this script manages sit relative to each other. Plots it
    # does not manage are left out rather than counted, so somebody's own chart
    # pinned in the middle does not make every managed plot look misplaced and
    # get rewritten on every run.
    managed = {plot["uuid"] for plot in existing.values()}
    placed = [key_of(plot) for plot in plots if plot["uuid"] in managed]

    # Bencher indexes plots 0..64. Reaching that would mean roughly thirty
    # machines, but a run that silently stopped charting the newest one is the
    # exact failure this script exists to prevent, so say so instead.
    if len(wanted) > 64:
        die(f"{len(wanted)} measure/testbed/model triples, and Bencher pins at most 64 plots")

    # Measure first, in MEASURES order, so core_percent keeps the indices it has
    # and a measure added later lands after everything already on the dashboard
    # instead of pushing it all down.
    measure_order = list(MEASURES)

    def order(key: tuple[str, str, str]) -> tuple[int, str, str]:
        return (measure_order.index(key[0]), key[1], key[2])

    for index, key in enumerate(sorted(wanted, key=order)):
        measure_name, testbed_slug, model = key
        style = MEASURES[measure_name]["style"]
        series = wanted[key]
        current = existing.get(key)
        # Testbed first, so the list sorts into one block per machine — which is
        # how anyone reads it, having usually come to look at one machine. Only
        # for a new plot: an existing one keeps whatever it has been renamed to.
        title = (
            current["title"]
            if current and current.get("title")
            else f"{testbed_slug} : {model}{MEASURES[measure_name]['suffix']}"
        )
        if len(title) > 64:
            print(f"skipping plot {title!r}: Bencher titles are limited to 64 characters")
            continue

        testbed_uuid = next(t["uuid"] for t in testbeds if t["slug"] == testbed_slug)
        measure_uuid = uuid_of_measure[measure_name]
        desired = {
            "title": title,
            "index": index,
            "window": PLOT_WINDOW_SECONDS,
            "branches": [branch["uuid"]],
            "testbeds": [testbed_uuid],
            "measures": [measure_uuid],
            "benchmarks": [b["uuid"] for b in series],
            **style,
        }

        if current and index < len(placed) and placed[index] == key and all(
            sorted(current.get(k, [])) == sorted(v) if isinstance(v, list)
            else current.get(k) == v
            for k, v in desired.items()
            if k != "index"  # not reported by the API; checked by position above
        ):
            print(f"plot {title!r}: up to date")
            continue

        flags: list[str] = [
            "--title", title,
            "--index", str(index),
            "--window", str(PLOT_WINDOW_SECONDS),
            "--x-axis", style["x_axis"],
            "--y-axis", style["y_axis"],
        ]
        for k in ("lower_value", "upper_value", "lower_boundary", "upper_boundary"):
            # `create` takes these as bare switches, `update` as explicit
            # booleans, because update has to be able to turn one back off.
            flag = "--" + k.replace("_", "-")
            if current:
                flags += [flag, "true" if style[k] else "false"]
            elif style[k]:
                flags += [flag]
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



# The impulse-response plots: titled as the WaveNet project's are, by
# architecture, and in that order.
IR_PLOT_TITLES = {"m2-air": "Apple M2", "pi500": "Cortex-A76", "tinker": "ARMv7"}
IR_PLOT_BENCHMARKS = ("ir_8192/adt_upstream", "ir_8192/adt_partitioned_fft")


def sync_ir_plots(
    project: str,
    branch: dict[str, Any],
    testbeds: list[dict[str, Any]],
    benchmarks: list[dict[str, Any]],
    measures: dict[str, str],
    dry_run: bool,
) -> None:
    """Pin one plot per testbed: mean and p99, upstream and FFT, 8192 taps.

    No error bars. The p99 has no interval to draw, and a plot styles every
    series alike, so bars for the mean would be the odd one out on a chart
    that is about the gap between two lines.
    """
    wanted_measures = [measures[m] for m in MEASURES if m in measures]
    uuid_of_name = {b["name"]: b["uuid"] for b in benchmarks}
    wanted_benchmarks = [uuid_of_name[n] for n in IR_PLOT_BENCHMARKS if n in uuid_of_name]
    if not wanted_benchmarks:
        print(f"no plots: {project} has none of {', '.join(IR_PLOT_BENCHMARKS)} yet")
        return

    by_slug = {t["slug"]: t for t in testbeds}
    order = [slug for slug in IR_PLOT_TITLES if slug in by_slug]
    order += sorted(slug for slug in by_slug if slug not in IR_PLOT_TITLES)

    # Recognised by the one testbed it draws, so a plot renamed on the
    # dashboard is updated in place rather than joined by a duplicate.
    plots = listing(project, "plot", "--sort", "index", "--direction", "asc")
    existing = {}
    for plot in plots:
        if len(plot.get("testbeds", [])) == 1 and set(plot.get("benchmarks", [])) <= set(wanted_benchmarks):
            existing.setdefault(plot["testbeds"][0], plot)

    style = P99_PLOT_STYLE
    for index, slug in enumerate(order):
        testbed_uuid = by_slug[slug]["uuid"]
        current = existing.get(testbed_uuid)
        title = current["title"] if current and current.get("title") else IR_PLOT_TITLES.get(slug, slug)
        desired = {
            "title": title,
            "window": PLOT_WINDOW_SECONDS,
            "branches": [branch["uuid"]],
            "testbeds": [testbed_uuid],
            "measures": wanted_measures,
            "benchmarks": wanted_benchmarks,
            **style,
        }
        if current and all(
            sorted(current.get(k, [])) == sorted(v) if isinstance(v, list) else current.get(k) == v
            for k, v in desired.items()
        ):
            print(f"plot {title!r}: up to date")
            continue

        flags: list[str] = [
            "--title", title,
            "--index", str(index),
            "--window", str(PLOT_WINDOW_SECONDS),
            "--x-axis", style["x_axis"],
            "--y-axis", style["y_axis"],
        ]
        for k in ("lower_value", "upper_value", "lower_boundary", "upper_boundary"):
            flag = "--" + k.replace("_", "-")
            if current:
                flags += [flag, "true" if style[k] else "false"]
            elif style[k]:
                flags += [flag]
        flags += ["--branches", branch["uuid"], "--testbeds", testbed_uuid]
        for uuid in wanted_measures:
            flags += ["--measures", uuid]
        for uuid in wanted_benchmarks:
            flags += ["--benchmarks", uuid]

        if current:
            print(f"plot {title!r}: updating")
            action = ["plot", "update", project, current["uuid"], *flags]
        else:
            print(f"plot {title!r}: creating")
            action = ["plot", "create", project, *flags]
        if not dry_run:
            bencher(*action)


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
    parser.add_argument(
        "--ir",
        action="store_true",
        help="an impulse-response project: one plot per testbed at 8192 taps",
    )
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

    # Which of the measures this script manages the project actually has. A
    # measure exists once a run has uploaded one, so a project that has never
    # seen an impulse-response run simply has no block_p99_percent to watch.
    listed = listing(args.project, "measure")
    measures = {m["name"]: m["uuid"] for m in listed if m["name"] in MEASURES}
    if not measures:
        die(
            f"{args.project} has none of {', '.join(MEASURES)}; upload a run before syncing"
        )

    if not args.skip_thresholds:
        sync_thresholds(args.project, args.branch, testbeds, measures, args.dry_run)

    if not args.skip_plots:
        (sync_ir_plots if args.ir else sync_plots)(
            args.project, branch, testbeds, listing(args.project, "benchmark"),
            measures, args.dry_run,
        )

    if args.dry_run:
        print("dry run: nothing was changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
