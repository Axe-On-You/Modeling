"""Stage individual GPSS World experiments and collect validated reports."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
from statistics import mean

from gpss_runner import TEMPLATE, REPORT_NAME, report_text, validate_report
from part1_models import SEEDS, load_trace, make_experiments, create_model
from part2 import CAPACITY, ROOT, canonical_scenarios, closed_metrics, paired_scenarios, render_model


def programs(part, replications):
    if part == "part1":
        trace, _ = load_trace()
        for case in make_experiments(mean(trace)):
            for rep in range(1, min(replications, case.replicates)+1):
                yield case.name, rep, create_model(case, trace, replication=rep)
        return

    cases, threshold = canonical_scenarios()
    # Common control rate for native open/closed comparisons.
    exact_rate = closed_metrics(threshold)[0]
    cases += paired_scenarios(threshold, dict.fromkeys(("exp", "erl", "h2"), exact_rate))
    for case in cases:
        warmup, duration = ((100_000, 800_000) if case.arrival_rate >= .95*CAPACITY
                            else (10_000, 200_000))
        for rep in range(1, replications+1):
            trial = replace(case, seed=3 + 1_000_003*(rep-1))
            yield case.name, rep, render_model(trial, warmup=warmup, duration=duration)


def stage(part, target, requested, replications):
    items = [(name, rep, source) for name, rep, source in programs(part, replications)
             if not requested or name in requested]
    if requested and set(requested) - {name for name, _, _ in items}:
        raise ValueError("Unknown scenario: " + ", ".join(set(requested)-{name for name, _, _ in items}))
    target.mkdir(parents=True, exist_ok=False)
    for name, rep, source in items:
        workdir = target / name / f"{rep:02d}"
        workdir.mkdir(parents=True)
        shutil.copyfile(TEMPLATE, workdir / f"{name}.gps")
        (workdir / "model.txt").write_text(source, encoding="ascii", newline="\r\n")
    manifest = [{"name": name, "replication": rep} for name, rep, _ in items]
    (target / "runs.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    return len(items)


def collect(staging, destination):
    count = 0
    manifest = json.loads((staging / "runs.json").read_text(encoding="utf-8"))
    for item in manifest:
        name, rep = item["name"], item["replication"]
        paths = list((staging / name / f"{rep:02d}").glob("*.gpr"))
        if not paths:
            continue
        reports = [(REPORT_NAME.fullmatch(p.name), p) for p in paths]
        published = []
        for match, path in reports:
            if not match:
                raise ValueError(f"Unexpected report name: {path}")
            stem, simulation, serial = match.groups()
            content = report_text(path)
            validate_report(path, content, stem, int(simulation), int(serial))
            published.append(path)
        if len(published) != 1:
            raise ValueError(f"Expected one report for {name}, repeat {rep}")
        output = destination / name / f"{rep:02d}.gpr"
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and output.read_bytes() != published[0].read_bytes():
            raise ValueError(f"Existing report conflicts with the new run: {output}")
        shutil.copyfile(published[0], output)
        count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("stage", "collect"))
    parser.add_argument("part", choices=("part1", "part2"))
    parser.add_argument("staging", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--replications", type=int, choices=range(1, len(SEEDS)+1), default=1)
    args = parser.parse_args()
    if args.mode == "stage":
        print(f"{stage(args.part, args.staging, args.case, args.replications)} staged runs")
    else:
        print(f"{collect(args.staging, args.output or ROOT/'out'/args.part/'raw')} verified reports")
