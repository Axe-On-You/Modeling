"""Read Part 1 GPSS campaigns and standalone GPSS Standard Reports."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import re
from statistics import mean

from native_campaign import HEADER_1
from native_metrics import campaign_rows, count, number
from part1_models import OUT, load_trace, make_experiments
from part1_simulation import BASELINE_MARKOV, stability_check, summarize, write_csv

LABELLED_TERMINATE = re.compile(r"^\s*(DONE\d+|LOST\d+)\s+\d+\s+TERMINATE\s+(\d+)")
TABLE_ROW = re.compile(r"^\s*(WTALL|STALL)\s+([\d.]+)\s+([\d.]+)")
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"


def plain_report(path: Path) -> str:
    content = path.read_bytes()
    rtf = content.find(b"{\\rtf")
    if rtf >= 0:
        text = content[rtf:].decode("cp1251", errors="replace")
        return text.replace("\\tab", " ").replace("\\par", "\n")
    return content.decode("cp1251")


def values(lines, prefix, count):
    found = {}
    for line in lines:
        columns = line.split()
        if len(columns) == count + 1 and re.fullmatch(prefix + r"\d+", columns[0]):
            found[columns[0]] = [float(value) for value in columns[1:]]
    return found


def parse_report(path, case, expected):
    raw = plain_report(path)
    clock = re.search(rf"^\s*START TIME[^\n]*\n\s*({NUMBER})\s+({NUMBER})", raw, re.M)
    if not clock:
        raise ValueError(f"{path}: invalid Standard Report header")
    start, end = map(float, clock.groups())
    if end <= start:
        raise ValueError(f"{path}: empty measurement period")
    lines = raw.upper().splitlines()
    terminal = {}
    tables = {}
    for line in lines:
        if found := LABELLED_TERMINATE.match(line):
            terminal[found[1]] = int(found[2])
        if found := TABLE_ROW.match(line):
            tables[found[1]] = float(found[2]), float(found[3])
    nodes = range(1, len(case.channels) + 1)
    lost = sum(terminal[f"LOST{node}"] for node in nodes)
    served = sum(terminal[f"DONE{node}"] for node in nodes)
    if lost + served != expected:
        raise ValueError(f"{path}: expected {expected} outcomes, got {lost} losses and {served} departures")
    queues = values(lines, "QUEUE", 8)
    storage = values(lines, "SRV", 10)
    length = sum(queues.get(f"QUEUE{node}", [0] * 8)[4] for node in nodes)
    busy = sum(storage.get(f"SRV{node}", [0] * 10)[6] for node in nodes)
    wait, wait_sd = tables["WTALL"]
    stay, stay_sd = tables["STALL"]
    # The WAIT table's first bin ends at 0 and isolates exactly-zero delays.
    wait_beg = next(index for index, line in enumerate(lines) if re.match(r"^\s*WTALL\s", line) and TABLE_ROW.match(line))
    first_bin = next((line for line in lines[wait_beg:wait_beg + 4] if re.search(r"-\s+0\.0+\s+\d+\s", line)), "")
    zeros = re.search(r"-\s+0\.0+\s+(\d+)\s+[\d.]+", first_bin)
    wait_zero = int(zeros[1]) / served if zeros and served else 0.0
    if not 0 <= wait_zero <= 1:
        raise ValueError(f"{path}: impossible zero-delay fraction {wait_zero}")
    positive = 1 - wait_zero
    positive_cv = ((max(0, ((wait_sd * wait_sd + wait * wait) * positive / (wait * wait) - 1))) ** 0.5
                   if wait and positive else 0.0)
    return {
        "count": expected,
        "lost": lost,
        "served": served,
        "duration": end - start,
        "busy": busy,
        "throughput": served / (end - start),
        "loss_probability": lost / expected,
        "queue_length": length,
        "wait": wait,
        "wait_sd": wait_sd,
        "wait_cv": wait_sd / wait if wait else 0.0,
        "positive_wait_cv": positive_cv,
        "wait_zero": wait_zero,
        "stay": stay,
        "stay_sd": stay_sd,
        "stay_cv": stay_sd / stay if stay else 0.0,
        "max_waiting": max((int(row[0]) for row in queues.values()), default=0),
    }


def paths_for(case):
    directory = OUT / "raw" / case.name
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir()
                  if path.suffix.lower() in {".txt", ".gpr"})


def parse_campaign(path_list, cases):
    measured, transition, known = [], [], set()
    for row in campaign_rows(path_list, HEADER_1):
        case = cases[row["case"]]
        checkpoint = count(row, "checkpoint")
        rep = count(row, "replication")
        key = case.name, rep, checkpoint
        if not 1 <= rep <= case.replicates or key in known:
            raise ValueError(f"{key}: invalid or duplicate GPSS repeat")
        if checkpoint and (rep != 1 or checkpoint not in case.checkpoints):
            raise ValueError(f"{key}: invalid GPSS checkpoint")
        known.add(key)
        duration = number(row, "duration")
        served = count(row, "served")
        lost = count(row, "lost")
        outcomes = served + lost
        zero_count = count(row, "wait_zero")
        if duration <= 0 or outcomes != (checkpoint or case.observations) or zero_count > served:
            raise ValueError(f"{key}: inconsistent GPSS counters")
        wait, wait_sd = number(row, "wait"), number(row, "wait_sd")
        stay, stay_sd = number(row, "stay"), number(row, "stay_sd")
        wait_zero = zero_count / served if served else 0.0
        positive = 1 - wait_zero
        positive_cv = (max(0, (wait_sd**2 + wait**2) * positive / wait**2 - 1)**0.5
                       if wait and positive else 0.0)
        value = {
            "case": case.name,
            "checkpoint" if checkpoint else "replicate": checkpoint or rep,
            "count": outcomes, "lost": lost, "served": served, "duration": duration,
            "busy": number(row, "busy"), "throughput": served / duration,
            "loss_probability": lost / outcomes, "queue_length": number(row, "queue_length"),
            "wait": wait, "wait_sd": wait_sd, "wait_cv": wait_sd / wait if wait else 0.0,
            "positive_wait_cv": positive_cv, "wait_zero": wait_zero,
            "stay": stay, "stay_sd": stay_sd, "stay_cv": stay_sd / stay if stay else 0.0,
            "max_waiting": count(row, "max_waiting"),
        }
        (transition if checkpoint else measured).append(value)
    return measured, transition, known


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--campaign", action="append", type=Path, default=[],
                     help="GPSS campaign CSV. Defaults to out/part1/raw/campaign/*.csv.")
    args = cli.parse_args()
    trace, _ = load_trace()
    cases = {case.name: case for case in make_experiments(mean(trace))}
    campaigns = args.campaign or sorted((OUT / "raw" / "campaign").glob("*.csv"))
    measured, transition, campaign_keys = parse_campaign(campaigns, cases)
    rejected = []
    for case in cases.values():
        files = paths_for(case)
        if not files:
            continue
        check_count = len(case.checkpoints)
        regular = len(files) == check_count + case.replicates
        # Independently launched model objects may contain only a subset of the repeats.
        if not regular and len(files) > case.replicates:
            rejected.append(f"{case.name}: unexpected report count {len(files)}")
            continue
        checkpoints = case.checkpoints if regular else ()
        if not regular:
            case = replace(case, replicates=len(files), checkpoints=())
        labels = tuple(checkpoints) + (case.observations,) * case.replicates
        try:
            rows = [parse_report(path, case, expected) for path, expected in zip(files, labels, strict=True)]
        except (ValueError, KeyError) as error:
            rejected.append(f"{case.name}: {error}")
            continue
        check_count = len(checkpoints)
        transition.extend(
            {"case": case.name, "checkpoint": n, **row}
            for n, row in zip(checkpoints, rows[:check_count], strict=True)
            if (case.name, 1, n) not in campaign_keys
        )
        measured.extend(
            {"case": case.name, "replicate": i + 1, **row}
            for i, row in enumerate(rows[check_count:])
            if (case.name, i + 1, 0) not in campaign_keys
        )
    for failure in rejected:
        print(f"Rejected {failure}")
    if not measured:
        raise RuntimeError("No valid Part 1 native reports found")
    measured.sort(key=lambda row: (row["case"], row["replicate"]))
    transition.sort(key=lambda row: (row["case"], row["checkpoint"]))
    selected_cases = []
    for case in cases.values():
        replicates = sum(row["case"] == case.name for row in measured)
        if replicates:
            checkpoints = tuple(row["checkpoint"] for row in transition if row["case"] == case.name)
            selected_cases.append(replace(case, replicates=replicates, checkpoints=checkpoints))
    summary = summarize(selected_cases, measured)
    result = {
        "engine": "GPSS World Student",
        "markov": BASELINE_MARKOV,
        "experiments": summary,
        "stability": {
            case.name: stability_check(
                case, [row for row in transition if row["case"] == case.name],
                summary[case.name]["metrics"]
            )
            for case in selected_cases if case.checkpoints
        },
    }
    write_csv(OUT / "gpss_measurements.csv", measured)
    write_csv(OUT / "gpss_transient.csv", transition)
    (OUT / "gpss_results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"{len(selected_cases)} valid native cases, {len(measured)} observations")


if __name__ == "__main__":
    main()
