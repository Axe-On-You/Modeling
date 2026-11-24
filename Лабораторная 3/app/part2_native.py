"""Analyze verified GPSS reports from individual M3 experiments."""
from collections import Counter, defaultdict
from dataclasses import asdict
import argparse
from pathlib import Path
from math import isclose

from native_campaign import HEADER_2
from native_metrics import campaign_rows, count, number
from part2 import ROOT, canonical_scenarios, closed_metrics, paired_scenarios
from part2_results import parse_report, compute_summary, compare_theory, draw, write_csv


OUT = ROOT / "out" / "part2"


def histogram(ax, bins, width, title):
    count = sum(frequency for _, _, frequency in bins)
    by_edge = Counter({hi if hi is not None else lo+width: n for lo, hi, n in bins})
    edges = list(range(0, round(max(by_edge))+1, width))
    ax.bar(edges, [by_edge[x]/count for x in edges], width * .9, color="#35668d")
    ax.set(title=title, xlabel="Правая граница интервала, с", ylabel="Доля наблюдений")
    ax.grid(alpha=.25, axis="y")


def native_histogram(distribution, network):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.5), layout="constrained")
    scope = ("за цикл", "Длительность цикла") if network == "closed" else ("в сети", "Пребывание в сети")
    for ax, name, step, title in zip(
        axes.flat, ("wait", "time", "wait_u2", "time_u2"), (25, 25, 1, 5),
        (f"Ожидание {scope[0]}", scope[1], "Ожидание в У2", "Пребывание в У2"), strict=True
    ):
        histogram(ax, distribution[name], step, title)
    fig.savefig(OUT / f"gpss_{network}_histograms.pdf", bbox_inches="tight", pad_inches=.06)
    plt.close(fig)


def parse_campaign(path_list, cases):
    results, known = defaultdict(list), set()
    for row in campaign_rows(path_list, HEADER_2):
        case = cases[row["case"]]
        rep = count(row, "replication")
        key = case.name, rep
        if not 1 <= rep <= 5 or key in known:
            raise ValueError(f"{key}: invalid or duplicate GPSS repeat")
        known.add(key)
        duration = number(row, "duration")
        completed = count(row, "completed")
        rate = number(row, "arrival_rate")
        if not duration or not completed or bool(case.population) == bool(rate):
            raise ValueError(f"{key}: inconsistent GPSS network counters")
        old = results[case.name]
        if old and not isclose(old[0]["arrival_rate"], rate):
            raise ValueError(f"{key}: different arrival rates for one GPSS case")
        values = {**asdict(case), "replication": rep, "arrival_rate": rate,
                  "duration": duration, "completed": completed, "throughput": completed / duration}
        for suffix in ("", "_u1", "_u2", "_u3", "_u4"):
            for field in ("waiting", "residence"):
                average = number(row, field+suffix)
                deviation = number(row, field+"_sd"+suffix)
                values[field+suffix] = average
                values[field+"_cv"+suffix] = deviation / average if average else 0.0
            if suffix:
                for field in ("utilization", "queue"):
                    values[field+suffix] = number(row, field+suffix)
        old.append(values)
    return results


def run(raw, campaigns=()):
    scenarios, threshold = canonical_scenarios()
    # These control models use the exact exponential reference rate at M*.
    scenarios += paired_scenarios(threshold, {law: closed_metrics(threshold)[0]
                                             for law in ("exp", "erl", "h2")})
    observations = parse_campaign(campaigns, {case.name: case for case in scenarios})
    histograms = {}
    for case in scenarios:
        folder = raw / case.name
        if not folder.is_dir():
            continue
        values = observations.get(case.name, [])
        if values and not isclose(values[0]["arrival_rate"], case.arrival_rate, rel_tol=1e-6):
            continue
        have_repeats = {value["replication"] for value in values}
        distributions = []
        for rep, path in enumerate(sorted(folder.glob("*.gpr")), 1):
            metrics, distribution = parse_report(path)
            distributions.append(distribution)
            if rep == 1 and case.name in (f"closed_m{threshold:02d}", "open_rho90"):
                native_histogram(distribution, "closed" if case.population else "open")
            if rep not in have_repeats:
                values.append({"replication": rep, **asdict(case), **metrics})
        if values:
            observations[case.name] = sorted(values, key=lambda row: row["replication"])
        if distributions:
            histograms[case.name] = distributions
    if not observations:
        raise RuntimeError("No native GPSS reports found")
    summary = compute_summary(observations)
    write_csv(OUT / "gpss_replications.csv", (
        {"group": name, **row} for name, rows in observations.items() for row in rows
    ))
    write_csv(OUT / "gpss_metrics.csv", ({"group": name, **values} for name, values in summary.items()))
    figure_dir = OUT / "gpss"
    figure_dir.mkdir(parents=True, exist_ok=True)
    draw(summary, histograms, figure_dir, threshold)
    theoretical = compare_theory(summary)
    if theoretical:
        write_csv(OUT / "gpss_analytical_check.csv", theoretical)
    print(f"{len(summary)} verified native network scenarios")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--raw", type=Path, default=OUT / "raw")
    cli.add_argument("--campaign", action="append", type=Path, default=[])
    args = cli.parse_args()
    run(args.raw, args.campaign)
