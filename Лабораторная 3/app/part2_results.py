"""Read GPSS network Standard Reports and build the report's tables and plots."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
from statistics import mean, stdev

from gpss_runner import report_text
from part2 import CAPACITY, ROOT, SERVERS, VISITS, closed_metrics, open_metrics

OUT = ROOT / "out" / "part2"
NODE_FIELDS = ("utilization", "queue", "waiting", "residence")
NUMERIC = r"-?(?:\d+(?:\.\d*)?|\.\d+)"
INTERVAL = re.compile(
    rf"(?P<lo>{NUMERIC}|_)\s*-\s*(?P<hi>{NUMERIC}|_)\s+"
    r"(?:\d+\s+)?(?P<count>\d+)\s+(?P<cumulative>[\d.]+)\s*$"
)


def positive_duration(lines):
    for i, line in enumerate(lines):
        if re.search(r"START TIME\s+END TIME", line):
            start, stop = map(float, lines[i + 1].split()[:2])
            if start <= 0 or stop <= start:
                raise ValueError("Expected a Standard Report from the measurement interval")
            return stop - start
    raise ValueError("Standard Report header not found")


def read_table(lines, table):
    """Return exact count, rounded mean and histogram bins from a standard report."""
    label = table.upper()
    table_header = next(i for i, line in enumerate(lines) if line.split()[:2] == ["TABLE", "MEAN"])
    start = next(i for i in range(table_header + 1, len(lines)) if lines[i].split()[:1] == [label])
    first = lines[start].split()
    average, deviation = map(float, first[1:3])
    bins = []
    for i, line in enumerate(lines[start:], start):
        match = INTERVAL.search(line[36:] if i == start else line)
        if not match and i != start:
            break
        if match:
            lo, hi, count, _ = match.groups()
            bins.append((float(lo) if lo != "_" else 0.0, float(hi) if hi != "_" else None, int(count)))
    if not bins or not sum(count for _, _, count in bins):
        raise ValueError(f"Table {table} contains no completed observations")
    return sum(count for _, _, count in bins), average, deviation, bins


def parse_report(path):
    lines = report_text(path).decode("cp1252", errors="replace").splitlines()
    duration = positive_duration(lines)
    done, residence, residence_sd, time_bins = read_table(lines, "Residence")
    wait_count, waiting, waiting_sd, wait_bins = read_table(lines, "WaitNet")
    if done != wait_count:
        raise ValueError(f"Network tables have different exit counts: {path}")
    result = {
        "duration": duration,
        "completed": done,
        "throughput": done / duration,
        "waiting": waiting,
        "residence": residence,
        "waiting_cv": waiting_sd / waiting if waiting else 0.0,
        "residence_cv": residence_sd / residence,
    }
    histograms = {"time": time_bins, "wait": wait_bins}
    for i in range(1, 5):
        queue = next(line.split() for line in lines if line.split()[:1] == [f"QUEUEU{i}"]
                     and len(line.split()) >= 9)
        store = next(line.split() for line in lines if line.split()[:1] == [f"STOREU{i}"]
                     and len(line.split()) >= 10)
        _, node_residence, visit_sd, visit_bins = read_table(lines, f"Visit{i}")
        _, wait_avg, wait_sd, delay_bins = read_table(lines, f"Delay{i}")
        result[f"residence_cv_u{i}"] = visit_sd / node_residence
        result[f"waiting_cv_u{i}"] = wait_sd / wait_avg if wait_avg else 0.0
        histograms[f"time_u{i}"] = visit_bins
        histograms[f"wait_u{i}"] = delay_bins
        for field, value in zip(NODE_FIELDS, (store[8], queue[5], queue[6], node_residence)):
            result[f"{field}_u{i}"] = float(value)
    return result, histograms


def load_runs(folder, scenarios):
    result = defaultdict(list)
    histograms = defaultdict(list)
    for scenario in scenarios:
        path = folder / scenario["name"] / "01.txt"
        measurements, observed = parse_report(path)
        values = {**scenario, **measurements}
        result[scenario["group"]].append(values)
        histograms[scenario["group"]].append(observed)
    return result, histograms


def compute_summary(runs):
    summary = {}
    for group, reps in runs.items():
        first = reps[0]
        fields = ["throughput", "residence", "waiting", "completed", "duration", "waiting_cv", "residence_cv"] + [
            f"{key}_u{i}" for i in range(1, 5) for key in (*NODE_FIELDS, "residence_cv", "waiting_cv")
        ]
        result = {key: first[key] for key in
                  ("population", "arrival_rate", "arrival", "service2", "servers3", "service3")}
        result["replications"] = len(reps)
        for field in fields:
            nums = [r[field] for r in reps]
            result[field] = mean(nums)
            if field in ("throughput", "residence", "waiting"):
                result[field + "_sd"] = stdev(nums) if len(nums) > 1 else 0.0
        summary[group] = result
    return summary


def write_csv(path, rows):
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_theory(summary):
    rows = []
    for group, result in summary.items():
        if result["service2"] != "exp" or result["arrival"] != "exp":
            continue
        n = result["population"]
        if n:
            servers = (3, 2, result["servers3"], 1)
            duration = (4.0, 2.5, result["service3"], 5.0)
            throughput, residence, waits = closed_metrics(n, servers, duration)
            waiting = sum(v*w for v, w in zip(VISITS, waits))
        else:
            throughput = result["arrival_rate"]
            waiting, residence = open_metrics(throughput)
        row = {"group": group}
        for key, exact in zip(("throughput", "residence", "waiting"), (throughput, residence, waiting)):
            observed = result[key]
            row.update({key: observed, key + "_exact": exact,
                        key + "_relative_pct": (observed / exact - 1) * 100 if exact else ""})
        rows.append(row)
    return rows


def draw_topology(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch

    nodes = [(0.15, 0), (1.60, 0), (3.35, 1.06), (3.35, 0), (3.35, -1.06)]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.45), layout="constrained")
    for closed, ax in zip((True, False), axes):
        for i, (x, y) in enumerate(nodes):
            circle = Circle((x, y), .225, facecolor="#e7f0fa", edgecolor="#295782", lw=1.2)
            ax.add_patch(circle)
            ax.text(x, y, "0" if i == 0 else f"У{i}", ha="center", va="center", weight="bold")
        for target in (0, 2, 3, 4):
            for reverse, label in ((False, "0,25"), (True, "1")):
                start, end = ((nodes[target], nodes[1]) if reverse else (nodes[1], nodes[target]))
                arc = FancyArrowPatch(start, end, connectionstyle="arc3,rad=.16",
                                      arrowstyle="-|>", color="#3b5272", mutation_scale=11,
                                      shrinkA=16, shrinkB=16)
                ax.add_patch(arc)
                dx, dy = end[0] - start[0], end[1] - start[1]
                mag = (dx*dx + dy*dy)**.5
                x, y = (start[0] + end[0])/2 + .20*dy/mag, (start[1] + end[1])/2 - .20*dx/mag
                ax.text(x, y, label, ha="center", va="center", fontsize=8.4, color="#355379")
        for node, service in enumerate(("3 × 4 с", "2 × 2,5 с", "1 × 14 с", "1 × 5 с"), 1):
            x, y = nodes[node]
            ax.text(x + (0.29 if node != 1 else -.03), y - (.34 if node == 1 else 0),
                    service, ha="center" if node == 1 else "left", va="center", fontsize=8.2)
        ax.text(nodes[0][0], -.42,
                "Возврат\nбез задержки" if closed else "Внешний\nисточник и сток",
                ha="center", va="top", fontsize=8.4)
        ax.set(title="Замкнутая сеть" if closed else "Разомкнутая сеть",
               xlim=(-.44,4.22), ylim=(-1.46,1.45), aspect="equal")
        ax.axis("off")
    fig.savefig(directory / "topology.pdf")
    plt.close(fig)


def draw(summary, histograms, directory, threshold, *, result_label="GPSS World"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({
        "font.size": 11, "axes.grid": True, "grid.alpha": 0.22,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.constrained_layout.use": True,
    })
    palette = ("#235889", "#bb632a", "#46845a")

    def finish(fig, name):
        fig.savefig(directory / (name + ".pdf"))
        plt.close(fig)

    closed = sorted(
        (r for name, r in summary.items() if re.fullmatch(r"closed_m\d+", name)),
        key=lambda r: r["population"],
    )
    if closed:
        fig, (a, b) = plt.subplots(1, 2, figsize=(9.4, 3.2))
        m = [r["population"] for r in closed]
        a.errorbar(m, [r["throughput"] for r in closed],
                   yerr=[r["throughput_sd"] for r in closed], marker="o",
                   ms=4, color=palette[0], label=result_label)
        a.plot(m, [closed_metrics(n)[0] for n in m], color="black", ls=":", label="Расчёт")
        a.axhline(CAPACITY, ls="--", color="#777777", label="Предел")
        a.axvline(threshold, ls="--", color=palette[1], lw=1)
        a.set(xlabel="Число заявок M", ylabel="Производительность, заявок/с")
        a.legend(loc="lower right", fontsize=9)
        for field, text, color in [("residence", "Пребывание", palette[0]),
                                   ("waiting", "Ожидание", palette[1])]:
            b.errorbar(m, [r[field] for r in closed], yerr=[r[field + "_sd"] for r in closed],
                       marker="o", ms=4, color=color, label=text)
        b.legend(loc="upper left", fontsize=9)
        b.set(xlabel="Число заявок M", ylabel="Время на цикл, с")
        for ax in (a, b):
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        finish(fig, "closed_scaling")

        fig, (a, b) = plt.subplots(1, 2, figsize=(9.4, 3.2))
        for calc, text, color in zip(("exp", "erl", "h2"),
                                     ("Экспоненциальное", "Эрланга E₂", "Гиперэксп. H₂"), palette):
            filter_name = (r"closed_m\d+" if calc == "exp" else rf"closed_{calc}_m\d+")
            reps = sorted((r for name, r in summary.items() if re.fullmatch(filter_name, name)),
                          key=lambda r: r["population"])
            if reps:
                m = [r["population"] for r in reps]
                a.plot(m, [r["throughput"] for r in reps], "o-", ms=4, color=color, label=text)
                b.plot(m, [r["waiting_u2"] for r in reps], "o-", ms=4, color=color, label=text)
        a.set(xlabel="Число заявок M", ylabel="Производительность, заявок/с")
        b.set(xlabel="Число заявок M", ylabel="Среднее ожидание в У2, с")
        a.legend(loc="lower right", fontsize=8)
        b.legend(loc="upper left", fontsize=8)
        for ax in (a, b):
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        finish(fig, "closed_service")

    open_sweep = sorted(
        (r for name, r in summary.items() if re.fullmatch(r"open_rho\d+", name)),
        key=lambda r: r["arrival_rate"],
    )
    if open_sweep:
        fig, (a, b) = plt.subplots(1, 2, figsize=(9.4, 3.2))
        x = [r["arrival_rate"] / CAPACITY for r in open_sweep]
        for field, text, color in [("residence", "Пребывание", palette[0]),
                                   ("waiting", "Ожидание", palette[1])]:
            a.errorbar(x, [r[field] for r in open_sweep],
                       yerr=[r[field + "_sd"] for r in open_sweep],
                       marker="o", ms=4, color=color, label=text)
        a.set(xlabel="Заданная загрузка У3", ylabel="Время в сети, с")
        a.legend(fontsize=9)
        for i, color in enumerate(("#235889", "#bb632a", "#46845a", "#756096"), 1):
            b.plot(x, [r[f"utilization_u{i}"] for r in open_sweep],
                   marker="o", ms=4, color=color, label=f"У{i}")
        b.set(xlabel="Заданная загрузка У3", ylabel="Измеренная загрузка прибора")
        b.legend(fontsize=9)
        finish(fig, "open_load")

    matches = [f"open_match_{law}_{arr}" for law in ("exp", "erl", "h2") for arr in ("exp", "det")]
    closed_matches = [f"closed_{tag}m{threshold:02d}" for tag in ("", "erl_", "h2_")]
    if all(name in summary for name in matches + closed_matches):
        fig, (a, b) = plt.subplots(1, 2, figsize=(9.4, 3.2))
        labels = ("Эксп.", "Эрланг", "Гиперэксп.")
        for sub, key, ylabel in ((a, "residence", "Среднее пребывание, с"), (b, "waiting_u2", "Ожидание в У2, с")):
            series = [
                [summary[name][key] for name in closed_matches],
                [summary[name][key] for name in matches[::2]],
                [summary[name][key] for name in matches[1::2]],
            ]
            for j, (values, text, color) in enumerate(zip(series, ("Замкнутая", "РСеМО, Пуассон",
                                                       "РСеМО, регулярный"), palette)):
                sub.bar([i+(j-1)*0.25 for i in range(3)], values, width=.24, label=text, color=color)
            sub.set(xticks=range(3), xticklabels=labels, ylabel=ylabel)
        a.legend(fontsize=8)
        finish(fig, "matched_service")

    histogram_matches = (f"closed_m{threshold:02d}", "open_match_exp_exp", "open_match_exp_det")
    if all(name in histograms for name in histogram_matches):
        fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.0))
        for ax, key, width, title in zip(
            axes.flat, ("wait", "time", "wait_u2", "time_u2"), (25, 25, 1, 5),
            ("Ожидание в сети", "Пребывание в сети", "Ожидание в У2", "Пребывание в У2")
        ):
            for name, label, color in zip(
                histogram_matches, ("Замкнутая", "РСеМО, Пуассон", "РСеМО, регулярный"), palette
            ):
                counts = Counter()
                for replication in histograms[name]:
                    for lo, hi, count in replication[key]:
                        counts[hi if hi is not None else lo+width] += count
                overall = sum(counts.values())
                x = list(range(0, round(max(counts)) + 1, width))
                ax.step(x, [counts[edge] / overall for edge in x],
                        where="mid", label=label, color=color)
            ax.set(title=title, xlabel="Правая граница интервала, с", ylabel="Доля заявок")
        axes[0, 0].legend(fontsize=8)
        finish(fig, "distributions")


def analyze(reports, manifest, output):
    data = json.loads(manifest.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    draw_topology(output)
    runs, histograms = load_runs(reports, data["scenarios"])
    summary = compute_summary(runs)
    write_csv(output / "replications.csv", (r for reps in runs.values() for r in reps))
    write_csv(output / "metrics.csv", ({"group": group, **r} for group, r in summary.items()))
    write_csv(output / "analytical_check.csv", compare_theory(summary))
    draw(summary, histograms, output, data["threshold"])
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    summary = analyze(args.reports, args.manifest, args.output)
    print(f"Processed {len(summary)} network scenarios")
