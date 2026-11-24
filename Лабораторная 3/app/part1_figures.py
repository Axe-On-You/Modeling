"""Create figures for the system experiments from exported measurements."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "out" / "part1"
plt.rcParams.update({"font.size": 10, "figure.dpi": 130, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": .22})
COLORS = ("#235889", "#b55837", "#387a58", "#745890")
LOADS = (50, 75, 95, 99)


def save(fig, name, output):
    fig.savefig(output / (name + ".pdf"), bbox_inches="tight", pad_inches=.05)
    plt.close(fig)


def available_points(experiments, axis, keys):
    return [(value, experiments[key]["metrics"]) for value, key in zip(axis, keys, strict=True)
            if key in experiments]


def finish(fig, legend_ax, identifier, output):
    if legend_ax.lines:
        legend_ax.legend(fontsize=8)
        save(fig, identifier, output)
    else:
        plt.close(fig)


def band(ax, records, x, field, label, color, scale=1, style="-"):
    y = [r[field]["mean"] * scale for r in records]
    intervals = [(r[field]["ci95_halfwidth"] or 0) * scale for r in records]
    ax.plot(x, y, style, marker="o", ms=4, color=color, label=label)
    if any(intervals):
        ax.fill_between(x, [a-b for a, b in zip(y, intervals)],
                        [a+b for a, b in zip(y, intervals)], alpha=.13, color=color)


def flows(experiments, family, identifier, output):
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.25), layout="constrained")
    series = ([("exp", "Простейший"), ("trace", "Трасса"), ("h2", "$H_2$ входа"), ("uniform", "Равномерный")]
              if family == "flow" else [("exp", "Экспоненциальное"), ("e2", "$E_2$"), ("h2", "$H_2$")])
    for (key, label), color in zip(series, COLORS):
        prefix = "flow" if family == "flow" or key == "exp" else "service"
        pairs = available_points(experiments, [load/100 for load in LOADS],
                                 [f"{prefix}_{key}_{load:03d}" for load in LOADS])
        if not pairs:
            continue
        x, records = zip(*pairs)
        for ax, (metric, title, scale) in zip(axes, (("wait", "Ожидание, с", 1),
                                ("stay", "Пребывание, с", 1), ("loss_probability", "Потери, %", 100))):
            band(ax, records, x, metric, label, color, scale)
            ax.set(xlabel=r"Предложенная загрузка $\rho_1$", ylabel=title)
    finish(fig, axes[0], identifier, output)


def buffers(experiments, output):
    capacities = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)
    fig, axes = plt.subplots(1, 3, figsize=(10.25, 3.25), layout="constrained")
    for load, color in zip((50, 90), COLORS):
        pairs = available_points(experiments, capacities,
                                 [f"buffer_{load:03d}_{b:03d}" for b in capacities])
        if not pairs:
            continue
        x, records = zip(*pairs)
        for ax, (metric, title, scale) in zip(axes, (("loss_probability", "Потери, %", 100),
                                ("wait", "Ожидание, с", 1), ("stay", "Пребывание, с", 1))):
            band(ax, records, x, metric, fr"$\rho_1={load/100:g}$", color, scale)
            ax.set(xlabel="Мест ожидания перед каждым прибором", ylabel=title,
                   xscale="symlog", xticks=(0, 2, 8, 32, 128, 256))
            ax.set_xticklabels(("0", "2", "8", "32", "128", "256"))
    finish(fig, axes[0], "buffers", output)


def servers(experiments, output):
    fig, axes = plt.subplots(2, 2, figsize=(8.9, 5.4), layout="constrained")
    for row_axes, (family, label) in zip(axes, (("work", "Среднее обслуживание фиксировано"),
                                              ("load", "Загрузка первой ветви фиксирована"))):
        for load, color in zip((50, 75, 95, 99), COLORS):
            keys = (f"servers_{family}_{load:03d}_1", f"flow_exp_{load:03d}", f"servers_{family}_{load:03d}_3")
            if not any(key in experiments for key in (keys[0], keys[-1])):
                continue
            x, records = zip(*available_points(experiments, (1, 2, 3), keys))
            for ax, (metric, title) in zip(row_axes, (("wait", "Ожидание, с"), ("stay", "Пребывание, с"))):
                band(ax, records, x, metric, f"{load/100:g}", color)
                ax.set(xlabel="Число приборов", ylabel=title, xticks=(1, 2, 3))
        row_axes[0].set_title(label, loc="left")
    populated = next((ax for ax in (axes[0][1], axes[1][1]) if ax.lines), None)
    if populated is None:
        plt.close(fig)
    else:
        populated.legend(title=r"Исходная $\rho_1$", fontsize=8)
        save(fig, "servers", output)


def transient(experiments, source, output):
    path = OUT / f"{source}_transient.csv"
    if not path.is_file():
        return
    with path.open(encoding="utf-8", newline="") as stream:
        data = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.35), layout="constrained")
    for case, label, color in zip(("flow_exp_075", "flow_trace_075", "flow_h2_075"),
                              ("Простейший", "Трасса", "$H_2$"), COLORS):
        selected = [r for r in data if r["case"] == case]
        if not selected or case not in experiments:
            continue
        x = [int(r["checkpoint"]) for r in selected]
        for ax, (metric, title, scale) in zip(axes, (("loss_probability", "Накопленная доля потерь, %", 100),
                                                ("wait", "Накопленное среднее ожидание, с", 1))):
            ax.plot(x, [float(r[metric])*scale for r in selected], "o-", ms=3.5, color=color, label=label)
            ax.axhline(experiments[case]["metrics"][metric]["mean"]*scale, color=color, linestyle=":", lw=1)
            ax.set(xscale="log", xlabel="Число завершений и отказов", ylabel=title)
    finish(fig, axes[0], "transition", output)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source", choices=("python", "gpss"), default="gpss")
    args = cli.parse_args()
    output = OUT if args.source == "python" else OUT / "gpss"
    output.mkdir(exist_ok=True)
    series = json.loads((OUT / f"{args.source}_results.json").read_text(encoding="utf-8"))["experiments"]
    flows(series, "flow", "flows", output)
    flows(series, "service", "services", output)
    buffers(series, output)
    servers(series, output)
    transient(series, args.source, output)
