"""Build the GPSS World models and experiment manifest for UIR 3, part 1."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from itertools import product
from pathlib import Path
import random
from statistics import mean, stdev

LAB = Path(__file__).resolve().parents[1]
MODELS = LAB / "models" / "part1"
OUT = LAB / "out" / "part1"
TRACE_SOURCE = LAB.parent / "Лабораторная 1/Вариант/Вариант_207.csv"
CHECKPOINTS = (10, 50, 100, 200, 300, 600, 1200, 2500, 5000, 10000)
OBSERVATIONS = 120000
CORE_LOADS = (0.50, 0.75, 0.95, 0.99)
H2_P, H2_M1, H2_M2 = 0.2319, 44.3201, 1.6122
random_sequence = random.Random(20261123)
SEEDS = [
    tuple(random_sequence.sample(range(1, 2_147_483_646), 7))
    for _ in range(5)
]


@dataclass(frozen=True)
class Experiment:
    name: str
    family: str
    incoming: str
    gap: float
    service: str
    service_time: float
    channels: tuple[int, ...]
    waiting_places: tuple[int, ...]
    offered_load: float
    warmup: int = 10000
    observations: int = OBSERVATIONS
    replicates: int = 5
    checkpoints: tuple[int, ...] = ()


def load_trace() -> tuple[list[float], list[float]]:
    source = TRACE_SOURCE.read_text(encoding="utf-8-sig").splitlines()
    if source[0].strip() != "207":
        raise ValueError("Expected variant 207 in the first line of the trace")
    lines = [line.strip() for line in source[1:] if line.strip()][:300]
    # Match step1_stats.py in UIR 1, which read only the first CSV column.
    uir1 = [float(line.split(",")[0]) for line in lines]
    original = [float(line.replace(",", ".")) for line in lines]
    if len(uir1) != 300 or any(value < 0 for value in original):
        raise ValueError("Expected 300 nonnegative trace intervals")
    return uir1, original


def load_code(value: float) -> str:
    return f"{round(value * 100):03d}"


def make_experiments(trace_mean: float) -> list[Experiment]:
    baseline = Experiment("baseline", "baseline", "exp", 1 / 0.6, "exp", 5, (1, 1), (2, 1), 2.25)
    cases = [baseline]
    for stream, load in product(("exp", "trace", "h2", "uniform"), CORE_LOADS):
        cases.append(Experiment(
            f"flow_{stream}_{load_code(load)}", "flow", stream, trace_mean, "exp",
            load * trace_mean / 0.75, (1, 1), (2, 1), load, checkpoints=CHECKPOINTS,
        ))
    for load in (0.1, 0.3, 0.7, 0.9):
        cases.append(Experiment(
            f"transient_{load_code(load)}", "transient", "exp", trace_mean,
            "exp", load * trace_mean / 0.75, (1, 1), (2, 1), load,
            checkpoints=CHECKPOINTS, replicates=1,
        ))
        gap = 5 * 0.75 / load
        cases.append(Experiment(
            f"transient_gap_{load_code(load)}", "transient_gap", "exp", gap,
            "exp", 5, (1, 1), (2, 1), load, checkpoints=CHECKPOINTS, replicates=1,
        ))
    for service, load in product(("e2", "h2"), CORE_LOADS):
        cases.append(Experiment(
            f"service_{service}_{load_code(load)}", "service", "exp", trace_mean,
            service, load * trace_mean / 0.75, (1, 1), (2, 1), load,
        ))
    for load, capacity in product((0.5, 0.9), (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)):
        cases.append(Experiment(
            f"buffer_{load_code(load)}_{capacity:03d}", "buffer", "exp", trace_mean,
            "exp", load * trace_mean / 0.75, (1, 1), (capacity, capacity), load,
        ))
    for load, count in product(CORE_LOADS, (1, 2, 3)):
        count_config = {1: ((1,), (2,), 1.0), 2: ((1, 1), (2, 1), 0.75), 3: ((2, 1), (2, 1), 0.75)}
        channels, capacities, first_fraction = count_config[count]
        service_at_fixed_work = load * trace_mean / 0.75
        service_at_fixed_load = load * trace_mean * channels[0] / first_fraction
        for family, service_mean in (
            ("servers_work", service_at_fixed_work),
            ("servers_load", service_at_fixed_load),
        ):
            if count == 2:  # Use the identical exponential-flow scenario.
                continue
            cases.append(Experiment(
                f"{family}_{load_code(load)}_{count}", family, "exp", trace_mean, "exp",
                service_mean, channels, capacities, first_fraction * service_mean / (trace_mean * channels[0]),
            ))
    for load, service in product((0.50, 0.75), ("exp", "h2")):
        cases.append(Experiment(
            f"control_{service}_{load_code(load)}", "control", "exp", trace_mean, service,
            load * trace_mean, (1,), (256,), load,
        ))
    return cases


def exp_duration(stream: int, average: float) -> str:
    return f"Exponential({stream},0,{average:.11g})"


def mix_service(mean_service: float) -> tuple[float, float, float]:
    p = (1 + (3 / 5) ** 0.5) / 2
    return p, mean_service / (2 * p), mean_service / (2 * (1 - p))


def append_mixture(lines, tag, choose, stream, p, first, second):
    lines.extend([
        f"       TEST L (Uniform({choose},0,1)),{p:.11g},{tag}Second",
        f"       ADVANCE ({exp_duration(stream, first)})",
        f"       TRANSFER ,{tag}Join",
        f"{tag}Second ADVANCE ({exp_duration(stream, second)})",
    ])
    return f"{tag}Join"


def create_model(case: Experiment, trace: list[float], replication: int = 1) -> str:
    """Return one experiment with a fresh generator state and one native report."""
    lines = [
        f"* UIR 3 Part 1: {case.name}, replication {replication}",
        "* Time unit is one second.",
        "       RMULT " + ",".join(map(str, SEEDS[replication - 1])),
    ]
    if case.incoming == "trace":
        lines.append(f"TraceFn FUNCTION P1,L{len(trace)}")
        pairs = [f"{index},{value:.11g}" for index, value in enumerate(trace, 1)]
        lines.extend("/".join(pairs[index:index + 6]) for index in range(0, len(pairs), 6))
    for node, (servers, waiting) in enumerate(zip(case.channels, case.waiting_places, strict=True), 1):
        lines.extend([f"Cap{node} STORAGE {servers + waiting}", f"Srv{node} STORAGE {servers}"])
    step = max(.25, case.service_time / 4)
    lines.extend([f"WTall TABLE P3,0,{step:.11g},65", f"STall TABLE M1,0,{step:.11g},65", ""])
    if case.incoming == "trace":
        lines += [
            "       GENERATE ,,,1",
            "IdxOne ASSIGN 1,1",
            "TNext  ADVANCE FN$TraceFn",
            "       SPLIT 1,Route",
            "       ASSIGN 1+,1",
            f"       TEST LE P1,{len(trace)},IdxOne",
            "       TRANSFER ,TNext",
        ]
    elif case.incoming == "h2":
        correction = case.gap / (H2_P * H2_M1 + (1 - H2_P) * H2_M2)
        lines += ["       GENERATE ,,,1", "NextGap ASSIGN 4,0"]
        join = append_mixture(lines, "Gap", 7, 2, H2_P, H2_M1 * correction, H2_M2 * correction)
        lines += [f"{join} SPLIT 1,Route", "       TRANSFER ,NextGap"]
    elif case.incoming == "uniform":
        lines.append(f"       GENERATE (Uniform(2,0,{2*case.gap:.11g}))")
    else:
        lines.append(f"       GENERATE ({exp_duration(2, case.gap)})")
    lines.append("Route  MARK")
    if len(case.channels) == 2:
        lines.append("       TRANSFER .75,In2,In1")
    for node, (servers, waiting) in enumerate(zip(case.channels, case.waiting_places, strict=True), 1):
        lines.extend([
            f"In{node}    TEST L S$Cap{node},{servers + waiting},Lost{node}",
            f"       ENTER Cap{node}",
            f"       QUEUE Queue{node}",
            f"       ENTER Srv{node}",
            f"       DEPART Queue{node}",
            "       ASSIGN 3,M1",
        ])
        stream = 3 if node == 1 else 5
        join = ""
        if case.service == "h2":
            join = append_mixture(lines, f"Svc{node}", stream+1, stream, *mix_service(case.service_time))
        elif case.service == "e2":
            for generator in (stream, stream+1):
                lines.append(f"       ADVANCE ({exp_duration(generator, case.service_time / 2)})")
        else:
            lines.append(f"       ADVANCE ({exp_duration(stream, case.service_time)})")
        lines.extend([
            f"{join:<12}TABULATE WTall",
            "       TABULATE STall",
            f"       LEAVE Srv{node}",
            f"       LEAVE Cap{node}",
            f"Done{node}  TERMINATE 1",
            f"Lost{node}  TERMINATE 1",
        ])
    lines.append("")
    if case.warmup:
        lines.extend([f"       START {case.warmup},NP", "       RESET"])
    lines.append(f"       START {case.observations}")
    return "\n".join(lines) + "\n"

def main() -> None:
    trace, original = load_trace()
    trace_mean = mean(trace)
    cases = make_experiments(trace_mean)
    MODELS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for case in cases:
        (MODELS / f"{case.name}.gpss").write_text(create_model(case, trace), encoding="ascii", newline="\r\n")
    manifest = {
        "trace": {
            "variant": 207,
            "count": len(trace),
            "uir1_mean": trace_mean,
            "uir1_cv": stdev(trace) / trace_mean,
            "decimal_mean": mean(original),
            "decimal_cv": stdev(original) / mean(original),
        },
        "seeds": SEEDS,
        "experiments": [asdict(case) for case in cases],
    }
    (OUT / "experiments.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(cases)} models in {MODELS}")


if __name__ == "__main__":
    main()
