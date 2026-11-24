"""Independent event simulation for validating Part 1 GPSS World results."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field
import heapq
import json
import math
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from part1_models import Experiment, H2_M1, H2_M2, H2_P, OUT, SEEDS, load_trace, make_experiments


BASELINE_MARKOV = {
    "busy": 1.516814197226568,
    "throughput": 0.3033628394453136,
    "loss_probability": 0.4943952675911439,
    "queue_length": 1.6564073986754394,
    "wait": 5.460152607036879,
    "stay": 10.46015260703688,
}


@dataclass
class JobMoments:
    n: int = 0
    positive: int = 0
    total: float = 0.0
    square: float = 0.0
    positive_total: float = 0.0
    positive_square: float = 0.0

    def add(self, value):
        self.n += 1
        self.total += value
        self.square += value * value
        if value > 0:
            self.positive += 1
            self.positive_total += value
            self.positive_square += value * value

    def average(self):
        return self.total / self.n if self.n else 0.0

    def std(self):
        return standard_deviation(self.n, self.total, self.square)

    def positive_cv(self):
        if not self.positive:
            return 0.0
        return standard_deviation(self.positive, self.positive_total, self.positive_square) / (
            self.positive_total / self.positive
        )


def standard_deviation(n: int, total: float, square: float) -> float:
    return math.sqrt(max(0.0, (square - total * total / n) / (n - 1))) if n > 1 else 0.0


@dataclass
class Period:
    start: float = 0.0
    count: int = 0
    lost: int = 0
    waits: JobMoments = field(default_factory=JobMoments)
    stays: JobMoments = field(default_factory=JobMoments)
    queue_area: float = 0.0
    busy_area: float = 0.0

    def depart(self, arrival: float, service_start: float, departure: float):
        self.count += 1
        self.waits.add(service_start - arrival)
        self.stays.add(departure - arrival)
        self.queue_area += clipped_duration(arrival, service_start, self.start, departure)
        self.busy_area += clipped_duration(service_start, departure, self.start, departure)

    def lose(self):
        self.count += 1
        self.lost += 1

    def metrics(self, clock, pending, maximum):
        duration = clock - self.start
        if duration <= 0:
            raise ValueError("A measurement period must have positive duration")
        queue_area = self.queue_area
        busy_area = self.busy_area
        for departure, _, service_start, arrival in pending:
            queue_area += clipped_duration(arrival, service_start, self.start, clock)
            busy_area += clipped_duration(service_start, departure, self.start, clock)
        average_wait = self.waits.average()
        average_stay = self.stays.average()
        return {
            "count": self.count,
            "lost": self.lost,
            "served": self.waits.n,
            "duration": duration,
            "busy": busy_area / duration,
            "throughput": self.waits.n / duration,
            "loss_probability": self.lost / self.count,
            "queue_length": queue_area / duration,
            "wait": average_wait,
            "wait_sd": self.waits.std(),
            "wait_cv": self.waits.std() / average_wait if average_wait else 0.0,
            "positive_wait_cv": self.waits.positive_cv(),
            "wait_zero": 1 - self.waits.positive / self.waits.n if self.waits.n else 0.0,
            "stay": average_stay,
            "stay_sd": self.stays.std(),
            "stay_cv": self.stays.std() / average_stay if average_stay else 0.0,
            "max_waiting": maximum,
        }


def clipped_duration(begin: float, end: float, left: float, right: float) -> float:
    return max(0.0, min(right, end) - max(left, begin))


def h2_scales(mean_service: float):
    prob = (1 + math.sqrt(3 / 5)) / 2
    return prob, mean_service / (2 * prob), mean_service / (2 * (1 - prob))


def random_durations(generator, chooser, law, average, amount, for_arrivals=False):
    if law == "exp":
        return generator.exponential(average, amount).tolist()
    if law == "uniform":
        return generator.uniform(0, 2 * average, amount).tolist()
    if law == "e2":
        return (generator.exponential(average / 2, amount) +
                chooser.exponential(average / 2, amount)).tolist()
    if law == "h2":
        if for_arrivals:
            prob, m1, m2 = H2_P, H2_M1, H2_M2
            adjustment = average / (prob * m1 + (1 - prob) * m2)
            m1, m2 = m1 * adjustment, m2 * adjustment
        else:
            prob, m1, m2 = h2_scales(average)
        factors = np.where(chooser.random(amount) < prob, m1, m2)
        return (generator.standard_exponential(amount) * factors).tolist()
    raise ValueError(f"Unknown law: {law}")


def simulate(case: Experiment, seed_set: tuple[int, ...], trace: list[float], capture: bool):
    target = case.warmup + case.observations
    capacity = [n + e for n, e in zip(case.channels, case.waiting_places, strict=True)]
    rng = [np.random.default_rng(seed) for seed in seed_set]
    amount = target + sum(capacity) + 1
    if case.incoming != "trace":
        gaps = random_durations(rng[1], rng[6], case.incoming, case.gap, amount, True)
    routes = (rng[0].random(amount) >= 0.75).tolist()
    durations = [
        random_durations(rng[2 + 2 * node], rng[3 + 2 * node], case.service, case.service_time, amount)
        for node in range(len(case.channels))
    ]
    index = [0] * len(capacity)
    wip = [0] * len(capacity)
    free = [[0.0] * n for n in case.channels]
    pending = []
    maximum = 0
    total_lost = 0
    period = Period()
    checkpoints = set(case.checkpoints if capture else ())
    transition = []
    total = 0
    arrival = 0.0

    def record(clock):
        nonlocal period, total
        total += 1
        if total in checkpoints:
            transition.append({"checkpoint": total, **period.metrics(clock, pending, maximum)})
        if total == case.warmup:
            period = Period(start=clock)
        if total == target:
            return {**period.metrics(clock, pending, maximum), "lost_including_warmup": total_lost}
        return None

    for i in range(amount):
        gap = trace[i % len(trace)] if case.incoming == "trace" else gaps[i]
        arrival += gap
        while pending and pending[0][0] <= arrival:
            departure, node, service_start, born = heapq.heappop(pending)
            wip[node] -= 1
            period.depart(born, service_start, departure)
            final = record(departure)
            if final is not None:
                return final, transition

        node = int(routes[i]) if len(capacity) == 2 else 0
        if wip[node] == capacity[node]:
            period.lose()
            total_lost += 1
            final = record(arrival)
            if final is not None:
                return final, transition
            continue

        queue = free[node]
        service_start = max(arrival, queue[0])
        departure = service_start + durations[node][index[node]]
        heapq.heapreplace(queue, departure)
        index[node] += 1
        wip[node] += 1
        if service_start > arrival:
            maximum = max(maximum, wip[node] - case.channels[node])
        heapq.heappush(pending, (departure, node, service_start, arrival))

    raise RuntimeError(f"Insufficient planned arrivals for {case.name}")


def collect(cases, trace):
    measured, transition = [], []
    for case in cases:
        for replicate in range(case.replicates):
            stats, samples = simulate(case, SEEDS[replicate], trace, capture=replicate == 0)
            measured.append({"case": case.name, "replicate": replicate + 1, **stats})
            transition.extend({"case": case.name, **row} for row in samples)
        print(case.name, flush=True)
    return measured, transition


METRICS = (
    "busy", "throughput", "loss_probability", "queue_length", "wait", "wait_sd",
    "wait_cv", "positive_wait_cv", "wait_zero", "stay", "stay_sd", "stay_cv",
)


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(cases, measured):
    multiplier = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
    output = {}
    for case in cases:
        group = [row for row in measured if row["case"] == case.name]
        size = len(group)
        summary = {}
        for metric in METRICS:
            observations = [row[metric] for row in group]
            center = mean(observations)
            halfwidth = multiplier[size] * stdev(observations) / math.sqrt(size) if size > 1 else None
            summary[metric] = {"mean": center, "ci95_halfwidth": halfwidth}
        summary["observed_lost"] = sum(row["lost"] for row in group)
        if "lost_including_warmup" in group[0]:
            summary["lost_including_warmup"] = sum(row["lost_including_warmup"] for row in group)
        summary["max_waiting"] = max(row["max_waiting"] for row in group)
        output[case.name] = {"parameters": asdict(case), "metrics": summary}
    return output


def stability_check(case, measurements, reference, loss_absolute=0.005, wait_relative=0.05):
    mean_loss = reference["loss_probability"]["mean"]
    mean_wait = reference["wait"]["mean"]
    max_wait_deviation = max(0.02, wait_relative * mean_wait)
    valid = [
        abs(m["loss_probability"] - mean_loss) <= loss_absolute
        and abs(m["wait"] - mean_wait) <= max_wait_deviation
        for m in measurements
    ]
    # Reaching the tolerance at one checkpoint alone does not establish stability.
    for index, row in enumerate(measurements):
        if len(measurements) - index >= 2 and all(valid[index:]):
            return row["checkpoint"]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", default=[], help="Use selected experiment names.")
    args = parser.parse_args()
    trace, original_trace = load_trace()
    cases = make_experiments(mean(trace))
    if args.case:
        cases = [case for case in cases if case.name in args.case]
        if {case.name for case in cases} != set(args.case):
            parser.error("Unknown experiment name")
    measured, transition = collect(cases, trace)
    # Verify the least tested lossless buffer and the immediately smaller one for
    # the same streams, within this experiment's finite horizon.
    exact_buffers = {}
    for load in (0.50, 0.90):
        large = next((case for case in cases if case.family == "buffer"
                      and case.offered_load == load and case.waiting_places == (256, 256)), None)
        if large is None:
            continue
        maximum = max(row["max_waiting"] for row in measured if row["case"] == large.name)
        probes = [
            Experiment(
                f"buffer_exact_{round(load * 100):03d}_{capacity:03d}", "buffer_exact",
                large.incoming, large.gap, large.service, large.service_time,
                large.channels, (capacity, capacity), large.offered_load,
            )
            for capacity in sorted({max(0, maximum - 1), maximum})
        ]
        cases.extend(probes)
        measurements, _ = collect(probes, trace)
        measured.extend(measurements)
        exact_buffers[round(load * 100)] = maximum
    summary = summarize(cases, measured)
    result = {
        "engine": "Independent Python discrete-event simulator",
        "markov": BASELINE_MARKOV,
        "empirical_lossless_capacities": exact_buffers,
        "experiments": summary,
        "stability": {
            case.name: stability_check(
                case, [row for row in transition if row["case"] == case.name],
                summary[case.name]["metrics"]
            )
            for case in cases if case.checkpoints
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.case:
        exact = [value * mean(trace)/mean(original_trace) for value in original_trace]
        case = next(case for case in cases if case.name == "flow_trace_075")
        sensitivity = []
        for label, values in (("uir1_first_column", trace), ("decimal_rescaled", exact)):
            for rep, seeds in enumerate(SEEDS, 1):
                metrics, _ = simulate(case, seeds, values, capture=False)
                sensitivity.append({"trace": label, "replicate": rep, **metrics})
        write_csv(OUT / "trace_sensitivity.csv", sensitivity)

    prefix = "partial_" if args.case else ""
    write_csv(OUT / f"{prefix}python_measurements.csv", measured)
    write_csv(OUT / f"{prefix}python_transient.csv", transition)
    (OUT / f"{prefix}python_results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
