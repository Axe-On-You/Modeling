"""Independently simulate M3 for statistical and conservation checks."""
from __future__ import annotations

import argparse
from collections import defaultdict
import heapq
import math
from pathlib import Path
import random

from part2 import CAPACITY, MEANS, ROOT, SERVERS, canonical_scenarios, paired_scenarios
from part2_results import compare_theory, compute_summary, draw, write_csv


class Times:
    def __init__(self):
        self.values = []

    def mean(self):
        return sum(self.values) / len(self.values) if self.values else 0.0

    def cv(self):
        average = self.mean()
        if len(self.values) < 2 or average == 0:
            return 0.0
        variance = sum((value - average)**2 for value in self.values) / (len(self.values) - 1)
        return math.sqrt(variance) / average

    def histogram(self, width, maximum):
        counts = [0] * (maximum // width + 2)
        for value in self.values:
            bin_number = min(math.ceil(value / width), len(counts) - 1)
            counts[bin_number] += 1
        return [
            (max(0, (index - 1) * width), index * width if index < len(counts) - 1 else None, count)
            for index, count in enumerate(counts) if count
        ]


def simulate(scenario, warmup, duration, seed):
    rng = random.Random(seed)
    routing = random.Random(seed + 252_715)
    input_rng = random.Random(seed + 495_337)
    means = (MEANS[0], MEANS[1], scenario.service3, MEANS[3])
    servers = (SERVERS[0], SERVERS[1], scenario.servers3, SERVERS[3])
    horizon = warmup + duration
    facilities = [[0.0] * count for count in servers]
    events = []
    sequence = 0
    waits = [Times() for _ in servers]
    stays = [Times() for _ in servers]
    wait_net, stay_net = Times(), Times()
    busy_area, queue_area = [0.0] * 4, [0.0] * 4
    phase_probability = (1 - math.sqrt(3/5))/2

    def interval():
        return (1 / scenario.arrival_rate if scenario.arrival == "det"
                else input_rng.expovariate(scenario.arrival_rate))

    def scheduled_service(node, born, network_wait, clock):
        nonlocal sequence
        queue = facilities[node]
        start = max(queue[0], clock)
        b = means[node]
        if node == 1 and scenario.service2 == "erl":
            service = rng.expovariate(2/b) + rng.expovariate(2/b)
        else:
            if node == 1 and scenario.service2 == "h2":
                probability = phase_probability if rng.random() < phase_probability else 1-phase_probability
                b /= 2 * probability
            service = rng.expovariate(1/b)
        end = start + service
        heapq.heapreplace(queue, end)
        sequence += 1
        heapq.heappush(events, (end, sequence, node, clock, start, born, network_wait + start - clock))
        queue_area[node] += max(0.0, min(start, horizon) - max(warmup, clock))
        busy_area[node] += max(0.0, min(end, horizon) - max(warmup, start))

    next_arrival = math.inf if scenario.population else interval()
    for _ in range(scenario.population):
        scheduled_service(0, 0.0, 0.0, 0.0)

    while next_arrival <= horizon or (events and events[0][0] <= horizon):
        if next_arrival <= horizon and (not events or next_arrival < events[0][0]):
            clock = next_arrival
            scheduled_service(0, clock, 0.0, clock)
            next_arrival += interval()
            continue
        clock, _, node, arrival, start, born, network_wait = heapq.heappop(events)
        if clock >= warmup:
            waits[node].values.append(start - arrival)
            stays[node].values.append(clock - arrival)
        destination = routing.randrange(4) if node == 0 else 0
        if node == 0 and destination == 0:
            if clock >= warmup:
                wait_net.values.append(network_wait)
                stay_net.values.append(clock - born)
            if scenario.population:
                scheduled_service(0, clock, 0.0, clock)
        else:
            scheduled_service(destination, born, network_wait, clock)

    if not stay_net.values:
        raise RuntimeError(f"No cycles completed: {scenario.name}")
    if scenario.population and len(events) != scenario.population:
        raise ValueError(f"Population conservation failed: {scenario.name}")
    result = {
        "population": scenario.population, "arrival_rate": scenario.arrival_rate,
        "arrival": scenario.arrival, "service2": scenario.service2,
        "servers3": scenario.servers3, "service3": scenario.service3, "duration": duration,
        "completed": len(stay_net.values), "throughput": len(stay_net.values) / duration,
        "waiting": wait_net.mean(), "residence": stay_net.mean(),
        "waiting_cv": wait_net.cv(), "residence_cv": stay_net.cv(),
    }
    histograms = {"wait": wait_net.histogram(25, 400), "time": stay_net.histogram(25, 400)}
    for i, (wait, stay) in enumerate(zip(waits, stays, strict=True), 1):
        result[f"utilization_u{i}"] = busy_area[i-1] / (duration * servers[i-1])
        result[f"queue_u{i}"] = queue_area[i-1] / duration
        result[f"waiting_u{i}"] = wait.mean()
        result[f"residence_u{i}"] = stay.mean()
        result[f"waiting_cv_u{i}"] = wait.cv()
        result[f"residence_cv_u{i}"] = stay.cv()
        histograms[f"wait_u{i}"] = wait.histogram(1, 14)
        histograms[f"time_u{i}"] = stay.histogram(5, 50)
    return result, histograms


def run(output, repeats=3, warmup=25000, duration=250000):
    scenarios, threshold = canonical_scenarios()
    all_runs, histograms = defaultdict(list), defaultdict(list)

    def collect(group):
        for case in group:
            for rep in range(1, repeats+1):
                warm = max(warmup, 100000) if case.arrival_rate >= .95 * CAPACITY else warmup
                measure = max(duration, 800000) if case.arrival_rate >= .95 * CAPACITY else duration
                values, hist = simulate(case, warm, measure, 232189 + 1000003 * rep)
                all_runs[case.name].append({"replication": rep, "warmup": warm, **values})
                histograms[case.name].append(hist)
            print(case.name, flush=True)

    collect(scenarios)
    base = compute_summary(all_runs)
    rates = {
        law: base[f"closed_{tag}m{threshold:02d}"]["throughput"]
        for law, tag in [("exp", ""), ("erl", "erl_"), ("h2", "h2_")]
    }
    if not all(0 < rate < CAPACITY for rate in rates.values()):
        raise ValueError("Matched open load must be stable for every service law")
    collect(paired_scenarios(threshold, rates))
    observed = compute_summary(all_runs)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "replications.csv", (
        {"group": name, **row} for name, runs in all_runs.items() for row in runs
    ))
    write_csv(output / "metrics.csv", ({"group": name, **v} for name, v in observed.items()))
    write_csv(output / "analytical_check.csv", compare_theory(observed))
    draw(observed, histograms, output, threshold, result_label="Событийный расчёт")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "out" / "part2" / "independent")
    parser.add_argument("--replications", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=25000)
    parser.add_argument("--duration", type=int, default=250000)
    args = parser.parse_args()
    if min(args.replications, args.warmup, args.duration) < 1:
        parser.error("Replication count and horizons must be positive")
    run(args.output, args.replications, args.warmup, args.duration)

