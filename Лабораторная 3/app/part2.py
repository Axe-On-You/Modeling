"""GPSS World model and experiment generation for queueing network M3."""

from dataclasses import dataclass, replace
from itertools import combinations
import json
from math import factorial, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVERS = (3, 2, 1, 1)
MEANS = (4.0, 2.5, 14.0, 5.0)
VISITS = (4, 1, 1, 1)
CAPACITY = min(c / (v * b) for c, v, b in zip(SERVERS, VISITS, MEANS))
WARMUP = 100_000
DURATION = 1_000_000


@dataclass(frozen=True)
class Scenario:
    name: str
    population: int = 0
    arrival_rate: float = 0
    arrival: str = "exp"
    service2: str = "exp"
    servers3: int = 1
    service3: float = 14.0
    seed: int = 3


def populations(total, units=4):
    """Enumerate network states with a fixed number of jobs."""
    for cuts in combinations(range(total + units - 1), units - 1):
        boundaries = (-1, *cuts, total + units - 1)
        yield tuple(b - a - 1 for a, b in zip(boundaries, boundaries[1:]))


def closed_metrics(total, servers=SERVERS, means=MEANS):
    """Product-form benchmark for the exponential closed network."""
    weights = [[1.0] for _ in servers]
    for n in range(1, total + 1):
        for i, (v, b, c) in enumerate(zip(VISITS, means, servers)):
            weights[i].append(weights[i][-1] * (v * b / min(n, c)))

    def normalization(m):
        # Convolution avoids enumerating every occupancy vector.
        g = [1.0] + [0.0] * m
        for w in weights:
            g = [sum(g[j - k] * w[k] for k in range(j + 1)) for j in range(m + 1)]
        return g[m]

    g = normalization(total)
    throughput = normalization(total - 1) / g
    queues = [0.0] * len(servers)
    for state in populations(total):
        prob = 1.0
        for i, n in enumerate(state):
            prob *= weights[i][n]
        for i, n in enumerate(state):
            queues[i] += prob / g * max(0, n - servers[i])
    return throughput, total / throughput, tuple(q / (throughput * v) for q, v in zip(queues, VISITS))


def open_metrics(rate):
    """Jackson network benchmark with exponential arrivals and service."""
    waits = []
    for v, b, c in zip(VISITS, MEANS, SERVERS):
        a = rate * v * b
        rho = a / c
        if rho >= 1:
            raise ValueError("The open network is not stable at this arrival rate.")
        tail = a**c / (factorial(c) * (1 - rho))
        p_wait = tail / (sum(a**k / factorial(k) for k in range(c)) + tail)
        waits.append(p_wait * b / (c - a))
    waiting = sum(v * w for v, w in zip(VISITS, waits))
    return waiting, waiting + sum(v * b for v, b in zip(VISITS, MEANS))


def render_model(scenario, warmup=WARMUP, duration=DURATION):
    """Return the complete GPSS model. Node 0 denotes a network-cycle boundary."""
    if bool(scenario.population) == bool(scenario.arrival_rate):
        raise ValueError("Choose closed population or open arrival rate.")
    means = (4.0, 2.5, scenario.service3, 5.0)
    servers = (3, 2, scenario.servers3, 1)
    seeds = ",".join(str(scenario.seed + 64007 * (i + 1)) for i in range(7))
    lines = [
        f"* UIR 3, variant 41/3, M3, {scenario.name}",
        "* Stream 1 routes; streams 2-5 serve U1-U4; 6 arrives; 7 selects a phase.",
        f"                RMULT {seeds}",
        "* P2 is the node-arrival clock. P3 accumulates waiting over the cycle.",
    ]
    for i, channels in enumerate(servers, 1):
        lines.extend([
            f"StoreU{i}        STORAGE {channels}",
            f"Delay{i}          QTABLE QueueU{i},0,1,16",
            f"Visit{i}          TABLE MP2,0,5,12",
        ])
    lines.extend([
        "Residence       TABLE M1,0,25,18",
        "WaitNet         TABLE P3,0,25,18",
        "",
        "* Create the fixed population or generate external arrivals.",
    ])
    if scenario.population:
        lines.append(f"                GENERATE ,,,{scenario.population}")
    elif scenario.arrival == "det":
        lines.append(f"                GENERATE {1/scenario.arrival_rate:.10f}")
    else:
        lines.append(f"                GENERATE (Exponential(6,0,{1/scenario.arrival_rate:.10f}))")
    lines.extend(["Cycle           MARK", "                ASSIGN 3,0"])
    for i, mean in enumerate(means, 1):
        lines.extend([
            f"* Node {i}: FCFS queue and {servers[i-1]} parallel server(s).",
            f"NodeU{i}         MARK 2",
            f"                QUEUE QueueU{i}",
            f"                ENTER StoreU{i}",
            f"                DEPART QueueU{i}",
            "                ASSIGN 3+,MP2",
        ])
        stream = i + 1
        if i == 2 and scenario.service2 == "erl":
            lines.extend([f"                ADVANCE (Exponential(3,0,{mean/2}))"] * 2)
        elif i == 2 and scenario.service2 == "h2":
            prob = (1 - sqrt(3 / 5)) / 2
            lines.extend([
                f"                TEST L (Uniform(7,0,1)),{prob:.10f},N2Fast",
                f"                ADVANCE (Exponential(3,0,{mean/(2*prob):.10f}))",
                "                TRANSFER ,N2Done",
                f"N2Fast          ADVANCE (Exponential(3,0,{mean/(2*(1-prob)):.10f}))",
                "N2Done          EQU 0",
            ])
            # Attach the join label to LEAVE rather than defining a separate entity.
            lines[-1] = f"N2Done          LEAVE StoreU{i}"
        else:
            lines.append(f"                ADVANCE (Exponential({stream},0,{mean:g}))")
        if not (i == 2 and scenario.service2 == "h2"):
            lines.append(f"                LEAVE StoreU{i}")
        lines.append(f"                TABULATE Visit{i}")
        if i == 1:
            lines.extend([
                "* One draw routes with equal probabilities to 0, 2, 3, 4.",
                "                ASSIGN 1,RN1",
                "                TEST GE P1,250,NetExit",
                "                TEST GE P1,500,NodeU2",
                "                TEST GE P1,750,NodeU3",
                "                TRANSFER ,NodeU4",
            ])
        else:
            lines.append("                TRANSFER ,NodeU1")
    lines.extend([
        "NetExit         TABULATE Residence",
        "                TABULATE WaitNet",
        "                TRANSFER ,Cycle" if scenario.population else "                TERMINATE",
        "",
        "* Stop once for warm-up and once at the end of measurement.",
        f"                GENERATE {warmup},,,1",
        "                TERMINATE 1",
        f"                GENERATE {warmup+duration},,,1",
        "                TERMINATE 1",
        "                START 1,NP",
        "                RESET",
        "                START 1",
        "",
    ])
    return "\n".join(lines)


def canonical_scenarios():
    """Use the selected populations, perturbations and paired open rates."""
    base = [Scenario(f"closed_m{n:02d}", population=n) for n in range(1, 17)]
    threshold = next(s.population for s in base
                     if closed_metrics(s.population)[0] >= 0.98 * CAPACITY)
    for service in ("erl", "h2"):
        base.extend(Scenario(f"closed_{service}_m{n:02d}", population=n, service2=service)
                    for n in sorted({2, 4, 6, threshold, 16}))
    base.extend([
        Scenario("closed_faster_u3", population=threshold, service3=5.0),
        Scenario("closed_more_u3", population=threshold, servers3=3),
    ])
    base.extend(Scenario(f"open_rho{round(rho*100):02d}", arrival_rate=rho*CAPACITY)
                for rho in (0.4, 0.6, 0.8, 0.85, 0.9, 0.95))
    # Hold offered load constant when isolating the service and input laws.
    base.extend(
        Scenario(f"open_fixed_{service}_{arrival}", arrival_rate=0.90 * CAPACITY,
                 service2=service, arrival=arrival)
        for service in ("exp", "erl", "h2") for arrival in ("exp", "det")
        if (service, arrival) != ("exp", "exp")
    )
    return base, threshold


def paired_scenarios(threshold, rates):
    return [
        Scenario(
            f"open_match_{service}_{arrival}", arrival_rate=rates[service],
            service2=service, arrival=arrival
        )
        for service in ("exp", "erl", "h2") for arrival in ("exp", "det")
    ]


def write_models(scenarios, folder, **kwargs):
    folder.mkdir(parents=True, exist_ok=True)
    for scenario in scenarios:
        (folder / (scenario.name + ".gpss")).write_text(
            render_model(scenario, **kwargs), encoding="ascii"
        )


if __name__ == "__main__":
    base, threshold = canonical_scenarios()
    print(f"Theoretical threshold: {threshold}; capacity: {CAPACITY:.8f} jobs/s")
    write_models(
        [Scenario("closed", population=threshold), Scenario("open", arrival_rate=0.9*CAPACITY)],
        ROOT / "models" / "part2"
    )
    print("Representative models created.")
