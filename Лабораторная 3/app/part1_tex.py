"""Generate the report's Part 1 tables from checked GPSS World measurements."""
from __future__ import annotations

import json

from part1_models import OUT


FLOW_NAMES = (
    ("exp", "Простейший"),
    ("trace", "Трасса"),
    ("h2", r"\(H_2\) входа"),
    ("uniform", "Равномерный"),
)
SERVICE_NAMES = (
    ("e2", r"\(E_2\)"),
    ("h2", r"\(H_2\) обслуживания"),
)


def decimal(value, digits=2):
    return f"{value:.{digits}f}".replace(".", ",")


def integer(value):
    return f"{value:,d}".replace(",", r"\,")


def value(record, metric):
    return record["metrics"][metric]["mean"]


def row(*cells):
    return " & ".join(cells) + r" \\"


def time_pair(record, metric):
    return decimal(value(record, metric)), decimal(value(record, f"{metric}_sd"))


def flow_service_rows(cases):
    selected = [(label, cases[f"flow_{law}_075"]) for law, label in FLOW_NAMES]
    selected += [(label, cases[f"service_{law}_075"]) for law, label in SERVICE_NAMES]
    return [
        row(label, decimal(value(case, "busy")), integer(case["metrics"]["observed_lost"]),
            decimal(100 * value(case, "loss_probability")), *time_pair(case, "wait"),
            *time_pair(case, "stay"))
        for label, case in selected
    ]


def stability_rows(stability):
    loads = (("050", "0,50"), ("075", "0,75"), ("095", "0,95"), ("099", "0,99"))
    return [row(label, *(checkpoint(stability[f"flow_{law}_{code}"]) for law, _ in FLOW_NAMES[:3]))
            for code, label in loads]


def checkpoint(count):
    return integer(count) if count else "---"


def fixed_load_stability_rows(stability):
    loads = (("010", "0,10"), ("030", "0,30"), ("070", "0,70"), ("090", "0,90"))
    return [row(label, checkpoint(stability[f"transient_{code}"]),
                checkpoint(stability[f"transient_gap_{code}"])) for code, label in loads]


def buffer_rows(cases):
    lines = []
    for load in (0.50, 0.90):
        runs = [case for case in cases.values() if case["parameters"]["family"] == "buffer"
                and case["parameters"]["offered_load"] == load]
        runs.sort(key=lambda case: case["parameters"]["waiting_places"][0])
        first_zero = next((index for index, case in enumerate(runs)
                           if case["metrics"]["observed_lost"] == 0), len(runs))
        for index, case in enumerate(runs):
            capacity = case["parameters"]["waiting_places"][0]
            if capacity not in {0, 8} and index not in {first_zero - 1, first_zero}:
                continue
            lines.append(row(decimal(load), str(capacity),
                             integer(case["metrics"]["observed_lost"]),
                             decimal(100 * value(case, "loss_probability"), 3),
                             decimal(value(case, "wait")), decimal(value(case, "stay"))))
    return lines


def server_rows(cases):
    lines = []
    for family, label in (("work", "Среднее обслуживание"), ("load", "Загрузка первой ветви")):
        for count in (1, 2, 3):
            case = cases["flow_exp_075" if count == 2 else f"servers_{family}_075_{count}"]
            params = case["parameters"]
            lines.append(row(label if count == 1 else "", str(count),
                             decimal(params["offered_load"]), decimal(params["service_time"]),
                             decimal(value(case, "wait")), decimal(value(case, "stay"))))
    return lines


def control_rows(cases):
    lines = []
    for service, cv, name in (("exp", 1, "M/M/1"), ("h2", 2, "M/G/1")):
        for load in ("050", "075"):
            case = cases[f"control_{service}_{load}"]
            params = case["parameters"]
            rho = params["offered_load"]
            b = params["service_time"]
            wait = rho * b * (1 + cv**2) / (2 * (1 - rho))
            gp_wait = value(case, "wait")
            lines.append(row(name, decimal(rho), decimal(wait), decimal(gp_wait),
                             decimal(100 * (gp_wait / wait - 1)), decimal(wait + b),
                             decimal(value(case, "stay")), integer(case["metrics"]["observed_lost"])))
    return lines


def variation_rows(cases):
    return [row(label, *(decimal(value(cases[f"flow_{law}_075"], metric) * scale)
                          for metric, scale in (("wait_zero", 100), ("wait_cv", 1),
                                                ("positive_wait_cv", 1), ("stay_cv", 1))))
            for law, label in FLOW_NAMES]


def main():
    source = OUT / "gpss_results.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    if data["engine"] != "GPSS World Student":
        raise ValueError("Part 1 report requires GPSS World measurements")
    cases, stability = data["experiments"], data["stability"]
    tables = {
        "flow_service": flow_service_rows(cases),
        "transition": stability_rows(stability),
        "transition_load": fixed_load_stability_rows(stability),
        "buffers": buffer_rows(cases),
        "servers": server_rows(cases),
        "controls": control_rows(cases),
        "variation": variation_rows(cases),
    }
    folder = OUT / "gpss"
    folder.mkdir(exist_ok=True)
    for name, lines in tables.items():
        (folder / f"{name}_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
