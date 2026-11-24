"""Generate native GPSS World campaigns using PLUS and direct file I/O."""

from __future__ import annotations

import argparse
import csv
import io
from dataclasses import replace
from pathlib import Path
import shutil
from statistics import mean

from gpss_runner import TEMPLATE
from part1_models import H2_M1, H2_M2, H2_P, SEEDS, load_trace, make_experiments
from part2 import CAPACITY, canonical_scenarios, paired_scenarios, render_model


HEADER_1 = [
    "case", "replication", "checkpoint", "duration", "served", "lost", "busy",
    "queue_length", "wait", "wait_sd", "stay", "stay_sd", "wait_zero", "max_waiting",
]
HEADER_2 = [
    "case", "replication", "arrival_rate", "duration", "completed", "waiting", "waiting_sd",
    "residence", "residence_sd",
] + [
    f"{field}_u{i}" for i in range(1, 5)
    for field in ("utilization", "queue", "waiting", "waiting_sd", "residence", "residence_sd")
]


def p1_model(channels, trace):
    sc = H2_P * H2_M1 + (1 - H2_P) * H2_M2
    lines = [
        "* UIR3, GPSS built-in experiment. Model times are in seconds.",
        "TraceFn FUNCTION P1,L300",
    ]
    pairs = [f"{i},{number:.10g}" for i, number in enumerate(trace, 1)]
    lines.extend("/".join(pairs[i:i+6]) for i in range(0, len(pairs), 6))
    for j, count in enumerate(channels, 1):
        lines.extend([f"Cap{j} STORAGE 1024", f"Srv{j} STORAGE {count}"])
    lines.extend([
        "WTall TABLE P3,0,5,101",
        "STall TABLE M1,0,5,101",
        "       GENERATE ,,,1",
        "TraceReset ASSIGN 1,1",
        "NextGap TEST E X$ArrLaw,1,GapTrace",
        "       ADVANCE (Exponential(2,0,X$InputMean))",
        "       TRANSFER ,Arrive",
        "GapTrace TEST E X$ArrLaw,2,GapH2",
        "       ADVANCE FN$TraceFn",
        "       TRANSFER ,Arrive",
        "GapH2 TEST E X$ArrLaw,3,GapUniform",
        f"       TEST L (Uniform(7,0,1)),{H2_P:.8g},GapShort",
        f"       ADVANCE (Exponential(2,0,X$InputMean#{H2_M1/sc:.10g}))",
        "       TRANSFER ,Arrive",
        f"GapShort ADVANCE (Exponential(2,0,X$InputMean#{H2_M2/sc:.10g}))",
        "       TRANSFER ,Arrive",
        "GapUniform ADVANCE (Uniform(2,0,X$InputMean#2))",
        "Arrive SPLIT 1,Route",
        "       ASSIGN 1+,1",
        "       TEST LE P1,300,TraceReset",
        "       TRANSFER ,NextGap",
        "Route MARK",
    ])
    if len(channels) > 1:
        lines.append("       TRANSFER .75,In2,In1")
    for j in range(1, len(channels) + 1):
        stream = 1 + 2*j
        lines.extend([
            f"In{j} TEST L S$Cap{j},X$Limit{j},Lost{j}",
            f"       ENTER Cap{j}",
            f"       QUEUE Queue{j}",
            f"       ENTER Srv{j}",
            f"       DEPART Queue{j}",
            "       ASSIGN 3,M1",
            f"       TEST E X$SvcLaw,1,SvcE{j}",
            f"       ADVANCE (Exponential({stream},0,X$ServiceMean))",
            f"       TRANSFER ,SvcDone{j}",
            f"SvcE{j} TEST E X$SvcLaw,2,SvcMix{j}",
            f"       ADVANCE (Exponential({stream},0,X$ServiceMean/2))",
            f"       ADVANCE (Exponential({stream+1},0,X$ServiceMean/2))",
            f"       TRANSFER ,SvcDone{j}",
            f"SvcMix{j} TEST L (Uniform({stream+1},0,1)),0.8872983346,SvcLong{j}",
            f"       ADVANCE (Exponential({stream},0,X$ServiceMean/1.774596669))",
            f"       TRANSFER ,SvcDone{j}",
            f"SvcLong{j} ADVANCE (Exponential({stream},0,X$ServiceMean/0.2254033308))",
            f"SvcDone{j} TEST E P3,0,Measure{j}",
            f"ZeroWait{j} SAVEVALUE WaitZeroDummy,1",
            f"Measure{j} TABULATE WTall",
            "       TABULATE STall",
            f"       LEAVE Srv{j}",
            f"       LEAVE Cap{j}",
            f"Done{j}  TERMINATE 1",
            f"Lost{j}  TERMINATE 1",
        ])
    return lines


def p2_model(warmup, duration):
    sample = next(case for case in canonical_scenarios()[0] if case.service2 == "h2")
    sample = replace(sample, population=1)
    full = render_model(sample, warmup=warmup, duration=duration)
    head, _, _ = full.rpartition("                START 1,NP\n")
    substitutions = (
        ("                GENERATE ,,,1\nCycle", """                GENERATE (X$IsExp#Exponential(6,0,1+X$ExpMean)+10000000#(1-X$IsExp))
                TRANSFER ,Cycle
                GENERATE (1+X$DetMean)
                TRANSFER ,Cycle
                GENERATE ,,,1
                TEST G X$Pop,0,DropJob
                TEST G X$Pop,1,Cycle
                SPLIT (X$Pop-1)
Cycle"""),
        ("                TEST L (Uniform(7,0,1))", """                TEST E X$Law,0,NodeU2Option
                ADVANCE (Exponential(3,0,2.5))
                TRANSFER ,N2Done
NodeU2Option    TEST E X$Law,1,NodeU2H2
                ADVANCE (Exponential(3,0,1.25))
                ADVANCE (Exponential(3,0,1.25))
                TRANSFER ,N2Done
NodeU2H2        TEST L (Uniform(7,0,1))"""),
        ("                ADVANCE (Exponential(4,0,14))",
         "                ADVANCE (Exponential(4,0,X$MeanNodeThree))"),
        ("                TABULATE WaitNet\n                TRANSFER ,Cycle",
         "                TABULATE WaitNet\n                TEST G X$Pop,0,DropJob\n"
         "                TRANSFER ,Cycle\nDropJob         TERMINATE"),
    )
    for old, new in substitutions:
        if old not in head:
            raise ValueError(f"Missing native statement: {old}")
        head = head.replace(old, new, 1)
    # GENERATE primes before CONDUCT dispatch.
    defaults = [
        "                INITIAL X$IsExp,0",
        "                INITIAL X$ExpMean,9",
        "                INITIAL X$DetMean,9999999",
        "                INITIAL X$MeanNodeThree,14",
    ]
    return defaults + head.splitlines()


def do(statement):
    return f'  DoCommand("{statement}");'


def export_procedure(name, parameters, fields, prologue=(), temporary="Result_Line"):
    # A line is assembled a field at a time to keep GPSS source lines short.
    rows = [
        f'PROCEDURE {name}({",".join(parameters)}) BEGIN', f"  TEMPORARY {temporary};",
        *prologue, f'  Result_Line = Catenate({fields[0]},",");',
    ]
    for expression in fields[1:-1]:
        rows.append(f'  Result_Line = PolyCatenate(Result_Line,{expression},",");')
    rows.extend([
        f"  Result_Line = Catenate(Result_Line,{fields[-1]});",
        "  RETURN Write(97,Result_Line);",
        "END;",
    ])
    return rows


def p1_export(channels):
    nodes = range(1, len(channels)+1)
    def s(sna, item):
        return "+".join(f"{sna}\x24{item}{i}" for i in nodes)

    setup = ["  Max_Queue = QM$Queue1;"]
    if len(channels) > 1:
        setup.append("  IF (QM$Queue2 > Max_Queue) THEN Max_Queue = QM$Queue2;")
    fields = [
        "Case_Name", "Replication_Index", "Checkpoint_Count", "C1",
        s("N", "Done"), s("N", "Lost"), s("SA", "Srv"), s("QA", "Queue"),
        "TB$WTall", "TD$WTall", "TB$STall", "TD$STall", s("N", "ZeroWait"), "Max_Queue",
    ]
    return export_procedure("ExportResults", ("Case_Name", "Replication_Index", "Checkpoint_Count"), fields, setup, "Result_Line,Max_Queue")


def p2_export():
    fields = [
        "Case_Name", "Replication_Index", "Native_Rate", "C1", "TC$Residence", "TB$WaitNet",
        "TD$WaitNet", "TB$Residence", "TD$Residence",
    ]
    for j in range(1, 5):
        fields.extend([f"SR$StoreU{j}/1000", f"QA$QueueU{j}", f"TB$Delay{j}", f"TD$Delay{j}",
                       f"TB$Visit{j}", f"TD$Visit{j}"])
    return export_procedure("ExportResults", ("Case_Name", "Replication_Index", "Native_Rate"), fields)


def p1_commands(cases, repeat_max, checkpoints):
    cmds = []
    for case in cases:
        for rep in range(min(repeat_max, case.replicates)):
            cmds += [
                f"  /* {case.name} run {rep+1} */",
                do("RMULT " + ",".join(str(seed) for seed in SEEDS[rep])),
                do("CLEAR OFF"),
                do("INITIAL X$ArrLaw," + str({"exp": 1, "trace": 2, "h2": 3, "uniform": 4}[case.incoming])),
                do(f"INITIAL X$InputMean,{case.gap:.10g}"),
                do("INITIAL X$SvcLaw," + str({"exp": 1, "e2": 2, "h2": 3}[case.service])),
                do(f"INITIAL X$ServiceMean,{case.service_time:.10g}"),
            ]
            cmds += [
                do(f"INITIAL X$Limit{i},{count+capacity}")
                for i, (count, capacity) in enumerate(zip(case.channels, case.waiting_places, strict=True), 1)
            ]
            elapsed = 0
            if not rep and checkpoints:
                for n in case.checkpoints:
                    cmds.extend([
                        do(f"START {n-elapsed},NP"),
                        f'  File_Code = ExportResults("{case.name}",{rep+1},{n});',
                    ])
                    elapsed = n
            if case.warmup > elapsed:
                cmds.append(do(f"START {case.warmup-elapsed},NP"))
            cmds.extend([
                do("RESET"), do(f"START {case.observations},NP"),
                f'  File_Code = ExportResults("{case.name}",{rep+1},0);',
            ])
    return cmds


def p2_commands(cases, repeat_max):
    cmds = []
    for case in cases:
        for rep in range(repeat_max):
            seeded = case.seed + 1000003 * rep
            exp_input = bool(case.arrival_rate) and case.arrival == "exp"
            det_mean = (1/case.arrival_rate) if case.arrival == "det" and case.arrival_rate else 10_000_000
            exp_mean = (1/case.arrival_rate) if case.arrival_rate else 10
            cmds.extend([
                f"  /* {case.name} run {rep+1} */",
                do("RMULT " + ",".join(str(seeded+64007*(i+1)) for i in range(7))),
                do(f"INITIAL X$Pop,{case.population}"),
                do(f"INITIAL X$IsExp,{int(exp_input)}"),
                do(f"INITIAL X$ExpMean,{exp_mean-1:.10g}"),
                do(f"INITIAL X$DetMean,{det_mean-1:.10g}"),
                do(f"INITIAL X$MeanNodeThree,{case.service3:g}"),
                do("INITIAL X$Law," + str({"exp": 0, "erl": 1, "h2": 2}[case.service2])),
                do("CLEAR OFF"),
                do(f"StoreU3 STORAGE {case.servers3}"),
                do("START 1,NP"),do("RESET"),do("START 1,NP"),
                f'  File_Code = ExportResults("{case.name}",{rep+1},"{case.arrival_rate:.10g}");',
            ])
    return cmds


def stage(case_dir, name, body, procedure, header, commands):
    target = case_dir / name
    target.mkdir(parents=True, exist_ok=True)
    if (target / "native_results.csv").exists():
        raise FileExistsError(f"Choose a fresh workspace to preserve the existing export: {target}")
    shutil.copyfile(TEMPLATE, target / f"{name}.gps")
    lines = body + [""] + procedure
    chunks = []
    pending = []
    for line in commands:
        if line.startswith("  /*") and len(pending) >= 120:
            chunks.append(pending)
            pending = []
        pending.append(line)
    if pending:
        chunks.append(pending)
    for number, chunk in enumerate(chunks, 1):
        lines += ["", f"PROCEDURE Campaign_Segment_{number}() BEGIN", "  TEMPORARY File_Code;", *chunk, "END;"]
    lines += [
        "", "EXPERIMENT RunCampaign() BEGIN", "  TEMPORARY File_Code,Header_Line;",
        '  File_Code = Open(97,"native_results.csv");',
    ]
    # Group CSV headers into short strings for the native parser.
    lines.append(f'  Header_Line = "{",".join(header[:6])}";')
    for start in range(6, len(header), 6):
        fields = ",".join(header[start:start+6])
        lines.append(f'  Header_Line = PolyCatenate(Header_Line,",","{fields}");')
    lines.append('  File_Code = Write(97,Header_Line);')
    lines += [f"  Campaign_Segment_{number}();" for number in range(1, len(chunks)+1)]
    lines += [
        '  File_Code = Write(97,"COMPLETED");', "  File_Code = Close(97);",
        "END;", "* Run CONDUCT RunCampaign() once GPSS displays Ready.",
    ]
    (target / "model.txt").write_text("\n".join(lines)+"\n", encoding="ascii", newline="\r\n")
    return target / f"{name}.gps"


def paired_native_rates(path: Path, population: int) -> dict[str, float]:
    records = path.read_text(encoding="ascii").splitlines()
    if not records or records[-1] != "COMPLETED":
        raise ValueError(f"Incomplete native GPSS export: {path}")
    rows = list(csv.DictReader(io.StringIO("\n".join(records[:-1]))))
    rates = {}
    for service, prefix in (("exp", "closed"), ("erl", "closed_erl"), ("h2", "closed_h2")):
        selected = [row for row in rows if row["case"] == f"{prefix}_m{population:02d}"]
        if not selected:
            raise ValueError(f"No native closed observations of {service} at population {population}")
        exposure = sum(float(row["duration"]) for row in selected)
        count = sum(float(row["completed"]) for row in selected)
        if exposure <= 0 or count <= 0:
            raise ValueError(f"Invalid closed observations of {service}")
        rate = count / exposure
        if not 0 < rate < CAPACITY:
            raise ValueError(f"Matched open run would be unstable for {service}: {rate:g}")
        rates[service] = rate
    return rates

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part", choices=("part1", "part2"))
    parser.add_argument("destination", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument("--warmup", type=int, default=25000)
    parser.add_argument("--duration", type=int, default=250000)
    pairing = parser.add_mutually_exclusive_group()
    pairing.add_argument("--paired", type=float)
    pairing.add_argument("--paired-from", type=Path, help="Measured GPSS CSV for the closed network")
    args = parser.parse_args()
    if not 1 <= args.repetitions <= len(SEEDS):
        parser.error("repetitions is out of range")
    if args.part == "part1":
        trace, _ = load_trace()
        cases = make_experiments(mean(trace))
        for channels in dict.fromkeys(case.channels for case in cases):
            selected = [
                case for case in cases if case.channels == channels
                and (not args.case or case.name in args.case)
            ]
            if selected:
                name = "system_" + "_".join(map(str, channels))
                print(stage(args.destination, name, p1_model(channels, trace), p1_export(channels),
                            HEADER_1, p1_commands(selected, args.repetitions, not args.no_checkpoints)))
    else:
        base, threshold = canonical_scenarios()
        if args.paired_from:
            rates = paired_native_rates(args.paired_from, threshold)
            base = paired_scenarios(threshold, rates)
        elif args.paired is not None:
            base = paired_scenarios(threshold, dict.fromkeys(("exp", "erl", "h2"), args.paired))
        cases = [case for case in base if not args.case or case.name in args.case]
        if cases:
            print(stage(args.destination, "network", p2_model(args.warmup, args.duration), p2_export(),
                        HEADER_2,p2_commands(cases, args.repetitions)))


if __name__ == "__main__":
    main()

