"""Prepare GPSS World Model Objects and extract their Standard Reports."""

import argparse
from decimal import Decimal
from pathlib import Path
import re
import shutil

TEMPLATE = Path(__file__).with_name("batch_template.gps")
REPORT_NAME = re.compile(r"(.+)\.(\d+)\.(\d+)(?:\.gpr)?", re.IGNORECASE)
REPORT_HEADER = b"GPSS World Simulation Report"
NUMBER = rb"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
REPORT_TIMES = re.compile(
    rb"START TIME[ \t]+END TIME[ \t]+BLOCKS[ \t]+FACILITIES[ \t]+STORAGES[ \t]*\r?\n"
    rb"[ \t]*" + NUMBER + rb"[ \t]+" + NUMBER + rb"[ \t]+\d+",
)
RTF_TOKEN = re.compile(
    rb"\\'([0-9a-fA-F]{2})|\\([a-zA-Z]+)(-?\d+)? ?|\\([{}\\])|[{}]|[\r\n]+"
)


def report_text(path):
    data = path.read_bytes()
    if data.lstrip().startswith(REPORT_HEADER):
        return data
    start = data.find(br"{\rtf")
    end = data.find(b"\x00", start)
    if end >= 0:
        data = data[:end]
    header = data.find(REPORT_HEADER, start)
    if start < 0 or header < 0:
        raise ValueError(f"Unexpected GPSS report format: {path.name}")

    # GPSS stores its fixed-width report columns as RichEdit paragraphs.
    text = bytearray()
    position = header
    for token in RTF_TOKEN.finditer(data, header):
        text.extend(data[position:token.start()])
        hex_byte, control, _, escaped_symbol = token.groups()
        if hex_byte:
            text.extend(bytes.fromhex(hex_byte.decode("ascii")))
        elif control in {b"par", b"line"}:
            text.extend(b"\n")
        elif control == b"tab":
            text.extend(b"\t")
        elif escaped_symbol:
            text.extend(escaped_symbol)
        position = token.end()
    text.extend(data[position:])
    return bytes(text).rstrip() + b"\n"


def validate_report(path, text, native_stem, simulation, serial):
    title = text.splitlines()[0].strip()
    expected_title = f" - {native_stem}.{simulation}.{serial}".encode("ascii")
    if not title.endswith(expected_title):
        raise ValueError(f"Report title does not match its native file: {path.name}")
    match = REPORT_TIMES.search(text)
    if not match or b"BLOCK TYPE" not in text or b"ENTRY COUNT" not in text:
        raise ValueError(f"Missing Standard Report sections: {path.name}")
    start, end = (Decimal(number.decode("ascii")) for number in match.groups())
    if end <= start:
        raise ValueError(f"Nonpositive observation time in {path.name}: {start} to {end}")


def model_paths(arguments):
    sources = []
    for argument in arguments:
        if argument.is_dir():
            sources.extend(sorted(
                file for file in argument.iterdir()
                if file.is_file() and file.suffix.lower() in {".gps", ".gpss"}
            ))
        else:
            sources.append(argument)
    if not sources or any(not source.is_file() for source in sources):
        raise ValueError("Provide existing GPSS source files or a directory containing them.")
    if len({source.stem.casefold() for source in sources}) != len(sources):
        raise ValueError("Every GPSS source needs a distinct file name.")
    return sources


def stage_model(source, inputs, workdir):
    program = source.read_text(encoding="ascii").rstrip() + "\n"
    names = {"batch.gps", "model.txt"}
    for data in inputs:
        if not data.is_file():
            raise FileNotFoundError(f"Input file not found: {data}")
        if data.name.casefold() in names:
            raise ValueError(f"Input file name collision: {data.name}")
        names.add(data.name.casefold())

    workdir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(TEMPLATE, workdir / "batch.gps")
    for data in inputs:
        shutil.copyfile(data, workdir / data.name)
    (workdir / "model.txt").write_text(program, encoding="ascii")


def collect_reports(workdir, destination, expected):
    reports = []
    for file in workdir.iterdir():
        match = REPORT_NAME.fullmatch(file.name)
        if match and file.is_file():
            stem, simulation, serial = match.groups()
            reports.append((stem, int(simulation), int(serial), file))
    if len(reports) != expected:
        raise RuntimeError(f"Found {len(reports)} of {expected} expected reports in {workdir}.")
    if len({(stem, simulation) for stem, simulation, _, _ in reports}) != 1:
        raise ValueError(f"Multiple simulation objects in {workdir}. Use a fresh working directory.")

    extracted = []
    for index, (stem, simulation, serial, file) in enumerate(sorted(reports), start=1):
        if serial != index:
            raise ValueError(f"Missing or duplicate report serial {index} in {workdir}.")
        text = report_text(file)
        validate_report(file, text, stem, simulation, serial)
        extracted.append(text)

    destination.mkdir(parents=True, exist_ok=False)
    for index, text in enumerate(extracted, start=1):
        (destination / f"{index:02d}.txt").write_bytes(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", type=Path, help="Plain-text .gps/.gpss sources or their directories.")
    parser.add_argument("--output", type=Path, help="Destination for the extracted reports.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", type=Path, help="Stage Model Objects for interactive runs in this directory.")
    action.add_argument("--collect", type=Path, help="Collect reports after interactive runs from this directory.")
    parser.add_argument("--input", action="append", type=Path, default=[], help="Additional model input file.")
    parser.add_argument("--expected-reports", default=1, type=int, help="Expected Standard Reports per source.")
    args = parser.parse_args()
    if args.collect and args.output is None:
        parser.error("--output is required with --collect.")
    if args.prepare and not TEMPLATE.is_file():
        parser.error(f"GPSS World Model Object not found: {TEMPLATE}")
    if args.expected_reports <= 0:
        parser.error("Expected report count must be positive.")
    try:
        for source in model_paths(args.models):
            if args.prepare:
                workdir = args.prepare / source.stem
                stage_model(source, args.input, workdir)
                print(f"{source.name}: {workdir / 'batch.gps'}", flush=True)
            else:
                destination = args.output / source.stem
                collect_reports(args.collect / source.stem, destination, args.expected_reports)
                print(f"{source.name}: {destination}", flush=True)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
