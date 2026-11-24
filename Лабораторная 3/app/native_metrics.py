"""Read completed CSV exports from a GPSS World PLUS campaign."""

import csv
from math import isfinite


def campaign_rows(paths, required):
    """Return verified CSV records from explicitly supplied native exports."""
    for path in paths:
        lines = path.read_text(encoding="ascii").splitlines()
        if not lines or lines[-1].strip() != "COMPLETED":
            raise ValueError(f"{path}: GPSS campaign did not finish")
        reader = csv.DictReader(lines[:-1])
        if reader.fieldnames is None or not set(required).issubset(reader.fieldnames):
            raise ValueError(f"{path}: missing expected GPSS columns")
        for row in reader:
            if None in row or any(row[field] is None for field in required):
                raise ValueError(f"{path}:{reader.line_num}: incomplete GPSS record")
            yield row


def number(row, field):
    value = float(row[field])
    if not isfinite(value) or value < 0:
        raise ValueError(f"{row['case']}: {field} must be finite and nonnegative")
    return value


def count(row, field):
    value = number(row, field)
    if not value.is_integer():
        raise ValueError(f"{row['case']}: {field} must be an integer")
    return int(value)
