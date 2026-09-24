#!/usr/bin/env python3
"""Create a public CSV copy without local paths or private metadata."""

import argparse
import csv
from pathlib import Path
import re


def portable_input_path(value: str) -> str:
    if not value or value == "N/A":
        return value
    return f"inputs/{Path(value).name}"


def sanitize_value(column: str, value: str) -> str:
    if column == "hostname":
        return "MI210"
    if column in {"weight_file", "input_file"}:
        return portable_input_path(value)
    if column == "notes":
        return re.sub(
            r"source=([^;]+)",
            lambda match: f"source={portable_input_path(match.group(1))}",
            value,
        )
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must be different files")

    with args.input.open(newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise SystemExit("input CSV has no header")
        fieldnames = [name for name in reader.fieldnames if name != "private_commit"]
        rows = [
            {
                name: sanitize_value(name, row.get(name, ""))
                for name in fieldnames
            }
            for row in reader
        ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
