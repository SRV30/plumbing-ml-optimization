"""Classify and validate all travel-time key types in the TDC ZIP.

Run from the repository root:
    python src/summarize_travel_times.py

The script streams the travel-time section and does not store the full matrix.
It does not create or modify any files.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from pathlib import Path
from zipfile import BadZipFile, ZipFile
import io

ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = ROOT / "data" / "raw" / "TDCData_instance.zip"
TASKS_PATH = ROOT / "data" / "processed" / "tasks.csv"
TECHNICIANS_PATH = ROOT / "data" / "processed" / "technicians.csv"
MEMBER = "TDCData.txt"
TRAVEL_HEADER = "Travel time:"
PROGRESS_EVERY = 2_000_000

# Examples: task1000task1: 186.0; task1000technician1: 16.0
TRAVEL_RE = re.compile(
    r"^(?P<origin_type>task|technician)(?P<origin_id>\d+)"
    r"(?P<destination_type>task|technician)(?P<destination_id>\d+):\s*"
    r"(?P<minutes>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$",
    re.IGNORECASE,
)


def load_ids(path: Path, column_name: str) -> set[int]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {int(row[column_name]) for row in csv.DictReader(handle)}


def main() -> None:
    for path in (ZIP_PATH, TASKS_PATH, TECHNICIANS_PATH):
        if not path.exists():
            print(f"Required file not found: {path.relative_to(ROOT)}")
            if path == ZIP_PATH:
                print("Download TDCData_instance.zip from https://zenodo.org/records/7904015")
            else:
                print("Run python src/parse_tdc_records.py first.")
            return

    task_ids = load_ids(TASKS_PATH, "task_id")
    technician_ids = load_ids(TECHNICIANS_PATH, "technician_id")

    category_counts: Counter[tuple[str, str]] = Counter()
    malformed_count = 0
    malformed_samples: list[tuple[int, str]] = []
    bad_reference_counts: Counter[tuple[str, str]] = Counter()
    negative_count = 0
    zero_count = 0
    self_pair_count = 0
    minimum = math.inf
    maximum = -math.inf
    total_minutes = 0.0
    sample_by_category: dict[tuple[str, str], list[str]] = {}
    travel_header_found = False
    total_parsed = 0

    try:
        with ZipFile(ZIP_PATH, "r") as archive:
            with archive.open(MEMBER, "r") as binary_stream:
                with io.TextIOWrapper(binary_stream, encoding="utf-8", errors="replace") as stream:
                    for line_number, raw_line in enumerate(stream, start=1):
                        line = raw_line.strip()

                        if not travel_header_found:
                            if line == TRAVEL_HEADER:
                                travel_header_found = True
                            continue

                        if not line:
                            continue

                        match = TRAVEL_RE.match(line)
                        if not match:
                            malformed_count += 1
                            if len(malformed_samples) < 10:
                                malformed_samples.append((line_number, line[:300]))
                            continue

                        origin_type = match.group("origin_type").lower()
                        destination_type = match.group("destination_type").lower()
                        origin_id = int(match.group("origin_id"))
                        destination_id = int(match.group("destination_id"))
                        minutes = float(match.group("minutes"))
                        category = (origin_type, destination_type)

                        total_parsed += 1
                        category_counts[category] += 1
                        total_minutes += minutes
                        minimum = min(minimum, minutes)
                        maximum = max(maximum, minutes)
                        negative_count += minutes < 0
                        zero_count += minutes == 0
                        self_pair_count += origin_type == destination_type and origin_id == destination_id

                        origin_pool = task_ids if origin_type == "task" else technician_ids
                        destination_pool = task_ids if destination_type == "task" else technician_ids
                        if origin_id not in origin_pool:
                            bad_reference_counts[(origin_type, "origin")] += 1
                        if destination_id not in destination_pool:
                            bad_reference_counts[(destination_type, "destination")] += 1

                        samples = sample_by_category.setdefault(category, [])
                        if len(samples) < 3:
                            samples.append(line)

                        if total_parsed % PROGRESS_EVERY == 0:
                            print(f"Classified {total_parsed:,} travel-time entries...")

    except KeyError:
        print(f"File {MEMBER!r} was not found inside the ZIP.")
        return
    except BadZipFile:
        print("The ZIP archive is invalid or corrupted.")
        return
    except OSError as error:
        print(f"Could not read dataset: {error}")
        return

    print("\nTRAVEL-TIME DATA SUMMARY")
    print("========================")
    print(f"Travel-time header found: {travel_header_found}")
    print(f"Classified entries: {total_parsed:,}")
    print(f"Unrecognized/malformed entries: {malformed_count:,}")

    labels = {
        ("task", "task"): "Task -> task",
        ("task", "technician"): "Task -> technician",
        ("technician", "task"): "Technician -> task",
        ("technician", "technician"): "Technician -> technician",
    }
    print("\nEntries by key type:")
    for key in labels:
        print(f"  {labels[key]}: {category_counts[key]:,}")

    print("\nReference consistency:")
    print(f"  Task IDs loaded from CSV: {len(task_ids):,}")
    print(f"  Technician IDs loaded from CSV: {len(technician_ids):,}")
    print(f"  Entries with bad typed ID references: {sum(bad_reference_counts.values()):,}")
    if bad_reference_counts:
        for key, count in sorted(bad_reference_counts.items()):
            print(f"    {key[0]} as {key[1]}: {count:,}")

    print("\nNumeric checks:")
    print(f"  Self-pairs: {self_pair_count:,}")
    print(f"  Zero values: {zero_count:,}")
    print(f"  Negative values: {negative_count:,}")
    if total_parsed:
        print(f"  Minimum value: {minimum}")
        print(f"  Maximum value: {maximum}")
        print(f"  Mean value: {total_minutes / total_parsed:.4f}")
        print("  Travel-time unit: the research paper describes these values in minutes.")

    print("\nExamples by key type:")
    for key in labels:
        print(f"  {labels[key]}:")
        for example in sample_by_category.get(key, []):
            print(f"    {example}")

    if malformed_samples:
        print("\nFirst unrecognized entries:")
        for number, sample in malformed_samples:
            print(f"  Line {number}: {sample}")

    print("\nThis was a streaming scan; no travel-time files were written.")


if __name__ == "__main__":
    main()
