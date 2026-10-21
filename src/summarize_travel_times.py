"""Summarize the travel-time section without loading it into memory.

Run from the repository root:
    python src/summarize_travel_times.py

This scans the travel-time section in the ZIP and prints diagnostics only.
It does not create or modify files.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from zipfile import BadZipFile, ZipFile

ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = ROOT / "data" / "raw" / "TDCData_instance.zip"
TASKS_PATH = ROOT / "data" / "processed" / "tasks.csv"
MEMBER = "TDCData.txt"
TRAVEL_HEADER = "Travel time:"
PROGRESS_EVERY = 2_000_000

TRAVEL_RE = re.compile(
    r"^task(?P<origin>\d+)task(?P<destination>\d+):\s*"
    r"(?P<minutes>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)


def load_task_ids(path: Path) -> set[int]:
    """Load parsed task IDs for cross-checking travel-time references."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {int(row["task_id"]) for row in csv.DictReader(handle)}


def main() -> None:
    if not ZIP_PATH.exists():
        print(f"Dataset ZIP not found: {ZIP_PATH}")
        return
    if not TASKS_PATH.exists():
        print(f"Parsed tasks CSV not found: {TASKS_PATH}")
        print("Run python src/parse_tdc_records.py first.")
        return

    task_ids = load_task_ids(TASKS_PATH)
    record_count = 0
    malformed_count = 0
    negative_count = 0
    zero_count = 0
    self_pair_count = 0
    from_missing_count = 0
    to_missing_count = 0
    total_minutes = 0.0
    minimum = math.inf
    maximum = -math.inf
    first_samples: list[str] = []
    malformed_samples: list[tuple[int, str]] = []
    origin_ids: set[int] = set()
    destination_ids: set[int] = set()
    found_header = False
    line_number = 0

    try:
        with ZipFile(ZIP_PATH, "r") as archive:
            with archive.open(MEMBER, "r") as binary_stream:
                import io

                with io.TextIOWrapper(
                    binary_stream, encoding="utf-8", errors="replace"
                ) as stream:
                    for line_number, raw_line in enumerate(stream, start=1):
                        line = raw_line.strip()
                        if not found_header:
                            if line == TRAVEL_HEADER:
                                found_header = True
                            continue

                        if not line:
                            continue

                        match = TRAVEL_RE.match(line)
                        if not match:
                            malformed_count += 1
                            if len(malformed_samples) < 10:
                                malformed_samples.append((line_number, line[:300]))
                            continue

                        origin = int(match.group("origin"))
                        destination = int(match.group("destination"))
                        minutes = float(match.group("minutes"))

                        record_count += 1
                        origin_ids.add(origin)
                        destination_ids.add(destination)
                        from_missing_count += origin not in task_ids
                        to_missing_count += destination not in task_ids
                        self_pair_count += origin == destination
                        negative_count += minutes < 0
                        zero_count += minutes == 0
                        total_minutes += minutes
                        minimum = min(minimum, minutes)
                        maximum = max(maximum, minutes)

                        if len(first_samples) < 10:
                            first_samples.append(line)

                        if record_count % PROGRESS_EVERY == 0:
                            print(f"Scanned {record_count:,} travel-time entries...")

    except KeyError:
        print(f"File {MEMBER!r} was not found in the ZIP.")
        return
    except BadZipFile:
        print("The ZIP archive is invalid or corrupted.")
        return
    except OSError as error:
        print(f"Could not read dataset: {error}")
        return

    print("\nTRAVEL-TIME DATA SUMMARY")
    print("========================")
    print(f"Travel-time header found: {found_header}")
    print(f"Well-formed entries: {record_count:,}")
    print(f"Malformed entries: {malformed_count:,}")
    print(f"Distinct origin IDs observed: {len(origin_ids):,}")
    print(f"Distinct destination IDs observed: {len(destination_ids):,}")
    print(f"Origins absent from parsed task IDs (entry count): {from_missing_count:,}")
    print(f"Destinations absent from parsed task IDs (entry count): {to_missing_count:,}")
    print(f"Self-pairs (origin = destination): {self_pair_count:,}")
    print(f"Zero travel times: {zero_count:,}")
    print(f"Negative travel times: {negative_count:,}")

    if record_count:
        print(f"Minimum value: {minimum}")
        print(f"Maximum value: {maximum}")
        print(f"Mean value: {total_minutes / record_count:.4f}")
        print("Research paper states travel time is measured in minutes; confirm data semantics before operational interpretation.")

    print("\nFirst 10 well-formed entries:")
    for sample in first_samples:
        print("  " + sample)

    if malformed_samples:
        print("\nFirst malformed entries:")
        for number, sample in malformed_samples:
            print(f"  Line {number}: {sample}")

    print("\nThis was a streaming scan. Travel-time records were not stored or modified.")


if __name__ == "__main__":
    main()
