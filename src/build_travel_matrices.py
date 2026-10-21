"""Build dense NumPy travel-time matrices from the TDC text archive.

Run from the repository root:
    python src/build_travel_matrices.py

Creates these local, Git-ignored outputs in data/processed/:
    task_to_task.npy
    task_to_technician.npy
    technician_to_task.npy

The three matrices are initialized with NaN so missing or duplicate pairs can be
checked. A final matrix is published only if each expected pair is present once.
The ZIP and source records are never modified.
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = ROOT / "data" / "raw" / "TDCData_instance.zip"
DATA_DIR = ROOT / "data" / "processed"
TASKS_CSV = DATA_DIR / "tasks.csv"
TECHNICIANS_CSV = DATA_DIR / "technicians.csv"
MEMBER = "TDCData.txt"
PROGRESS_EVERY = 2_000_000

TRAVEL_RE = re.compile(
    r"^(?P<origin_type>task|technician)(?P<origin_id>\d+)"
    r"(?P<destination_type>task|technician)(?P<destination_id>\d+):\s*"
    r"(?P<minutes>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$",
    re.IGNORECASE,
)

MATRIX_NAMES = ("task_to_task", "task_to_technician", "technician_to_task")


def load_ids(path: Path, column: str) -> list[int]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        values = [int(row[column]) for row in csv.DictReader(handle)]
    if not values:
        raise ValueError(f"No IDs found in {path.name}.")
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate IDs found in {path.name}.")
    expected = set(range(1, len(values) + 1))
    if set(values) != expected:
        raise ValueError(
            f"{path.name} IDs are not contiguous from 1 to {len(values)}; "
            "use an explicit ID-to-index mapping before building matrices."
        )
    return values


def close_memmap(matrix: np.memmap | None) -> None:
    if matrix is not None:
        matrix.flush()
        matrix._mmap.close()


def main() -> None:
    if not ZIP_PATH.exists():
        print(f"Dataset ZIP not found: {ZIP_PATH}")
        return
    if not TASKS_CSV.exists() or not TECHNICIANS_CSV.exists():
        print("Processed CSV files are missing. Run python src/parse_tdc_records.py first.")
        return

    try:
        task_ids = load_ids(TASKS_CSV, "task_id")
        technician_ids = load_ids(TECHNICIANS_CSV, "technician_id")
    except (OSError, ValueError) as error:
        print(f"Could not validate ID tables: {error}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    shapes = {
        "task_to_task": (len(task_ids), len(task_ids)),
        "task_to_technician": (len(task_ids), len(technician_ids)),
        "technician_to_task": (len(technician_ids), len(task_ids)),
    }
    temp_paths = {
        name: DATA_DIR / f".{name}.building.npy" for name in MATRIX_NAMES
    }
    final_paths = {
        name: DATA_DIR / f"{name}.npy" for name in MATRIX_NAMES
    }
    matrices: dict[str, np.memmap] = {}

    try:
        for name in MATRIX_NAMES:
            matrices[name] = np.lib.format.open_memmap(
                temp_paths[name], mode="w+", dtype=np.float32, shape=shapes[name]
            )
            matrices[name][:] = np.nan

        category_counts = {
            "task_to_task": 0,
            "task_to_technician": 0,
            "technician_to_task": 0,
            "technician_to_technician": 0,
        }
        duplicate_counts = {name: 0 for name in MATRIX_NAMES}
        malformed_count = 0
        invalid_reference_count = 0
        invalid_tech_to_tech_count = 0
        tech_self_count = 0
        progress_count = 0
        header_found = False
        malformed_samples: list[tuple[int, str]] = []

        with ZipFile(ZIP_PATH, "r") as archive:
            with archive.open(MEMBER, "r") as binary_stream:
                with io.TextIOWrapper(
                    binary_stream, encoding="utf-8", errors="replace"
                ) as stream:
                    for line_number, raw_line in enumerate(stream, start=1):
                        line = raw_line.strip()
                        if not header_found:
                            if line == "Travel time:":
                                header_found = True
                            continue

                        if not line:
                            continue

                        match = TRAVEL_RE.match(line)
                        if not match:
                            malformed_count += 1
                            if len(malformed_samples) < 5:
                                malformed_samples.append((line_number, line[:300]))
                            continue

                        origin_type = match.group("origin_type").lower()
                        destination_type = match.group("destination_type").lower()
                        origin_id = int(match.group("origin_id"))
                        destination_id = int(match.group("destination_id"))
                        minutes = float(match.group("minutes"))
                        progress_count += 1

                        if not np.isfinite(minutes) or minutes < 0:
                            invalid_reference_count += 1
                            continue

                        if origin_type == "task" and destination_type == "task":
                            name = "task_to_task"
                            row, col = origin_id - 1, destination_id - 1
                        elif origin_type == "task" and destination_type == "technician":
                            name = "task_to_technician"
                            row, col = origin_id - 1, destination_id - 1
                        elif origin_type == "technician" and destination_type == "task":
                            name = "technician_to_task"
                            row, col = origin_id - 1, destination_id - 1
                        else:
                            category_counts["technician_to_technician"] += 1
                            if origin_id != destination_id or minutes != 0:
                                invalid_tech_to_tech_count += 1
                            else:
                                tech_self_count += 1
                            if progress_count % PROGRESS_EVERY == 0:
                                print(f"Read {progress_count:,} travel-time entries...")
                            continue

                        if not (0 <= row < shapes[name][0] and 0 <= col < shapes[name][1]):
                            invalid_reference_count += 1
                            continue

                        category_counts[name] += 1
                        matrix = matrices[name]
                        if not np.isnan(matrix[row, col]):
                            duplicate_counts[name] += 1
                        else:
                            matrix[row, col] = minutes

                        if progress_count % PROGRESS_EVERY == 0:
                            print(f"Read {progress_count:,} travel-time entries...")

        missing_counts = {
            name: int(np.isnan(matrix).sum()) for name, matrix in matrices.items()
        }

        expected_counts = {
            "task_to_task": len(task_ids) * len(task_ids),
            "task_to_technician": len(task_ids) * len(technician_ids),
            "technician_to_task": len(technician_ids) * len(task_ids),
            "technician_to_technician": len(technician_ids),
        }

        print("\nTRAVEL MATRIX BUILD SUMMARY")
        print("===========================")
        print(f"Travel-time header found: {header_found}")
        print(f"Task IDs: {len(task_ids):,}; technician IDs: {len(technician_ids):,}")
        print("\nCategory counts (observed / expected):")
        for name in ("task_to_task", "task_to_technician", "technician_to_task", "technician_to_technician"):
            print(f"  {name}: {category_counts[name]:,} / {expected_counts[name]:,}")
        print("\nMatrix quality:")
        for name in MATRIX_NAMES:
            print(f"  {name} missing pairs: {missing_counts[name]:,}")
            print(f"  {name} duplicate pairs: {duplicate_counts[name]:,}")
        print(f"  Malformed entries: {malformed_count:,}")
        print(f"  Invalid IDs/values: {invalid_reference_count:,}")
        print(f"  Nonzero or non-self technician-to-technician entries: {invalid_tech_to_tech_count:,}")
        print(f"  Technician self-pairs: {tech_self_count:,}")

        is_valid = (
            header_found
            and malformed_count == 0
            and invalid_reference_count == 0
            and invalid_tech_to_tech_count == 0
            and category_counts == expected_counts
            and all(count == 0 for count in missing_counts.values())
            and all(count == 0 for count in duplicate_counts.values())
            and tech_self_count == len(technician_ids)
        )

        for matrix in matrices.values():
            close_memmap(matrix)
        matrices.clear()

        if not is_valid:
            print("\nValidation FAILED; temporary matrices will be removed.")
            if malformed_samples:
                print("First unrecognized entries:")
                for number, sample in malformed_samples:
                    print(f"  Line {number}: {sample}")
            for path in temp_paths.values():
                path.unlink(missing_ok=True)
            print("No final matrices were published.")
            return

        for name in MATRIX_NAMES:
            temp_paths[name].replace(final_paths[name])

        print("\nValidation PASSED.")
        for name in MATRIX_NAMES:
            print(f"  Wrote data/processed/{name}.npy (shape={shapes[name]})")
        print("These .npy files are local and excluded from Git.")

    except (BadZipFile, OSError, ValueError) as error:
        for matrix in matrices.values():
            try:
                close_memmap(matrix)
            except (OSError, ValueError):
                pass
        for path in temp_paths.values():
            path.unlink(missing_ok=True)
        print(f"Matrix build failed: {error}")


if __name__ == "__main__":
    main()
