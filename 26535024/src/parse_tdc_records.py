"""Parse technician and task records from the TDC research dataset.

Run from the repository root:
    python src/parse_tdc_records.py

The script reads TDCData.txt directly from the ZIP and writes local CSVs to
 data/processed/. Travel-time rows are intentionally not expanded here.
"""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from zipfile import BadZipFile, ZipFile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = PROJECT_ROOT / "data" / "raw" / "TDCData_instance.zip"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
MAIN_MEMBER = "TDCData.txt"

TECHNICIAN_RE = re.compile(
    r'^Technician\s+id\s+(?P<id>\d+),\s*'
    r'skills\s+(?P<skills>\[.*?\]),\s*'
    r'time\s*window\s*\[(?P<start>[^,\]]+),\s*(?P<end>[^\]]+)\],\s*'
    r'investment option\s+(?P<investment>true|false)\s*$',
    re.IGNORECASE,
)

TASK_RE = re.compile(
    r'^Task\s+id\s+(?P<id>\d+),\s*'
    r'skills\s+(?P<skills>\[.*?\]),\s*'
    r'duration\s+(?P<duration>\d+),\s*'
    r'time\s*window\s*\[(?P<start>[^,\]]+),\s*(?P<end>[^\]]+)\],\s*'
    r'priority\s+(?P<priority>\d+),\s*'
    r'penalty\s+(?P<penalty>[-+]?(?:\d+(?:\.\d*)?|\.\d+)),\s*'
    r'created\s+(?P<created>[^,]+),\s*'
    r'modified\s+(?P<modified>[^,]+),\s*'
    r'cancelled\s+(?P<cancelled>true|false),\s*'
    r'relevant\s*technician\s*ids\s+(?:Int64\s*)?(?P<technicians>\[.*\])\s*$',
    re.IGNORECASE,
)

TECHNICIAN_FIELDS = [
    "technician_id",
    "skills_json",
    "time_window_start",
    "time_window_end",
    "investment_option",
]

TASK_FIELDS = [
    "task_id",
    "skills_json",
    "duration",
    "time_window_start",
    "time_window_end",
    "priority",
    "penalty",
    "created",
    "modified",
    "cancelled",
    "relevant_technician_ids_json",
]


def parse_list(raw: str, line_number: int, field_name: str) -> list:
    """Parse a JSON-style list and provide a useful error if it is malformed."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Line {line_number}: invalid list in field {field_name}: {error}"
        ) from error
    if not isinstance(value, list):
        raise ValueError(f"Line {line_number}: field {field_name} is not a list.")
    return value


def parse_technician(line: str, line_number: int) -> dict[str, object]:
    match = TECHNICIAN_RE.match(line)
    if not match:
        raise ValueError("technician record did not match expected format")

    skills = parse_list(match.group("skills"), line_number, "skills")
    return {
        "technician_id": int(match.group("id")),
        "skills_json": json.dumps(skills, ensure_ascii=False),
        "time_window_start": match.group("start").strip(),
        "time_window_end": match.group("end").strip(),
        "investment_option": match.group("investment").lower() == "true",
    }


def parse_task(line: str, line_number: int) -> dict[str, object]:
    match = TASK_RE.match(line)
    if not match:
        raise ValueError("task record did not match expected format")

    skills = parse_list(match.group("skills"), line_number, "skills")
    eligible_ids = parse_list(
        match.group("technicians"), line_number, "relevant technician IDs"
    )
    if not all(isinstance(value, int) for value in eligible_ids):
        raise ValueError("relevant technician IDs must be integers")

    modified_value = match.group("modified").strip()
    if modified_value.lower() == "nothing":
        modified_value = ""

    return {
        "task_id": int(match.group("id")),
        "skills_json": json.dumps(skills, ensure_ascii=False),
        "duration": int(match.group("duration")),
        "time_window_start": match.group("start").strip(),
        "time_window_end": match.group("end").strip(),
        "priority": int(match.group("priority")),
        "penalty": float(match.group("penalty")),
        "created": match.group("created").strip(),
        "modified": modified_value,
        "cancelled": match.group("cancelled").lower() == "true",
        "relevant_technician_ids_json": json.dumps(eligible_ids),
    }


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    """Write records to a UTF-8 CSV file."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not ZIP_PATH.exists():
        print(f"Dataset ZIP not found: {ZIP_PATH}")
        print("Download TDCData_instance.zip from https://zenodo.org/records/7904015")
        return

    technicians: list[dict[str, object]] = []
    tasks: list[dict[str, object]] = []
    issues: list[str] = []
    section = ""

    try:
        with ZipFile(ZIP_PATH, "r") as archive:
            with archive.open(MAIN_MEMBER, "r") as binary_stream:
                with io.TextIOWrapper(
                    binary_stream, encoding="utf-8", errors="replace"
                ) as stream:
                    for line_number, raw_line in enumerate(stream, start=1):
                        line = raw_line.strip()
                        if not line:
                            continue

                        if line in {"Technicians:", "Tasks:", "Travel time:"}:
                            section = line[:-1]
                            if section == "Travel time":
                                break
                            continue

                        try:
                            if section == "Technicians" and line.startswith("Technician"):
                                technicians.append(parse_technician(line, line_number))
                            elif section == "Tasks" and line.startswith("Task"):
                                tasks.append(parse_task(line, line_number))
                            elif section in {"Technicians", "Tasks"}:
                                issues.append(
                                    f"Line {line_number}: unexpected content in {section}: {line[:250]}"
                                )
                        except (ValueError, TypeError) as error:
                            
                            preview = line[:2500]
                            if len(line) > 2500:
                                preview += "... [record truncated]"
                            issues.append(
                                f"Line {line_number}: {error}\n    RAW RECORD: {preview}"
                            )

    except KeyError:
        print(f"Expected file {MAIN_MEMBER!r} not found inside the ZIP.")
        return
    except BadZipFile:
        print("The ZIP archive is invalid or corrupted.")
        return
    except OSError as error:
        print(f"Could not read dataset archive: {error}")
        return

    print("Dataset parsing summary")
    print("=======================")
    print(f"Technician records parsed: {len(technicians):,}")
    print(f"Task records parsed: {len(tasks):,}")
    print(f"Parsing issues: {len(issues):,}")

    if issues:
        print("\nFirst 10 issues (outputs not written):")
        for issue in issues[:10]:
            print("-", issue)
        print("\nNo output CSVs were written because parsing was incomplete.")
        return

    technician_ids = [row["technician_id"] for row in technicians]
    task_ids = [row["task_id"] for row in tasks]
    if len(technician_ids) != len(set(technician_ids)):
        print("Validation failed: duplicate technician IDs; no CSVs written.")
        return
    if len(task_ids) != len(set(task_ids)):
        print("Validation failed: duplicate task IDs; no CSVs written.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    technicians_path = OUTPUT_DIR / "technicians.csv"
    tasks_path = OUTPUT_DIR / "tasks.csv"
    write_csv(technicians_path, TECHNICIAN_FIELDS, technicians)
    write_csv(tasks_path, TASK_FIELDS, tasks)

    print("\nValidation passed: parsed records matched and IDs are unique.")
    print(f"Wrote {technicians_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {tasks_path.relative_to(PROJECT_ROOT)}")
    print("Travel-time records were left untouched.")


if __name__ == "__main__":
    main()
