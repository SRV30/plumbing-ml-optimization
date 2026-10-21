"""Parse technician and task rows from the TDC data ZIP into local CSV files.

Run from the repository root:
    python src/parse_tdc_records.py

Outputs are written to data/processed/ and are excluded from Git by .gitignore.
Travel-time entries are intentionally not parsed or expanded by this script.
"""

from __future__ import annotations

import csv
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
    r'time\s*window\s+\[(?P<start>[^,\]]+),\s*(?P<end>[^\]]+)\],\s*'
    r'investment option\s+(?P<investment>true|false)\s*$',
    re.IGNORECASE,
)

TASK_RE = re.compile(
    r'^Task\s+id\s+(?P<id>\d+),\s*'
    r'skills\s+(?P<skills>\[.*?\]),\s*'
    r'duration\s+(?P<duration>\d+),\s*'
    r'time\s*window\s+\[(?P<start>[^,\]]+),\s*(?P<end>[^\]]+)\],\s*'
    r'priority\s+(?P<priority>\d+),\s*'
    r'penalty\s+(?P<penalty>[-+]?(?:\d+(?:\.\d*)?|\.\d+)),\s*'
    r'created\s+(?P<created>[^,]+),\s*'
    r'modified\s+(?P<modified>[^,]+),\s*'
    r'cancelled\s+(?P<cancelled>true|false),\s*'
    r'relevant\s*technician\s*ids\s+(?P<technicians>\[.*\])\s*$',
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


def parse_json_list(raw: str, *, line_number: int, field: str) -> list:
    """Decode a list literal using JSON and validate its type."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Line {line_number}: invalid JSON list for {field}: {error}"
        ) from error

    if not isinstance(value, list):
        raise ValueError(f"Line {line_number}: {field} should be a list.")
    return value


def parse_technician(line: str, line_number: int) -> dict[str, object]:
    match = TECHNICIAN_RE.match(line)
    if not match:
        raise ValueError(f"Line {line_number}: technician record did not match expected format.")

    skills = parse_json_list(
        match.group("skills"), line_number=line_number, field="skills"
    )
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
        raise ValueError(f"Line {line_number}: task record did not match expected format.")

    skills = parse_json_list(
        match.group("skills"), line_number=line_number, field="skills"
    )
    relevant_technicians = parse_json_list(
        match.group("technicians"),
        line_number=line_number,
        field="relevant technician IDs",
    )

    if not all(isinstance(value, int) for value in relevant_technicians):
        raise ValueError(
            f"Line {line_number}: relevant technician IDs must all be integers."
        )

    modified_raw = match.group("modified").strip()
    modified = "" if modified_raw.lower() == "nothing" else modified_raw

    return {
        "task_id": int(match.group("id")),
        "skills_json": json.dumps(skills, ensure_ascii=False),
        "duration": int(match.group("duration")),
        "time_window_start": match.group("start").strip(),
        "time_window_end": match.group("end").strip(),
        "priority": int(match.group("priority")),
        "penalty": float(match.group("penalty")),
        "created": match.group("created").strip(),
        "modified": modified,
        "cancelled": match.group("cancelled").lower() == "true",
        "relevant_technician_ids_json": json.dumps(relevant_technicians),
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    """Write parsed records as UTF-8 CSV."""
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not ZIP_PATH.exists():
        print("Dataset ZIP not found.")
        print(f"Expected location: {ZIP_PATH}")
        print("Download TDCData_instance.zip from https://zenodo.org/records/7904015")
        return

    technicians: list[dict[str, object]] = []
    tasks: list[dict[str, object]] = []
    issues: list[str] = []
    section = ""
    seen_sections: set[str] = set()

    try:
        with ZipFile(ZIP_PATH, "r") as archive:
            try:
                with archive.open(MAIN_MEMBER, "r") as binary_stream:
                    import io

                    with io.TextIOWrapper(
                        binary_stream, encoding="utf-8", errors="replace"
                    ) as stream:
                        for line_number, raw_line in enumerate(stream, start=1):
                            line = raw_line.strip()
                            if not line:
                                continue

                            if line in {"Technicians:", "Tasks:", "Travel time:"}:
                                section = line[:-1]
                                seen_sections.add(section)
                                if section == "Travel time":
                                    break
                                continue

                            try:
                                if section == "Technicians":
                                    if line.startswith("Technician"):
                                        technicians.append(
                                            parse_technician(line, line_number)
                                        )
                                    else:
                                        issues.append(
                                            f"Line {line_number}: unexpected line in Technicians section: {line[:120]}"
                                        )
                                elif section == "Tasks":
                                    if line.startswith("Task"):
                                        tasks.append(parse_task(line, line_number))
                                    else:
                                        issues.append(
                                            f"Line {line_number}: unexpected line in Tasks section: {line[:120]}"
                                        )
                            except (ValueError, TypeError) as error:
                                # Include the source row so format variations can be fixed safely.
                                preview = line[:2500]
                                if len(line) > 2500:
                                    preview += "... [record truncated for display]"
                                issues.append(f"{error}\n    RAW RECORD: {preview}")

            except KeyError:
                print(f"Expected member {MAIN_MEMBER!r} was not found in the ZIP.")
                return

    except BadZipFile:
        print("The ZIP archive is invalid or corrupted.")
        return
    except OSError as error:
        print(f"Could not read dataset archive: {error}")
        return

    print("Dataset parsing summary")
    print("=======================")
    print(f"Sections detected before travel-time section: {sorted(seen_sections)}")
    print(f"Technician records parsed: {len(technicians):,}")
    print(f"Task records parsed: {len(tasks):,}")
    print(f"Parsing issues: {len(issues):,}")

    if issues:
        print("\nFirst 10 issues (outputs not written):")
        for issue in issues[:10]:
            print("-", issue)
        print(
            "\nPlease share these issues before proceeding. "
            "The script intentionally does not save partial results."
        )
        return

    # Basic sanity checks before writing output.
    technician_ids = [row["technician_id"] for row in technicians]
    task_ids = [row["task_id"] for row in tasks]
    if len(technician_ids) != len(set(technician_ids)):
        print("Validation failed: duplicate technician IDs found; no CSVs written.")
        return
    if len(task_ids) != len(set(task_ids)):
        print("Validation failed: duplicate task IDs found; no CSVs written.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    technician_path = OUTPUT_DIR / "technicians.csv"
    task_path = OUTPUT_DIR / "tasks.csv"

    write_csv(technician_path, TECHNICIAN_FIELDS, technicians)
    write_csv(task_path, TASK_FIELDS, tasks)

    print("\nValidation passed: record IDs are unique and every parsed row matched.")
    print(f"Wrote {technician_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {task_path.relative_to(PROJECT_ROOT)}")
    print("\nNote: travel-time entries were intentionally left untouched.")


if __name__ == "__main__":
    main()
