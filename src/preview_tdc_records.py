"""Preview full technician/task record examples and travel-time key format.

Run from the repository root:
    python src/preview_tdc_records.py

The script reads only the first few records from the ZIP and does not extract it.
"""

from pathlib import Path
from zipfile import BadZipFile, ZipFile
import io

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = PROJECT_ROOT / "data" / "raw" / "TDCData_instance.zip"
MAIN_MEMBER = "TDCData.txt"
MAX_RECORDS_PER_SECTION = 3
MAX_PRINT_CHARS = 1800
MAX_TRAVEL_EXAMPLES = 8


def main() -> None:
    if not ZIP_PATH.exists():
        print(f"Dataset ZIP not found: {ZIP_PATH}")
        return

    try:
        with ZipFile(ZIP_PATH, "r") as archive:
            with archive.open(MAIN_MEMBER, "r") as raw:
                with io.TextIOWrapper(raw, encoding="utf-8", errors="replace") as stream:
                    section = ""
                    section_samples = {"Technicians": 0, "Tasks": 0}
                    travel_samples = 0

                    print("Full-record preview (long lines may be truncated for display)\n")

                    for line_number, raw_line in enumerate(stream, start=1):
                        line = raw_line.rstrip("\r\n")

                        if line in {"Technicians:", "Tasks:", "Travel time:"}:
                            section = line[:-1]
                            print(f"\n=== {section} (starts at line {line_number}) ===")
                            if section == "Travel time" and travel_samples >= MAX_TRAVEL_EXAMPLES:
                                break
                            continue

                        if not line:
                            continue

                        if section in section_samples and section_samples[section] < MAX_RECORDS_PER_SECTION:
                            section_samples[section] += 1
                            shown = line[:MAX_PRINT_CHARS]
                            if len(line) > MAX_PRINT_CHARS:
                                shown += f"... [display truncated; full line has {len(line)} chars]"
                            print(f"Line {line_number}: {shown}")

                        elif section == "Travel time" and travel_samples < MAX_TRAVEL_EXAMPLES:
                            travel_samples += 1
                            print(f"Line {line_number}: {line}")

                        if (
                            all(section_samples[name] >= MAX_RECORDS_PER_SECTION for name in section_samples)
                            and travel_samples >= MAX_TRAVEL_EXAMPLES
                        ):
                            break

                    print("\nPreview complete. No data files were extracted or modified.")

    except KeyError as error:
        print(f"Expected archive member not found: {error}")
    except BadZipFile:
        print("The ZIP file could not be opened; it may be corrupted.")
    except OSError as error:
        print(f"Could not read the archive: {error}")


if __name__ == "__main__":
    main()
