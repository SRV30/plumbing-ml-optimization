"""Summarize the structure of TDCData.txt by streaming from the ZIP.

This avoids extracting the large text file or reading it all into RAM.

Run from the repository root:
    python src/summarize_tdc_structure.py
"""

from pathlib import Path
from zipfile import BadZipFile, ZipFile
import io
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = PROJECT_ROOT / "data" / "raw" / "TDCData_instance.zip"
MAIN_MEMBER = "TDCData.txt"
INVESTMENT_MEMBER = "TDCData_investment.txt"
SAMPLE_LINES_PER_SECTION = 2
MAX_SAMPLE_CHARS = 220


SECTION_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9 _()/-]{0,70}:$")


def print_investment_settings(archive: ZipFile) -> None:
    """Print the complete, small investment configuration file."""
    print("\n" + "=" * 78)
    print(f"CONFIGURATION FILE: {INVESTMENT_MEMBER}")

    try:
        with archive.open(INVESTMENT_MEMBER, "r") as binary_stream:
            with io.TextIOWrapper(
                binary_stream, encoding="utf-8", errors="replace"
            ) as text_stream:
                for number, line in enumerate(text_stream, start=1):
                    print(f"{number:>3}: {line.rstrip()}")
    except KeyError:
        print(f"{INVESTMENT_MEMBER} was not found in the ZIP.")


def summarize_main_file(archive: ZipFile) -> None:
    """Count lines/sections and retain only a few example records per section."""
    print("\n" + "=" * 78)
    print(f"STREAMING STRUCTURE SCAN: {MAIN_MEMBER}")
    print("Reading line by line; the whole file is not loaded into memory.")

    try:
        with archive.open(MAIN_MEMBER, "r") as binary_stream:
            with io.TextIOWrapper(
                binary_stream, encoding="utf-8", errors="replace"
            ) as text_stream:
                total_lines = 0
                blank_lines = 0
                current_section = "[before first section]"
                section_order: list[str] = []
                section_counts: dict[str, int] = {}
                section_examples: dict[str, list[str]] = {}
                max_line_length = 0

                for line_number, line in enumerate(text_stream, start=1):
                    total_lines = line_number
                    stripped = line.strip()
                    max_line_length = max(max_line_length, len(stripped))

                    if not stripped:
                        blank_lines += 1
                        continue

                    
                    if len(stripped) <= 80 and SECTION_PATTERN.fullmatch(stripped):
                        current_section = stripped[:-1]
                        if current_section not in section_counts:
                            section_order.append(current_section)
                            section_counts[current_section] = 0
                            section_examples[current_section] = []
                        print(f"Section header at line {line_number}: {stripped}")
                        continue

                    section_counts[current_section] = (
                        section_counts.get(current_section, 0) + 1
                    )
                    examples = section_examples.setdefault(current_section, [])
                    if len(examples) < SAMPLE_LINES_PER_SECTION:
                        preview = stripped[:MAX_SAMPLE_CHARS]
                        if len(stripped) > MAX_SAMPLE_CHARS:
                            preview += "... [truncated]"
                        examples.append(preview)

                print("\nOverall summary:")
                print(f"- Total lines: {total_lines:,}")
                print(f"- Blank lines: {blank_lines:,}")
                print(f"- Maximum line length: {max_line_length:,} characters")

                print("\nSection record counts and examples:")
                for section in section_order:
                    print(f"\n[{section}]")
                    print(f"Non-empty data lines counted: {section_counts[section]:,}")
                    for example in section_examples[section]:
                        print(f"  Example: {example}")

                if "[before first section]" in section_counts:
                    print("\n[Lines before first recognized section]")
                    print(
                        "Non-empty lines:",
                        section_counts["[before first section]"],
                    )
                    for example in section_examples["[before first section]"]:
                        print(f"  Example: {example}")

    except KeyError:
        print(f"{MAIN_MEMBER} was not found in the ZIP.")


def main() -> None:
    if not ZIP_PATH.exists():
        print(f"ZIP not found: {ZIP_PATH}")
        print("Download TDCData_instance.zip from https://zenodo.org/records/7904015")
        return

    try:
        with ZipFile(ZIP_PATH, "r") as archive:
            print(f"Archive: {ZIP_PATH.relative_to(PROJECT_ROOT)}")
            summarize_main_file(archive)
            print_investment_settings(archive)
    except BadZipFile:
        print("The ZIP archive could not be opened. It may be corrupted.")
    except OSError as error:
        print(f"Could not read the archive: {error}")


if __name__ == "__main__":
    main()
