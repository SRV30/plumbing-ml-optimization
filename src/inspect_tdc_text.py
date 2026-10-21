"""Preview the text files inside TDCData_instance.zip without extracting them.

Run from the repository root:
    python src/inspect_tdc_text.py

This script prints a small sample only. It does not modify the archive.
"""

from pathlib import Path
from zipfile import BadZipFile, ZipFile
import io

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = PROJECT_ROOT / "data" / "raw" / "TDCData_instance.zip"
SAMPLE_LINES = 25
MAX_CHARS_PER_LINE = 300
TARGET_FILES = ("TDCData.txt", "TDCData_investment.txt")


def show_sample(archive: ZipFile, member_name: str) -> None:
    """Print sample lines and basic delimiter clues for a ZIP member."""
    print("\n" + "=" * 78)
    print(f"FILE: {member_name}")

    try:
        with archive.open(member_name, "r") as binary_stream:
            # Replace undecodable characters during inspection; original bytes stay unchanged.
            with io.TextIOWrapper(binary_stream, encoding="utf-8", errors="replace") as text_stream:
                observed = 0
                token_counts: list[int] = []

                for line_number, line in enumerate(text_stream, start=1):
                    if line_number > SAMPLE_LINES:
                        break

                    observed += 1
                    stripped = line.rstrip("\r\n")
                    token_counts.append(len(stripped.split()))

                    preview = stripped[:MAX_CHARS_PER_LINE]
                    if len(stripped) > MAX_CHARS_PER_LINE:
                        preview += "... [line truncated for display]"

                    print(f"{line_number:>3}: {preview!r}")
                    print(
                        "     chars={}; whitespace_tokens={}; tabs={}; commas={}; semicolons={}".format(
                            len(stripped),
                            len(stripped.split()),
                            stripped.count("\t"),
                            stripped.count(","),
                            stripped.count(";"),
                        )
                    )

                if observed == 0:
                    print("File is empty.")
                else:
                    print("\nSample summary:")
                    print(f"- Lines displayed: {observed}")
                    print(f"- Whitespace token counts: {token_counts}")
                    print("- Check the samples above to identify headers, separators, and record structure.")

    except KeyError:
        print("This filename was not found inside the ZIP archive.")


def main() -> None:
    if not ZIP_PATH.exists():
        print("Dataset ZIP not found.")
        print(f"Expected: {ZIP_PATH}")
        print("Download TDCData_instance.zip from https://zenodo.org/records/7904015")
        return

    try:
        with ZipFile(ZIP_PATH, "r") as archive:
            archive_names = set(archive.namelist())
            print(f"Archive: {ZIP_PATH.relative_to(PROJECT_ROOT)}")
            print("This reads only a small preview from each text file; nothing is extracted.")

            for name in TARGET_FILES:
                if name in archive_names:
                    show_sample(archive, name)
                else:
                    matches = [candidate for candidate in archive_names if candidate.endswith("/" + name)]
                    if matches:
                        for match in sorted(matches):
                            show_sample(archive, match)
                    else:
                        print(f"\nCould not find {name!r}. Available entries:")
                        for candidate in sorted(archive_names)[:40]:
                            print("-", candidate)

    except BadZipFile:
        print("The ZIP archive could not be opened. It may be corrupted.")
    except OSError as error:
        print(f"Could not read the archive: {error}")


if __name__ == "__main__":
    main()
