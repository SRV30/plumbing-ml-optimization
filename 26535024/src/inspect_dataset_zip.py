"""Inspect a downloaded dataset ZIP without extracting it.

Expected default location:
    data/raw/TDCData_instance.zip

Run from the repository root:
    python src/inspect_dataset_zip.py
"""

from pathlib import Path
from zipfile import BadZipFile, ZipFile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ZIP_PATH = PROJECT_ROOT / "data" / "raw" / "TDCData_instance.zip"
MAX_ENTRIES_TO_PRINT = 80


def format_bytes(size: int) -> str:
    """Return a human-readable file size."""
    units = ("B", "KB", "MB", "GB")
    value = float(size)

    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024

    return f"{value:.1f} GB"


def main() -> None:
    """Print archive contents and identify files worth inspecting next."""
    zip_path = DEFAULT_ZIP_PATH

    if not zip_path.exists():
        print("Dataset ZIP not found.")
        print(f"Expected location: {zip_path}")
        print(
            "\nDownload TDCData_instance.zip from "
            "https://zenodo.org/records/7904015"
        )
        print("Place it in the data/raw/ folder, then run this script again.")
        return

    print(f"Dataset archive: {zip_path.relative_to(PROJECT_ROOT)}")
    print(f"Archive size: {format_bytes(zip_path.stat().st_size)}")

    try:
        with ZipFile(zip_path, "r") as archive:
            entries = [item for item in archive.infolist() if not item.is_dir()]
            print(f"File count (excluding folders): {len(entries)}")

            print("\nFirst archive entries:")
            for item in entries[:MAX_ENTRIES_TO_PRINT]:
                print(f"- {item.filename} ({format_bytes(item.file_size)})")

            if len(entries) > MAX_ENTRIES_TO_PRINT:
                print(
                    f"... {len(entries) - MAX_ENTRIES_TO_PRINT} more files "
                    "not shown."
                )

            extensions: dict[str, int] = {}
            for item in entries:
                suffix = Path(item.filename).suffix.lower() or "[no extension]"
                extensions[suffix] = extensions.get(suffix, 0) + 1

            print("\nFile types:")
            for suffix, count in sorted(extensions.items()):
                print(f"- {suffix}: {count}")

            tabular_files = [
                item.filename
                for item in entries
                if Path(item.filename).suffix.lower() in {".csv", ".xlsx", ".xls"}
            ]

            print("\nPotential tabular files:")
            if tabular_files:
                for filename in tabular_files:
                    print(f"- {filename}")
            else:
                print(
                    "No CSV or Excel files found inside the ZIP. "
                    "The dataset may use another format."
                )

            bad_member = archive.testzip()
            if bad_member is None:
                print("\nZIP integrity check: passed")
            else:
                print(f"\nZIP integrity check: failed at {bad_member}")

    except BadZipFile:
        print("The file is not a valid ZIP archive or is corrupted.")
    except OSError as error:
        print(f"Could not read the ZIP file: {error}")


if __name__ == "__main__":
    main()
