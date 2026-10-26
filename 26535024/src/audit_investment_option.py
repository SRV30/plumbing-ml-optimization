"""Audit the technician investment-option flag before changing the baseline.

Run from the repository root:
    python src/audit_investment_option.py

This script is read-only. It reports flag counts from technicians.csv, breaks
them down by workday date, and prints the small TDCData_investment.txt member
from the original ZIP if that member is present.
"""

from __future__ import annotations

from pathlib import Path
from zipfile import BadZipFile, ZipFile

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TECHNICIANS_PATH = ROOT / "data" / "processed" / "technicians.csv"
ZIP_PATH = ROOT / "data" / "raw" / "TDCData_instance.zip"
INVESTMENT_MEMBER_NAME = "TDCData_investment.txt"


def main() -> None:
    if not TECHNICIANS_PATH.exists():
        print(f"Processed technician CSV not found: {TECHNICIANS_PATH.relative_to(ROOT)}")
        print("Run python src/parse_tdc_records.py first.")
        return

    tech = pd.read_csv(TECHNICIANS_PATH, keep_default_na=False)
    required = {"technician_id", "time_window_start", "investment_option"}
    missing = sorted(required - set(tech.columns))
    if missing:
        print(f"Technician CSV is missing required column(s): {missing}")
        return

    flag = tech["investment_option"].astype(str).str.strip().str.lower()
    valid = flag.map({"true": True, "false": False, "1": True, "0": False})
    invalid_count = int(valid.isna().sum())

    print("INVESTMENT-OPTION FLAG AUDIT")
    print("============================")
    print(f"Technician rows: {len(tech):,}")
    print(f"Unique technician IDs: {tech['technician_id'].nunique():,}")
    print(f"True:  {int((valid == True).sum()):,}")  
    print(f"False: {int((valid == False).sum()):,}")  
    print(f"Missing/invalid flag values: {invalid_count:,}")

    starts = pd.to_datetime(tech["time_window_start"], errors="coerce")
    by_day = pd.DataFrame({
        "workday": starts.dt.strftime("%Y-%m-%d").fillna("[invalid date]"),
        "investment_option": valid.map({True: "True", False: "False"}).fillna("[invalid]"),
    })
    print("\nCounts by workday and investment-option flag:")
    print(
        by_day.groupby(["workday", "investment_option"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .to_string()
    )

    print("\nOriginal TDCData_investment.txt contents")
    print("========================================")
    if not ZIP_PATH.exists():
        print(f"ZIP not found: {ZIP_PATH.relative_to(ROOT)}")
        print("The flag counts above are still valid for the processed technician CSV.")
        return

    try:
        with ZipFile(ZIP_PATH, "r") as archive:
            matches = [
                name for name in archive.namelist()
                if Path(name).name.lower() == INVESTMENT_MEMBER_NAME.lower()
            ]
            if not matches:
                print(f"Member {INVESTMENT_MEMBER_NAME!r} was not found in the archive.")
                print("ZIP members:")
                for name in archive.namelist():
                    print(f"- {name}")
                return
            for member in matches:
                print(f"\n--- {member} ---")
                with archive.open(member, "r") as handle:
                    print(handle.read().decode("utf-8", errors="replace"))
    except (BadZipFile, OSError) as error:
        print(f"Could not inspect source ZIP: {error}")
        return

    true_count = int((valid == True).sum())  
    false_count = int((valid == False).sum())  
    if true_count == 710 and false_count == 720:
        print("\nCount pattern matches 710 True / 720 False.")
        print("This supports a possible explanation for the 720 vs. 1,430 count,")
        print("but do not exclude True records until the investment file semantics")
        print("and paper formulation establish that they are optional add-on technicians.")
    else:
        print("\nThe counts do not exactly match the hypothesized 710 True / 720 False split.")
        print("Do not infer the meaning of the flag from counts alone.")


if __name__ == "__main__":
    main()
