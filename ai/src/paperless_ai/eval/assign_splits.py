"""
Assign deterministic train/validation split to golden dataset entries.

This script runs once after ground truth review is complete. It assigns
a "split" field to each entry:
  - "validation": held out for hyperparameter research (10 entries)
  - "test": used for final evaluation (remaining entries)

The validation set is chosen for representativeness:
  - At least 1 manual entry (non-IDL)
  - At least 2 entries with both-null fields
  - At least 2 entries with null-date-only
  - Remaining slots filled with diverse complete entries

Usage:
    python assign_splits.py
"""

import json
from pathlib import Path

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"

# Validation set: 10 entries chosen for diversity
VALIDATION_KEYS = {
    "fkff0016",
    "fklm0254",  # Both-null (tests null handling)
    "gqhb0141",  # Both-null, second example
    "hlhj0239",  # Null-date-only (pharma, opioid)
    "fkhg0105",  # Null-date-only (Lorillard)
    "grmj0172",  # Complete, non-tobacco (GMR Marketing)
    "hgcb0104",  # Complete, media/newspaper (USA TODAY)
    "jmyg0244",  # Complete, recent date, pharma (Covidien)
    "frvb0205",  # Complete, old date (1976), UK company (BAT)
    "jjcj0064",  # Complete, individual person (Margaret Yates)
}


def main():
    if not GOLDEN_DATASET_PATH.exists():
        print(f"Error: Golden dataset not found at {GOLDEN_DATASET_PATH}")
        return 1

    with open(GOLDEN_DATASET_PATH, "r") as f:
        data = json.load(f)

    entries = data.get("entries", [])

    # Assign splits
    assigned_count = 0
    for entry in entries:
        key = entry.get("original_key") or entry.get("file_path", "")
        # Extract just the key from full paths
        if "/" in key:
            key = Path(key).stem

        if key in VALIDATION_KEYS:
            entry["split"] = "validation"
            assigned_count += 1
        else:
            entry["split"] = "test"

    data["entries"] = entries

    with open(GOLDEN_DATASET_PATH, "w") as f:
        json.dump(data, f, indent=2)

    # Summary
    validation_count = sum(1 for e in entries if e.get("split") == "validation")
    test_count = sum(1 for e in entries if e.get("split") == "test")

    print(f"\nAssigned splits to {len(entries)} entries:")
    print(f"  Validation: {validation_count}")
    print(f"  Test: {test_count}")
    print(f"\nSaved to {GOLDEN_DATASET_PATH}")

    # Verify the expected keys were found
    found_keys = {
        e.get("original_key") or Path(e.get("file_path")).stem for e in entries
    }
    not_found = VALIDATION_KEYS - found_keys
    if not_found:
        print(f"\nWarning: validation keys not found: {not_found}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
