"""
Assign deterministic validation/test splits to evaluation corpus entries.

This script assigns
a "split" field to each entry:
  - "validation": held out for hyperparameter research (10 entries)
  - "test": used for final evaluation (remaining entries)

The validation keys are chosen to cover varied document types.

Usage:
    python assign_splits.py
"""

import json
from pathlib import Path

EVAL_DATASET_PATH = Path(__file__).parent / "eval_dataset.json"

# Validation set: 10 entries chosen for diversity
VALIDATION_KEYS = {
    "fkff0016",
    "fklm0254",
    "gqhb0141",
    "hlhj0239",
    "fkhg0105",
    "grmj0172",
    "hgcb0104",
    "jmyg0244",
    "frvb0205",
    "jjcj0064",
}


def main():
    if not EVAL_DATASET_PATH.exists():
        print(f"Error: Evaluation corpus not found at {EVAL_DATASET_PATH}")
        return 1

    with open(EVAL_DATASET_PATH, "r") as f:
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

    with open(EVAL_DATASET_PATH, "w") as f:
        json.dump(data, f, indent=2)

    # Summary
    validation_count = sum(1 for e in entries if e.get("split") == "validation")
    test_count = sum(1 for e in entries if e.get("split") == "test")

    print(f"\nAssigned splits to {len(entries)} entries:")
    print(f"  Validation: {validation_count}")
    print(f"  Test: {test_count}")
    print(f"\nSaved to {EVAL_DATASET_PATH}")

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
