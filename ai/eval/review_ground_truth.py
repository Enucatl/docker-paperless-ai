"""
Interactive human-in-the-loop script to complete ground truth annotations.

For entries with null correspondent and/or date, this script:
1. Extracts text from the PDF
2. Calls LLM to propose correspondent and date
3. Prompts human for confirmation/correction
4. Writes back to golden_dataset.json with verified values or _verified_null flags

Usage:
    python review_ground_truth.py
"""

import json
import logging
import sys
import re
from pathlib import Path
from typing import Optional  # used in prompt_user_confirmation return type

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"


async def propose_ground_truth(entry: dict, agent) -> dict:
    """Run the full agent pipeline on the PDF, same as production, and return proposals."""
    file_path = entry["file_path"]
    if not Path(file_path).exists():
        log.warning(f"File not found: {file_path}")
        return {"correspondent": None, "date": None, "ocr_transcript": ""}

    try:
        result = await agent.process(file_path, existing_hints={})
        return {
            "title": result.metadata.title,
            "correspondent": result.metadata.correspondent,
            "date": result.metadata.document_date,
            "ocr_transcript": result.metadata.full_ocr_transcript,
        }
    except Exception as e:
        log.error(f"Agent failed for {entry.get('original_key')}: {e}")
        return {"title": None, "correspondent": None, "date": None, "ocr_transcript": ""}


def prompt_user_confirmation(
    key: str, field: str, proposed: Optional[str], current: Optional[str]
) -> tuple[bool, Optional[str]]:
    """
    Prompt user to confirm/correct proposed value.

    Returns: (accepted, value)
        accepted=True, value=proposed → user typed 'y'
        accepted=True, value=<custom> → user typed a custom value
        accepted=True, value=None → user typed 'n' (genuinely null)
        accepted=False, value=None → user typed 'skip' (abort this entry)
    """
    print(f"\n  {field.upper()} for {key}:")
    print(f"    Current: {current}")
    print(f"    Proposed: {proposed}")
    print(f"    Actions: [y]es / [n]o (null) / [s]kip this entry / type custom value")

    while True:
        user_input = input(f"    Your choice: ").strip().lower()

        if user_input == "y":
            return (True, proposed)
        elif user_input == "n":
            return (True, None)
        elif user_input == "s":
            return (False, None)
        elif user_input:
            # Custom value
            return (True, user_input)
        else:
            print("    Please enter y, n, s, or a custom value")


async def review_entry(entry: dict, agent) -> None:
    """Review a single entry: propose LLM values, prompt user, update entry in-place."""
    key = entry.get("original_key", entry.get("file_path", "unknown"))

    # Propose new values via full agent pipeline
    log.info(f"\nReviewing {key}...")
    proposed = await propose_ground_truth(entry, agent)

    # Show OCR transcript so reviewer can judge the proposals
    transcript = proposed.get("ocr_transcript", "").strip()
    if transcript:
        print("\n" + "─" * 60)
        print(transcript)
        print("─" * 60)

    # Prompt for title (always)
    accepted, title_value = prompt_user_confirmation(
        key, "title", proposed.get("title"), entry.get("expected_title_contains")
    )
    if not accepted:
        return
    if title_value is not None:
        entry["expected_title_contains"] = title_value

    # Prompt for correspondent (always)
    accepted, corr_value = prompt_user_confirmation(
        key, "correspondent", proposed["correspondent"], entry.get("expected_correspondent")
    )
    if not accepted:
        return
    if corr_value is None:
        entry["expected_correspondent"] = None
        entry["_verified_null_correspondent"] = True
    else:
        entry["expected_correspondent"] = corr_value
        entry.pop("_verified_null_correspondent", None)

    # Prompt for date (always)
    accepted, date_value = prompt_user_confirmation(
        key, "date", proposed["date"], entry.get("expected_date")
    )
    if not accepted:
        return
    if date_value is None:
        entry["expected_date"] = None
        entry["_verified_null_date"] = True
    else:
        if not re.match(r"\d{4}-\d{2}-\d{2}", date_value):
            log.error(f"Invalid date format: {date_value}. Expected YYYY-MM-DD.")
            return
        entry["expected_date"] = date_value
        entry.pop("_verified_null_date", None)


async def main():
    if not GOLDEN_DATASET_PATH.exists():
        log.error(f"Golden dataset not found: {GOLDEN_DATASET_PATH}")
        sys.exit(1)

    with open(GOLDEN_DATASET_PATH, "r") as f:
        data = json.load(f)

    entries = data.get("entries", [])

    # All entries are reviewed — title/correspondent/date are confirmed or corrected
    to_review = entries

    if not to_review:
        log.info("Golden dataset is empty.")
        return

    log.info(f"Found {len(to_review)} entries needing review")
    log.info("Loading agent...")

    from core.config import AgentConfig
    from agents.smart_graph_agent import SmartDocumentAgent
    config = AgentConfig.from_env()
    agent = SmartDocumentAgent(config)

    log.info("Starting interactive review. Press Ctrl+C to exit at any time.\n")

    try:
        for entry in to_review:
            # review_entry mutates entry in-place
            await review_entry(entry, agent)
            # Re-save after each entry in case of interruption
            with open(GOLDEN_DATASET_PATH, "w") as f:
                json.dump(data, f, indent=2)

        log.info(f"\nReview complete! Updated {GOLDEN_DATASET_PATH}")

    except KeyboardInterrupt:
        log.info("\nInterrupted. Saving progress...")
        with open(GOLDEN_DATASET_PATH, "w") as f:
            json.dump(data, f, indent=2)
        log.info(f"Progress saved to {GOLDEN_DATASET_PATH}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
