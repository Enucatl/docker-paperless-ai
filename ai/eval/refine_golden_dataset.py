
import json
import logging
import os
import sys
from pathlib import Path
import re

# Add the 'ai' directory to sys.path to import core modules
sys.path.append(str(Path(__file__).parent.parent))

from core.config import AgentConfig
import litellm
from pydantic import BaseModel
from typing import Optional

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"

class RefinedMetadata(BaseModel):
    correspondent: Optional[str] = None
    date: Optional[str] = None

async def refine_entry(entry, config: AgentConfig):
    transcript = entry.get("expected_ocr_transcript", "")
    if not transcript:
        return entry

    prompt = f"""
Analyze the following document OCR transcript and extract the primary Correspondent (Sender/Organization) and the Document Date.
Return only a JSON object with 'correspondent' and 'date' (YYYY-MM-DD).
If the date is ambiguous, choose the most likely one (e.g., from a header or signature).
If you cannot find a field, return null for it.

Transcript:
{transcript[:4000]}
"""

    try:
        response = await litellm.acompletion(
            model=config.effective_metadata_model,
            messages=[{"role": "user", "content": prompt}],
            response_format=RefinedMetadata,
            temperature=0,
            **config.get_litellm_kwargs()
        )
        
        raw = response.choices[0].message.content
        refined = RefinedMetadata.model_validate_json(raw)
        
        if refined.correspondent:
            entry["expected_correspondent"] = refined.correspondent
        if refined.date:
            # Simple YYYY-MM-DD validation
            if re.match(r"\d{4}-\d{2}-\d{2}", refined.date):
                entry["expected_date"] = refined.date
                
        log.info(f"Refined {entry.get('original_key')}: {entry['expected_correspondent']} | {entry['expected_date']}")
    except Exception as e:
        log.error(f"Failed to refine {entry.get('original_key')}: {e}")
        
    return entry

async def main():
    if not GOLDEN_DATASET_PATH.exists():
        log.error("Golden dataset not found")
        return

    # Mock/Load config to get API keys and model settings
    # We rely on env vars being present in the container
    config = AgentConfig()
    
    with open(GOLDEN_DATASET_PATH, "r") as f:
        data = json.load(f)
        
    entries = data.get("entries", [])
    refined_entries = []
    
    for entry in entries:
        if entry.get("expected_correspondent") == "Unknown (IDL Sample)":
            entry = await refine_entry(entry, config)
        refined_entries.append(entry)
        
    data["entries"] = refined_entries
    
    with open(GOLDEN_DATASET_PATH, "w") as f:
        json.dump(data, f, indent=2)
    
    log.info("Finished refining golden dataset")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
