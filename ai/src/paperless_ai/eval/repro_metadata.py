#!/usr/bin/env python3
"""Minimal reproduction script for metadata extraction via LiteLLM.

Usage:
    python repro_metadata.py --api-key AIza...
    python repro_metadata.py --api-key AIza... --model gemini/gemini-2.5-flash
"""

import argparse
import os
import logging
import sys
from datetime import datetime
from pprint import pprint
from typing import Optional

import litellm
from pydantic import BaseModel, Field, field_validator

# Configure comprehensive logging for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

# Enable litellm debug mode
litellm.set_verbose = True

OCR_TEXT = """\
BROWN & WILLIAMSON TOBACCO CORPORATION
RESEARCH & DEVELOPMENT
INTERNAL CORRESPONDENCE

<watermark>SECRET.</watermark>
<watermark>Do Not Copy Or Make Notes Of</watermark>
<watermark>This Page And Do Not Give Any Of The</watermark>
<watermark>Information Contained In This Document</watermark>
<watermark>To Anyone Except In Conformance With</watermark>
<watermark>The Moorgate Secrecy Protocol.</watermark>

TO: MR. P. H. HARPER
CC: MR. T. E. SANDEFUR
DR. P. L. AULBACH
FROM: MR. D. S. ROTH
DATE: JANUARY 13, 1983
SUBJECT: KENT 80/KS TAR REDUCTION - FINLAND/907
Ref S/R 107/82

After evaluations and rejections of tar reduction samples 49B, 50B, and 51B we have reviewed
deliveries and designs of several KENT products to find one which meets the 10.0 mg/cig. (DPM)
and 0.7 mg/cig. nicotine constraints.

We propose the following two trials.

<table>
  <tr><td></td><td>Trial 1</td><td>Trial 2</td></tr>
  <tr><td>•Blend (%):</td><td></td><td></td></tr>
  <tr><td></td><td>MGLF-31</td><td>69.9</td><td>69.9</td></tr>
  <tr><td></td><td>Oriental</td><td>10.3</td><td>10.3</td></tr>
  <tr><td></td><td>MRT</td><td>5.0</td><td>0.0</td></tr>
  <tr><td></td><td>F.C. WTS</td><td>14.8</td><td>19.8</td></tr>
  <tr><td>•Final Casing</td><td>MGE-651</td><td>MGE-651</td></tr>
  <tr><td>•Final Flavor</td><td>MGE-603</td><td>MGE-603</td></tr>
  <tr><td>•Cigarette Paper</td><td>Current KENT</td><td>---</td></tr>
  <tr><td>•Density</td><td>KENT GOLDEN LIGHTS (Less 2%)</td><td>---</td></tr>
  <tr><td>•Filter Tow</td><td>KENT GOLDEN LIGHTS</td><td>---</td></tr>
  <tr><td>•Plugwrap</td><td>9500 Coresta</td><td>---</td></tr>
  <tr><td>•Tipping</td><td>1100 Coresta MMP*</td><td>---</td></tr>
  <tr><td>•Vent Target</td><td>42-44%</td><td>---</td></tr>
</table>

*We recommend the micromechanically perforated tipping because this is the only type of tipping
with essentially invisible perforations at this high porosity level.

<image> Signature of D. S. R. </image>
D. S. R.

DSR/bfw
0014x
JAN 1 8 1983
620441151

Source: http://industrydocuments.library.ucsf.edu/tobacco/docs/klpb0135\
"""

METADATA_PROMPT = """\
Analyze the OCR text of the document below and extract the requested metadata.

Rules for extraction:
- Use null for any field you cannot determine with confidence. Do not invent or guess information.
- The "correspondent" is specifically the company, institution, or person who sent, authored, or issued the document (do not use the recipient).\
"""


class ExtractedMetadata(BaseModel):
    title: Optional[str] = Field(
        default=None,
        description=(
            "A clear, concise, and descriptive title summarizing the document's core subject "
            "or purpose for future retrieval. Maximum 100 characters. "
            "Do NOT include full sentences, conversational text, disclaimers, or notes about the extraction process."
        ),
    )
    date: Optional[str] = Field(
        default=None, description="Primary document date as YYYY-MM-DD."
    )
    correspondent: Optional[str] = Field(
        default=None,
        description="Name of the issuing organisation — not the recipient.",
    )

    @field_validator("date", mode="before")
    @classmethod
    def strip_time(cls, v):
        if not isinstance(v, str):
            return v
        try:
            return datetime.fromisoformat(v).date().isoformat()
        except ValueError:
            return v


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.environ.get("GOOGLE_API_KEY"))
    parser.add_argument("--model", default="gemini/gemini-3.1-flash-lite")
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--reasoning-effort", default=None)
    args = parser.parse_args()
    if not args.api_key:
        parser.error("--api-key is required or set GOOGLE_API_KEY")
    return args


def main() -> None:
    args = parse_args()
    api_key = args.api_key
    model = args.model
    max_tokens = args.max_tokens
    reasoning_effort = args.reasoning_effort
    logger = logging.getLogger(__name__)
    logger.info("Starting metadata extraction script")

    os.environ["GOOGLE_API_KEY"] = api_key
    logger.debug("API key set for environment")

    litellm.drop_params = True
    logger.debug("litellm.drop_params enabled")

    logger.debug(f"Checking model capabilities for: {model}")
    schema_support = litellm.supports_response_schema(model=model)
    json_support = "response_format" in (
        litellm.get_supported_openai_params(model=model) or []
    )
    logger.info(f"Model capabilities - Schema: {schema_support}, JSON: {json_support}")

    print(f"Model            : {model}")
    print(f"Schema support   : {schema_support}")
    print(f"JSON obj support : {json_support}")
    print()

    messages = [
        {"role": "system", "content": METADATA_PROMPT},
        {"role": "user", "content": OCR_TEXT},
    ]
    logger.debug(f"System message length: {len(METADATA_PROMPT)} chars")
    logger.debug(f"User message length: {len(OCR_TEXT)} chars")

    kwargs: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 1,
    }
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
        logger.debug(f"reasoning_effort set to: {reasoning_effort}")

    if schema_support:
        kwargs["response_format"] = ExtractedMetadata
        print("response_format  : Pydantic schema")
        logger.debug("Using Pydantic schema response format")
    elif json_support:
        kwargs["response_format"] = {"type": "json_object"}
        print("response_format  : json_object")
        logger.debug("Using json_object response format")
    else:
        print("response_format  : none (prompt only)")
        logger.debug("No response format specified")

    print()
    pprint(kwargs)

    logger.info("Making litellm.completion call...")
    logger.debug(
        f"Completion kwargs: model={model}, max_tokens={max_tokens}, temperature={kwargs['temperature']}"
    )
    response = litellm.completion(**kwargs)
    logger.info("Received response from litellm")

    message = response.choices[0].message
    logger.debug(
        f"Message type: {type(message)}, has reasoning: {hasattr(message, 'reasoning_content')}"
    )

    reasoning = getattr(message, "reasoning_content", None)
    if reasoning:
        logger.debug(f"Reasoning content present: {len(reasoning)} chars")
        print(
            f"reasoning_content ({len(reasoning)} chars):\n{reasoning[:500]}{'...' if len(reasoning) > 500 else ''}\n"
        )
    else:
        logger.debug("No reasoning content in response")

    raw = message.content or ""
    logger.debug(f"Raw content received: {len(raw)} chars")
    logger.debug(f"Raw content preview: {raw[:200]}")
    print(f"content ({len(raw)} chars):\n{raw}\n")

    try:
        logger.debug("Attempting strict JSON parse...")
        extracted = ExtractedMetadata.model_validate_json(raw)
        logger.info("Successfully parsed metadata with strict parse")
        print("Parsed OK:")
    except Exception as e:
        logger.warning(f"Strict parse failed: {e}, trying regex fallback...")
        print(f"Strict parse failed ({e}), trying regex fallback...")
        import re

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        try:
            logger.debug("Attempting regex-based JSON extraction...")
            extracted = ExtractedMetadata.model_validate_json(
                match.group() if match else "{}"
            )
            logger.info("Successfully parsed metadata via regex fallback")
            print("Parsed via regex:")
        except Exception as regex_e:
            logger.error(f"Regex fallback failed: {regex_e}, using empty result")
            extracted = ExtractedMetadata()
            print("Parsing failed entirely — empty result:")

    logger.info(
        f"Final extracted metadata - title: {extracted.title}, date: {extracted.date}, correspondent: {extracted.correspondent}"
    )
    print(f"  title       : {extracted.title}")
    print(f"  date        : {extracted.date}")
    print(f"  correspondent: {extracted.correspondent}")

    usage = response.usage
    if usage:
        logger.info(
            f"Token usage - input: {usage.prompt_tokens}, output: {usage.completion_tokens}"
        )
        print(f"\nTokens: in={usage.prompt_tokens} out={usage.completion_tokens}")
    else:
        logger.warning("No usage information in response")


if __name__ == "__main__":
    main()
