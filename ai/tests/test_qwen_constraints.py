"""Opt-in contract test for the configured Qwen-compatible endpoint."""

import os

import pytest

from paperless_ai.inference import complete


@pytest.mark.live
@pytest.mark.asyncio
async def test_qwen_json_object_contract() -> None:
    """Verify the explicit JSON-object request contract when enabled."""
    if os.environ.get("PAPERLESS_AI_LIVE") != "1":
        pytest.skip("set PAPERLESS_AI_LIVE=1 to contact the Qwen endpoint")

    response = await complete(
        model="Qwen/Qwen3.5-9B",
        api_base="http://complex.home.arpa:8107/v1",
        domain="metadata_contract",
        messages=[
            {"role": "system", "content": "Return only JSON."},
            {"role": "user", "content": 'Return {"ok": true}.'},
        ],
        response_format={"type": "json_object"},
        max_tokens=32,
        temperature=0,
    )

    assert response.content
