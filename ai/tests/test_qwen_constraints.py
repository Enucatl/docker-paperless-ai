"""
Test Qwen 3.5 support for Pydantic output constraints.

Checks whether Qwen actually supports response_schema (JSON schema constraints)
or falls back to basic JSON mode or unstructured text.
"""

import asyncio
import litellm
from pydantic import BaseModel, Field


class SampleOutput(BaseModel):
    """Sample structured output for testing."""

    title: str = Field(..., description="Document title")
    date: str = Field(..., description="Document date in YYYY-MM-DD format")
    correspondent: str | None = Field(None, description="Organization or person name")


async def test_qwen_response_schema_support():
    """
    Check if Qwen 3.5 supports response_schema (JSON schema constraints).

    Run this manually with:
        uv run pytest tests/test_qwen_constraints.py::test_qwen_response_schema_support -s
    """
    model = "openai/Qwen/Qwen3.5-9B"
    api_base = "http://complex.home.arpa:8107/v1"

    print(f"\n=== Testing {model} ===")

    # Check what litellm thinks the model supports
    print(
        f"\n1. litellm.supports_response_schema('{model}'): {litellm.supports_response_schema(model=model)}"
    )

    supported_params = litellm.get_supported_openai_params(model=model) or []
    print(f"2. Supported OpenAI params: {supported_params}")
    print(f"   - Has 'response_format': {'response_format' in supported_params}")

    # Try actual request with response_format
    print(f"\n3. Testing actual request...")
    try:
        response = await litellm.acompletion(
            model=model,
            api_base=api_base,
            messages=[
                {"role": "system", "content": "Extract metadata from this invoice."},
                {
                    "role": "user",
                    "content": "Invoice from Acme Corp dated 2024-01-15 for $100.",
                },
            ],
            response_format=SampleOutput,
            max_tokens=100,
            temperature=0,
        )
        print(f"   ✓ response_format=SampleOutput worked!")
        print(f"   Response type: {type(response.choices[0].message.content)}")
        print(f"   Content: {response.choices[0].message.content}")
    except Exception as e:
        print(f"   ✗ response_format=SampleOutput failed: {e}")

        # Try json_object fallback
        print(f"\n   Trying fallback: response_format='json_object'...")
        try:
            response = await litellm.acompletion(
                model=model,
                api_base=api_base,
                messages=[
                    {
                        "role": "system",
                        "content": 'Extract metadata as JSON: {"title": "...", "date": "YYYY-MM-DD", "correspondent": "..."}',
                    },
                    {
                        "role": "user",
                        "content": "Invoice from Acme Corp dated 2024-01-15 for $100.",
                    },
                ],
                response_format={"type": "json_object"},
                max_tokens=100,
                temperature=0,
            )
            print(f"   ✓ response_format='json_object' worked!")
            print(f"   Content: {response.choices[0].message.content}")
        except Exception as e2:
            print(f"   ✗ response_format='json_object' also failed: {e2}")
            print(
                f"\n   Conclusion: Qwen doesn't support structured output constraints."
            )
            print(f"   You'll need to parse unstructured text output instead.")


if __name__ == "__main__":
    asyncio.run(test_qwen_response_schema_support())
