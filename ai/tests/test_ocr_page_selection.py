"""Complete-transcript coverage for both configured OCR entry points."""

from types import SimpleNamespace
from unittest.mock import AsyncMock


from paperless_ai.agents import smart_graph_agent as agent
from paperless_ai.core.config import AgentConfig


def config(**kwargs) -> AgentConfig:
    """Build an isolated configuration without environment defaults."""
    return AgentConfig(
        _env_file=None,
        paperless_url="http://localhost",
        paperless_token="t",
        metadata_model="test-metadata",
        chat_model="test-chat",
        **kwargs,
    )


async def test_evaluation_uses_paddle_helper(monkeypatch):
    """Evaluation sends the full document through the production parser."""
    parse = AsyncMock(return_value=("# Table\n\n表", {"pages": []}, 43, 1.0))
    monkeypatch.setattr("paperless_ai.agents.paddle_ocr.run_paddle_ocr", parse)
    strategy = SimpleNamespace(
        extract=AsyncMock(return_value=agent._ExtractedMetadata(title="Parsed"))
    )
    cfg = config(ocr_endpoint="http://paddle")
    result = await agent.SmartDocumentAgent(cfg, strategy).process("whole.pdf", {})
    parse.assert_awaited_once_with("whole.pdf", cfg)
    assert result.pages == 43
    assert result.ocr_method == "layout-parsing"
    assert result.metadata.full_ocr_transcript == "# Table\n\n表"
    assert "# Table\n\n表" in strategy.extract.await_args.args[0]
