"""Complete-transcript coverage for both configured OCR entry points."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import fitz
import pytest

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


@pytest.mark.parametrize("blank", [False, True])
async def test_legacy_processes_every_page(tmp_path, monkeypatch, blank):
    """A long PDF includes middle pages; wholly empty OCR cannot replace content."""
    path = tmp_path / "long.pdf"
    with fitz.open() as pdf:
        for _ in range(43):
            pdf.new_page()
        pdf.save(path)
    rendered = []

    def render(file_path, page_idx, max_dim):
        """Record page coverage without rendering images."""
        rendered.append(page_idx)
        return "image"

    monkeypatch.setattr(agent, "_render_page_to_base64", render)
    monkeypatch.setattr(agent, "complete", AsyncMock(return_value=object()))
    monkeypatch.setattr(
        agent, "_get_completion_text", lambda _: "" if blank else "page"
    )
    if blank:
        with pytest.raises(ValueError, match="empty transcript"):
            await agent.run_vision_ocr_only(str(path), config())
    else:
        text, pages, _ = await agent.run_vision_ocr_only(str(path), config())
        assert pages == 43
        assert text.split("\n\n") == ["page"] * 43
    assert sorted(rendered) == list(range(43))


async def test_evaluation_uses_paddle_helper(monkeypatch):
    """Evaluation sends the full document through the production parser."""
    parse = AsyncMock(return_value=("# Table\n\n表", {"pages": []}, 43, 1.0))
    monkeypatch.setattr("paperless_ai.agents.paddle_ocr.run_paddle_ocr", parse)
    strategy = SimpleNamespace(
        extract=AsyncMock(return_value=agent._ExtractedMetadata(title="Parsed"))
    )
    cfg = config(ocr_backend="paddleocr", ocr_endpoint="http://paddle")
    result = await agent.SmartDocumentAgent(cfg, strategy).process("whole.pdf", {})
    parse.assert_awaited_once_with("whole.pdf", cfg)
    assert result.pages == 43
    assert result.ocr_method == "paddleocr"
    assert result.metadata.full_ocr_transcript == "# Table\n\n表"
    assert "# Table\n\n表" in strategy.extract.await_args.args[0]


async def test_legacy_evaluation_does_not_hit_graph_step_limit(tmp_path, monkeypatch):
    """Removing sampling also permits more than 25 graph iterations."""
    path = tmp_path / "long-eval.pdf"
    with fitz.open() as pdf:
        for _ in range(101):
            pdf.new_page()
        pdf.save(path)
    monkeypatch.setattr(agent, "_render_page_to_base64", lambda *args: "image")
    monkeypatch.setattr(agent, "complete", AsyncMock(return_value=object()))
    monkeypatch.setattr(agent, "_get_completion_text", lambda _: "page")
    strategy = SimpleNamespace(
        extract=AsyncMock(return_value=agent._ExtractedMetadata(title="Parsed"))
    )
    result = await agent.SmartDocumentAgent(config(), strategy).process(str(path), {})
    assert result.pages == 101
    assert result.metadata.full_ocr_transcript.split("\n\n") == ["page"] * 101
