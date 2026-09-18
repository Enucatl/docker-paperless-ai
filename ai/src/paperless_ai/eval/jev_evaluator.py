"""TypeSafe Jev evaluation for extracted Paperless metadata."""

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from paperless_common.telemetry import set_span_attributes, start_span
from typesafe_sdk import Noul


@dataclass(frozen=True)
class JevMetadataEvaluation:
    """Continuous Jev judgments for one metadata prediction."""

    date_score: float
    correspondent_score: float
    title_score: float
    date_confidence: float
    correspondent_confidence: float
    title_confidence: float
    model: str | None = None

    @property
    def aggregate_score(self) -> float:
        """Return the arithmetic mean of the three field scores."""
        return (self.date_score + self.correspondent_score + self.title_score) / 3


DATE_QUESTION = (
    "Is the predicted date an appropriate primary metadata date for this "
    "document? Judge only from the document evidence. The primary date is "
    "the date that best represents the document itself, such as an invoice, "
    "letter, issue, statement, contract, publication, or creation date. "
    "Do not select an incidental due date, deadline, birth date, service "
    "period boundary, transaction date, appointment date, historical date, "
    "or date belonging to another referenced document. If the prediction is "
    "null, answer yes when the document genuinely lacks a reasonably "
    "identifiable primary document date. A supported date need not appear "
    "verbatim if the document clearly supports it."
)

CORRESPONDENT_QUESTION = (
    "Is the predicted correspondent the appropriate Paperless correspondent "
    "for this document? Judge only from the document evidence. The "
    "correspondent is the primary person or organization associated with the "
    "document. Accept reasonable punctuation, capitalization, abbreviations, "
    "legal-suffix formatting, OCR errors, minor spelling differences, and "
    "reasonable organization or department presentation differences when "
    "they represent the same entity. Reject genuinely different entities. If "
    "the prediction is null, answer yes when the document genuinely lacks an "
    "identifiable correspondent."
)

TITLE_QUESTION = (
    "Is the predicted title an appropriate Paperless title for this document? "
    "Judge only from the document evidence. The title should be accurate, "
    "specific, concise enough to browse, and useful for finding the document "
    "later. It may be a semantic summary rather than a literal heading. "
    "Normally reject generic placeholders such as Document, Letter, Untitled, "
    "Scan, or PDF unless one is genuinely justified by the document. If the "
    "prediction is null or empty, answer yes only when no useful title can be "
    "supported from the document."
)


def _question(instructions: str) -> Noul:
    """Build a yes/no Jev question with explicit outcome semantics."""
    return Noul(
        instructions=instructions,
        criteria={
            "true": "The predicted metadata is appropriate and supported.",
            "false": "The predicted metadata is inappropriate, unsupported, or incidental.",
        },
    )


class JevMetadataEvaluator:
    """Evaluate all three extracted metadata fields in one Jev request.

    Attributes:
        client: Reused synchronous TypeSafe client.
        model: Fixed Jev model used for this experiment.
    """

    def __init__(self, client: Any, model: str) -> None:
        """Initialize a Jev evaluator.

        Args:
            client: A configured synchronous ``TypeSafeClient``.
            model: TypeSafe model name.
        """
        self.client = client
        self.model = model

    async def evaluate(
        self,
        *,
        document_context: str,
        title: str | None,
        date: str | None,
        correspondent: str | None,
    ) -> JevMetadataEvaluation:
        """Judge date, correspondent, and title with one System One call.

        Args:
            document_context: Exact context supplied to metadata extraction.
            title: Predicted Paperless title, including ``None`` when absent.
            date: Predicted ISO document date, including ``None`` when absent.
            correspondent: Predicted Paperless correspondent, including ``None``.

        Returns:
            Continuous positive-answer probabilities for all three fields.

        Raises:
            KeyError: If TypeSafe omits one of the requested answers.
            TypeError: If an answer does not expose a numeric probability.
        """
        state = {
            "document": document_context,
            "predicted_metadata": {
                "title": title,
                "date": date,
                "correspondent": correspondent,
            },
        }
        questions = {
            "date": _question(DATE_QUESTION),
            "correspondent": _question(CORRESPONDENT_QUESTION),
            "title": _question(TITLE_QUESTION),
        }

        trace_input = json.dumps(state, ensure_ascii=False, default=str)
        with start_span(
            "paperless_ai.eval.jev_metadata",
            **{
                "openinference.span.kind": "LLM",
                "llm.provider": "typesafe",
                "llm.model_name": self.model,
                "input.value": trace_input,
                "input.mime_type": "application/json",
            },
        ) as span:
            response = await asyncio.to_thread(
                self.client.system_one,
                state=state,
                questions=questions,
                model=self.model,
            )

            answers = response.nouls
            evaluation = JevMetadataEvaluation(
                date_score=float(answers["date"].noul),
                correspondent_score=float(answers["correspondent"].noul),
                title_score=float(answers["title"].noul),
                date_confidence=_confidence(answers["date"]),
                correspondent_confidence=_confidence(answers["correspondent"]),
                title_confidence=_confidence(answers["title"]),
                model=getattr(response, "model", self.model),
            )
            set_span_attributes(
                span,
                **{
                    "output.value": json.dumps(
                        {
                            "date": evaluation.date_score,
                            "correspondent": evaluation.correspondent_score,
                            "title": evaluation.title_score,
                            "date_confidence": evaluation.date_confidence,
                            "correspondent_confidence": evaluation.correspondent_confidence,
                            "title_confidence": evaluation.title_confidence,
                        },
                        ensure_ascii=False,
                    ),
                    "output.mime_type": "application/json",
                },
            )
            return evaluation


def _confidence(answer: Any) -> float:
    """Return Jev's confidence, deriving it from a binary probability if needed.

    Args:
        answer: A TypeSafe yes/no answer.

    Returns:
        The confidence associated with the answer's most likely outcome.
    """
    if (confidence := getattr(answer, "confidence", None)) is not None:
        return float(confidence)

    probability = float(answer.noul)
    return max(probability, 1 - probability)
