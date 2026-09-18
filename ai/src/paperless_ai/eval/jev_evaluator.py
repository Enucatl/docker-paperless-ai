"""TypeSafe Jev evaluation for extracted Paperless metadata."""

import asyncio
from dataclasses import dataclass
from typing import Any

from typesafe_sdk import Noul


@dataclass(frozen=True)
class JevMetadataEvaluation:
    """Continuous Jev judgments for one metadata prediction."""

    date_score: float
    correspondent_score: float
    title_score: float
    date_confidence: float | None = None
    correspondent_confidence: float | None = None
    title_confidence: float | None = None
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

    def __init__(self, client: Any, model: str, max_concurrency: int = 5) -> None:
        """Initialize a Jev evaluator.

        Args:
            client: A configured synchronous ``TypeSafeClient``.
            model: TypeSafe model name.
            max_concurrency: Maximum number of synchronous calls in flight.
        """
        self.client = client
        self.model = model
        self._semaphore = asyncio.Semaphore(max_concurrency)

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

        async with self._semaphore:
            response = await asyncio.to_thread(
                self.client.system_one,
                state=state,
                questions=questions,
                model=self.model,
            )

        answers = response.nouls
        return JevMetadataEvaluation(
            date_score=float(answers["date"].noul),
            correspondent_score=float(answers["correspondent"].noul),
            title_score=float(answers["title"].noul),
            model=getattr(response, "model", self.model),
        )
