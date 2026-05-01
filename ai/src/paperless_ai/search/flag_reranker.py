"""Small local compatibility wrapper matching FlagEmbedding's reranker API."""

from typing import Sequence


class FlagReranker:
    """Local cross-encoder reranker with a FlagEmbedding-like interface."""

    def __init__(
        self, model_name_or_path: str, use_fp16: bool = False, max_length: int = 512
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path
        )
        self._model.eval()
        self._model.to(self._device)
        self._max_length = max_length

        if use_fp16 and self._device != "cpu":
            self._model.half()

    def compute_score(
        self,
        sentence_pairs: Sequence[Sequence[str]] | Sequence[str],
        normalize: bool = False,
    ) -> list[float] | float:
        import torch

        pairs = sentence_pairs
        if pairs and isinstance(pairs[0], str):
            pairs = [pairs]  # type: ignore[assignment]

        inputs = self._tokenizer(
            list(pairs),
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self._max_length,
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}

        with torch.no_grad():
            scores = self._model(**inputs, return_dict=True).logits.view(-1).float()
            if normalize:
                scores = torch.sigmoid(scores)

        values = scores.cpu().tolist()
        return values[0] if len(values) == 1 else values
