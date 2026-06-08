from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_NLI_MODEL = "FacebookAI/roberta-large-mnli"
DEFAULT_CROSS_ENCODER_MODEL = "distilroberta-base"


def resolve_device(device: str | None = None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def semantic_interaction_features(
    question_embeddings: np.ndarray,
    answer_embeddings: np.ndarray,
) -> np.ndarray:
    if question_embeddings.shape != answer_embeddings.shape:
        raise ValueError(
            "Question and answer embedding matrices must have identical shapes; "
            f"got {question_embeddings.shape} and {answer_embeddings.shape}."
        )

    q = np.asarray(question_embeddings, dtype=np.float32)
    a = np.asarray(answer_embeddings, dtype=np.float32)
    numerator = np.sum(q * a, axis=1, keepdims=True)
    denominator = np.linalg.norm(q, axis=1, keepdims=True) * np.linalg.norm(a, axis=1, keepdims=True)
    cosine = numerator / np.clip(denominator, a_min=1e-12, a_max=None)
    return np.concatenate([q, a, np.abs(q - a), q * a, cosine], axis=1)


class TransformerBiEncoder:
    def __init__(
        self,
        model_name: str = DEFAULT_SEMANTIC_MODEL,
        device: str | None = None,
        max_length: int = 256,
    ) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            fallback = self.tokenizer.eos_token or self.tokenizer.unk_token
            if fallback is None:
                raise ValueError(f"{model_name} tokenizer does not define a usable padding token.")
            self.tokenizer.pad_token = fallback
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        batches: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                output = self.model(**encoded)
                pooled = self._mean_pool(output.last_hidden_state, encoded["attention_mask"])
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            batches.append(pooled.cpu().numpy().astype(np.float32))

        hidden_size = int(self.model.config.hidden_size)
        return np.vstack(batches) if batches else np.empty((0, hidden_size), dtype=np.float32)

    def interaction_features(
        self,
        questions: Sequence[str],
        answers: Sequence[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        question_embeddings = self.encode(questions, batch_size=batch_size)
        answer_embeddings = self.encode(answers, batch_size=batch_size)
        return semantic_interaction_features(question_embeddings, answer_embeddings)


@dataclass(frozen=True)
class NLILabelIndices:
    entailment: int
    neutral: int
    contradiction: int


def resolve_nli_label_indices(id2label: dict[int | str, str]) -> NLILabelIndices:
    normalized = {int(index): label.lower() for index, label in id2label.items()}

    def find(label_name: str) -> int | None:
        return next((index for index, label in normalized.items() if label_name in label), None)

    entailment = find("entail")
    neutral = find("neutral")
    contradiction = find("contrad")
    if entailment is not None and neutral is not None and contradiction is not None:
        return NLILabelIndices(entailment, neutral, contradiction)

    if set(normalized) == {0, 1, 2}:
        # Standard MNLI ordering used by BART/RoBERTa checkpoints.
        return NLILabelIndices(entailment=2, neutral=1, contradiction=0)

    raise ValueError(f"Could not identify entailment/neutral/contradiction labels from {id2label}.")


class NLIProbabilityExtractor:
    feature_names = ["nli_entailment", "nli_neutral", "nli_contradiction"]

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        device: str | None = None,
        max_length: int = 384,
    ) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.indices = resolve_nli_label_indices(self.model.config.id2label)

    def predict(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
        batch_size: int = 16,
    ) -> np.ndarray:
        if len(premises) != len(hypotheses):
            raise ValueError("Premise and hypothesis sequences must have the same length.")

        batches: list[np.ndarray] = []
        order = [self.indices.entailment, self.indices.neutral, self.indices.contradiction]
        for start in range(0, len(premises), batch_size):
            encoded = self.tokenizer(
                list(premises[start : start + batch_size]),
                list(hypotheses[start : start + batch_size]),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = self.model(**encoded).logits
                probabilities = torch.softmax(logits, dim=-1)[:, order]
            batches.append(probabilities.cpu().numpy().astype(np.float32))

        return np.vstack(batches) if batches else np.empty((0, 3), dtype=np.float32)


class SequencePairDataset(Dataset):
    def __init__(
        self,
        first_texts: Sequence[str],
        second_texts: Sequence[str],
        labels: Sequence[int],
        tokenizer,
        max_length: int,
    ) -> None:
        if not (len(first_texts) == len(second_texts) == len(labels)):
            raise ValueError("Text-pair and label sequences must have the same length.")
        self.first_texts = list(first_texts)
        self.second_texts = list(second_texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.first_texts[index],
            self.second_texts[index],
            truncation=True,
            max_length=self.max_length,
        )
        encoded["labels"] = int(self.labels[index])
        return encoded
