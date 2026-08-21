"""Ragas-based RAG answer validation (faithfulness + answer relevancy)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from tools.ragas_compat import ensure_ragas_langchain_compat

ensure_ragas_langchain_compat()

from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, Faithfulness

from config import (
    EMBEDDING_MODEL,
    JUDGE_MODEL,
    OLLAMA_URL,
)
from tools.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ValidationSample:
    question: str
    answer: str
    contexts: list[str]


@dataclass(frozen=True)
class ValidationScores:
    faithfulness: float | None
    answer_relevancy: float | None


@dataclass(frozen=True)
class ValidationResult:
    scores: ValidationScores
    per_sample: list[ValidationScores]
    num_samples: int
    judge_model: str
    embedding_model: str


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def _build_judge(
    judge_model: str,
    embedding_model: str,
) -> tuple[LangchainLLMWrapper, LangchainEmbeddingsWrapper]:
    llm = LangchainLLMWrapper(
        ChatOllama(
            model=judge_model,
            base_url=OLLAMA_URL,
            temperature=0,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(
            model=embedding_model,
            base_url=OLLAMA_URL,
        )
    )
    return llm, embeddings


def validate_samples(
    samples: Sequence[ValidationSample],
    *,
    judge_model: str | None = None,
    embedding_model: str | None = None,
) -> ValidationResult:
    if not samples:
        raise ValueError("At least one sample is required.")

    for index, sample in enumerate(samples):
        if not sample.question.strip():
            raise ValueError(f"Sample {index}: question is required.")
        if not sample.answer.strip():
            raise ValueError(f"Sample {index}: answer is required.")
        if not sample.contexts or not any(
            context.strip() for context in sample.contexts
        ):
            raise ValueError(
                f"Sample {index}: at least one non-empty context is required."
            )

    resolved_judge = judge_model or JUDGE_MODEL
    resolved_embeddings = embedding_model or EMBEDDING_MODEL

    llm, embeddings = _build_judge(
        resolved_judge,
        resolved_embeddings,
    )

    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=sample.question,
                response=sample.answer,
                retrieved_contexts=sample.contexts,
            )
            for sample in samples
        ]
    )

    faithfulness = Faithfulness()
    answer_relevancy = AnswerRelevancy()

    logger.info(
        "Running Ragas validation on %s sample(s) "
        "with judge=%s embeddings=%s",
        len(samples),
        resolved_judge,
        resolved_embeddings,
    )

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings,
    )

    table = result.to_pandas()
    per_sample: list[ValidationScores] = []
    for row in table.to_dict(orient="records"):
        per_sample.append(
            ValidationScores(
                faithfulness=_as_optional_float(
                    row.get("faithfulness")
                ),
                answer_relevancy=_as_optional_float(
                    row.get("answer_relevancy")
                ),
            )
        )

    faithfulness_values = [
        score.faithfulness
        for score in per_sample
        if score.faithfulness is not None
    ]
    relevancy_values = [
        score.answer_relevancy
        for score in per_sample
        if score.answer_relevancy is not None
    ]

    aggregate = ValidationScores(
        faithfulness=(
            sum(faithfulness_values) / len(faithfulness_values)
            if faithfulness_values
            else None
        ),
        answer_relevancy=(
            sum(relevancy_values) / len(relevancy_values)
            if relevancy_values
            else None
        ),
    )

    if (
        aggregate.faithfulness is None
        and aggregate.answer_relevancy is None
    ):
        raise RuntimeError(
            "Ragas evaluation produced no scores. "
            "Check that Ollama is reachable and the "
            "judge/embedding models are available."
        )

    return ValidationResult(
        scores=aggregate,
        per_sample=per_sample,
        num_samples=len(samples),
        judge_model=resolved_judge,
        embedding_model=resolved_embeddings,
    )
