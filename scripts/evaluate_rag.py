#!/usr/bin/env python3
"""Offline Ragas evaluation CLI for the Flexible RAG Platform.

Dataset JSON format (list of samples):

[
  {
    "question": "...",
    "answer": "...",
    "contexts": ["chunk 1", "chunk 2"]
  }
]

Example:

  python scripts/evaluate_rag.py \\
    --dataset scripts/sample_eval_dataset.json \\
    --fail-below 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ragas_validator import (  # noqa: E402
    ValidationSample,
    validate_samples,
)


def _load_dataset(path: Path) -> list[ValidationSample]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Dataset must be a JSON list of samples.")

    samples: list[ValidationSample] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Sample {index} must be an object.")
        try:
            samples.append(
                ValidationSample(
                    question=str(item["question"]),
                    answer=str(item["answer"]),
                    contexts=[str(c) for c in item["contexts"]],
                )
            )
        except KeyError as error:
            raise ValueError(
                f"Sample {index} missing required field: {error}"
            ) from error
    return samples


def _format_score(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _bar(value: float | None, width: int = 20) -> str:
    if value is None:
        return "[" + ("-" * width) + "]"
    filled = max(0, min(width, int(round(value * width))))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Ragas faithfulness and answer relevancy "
            "validation offline."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to JSON evaluation dataset.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Ollama judge model (default: JUDGE_MODEL / LLM_MODEL).",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Ollama embedding model (default: EMBEDDING_MODEL).",
    )
    parser.add_argument(
        "--fail-below",
        type=float,
        default=None,
        help=(
            "Exit with code 1 if faithfulness or answer_relevancy "
            "is below this threshold."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write JSON results.",
    )
    args = parser.parse_args()

    samples = _load_dataset(args.dataset)
    result = validate_samples(
        samples,
        judge_model=args.judge_model,
        embedding_model=args.embedding_model,
    )

    print("Ragas Evaluation Results")
    print("-" * 56)
    print(
        f"  Faithfulness       {_format_score(result.scores.faithfulness):>8}  "
        f"{_bar(result.scores.faithfulness)}"
    )
    print(
        f"  Answer Relevancy   {_format_score(result.scores.answer_relevancy):>8}  "
        f"{_bar(result.scores.answer_relevancy)}"
    )
    print(f"  Samples evaluated: {result.num_samples}")
    print(f"  Judge model:       {result.judge_model}")
    print(f"  Embedding model:   {result.embedding_model}")
    print("-" * 56)

    payload = {
        "faithfulness": result.scores.faithfulness,
        "answer_relevancy": result.scores.answer_relevancy,
        "num_samples": result.num_samples,
        "judge_model": result.judge_model,
        "embedding_model": result.embedding_model,
        "per_sample": [
            {
                "faithfulness": sample.faithfulness,
                "answer_relevancy": sample.answer_relevancy,
            }
            for sample in result.per_sample
        ],
    }

    if args.output is not None:
        args.output.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote results to {args.output}")

    if args.fail_below is not None:
        failing = []
        for name, value in (
            ("faithfulness", result.scores.faithfulness),
            ("answer_relevancy", result.scores.answer_relevancy),
        ):
            if value is None or value < args.fail_below:
                failing.append(name)
        if failing:
            print(
                "FAIL: metric(s) below "
                f"{args.fail_below}: {', '.join(failing)}"
            )
            return 1
        print(f"PASS: all metrics >= {args.fail_below}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
