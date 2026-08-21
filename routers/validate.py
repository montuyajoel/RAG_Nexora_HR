from fastapi import (
    APIRouter,
    HTTPException,
    Security,
)
from pydantic import BaseModel, Field

from security import verify_api_key
from tools.logger import get_logger
from tools.ragas_validator import (
    ValidationSample,
    validate_samples,
)

logger = get_logger(__name__)

router = APIRouter()


class ValidateRequest(BaseModel):
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    contexts: list[str] = Field(min_length=1)
    judge_model: str | None = None
    embedding_model: str | None = None


class ValidateBatchRequest(BaseModel):
    samples: list[ValidateRequest] = Field(min_length=1)
    judge_model: str | None = None
    embedding_model: str | None = None


def _scores_payload(result) -> dict:
    return {
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


@router.post("/validate")
def validate_answer(
    payload: ValidateRequest,
    api_key: str = Security(verify_api_key),
):
    try:
        result = validate_samples(
            [
                ValidationSample(
                    question=payload.question,
                    answer=payload.answer,
                    contexts=payload.contexts,
                )
            ],
            judge_model=payload.judge_model,
            embedding_model=payload.embedding_model,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error
    except Exception as error:
        logger.exception(
            "Ragas validation failed: %s",
            str(error),
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "Ragas validation is not available at the moment."
            ),
        ) from error

    return _scores_payload(result)


@router.post("/validate/batch")
def validate_answers_batch(
    payload: ValidateBatchRequest,
    api_key: str = Security(verify_api_key),
):
    try:
        result = validate_samples(
            [
                ValidationSample(
                    question=sample.question,
                    answer=sample.answer,
                    contexts=sample.contexts,
                )
                for sample in payload.samples
            ],
            judge_model=payload.judge_model,
            embedding_model=payload.embedding_model,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error
    except Exception as error:
        logger.exception(
            "Ragas batch validation failed: %s",
            str(error),
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "Ragas validation is not available at the moment."
            ),
        ) from error

    return _scores_payload(result)
