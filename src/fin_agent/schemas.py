from enum import StrEnum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, field_validator

from fin_agent.config import get_settings


class Status(StrEnum):
    OK = "ok"
    NOT_FOUND = "not_found"
    ERROR = "error"
    PARTIAL = "partial"


class Calculation(BaseModel):
    operation: str
    expression: str
    result: float
    currency: str = "RUB"

class DataSource(BaseModel):
    type: Literal["data"] = "data"
    file: str
    record_ids: list[str]


class DocumentSource(BaseModel):
    type: Literal["document"] = "document"
    file: str
    section: str
    quote: str


Source = Annotated[
    Union[DataSource, DocumentSource],
    Field(discriminator="type"),
]


class Answer(BaseModel):
    answer: str
    status: Status
    calculations: list[Calculation] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AskRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def _check(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Вопрос пустой")
        limit = get_settings().agent.max_question_length
        if len(v) > limit:
            raise ValueError(f"Вопрос длиннее {limit} символов")
        return v