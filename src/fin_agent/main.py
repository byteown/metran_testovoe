import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from fin_agent.config import get_settings
from fin_agent.schemas import Answer, AskRequest, Status

logging.basicConfig(level=get_settings().log.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = None
    yield


app = FastAPI(title="fin-agent", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/ask", response_model=Answer)
async def ask(request: AskRequest) -> Answer:
    return Answer(
        answer="",
        status=Status.NOT_FOUND,
    )

@app.exception_handler(Exception)
async def on_error(request, exc: Exception) -> JSONResponse:
    logger.exception("Необработанная ошибка")
    body = Answer(answer="Внутренняя ошибка сервиса", status=Status.ERROR)
    return JSONResponse(status_code=500, content=body.model_dump(mode="json"))
