import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from fin_agent.config import get_settings
from fin_agent.schemas import Answer, AskRequest, Status
from fin_agent.storage.db import load_database

logging.basicConfig(level=get_settings().log.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    conn, problems = load_database()
    app.state.db = conn
    app.state.load_warnings = problems
    yield
    conn.close()


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
