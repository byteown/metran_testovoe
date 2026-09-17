import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from fin_agent.agent.llm import LLMClient
from fin_agent.agent.loop import run_agent
from fin_agent.schemas import Answer, AskRequest, Status
from fin_agent.startup import build_context, configure_logging

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    context, load_warnings = build_context()
    app.state.context = context
    app.state.load_warnings = load_warnings
    app.state.llm = LLMClient()
    yield
    app.state.llm.close()
    context.conn.close()


app = FastAPI(title="fin-agent", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=Answer)
def ask(request: AskRequest, http_request: Request) -> Answer:
    state = http_request.app.state
    result = run_agent(request.question, state.context, state.llm)

    if state.load_warnings:
        result.answer.warnings.extend(state.load_warnings)
    return result.answer


@app.exception_handler(Exception)
async def on_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Необработанная ошибка")
    body = Answer(answer="Внутренняя ошибка сервиса", status=Status.ERROR)
    return JSONResponse(status_code=500, content=body.model_dump(mode="json"))
