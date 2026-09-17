import json
import sys

import typer

from fin_agent.agent.llm import LLMClient
from fin_agent.agent.loop import run_agent
from fin_agent.schemas import AskRequest
from fin_agent.startup import build_context, configure_logging

cli = typer.Typer(add_completion=False, help="Помощник финансовой службы")


@cli.command()
def ask(
        question: str = typer.Argument(..., help="Вопрос по данным 1С или регламентам"),
        as_json: bool = typer.Option(False, "--json", help="Вывести ответ целиком в JSON"),
        trace: bool = typer.Option(False, "--trace", help="Показать трассу: витки, инструменты, токены"),
) -> None:
    configure_logging()

    try:
        request = AskRequest(question=question)
    except ValueError as exc:
        typer.secho(f"Неверный вопрос: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    context, load_warnings = build_context()
    client = LLMClient()
    try:
        result = run_agent(request.question, context, client)
    finally:
        client.close()
        context.conn.close()

    answer = result.answer
    answer.warnings.extend(load_warnings)

    if as_json:
        typer.echo(answer.model_dump_json(indent=2))
    else:
        typer.echo(f"\n{answer.answer}\n")
        typer.echo(f"статус: {answer.status}")
        for calculation in answer.calculations:
            typer.echo(f"  расчёт: {calculation.operation} = {calculation.result:,.2f} {calculation.currency}")
            typer.echo(f"          {calculation.expression}")
        for source in answer.sources:
            if source.type == "data":
                typer.echo(f"  источник: {source.file} {', '.join(source.record_ids)}")
            else:
                typer.echo(f"  источник: {source.file} / {source.section}")
                typer.echo(f"            «{source.quote}»")
        for warning in answer.warnings:
            typer.secho(f"  ! {warning}", fg=typer.colors.YELLOW)

    if trace:
        typer.echo("\nтрасса:")
        typer.echo(json.dumps(result.trace.as_dict(), ensure_ascii=False, indent=2))

    if answer.status == "error":
        sys.exit(1)


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    cli()


if __name__ == "__main__":
    main()
