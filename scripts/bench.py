import argparse
import statistics
import time

from fin_agent.agent.llm import LLMClient
from fin_agent.agent.loop import run_agent
from fin_agent.startup import build_context, configure_logging

QUESTIONS: list[tuple[str, str, int]] = [
    ("простой расчётный", "Сколько дебиторской задолженности у контрагента «Ромашка»?", 10),
    ("RAG с цитатами", "Что делать при просрочке дебиторки 45 дней?", 12),
    ("многошаговый", "Топ-5 должников по сумме просрочки", 25),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=12, help="прогонов на вопрос")
    parser.add_argument("--price-in", type=float, default=0.0, help="цена за 1M входных токенов")
    parser.add_argument("--price-out", type=float, default=0.0, help="цена за 1M выходных токенов")
    args = parser.parse_args()

    configure_logging()
    context, _ = build_context()
    client = LLMClient()

    run_agent("Проверка связи", context, client)

    rows: list[dict] = []
    try:
        for label, question, target in QUESTIONS:
            durations: list[float] = []
            prompt_tokens: list[int] = []
            completion_tokens: list[int] = []
            turns: list[int] = []
            statuses: list[str] = []

            print(f"\n{label}: {question}")
            for run in range(args.runs):
                started = time.perf_counter()
                result = run_agent(question, context, client)
                durations.append(time.perf_counter() - started)
                prompt_tokens.append(result.trace.prompt_tokens)
                completion_tokens.append(result.trace.completion_tokens)
                turns.append(result.trace.turns)
                statuses.append(str(result.answer.status))
                print(f"  {run + 1:2}/{args.runs}  {durations[-1]:5.1f} с  "
                      f"{result.trace.total_tokens:5} ток.  {result.answer.status}")

            median_in = statistics.median(prompt_tokens)
            median_out = statistics.median(completion_tokens)
            rows.append(
                {
                    "label": label,
                    "target": target,
                    "median": statistics.median(durations),
                    "worst": max(durations),
                    "best": min(durations),
                    "tokens_in": median_in,
                    "tokens_out": median_out,
                    "turns": statistics.median(turns),
                    "ok_rate": statuses.count("ok") / len(statuses),
                    "cost": (median_in * args.price_in + median_out * args.price_out) / 1_000_000,
                }
            )
    finally:
        client.close()
        context.conn.close()

    _print_table(rows, args)


def _print_table(rows: list[dict], args: argparse.Namespace) -> None:
    print(f"\n\nРезультат, {args.runs} прогонов на вопрос\n")
    header = (
        "| Тип вопроса | Медиана | Худший | Лучший | Ориентир | Витков | "
        "Токенов вх./исх. | Доля ok |"
    )
    print(header)
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['label']} | {row['median']:.1f} с | {row['worst']:.1f} с | "
            f"{row['best']:.1f} с | {row['target']} с | {row['turns']:.0f} | "
            f"{row['tokens_in']:.0f} / {row['tokens_out']:.0f} | {row['ok_rate']:.0%} |"
        )

    if args.price_in or args.price_out:
        print("\n| Тип вопроса | Стоимость запроса |")
        print("|---|---:|")
        for row in rows:
            print(f"| {row['label']} | ${row['cost']:.6f} |")
    else:
        print(
            "\nПрямая стоимость нулевая: модель локальная (Ollama). "
            "Для оценки облачного варианта передайте --price-in и --price-out "
            "(цена за 1M токенов)."
        )


if __name__ == "__main__":
    main()
