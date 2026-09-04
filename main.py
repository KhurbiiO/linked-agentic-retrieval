from agent import create_retrieval_agent
from tools import StructuredDataExtractor


def main() -> None:
    agent = create_retrieval_agent(extractor=StructuredDataExtractor())
    context: list[dict[str, str]] = []

    print("Agent ready. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        trace_sink = None
        if agent.trace_enabled:
            trace_sink = lambda step: print(
                f"  [{step.sequence}] {step.stage}: {step.status} ({step.duration_ms:.1f} ms)"
            )
        result = agent.invoke(user_input, context=context, trace_sink=trace_sink)
        print(f"Agent: {result.answer}")
        print(f"Performance: {result.performance.model_dump_json()}")

        context.extend(
            [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": result.answer},
            ]
        )


if __name__ == "__main__":
    main()
