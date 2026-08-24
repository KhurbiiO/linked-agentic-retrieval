from agent import create_retrieval_agent


def main() -> None:
    agent = create_retrieval_agent()
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

        result = agent.invoke(
            user_input,
            context=context,
            trace_sink=lambda step: print(
                f"  [{step.sequence}] {step.stage}: {step.status} ({step.duration_ms:.1f} ms)"
            ),
        )
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
