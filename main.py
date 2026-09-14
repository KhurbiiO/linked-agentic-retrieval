"""Run the tri-agent workflow and save its RDF knowledge graphs."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agents import create_tri_agent


# Edit these values before running this file.
SEED_URL = "https://foodnetwork.co.uk"
QUESTION = "What useful information is available on this page?"

INSTRUCTOR_MODEL = "ollama:llama3.2"
CONTROLLER_MODEL = "ollama:llama3.2"
BUILD_MODEL = "ollama:llama3.2"

MAX_CONTROLLER_ACTIONS = 5
CONTROLLER_ACTION_TIMEOUT = 5
MAX_RETRIEVAL_ROUNDS = 3
PRELOAD_MODELS = True
OLLAMA_KEEP_ALIVE = "30m"

OUTPUT_DIRECTORY = Path("output")
TRACE_PROCESS = False
TRACE_FILE = OUTPUT_DIRECTORY / "process_trace.jsonl"


def run_tri_agent():
    prompt = f"{QUESTION}\nSeed URL: {SEED_URL}"
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    with create_tri_agent(
        model=INSTRUCTOR_MODEL,
        max_controller_actions=MAX_CONTROLLER_ACTIONS,
        controller_action_timeout=CONTROLLER_ACTION_TIMEOUT,
        max_retrieval_rounds=MAX_RETRIEVAL_ROUNDS,
        instructor_model=INSTRUCTOR_MODEL,
        controller_model=CONTROLLER_MODEL,
        builder_model=BUILD_MODEL,
        preload_models=PRELOAD_MODELS,
        ollama_keep_alive=OLLAMA_KEEP_ALIVE,
        trace=TRACE_PROCESS,
        trace_path=TRACE_FILE if TRACE_PROCESS else None,
    ) as tri_agent:
        result = tri_agent.invoke(prompt)
        graph_store = tri_agent.builder.graph_store

        graph_store.save(
            OUTPUT_DIRECTORY / "content_graph.ttl",
            kind="content",
            format="turtle",
        )

    return result, graph_store


def main() -> None:
    with ThreadPoolExecutor(max_workers=1) as executor:
        result, graph_store = executor.submit(run_tri_agent).result()

    if result.completed:
        print(result.answer)
    else:
        print("Goal not completed within the configured retrieval rounds.")
        print("Missing evidence:", "; ".join(result.verification.missing_information))
    print(f"Saved {graph_store.counts['content']} content triples")
    print(f"Output directory: {OUTPUT_DIRECTORY.resolve()}")


if __name__ == "__main__":
    main()
