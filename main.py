"""Run the tri-agent workflow and save its RDF knowledge graphs."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agents import create_tri_agent


# Edit these values before running this file.
SEED_URL = "https://foodnetwork.co.uk"
QUESTION = "What useful information is available on this page?"

INSTRUCTOR_MODEL = "deepseek-r1:8b"
CONTROLLER_MODEL = "qwen3:4b"
BUILD_MODEL = "qwen3:4b"
GRAPH_EMBEDDING_MODEL = "ollama:nomic-embed-text"

MAX_CONTROLLER_ACTIONS = 5
CONTROLLER_ACTION_TIMEOUT = 5
CONTROLLER_NAVIGATION_TIMEOUT = 15
MAX_RETRIEVAL_ROUNDS = 3
MAX_GRAPH_QUERY_STEPS = 4  # Maximum goal/criterion texts embedded per search
GRAPH_QUERY_RESULT_LIMIT = 12  # Maximum high-scoring entry facts
GRAPH_NEIGHBOR_LIMIT = 12  # Additional same-subject/linked facts per entry
GRAPH_MIN_SCORE = 0.08
GRAPH_VECTOR_DATABASE_PATH = ":memory:"  # Set a SQLite path to persist fact vectors
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
        controller_navigation_timeout=CONTROLLER_NAVIGATION_TIMEOUT,
        max_retrieval_rounds=MAX_RETRIEVAL_ROUNDS,
        max_graph_query_steps=MAX_GRAPH_QUERY_STEPS,
        graph_query_result_limit=GRAPH_QUERY_RESULT_LIMIT,
        graph_neighbor_limit=GRAPH_NEIGHBOR_LIMIT,
        graph_min_score=GRAPH_MIN_SCORE,
        graph_vector_database_path=GRAPH_VECTOR_DATABASE_PATH,
        instructor_model=INSTRUCTOR_MODEL,
        controller_model=CONTROLLER_MODEL,
        builder_model=BUILD_MODEL,
        graph_embedding_model=GRAPH_EMBEDDING_MODEL,
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

    print(result.answer)
    if not result.completed:
        print("Goal not completed within the configured retrieval rounds.")
        print("Missing evidence:", "; ".join(result.verification.missing_information))
    print(f"Saved {graph_store.counts['content']} content triples")
    print(
        "Chat-model tokens: "
        f"{sum(metric.total_tokens for metric in result.metrics)} total "
        f"({sum(metric.input_tokens for metric in result.metrics)} input, "
        f"{sum(metric.output_tokens for metric in result.metrics)} output)"
    )
    print(f"Output directory: {OUTPUT_DIRECTORY.resolve()}")


if __name__ == "__main__":
    main()
