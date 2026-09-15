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
EVIDENCE_EMBEDDING_MODEL = "ollama:nomic-embed-text"

MAX_CONTROLLER_ACTIONS = 5
CONTROLLER_ACTION_TIMEOUT = 5
CONTROLLER_NAVIGATION_TIMEOUT = 15
CONTROLLER_MAX_EVIDENCE_SEGMENTS = 12
EVIDENCE_CANDIDATE_LIMIT = 20
EVIDENCE_MIN_SCORE = 0.08
EVIDENCE_FALLBACK_BLOCKS = 3
EVIDENCE_CONTEXT_DEPTH = 2
EVIDENCE_MAX_CHARS = 16_000
EVIDENCE_BLOCK_MAX_CHARS = 3_000
EVIDENCE_SELECTION_MAX_CHARS = 24_000
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
        controller_navigation_timeout=CONTROLLER_NAVIGATION_TIMEOUT,
        controller_max_evidence_segments=CONTROLLER_MAX_EVIDENCE_SEGMENTS,
        controller_evidence_candidate_limit=EVIDENCE_CANDIDATE_LIMIT,
        controller_evidence_min_score=EVIDENCE_MIN_SCORE,
        controller_evidence_fallback_blocks=EVIDENCE_FALLBACK_BLOCKS,
        controller_evidence_context_depth=EVIDENCE_CONTEXT_DEPTH,
        controller_evidence_max_chars=EVIDENCE_MAX_CHARS,
        controller_evidence_block_max_chars=EVIDENCE_BLOCK_MAX_CHARS,
        controller_evidence_selection_max_chars=EVIDENCE_SELECTION_MAX_CHARS,
        max_retrieval_rounds=MAX_RETRIEVAL_ROUNDS,
        instructor_model=INSTRUCTOR_MODEL,
        controller_model=CONTROLLER_MODEL,
        builder_model=BUILD_MODEL,
        evidence_embedding_model=EVIDENCE_EMBEDDING_MODEL,
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
