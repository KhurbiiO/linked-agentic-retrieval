"""Run the tri-agent workflow and save its RDF knowledge graphs."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agents import create_tri_agent


# Edit these values before running this file.
SEED_URL = "https://foodnetwork.co.uk"
QUESTION = "What useful information is available on this page?"

INSTRUCTOR_MODEL = "ollama:llama3.2"
CONTROLLER_MODEL = "ollama:llama3.2"
BUILD_MODEL = "ollama:qwen2.5:32b-instruct"

MAX_CONTROLLER_ACTIONS = 5

OUTPUT_DIRECTORY = Path("output")
TRACE_PROCESS = False
TRACE_FILE = OUTPUT_DIRECTORY / "process_trace.jsonl"


def run_tri_agent():
    prompt = f"{QUESTION}\nSeed URL: {SEED_URL}"
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    with create_tri_agent(
        max_controller_actions=MAX_CONTROLLER_ACTIONS,
        trace=TRACE_PROCESS,
        trace_path=TRACE_FILE if TRACE_PROCESS else None,
    ) as tri_agent:
        result = tri_agent.invoke(prompt)
        graph_store = tri_agent.builder.graph_store

        graph_store.save(OUTPUT_DIRECTORY / "knowledge_graphs.trig")
        graph_store.save(
            OUTPUT_DIRECTORY / "content_graph.ttl",
            kind="content",
            format="turtle",
        )
        graph_store.save(
            OUTPUT_DIRECTORY / "layout_graph.ttl",
            kind="layout",
            format="turtle",
        )

    return result, graph_store


def main() -> None:
    with ThreadPoolExecutor(max_workers=1) as executor:
        result, graph_store = executor.submit(run_tri_agent).result()

    print(result.graph.summary)
    print(f"Saved {graph_store.counts['content']} content triples")
    print(f"Saved {graph_store.counts['layout']} layout triples")
    print(f"Output directory: {OUTPUT_DIRECTORY.resolve()}")


if __name__ == "__main__":
    main()
