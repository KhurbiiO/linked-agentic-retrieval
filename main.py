"""Run the tri-agent workflow and save its RDF knowledge graphs."""

from pathlib import Path

from agents import create_tri_agent


# Edit these values before running this file.
SEED_URL = "https://foodnetwork.co.uk"
QUESTION = "What useful information is available on this page?"
MODEL = "ollama:llama3.2"
MAX_CONTROLLER_ACTIONS = 5
OUTPUT_DIRECTORY = Path("output")


def main() -> None:
    prompt = f"{QUESTION}\nSeed URL: {SEED_URL}"
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    with create_tri_agent(
        model=MODEL,
        max_controller_actions=MAX_CONTROLLER_ACTIONS,
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

    print(result.graph.summary)
    print(f"Saved {graph_store.counts['content']} content triples")
    print(f"Saved {graph_store.counts['layout']} layout triples")
    print(f"Output directory: {OUTPUT_DIRECTORY.resolve()}")


if __name__ == "__main__":
    main()
