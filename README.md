# Linked Agentic Retrieval

A single reasoning agent that iteratively retrieves and verifies structured web
data. Extraction, traversal, and link discovery are deterministic tools—not
independent agents.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
ollama pull qwen3.5:4b
python main.py
```

Python 3.10+ is required. A request must contain a starting HTTP(S) URL, either
directly or in its conversation context.

## Configuration

The factory automatically loads `config.json` from the project root:

```json
{
  "model": {
    "identifier": "ollama:qwen3.5:4b",
    "temperature": 0
  },
  "agent": {
    "max_rounds": 5,
    "max_candidate_urls": 20
  },
  "retrieval": {
    "max_results_per_page": 12,
    "max_links_per_page": 20
  },
  "extractor": {
    "timeout_seconds": 30,
    "link_context_max_fields": 12,
    "link_context_max_chars": 1000
  }
}
```

All values are validated at startup and unknown keys are rejected. Load another
file or provide a Python override when needed:

```python
agent = create_retrieval_agent(
    config="configs/experiment.json",
    max_rounds=8,
)
```

Explicit Python arguments override the file. `AGENT_MODEL` can override only the
configured model identifier, which is useful for temporary model comparisons.

Discovered links include their immediate parent JSON path, inferred anchor text,
and bounded scalar sibling fields such as `name`, `title`, `label`, and
`description`. These fields participate in relevance ranking. The two
`link_context_*` settings prevent large parent objects from inflating prompts.

## Reasoning loop

1. `agent.question_analysis` derives goal, context, terms, URLs, and success criteria.
2. `agent.select_action` selects one allowed, unvisited URL.
3. `tool.extract_url` calls `StructuredDataExtractor.extract`.
4. `tool.traverse_data` ranks JSON paths and values against the search terms.
5. `tool.discover_links` finds possible next pages.
6. `agent.verify_evidence` decides `complete`, `continue`, or `failed`.
7. Steps 2-6 repeat as needed, then `agent.synthesize_answer` produces the answer.

There is one reasoning policy and one evolving evidence state. A single
`StructuredDataExtractor` in `tools/extract/webex.py` owns downloading,
structured-data extraction, ranked traversal, and link discovery. It contains
no model, memory, goal, or autonomous decision-making.

## Performance measurement

Every operation produces a `TraceStep` with its input, summarized output, actor,
UTC start time, duration, status, error, and provider-reported token usage.
Aggregate metrics are returned in `result.performance`:

- Total wall-clock, model, and tool time
- Input, output, and total tokens
- Reasoning rounds and visited URLs
- Successful and failed step counts

```python
from agent import create_retrieval_agent

agent = create_retrieval_agent(max_rounds=4)
result = agent.invoke(
    "Could I get a recipe for a chicken dish by Guy Fieri? Start at https://foodnetwork.co.uk",
    trace_sink=lambda step: print(step.stage, step.duration_ms, step.metrics),
)

print(result.answer)
print(result.performance.model_dump())

# Full run record for offline evaluation
with open("run.json", "w", encoding="utf-8") as output:
    output.write(result.model_dump_json(indent=2))
```

The optional callback receives completed steps immediately, including failed
steps. A successful run also includes the full trace in `result.trace`.

## Model selection

Use one LangChain model for the whole reasoning loop:

```text
AGENT_MODEL=ollama:qwen3.5:4b
```

The default uses Qwen 3.5 4B through the locally running Ollama service and
requires no API key. You can also inject it explicitly:

```python
from langchain_ollama import ChatOllama
from agent import create_retrieval_agent

agent = create_retrieval_agent(
    model=ChatOllama(model="qwen3.5:4b", temperature=0),
)
```

The model must support LangChain structured output because the loop validates
its analysis, action-selection, and verification responses.
