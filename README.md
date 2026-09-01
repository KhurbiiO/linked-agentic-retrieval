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
ollama pull llama3.2
python main.py
```

Python 3.10+ is required. A request must contain a starting HTTP(S) URL, either
directly or in its conversation context.

## Configuration

The factory automatically loads `config.json` from the project root:

```json
{
  "model": {
    "identifier": "ollama:llama3.2",
    "temperature": 0
  },
  "agent": {
    "max_rounds": 5,
    "max_candidate_urls": 20
  },
  "retrieval": {
    "max_results_per_page": 12,
    "max_links_per_page": 10,
    "scoring_method": "weighted_context",
    "traverse_links": true,
    "evidence_mode": "filtered",
    "extraction_prompt_max_chars_per_page": 12000,
    "excluded_url_extensions": [
      ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
      ".mp4", ".webm", ".mp3", ".css", ".js", ".woff2"
    ]
  },
  "extractor": {
    "timeout_seconds": 30,
    "link_context_max_fields": 12,
    "link_context_max_chars": 1000,
    "link_context_child_depth": 2
  },
  "tracing": {
    "enabled": false
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

`excluded_url_extensions` removes media and static-asset URLs before context
extraction, scoring, or candidate-pool insertion. Matching is case-insensitive
and checks only the URL path, so query strings such as `image.jpg?width=800` are
handled correctly. Edit the list to allow or exclude additional file types.

Discovered links include their immediate parent JSON path, inferred anchor text,
and bounded scalar sibling fields such as `name`, `title`, `label`, and
`description`. These fields participate in relevance ranking. The two
`link_context_*` settings prevent large parent objects from inflating prompts.

Candidate context is flattened recursively through child dictionaries and lists
up to `link_context_child_depth`. The default `weighted_context` scorer combines
URL matches, JSON-path matches, weighted contextual fields, and token overlap
with the complete retrieval goal. Every candidate exposes `score_components`
so rankings can be inspected during debugging.

Set `scoring_method` to `term_frequency` to use the original unweighted exact
substring counter. Additional strategies can implement `CandidateScorer` in
`tools/extract/scoring.py` without changing extraction or agent orchestration.

Set `traverse_links` to `false` to restrict retrieval to URLs supplied directly
in the request or conversation context. Seed pages are still extracted and
searched, and multiple supplied seeds may still be visited, but links discovered
inside their metadata are not collected or followed. This can also be overridden
in Python:

```python
agent = create_retrieval_agent(traverse_links=False)
```

### Evidence mode

`evidence_mode` controls how visited-page data is supplied to the model:

- `filtered` sends only ranked matches and contextual candidate links. This is
  the default and uses fewer tokens.
- `extraction` additionally sends a serialized portion of the raw structured
  extraction for every visited page. Filtered evidence remains attached for
  paths, scores, and citations.

```json
"retrieval": {
  "evidence_mode": "extraction",
  "extraction_prompt_max_chars_per_page": 12000
}
```

The character limit applies separately to every page. The payload tells the
model whether it was truncated and reports original and included character
counts. Extraction mode can substantially increase prompt-processing time and
may exceed a small Ollama context window when several pages are visited.

It can also be selected in Python:

```python
agent = create_retrieval_agent(
    evidence_mode="extraction",
    extraction_prompt_max_chars_per_page=8000,
)
```

### Debug tracing

Detailed tracing is disabled by default. Enable it in `config.json`:

```json
"tracing": {
  "enabled": true
}
```

It can also be toggled for one call:

```python
result = agent.invoke(question, trace_enabled=True)
print(result.trace)
```

Providing `trace_sink` automatically enables tracing for that invocation.
Aggregate `result.performance` timing and token metrics remain available when
debug tracing is disabled, but `result.trace` is empty and detailed inputs and
outputs are not retained.

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

If extraction fails because of an HTTP error, timeout, connection failure, or
redirect error, that URL is marked failed and removed from the pool. The loop
then selects another candidate instead of terminating. Failed downloads do not
consume `max_rounds`, which counts successful page retrievals, and their errors
remain available in debug traces and final-answer context.

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
    "Could I get a recipe by Guy Fieri without beans? Start at https://foodnetwork.co.uk/chefs/guy-fieri",
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
AGENT_MODEL=ollama:llama3.2
```

The default uses Llama 3.2 through the locally running Ollama service and
requires no API key. You can also inject Llama 3.2 explicitly:

```python
from langchain_ollama import ChatOllama
from agent import create_retrieval_agent

agent = create_retrieval_agent(
    model=ChatOllama(model="llama3.2", temperature=0),
)
```

The model must support LangChain structured output because the loop validates
its analysis, action-selection, and verification responses.
