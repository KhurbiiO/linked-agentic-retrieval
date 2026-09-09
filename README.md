# Linked Agentic Retrieval

A single reasoning agent that iteratively retrieves and verifies structured web
data. Extraction, traversal, and link discovery are deterministic tools—not
independent agents.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull llama3.2
python main.py
```

Python 3.10+ is required. A request must contain a starting HTTP(S) URL, either
directly or in its conversation context.

`RetrievalAgent` in `agent/host/host.py` is the abstract parent containing the
shared reasoning loop. Two concrete variants implement its evidence payload.
`FilteredRetrievalAgent` in `agent/filtered/filtered.py` supplies only ranked,
threshold-filtered evidence. `FullExtractionAgent` in `agent/full/full.py`
supplies the complete structured extraction from every
successfully visited page to navigation, verification, and final synthesis,
together with the score-filtered relevant evidence:

```python
from agent import create_full_extraction_agent

agent = create_full_extraction_agent()
result = agent.invoke("Question involving https://example.com")
```

The full variant can consume substantially more model context on large pages.

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
    "minimum_evidence_score": 0.0,
    "minimum_link_score": 0.0,
    "scoring_method": "semantic",
    "semantic_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "traverse_links": true,
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

Explicit Python arguments override the file, which is useful for temporary
model comparisons.

`excluded_url_extensions` removes media and static-asset URLs before context
extraction, scoring, or candidate-pool insertion. Matching is case-insensitive
and checks only the URL path, so query strings such as `image.jpg?width=800` are
handled correctly. Edit the list to allow or exclude additional file types.

When traversal is enabled, discovered links come from HTML elements carrying an
`href`, rather than only from URLs present in structured metadata. Each link is
ranked using only a naturalized version of its URL path and query string. URL
encoding is decoded, separators become spaces, camel-case words are split, and
obvious tracking parameters, numeric IDs, and opaque identifiers are removed.
For example, `/recipes/salmon-cooking-time?view=full` becomes
`recipes salmon cooking time view full`. The scheme, hostname, anchor
text, HTML context, page context, and page evidence do not affect link scores.

`minimum_evidence_score` and `minimum_link_score` are strict thresholds. Only
items with `score > threshold` are retained or supplied to the host. For
semantic cosine scoring, values such as `0.25` or `0.4` can remove weak matches,
but the cutoff should be calibrated for the selected embedding model. A value
of `0.0` preserves all positively scored items. The standard host supplies only
these filtered matches. The full-extraction agent additionally supplies each
complete structured extraction.

The default `semantic` scorer ranks candidates by cosine similarity between the
complete retrieval goal and this relative URL using
`all-MiniLM-L6-v2`. Every candidate exposes
the resulting `semantic_similarity` in `score_components`. The same semantic
strategy ranks extracted scalar evidence by comparing the complete retrieval
goal with each scalar's JSON path and value. Consequently,
`max_results_per_page` retains the most semantically relevant evidence from
each page rather than requiring exact term matches.

Set `scoring_method` to `weighted_context` to use goal-token and field-weighted
lexical matching without embeddings, or `term_frequency` for the original exact
substring counter. Additional strategies can implement `CandidateScorer` in
`tools/extract/scoring.py` without changing extraction or agent orchestration.

The embedding model is downloaded and loaded lazily on the first evidence- or link-scoring
operation. This makes the first semantic-scoring round slower; later calls reuse
the loaded model and cached embeddings.

Set `traverse_links` to `false` to restrict retrieval to URLs supplied directly
in the request or conversation context. Seed pages are still extracted and
searched, and multiple supplied seeds may still be visited, but links discovered
inside their metadata are not collected or followed. This can also be overridden
in Python:

```python
agent = create_retrieval_agent(traverse_links=False)
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
`StructuredDataExtractor` in `tools/extract/extract.py` owns downloading,
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
    "Does the dish in this page contain brocolli? Start at https://www.allrecipes.com/grilled-bruschetta-chicken-recipe-7509319grilled-bruschetta-chicken-recipe-7509319",
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

Set the default LangChain model in `config.json`. The default uses Llama 3.2
through the locally running Ollama service and requires no API key. You can also
inject Llama 3.2 explicitly:

```python
from langchain_ollama import ChatOllama
from agent import create_retrieval_agent

agent = create_retrieval_agent(
    model=ChatOllama(model="llama3.2", temperature=0),
)
```

The model must support LangChain structured output because the loop validates
its analysis, action-selection, and verification responses.
