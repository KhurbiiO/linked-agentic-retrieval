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
    "Who is Mary Berg? Start at https://foodnetwork.co.uk/chefs",
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

The default uses the locally running Ollama service and requires no API key.
You can also inject an explicitly configured Llama model:

```python
from langchain_ollama import ChatOllama
from agent import create_retrieval_agent

agent = create_retrieval_agent(
    model=ChatOllama(model="llama3.1", temperature=0),
)
```

The model must support LangChain structured output because the loop validates
its analysis, action-selection, and verification responses.
