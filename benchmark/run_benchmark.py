"""Run model configurations against the web-retrieval benchmark.

Edit the constants below, then run::

    python -m benchmark.run_benchmark

Results are appended after every task, so interrupted runs can be resumed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from benchmark.algorithms import BenchmarkAlgorithm, TriAgentAlgorithm
from benchmark.evaluate import evaluate_answer
from benchmark.judge import ModelAnswerJudge


DATASET_PATH = Path(__file__).parent / "tasks" / "web_retrieval_tasks_100_T2.json"
OUTPUT_DIRECTORY = Path("output") / "benchmark"
SCORABLE_ONLY = True
TASK_IDS: set[str] = set()  # Empty means all eligible tasks.
TASK_LIMIT: int | None = None
RESUME = True
JUDGE_MODEL: str | None = "ollama:deepseek-r1:8b"  # Set to None for deterministic scoring only.


ALGORITHMS: list[BenchmarkAlgorithm] = [
    TriAgentAlgorithm(
        name="TriAgent_V0_2",
        instructor_model="ollama:qwen3.5:9b",
        controller_model="ollama:qwen3.5:9b",
        builder_model="ollama:qwen3.5:9b",
    ),
]


def _load_tasks() -> list[dict[str, Any]]:
    tasks = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    selected = [
        task for task in tasks
        if (not SCORABLE_ONLY or task.get("scorable") is True)
        and (not TASK_IDS or task["id"] in TASK_IDS)
    ]
    return selected[:TASK_LIMIT] if TASK_LIMIT is not None else selected


def _load_completed(path: Path, *, require_model_judge: bool = False) -> set[str]:
    if not RESUME or not path.exists():
        return set()
    completed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            judged = record.get("model_judgement", {}).get("status") == "completed"
            if record.get("status") == "completed" and (not require_model_judge or judged):
                completed.add(record["task_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return completed


def _run_task(
    task: dict[str, Any], algorithm: BenchmarkAlgorithm,
    judge: ModelAnswerJudge | None = None,
) -> dict[str, Any]:
    started = perf_counter()
    try:
        response = algorithm.run(task["instruction"], task["start_url"])
        answer = response.answer
        evaluation = evaluate_answer(answer, task.get("gold_answer"))
        model_judgement = None
        if judge is not None:
            try:
                model_judgement = {
                    "status": "completed",
                    **judge.judge(
                        instruction=task["instruction"],
                        gold_answer=task.get("gold_answer"),
                        candidate_answer=answer,
                    ),
                }
            except Exception as exc:
                model_judgement = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        return {
            "task_id": task["id"],
            "algorithm": algorithm.name,
            "website": task.get("website"),
            "difficulty": task.get("difficulty"),
            "task_type": task.get("task_type"),
            "explicit_navigation": task.get("explicit_navigation"),
            "instruction": task["instruction"],
            "start_url": task["start_url"],
            "status": "completed",
            "answer": answer,
            "gold_answer": task.get("gold_answer"),
            "evaluation": evaluation,
            "model_judgement": model_judgement,
            "algorithm_completed": response.completed,
            "duration_ms": round((perf_counter() - started) * 1000, 3),
            "algorithm_duration_ms": response.duration_ms,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "total_tokens": response.total_tokens,
            "stage_metrics": response.stage_metrics,
            "algorithm_metadata": response.metadata,
        }
    except Exception as exc:
        return {
            "task_id": task["id"],
            "algorithm": algorithm.name,
            "website": task.get("website"),
            "difficulty": task.get("difficulty"),
            "task_type": task.get("task_type"),
            "instruction": task["instruction"],
            "start_url": task["start_url"],
            "status": "error",
            "gold_answer": task.get("gold_answer"),
            "error": f"{type(exc).__name__}: {exc}",
            "duration_ms": round((perf_counter() - started) * 1000, 3),
        }


def _write_summary(
    results_path: Path, summary_path: Path, algorithm: BenchmarkAlgorithm
) -> None:
    latest_by_task = {}
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if line:
            record = json.loads(line)
            latest_by_task[record["task_id"]] = record
    records = list(latest_by_task.values())
    completed = [record for record in records if record["status"] == "completed"]
    judged = [record for record in completed
              if record.get("model_judgement", {}).get("status") == "completed"]
    passed = sum(record["model_judgement"]["passed"] for record in judged)
    deterministic_passed = sum(record["evaluation"]["passed"] for record in completed)
    summary = {
        "algorithm": algorithm.name,
        "configuration": algorithm.configuration(),
        "generated_at": datetime.now(UTC).isoformat(),
        "tasks_recorded": len(records),
        "tasks_completed": len(completed),
        "errors": len(records) - len(completed),
        "judge_model": JUDGE_MODEL,
        "tasks_judged": len(judged),
        "judge_errors": len(completed) - len(judged) if JUDGE_MODEL else 0,
        "passed": passed,
        "accuracy": round(passed / len(judged), 4) if judged else 0.0,
        "mean_judge_score": round(
            sum(record["model_judgement"]["score"] for record in judged) / len(judged), 4
        ) if judged else 0.0,
        "deterministic_passed": deterministic_passed,
        "deterministic_accuracy": round(
            deterministic_passed / len(completed), 4
        ) if completed else 0.0,
        "mean_leaf_coverage": round(
            sum(record["evaluation"]["leaf_coverage"] for record in completed) / len(completed), 4
        ) if completed else 0.0,
        "mean_token_f1": round(
            sum(record["evaluation"]["token_f1"] for record in completed) / len(completed), 4
        ) if completed else 0.0,
        "total_tokens": sum(record.get("total_tokens", 0) for record in completed),
        "judge_total_tokens": sum(
            record["model_judgement"].get("total_tokens", 0) for record in judged
        ),
        "total_duration_ms": round(sum(record.get("duration_ms", 0) for record in records), 3),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    tasks = _load_tasks()
    judge = ModelAnswerJudge(JUDGE_MODEL) if JUDGE_MODEL else None
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for algorithm in ALGORITHMS:
        safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in algorithm.name)
        results_path = OUTPUT_DIRECTORY / f"{safe_name}.jsonl"
        summary_path = OUTPUT_DIRECTORY / f"{safe_name}.summary.json"
        completed_ids = _load_completed(
            results_path, require_model_judge=judge is not None
        )
        pending = [task for task in tasks if task["id"] not in completed_ids]
        print(f"[{algorithm.name}] {len(pending)} pending of {len(tasks)} selected tasks")
        with results_path.open("a", encoding="utf-8") as output:
            for index, task in enumerate(pending, start=1):
                print(f"[{algorithm.name}] {index}/{len(pending)} {task['id']}", flush=True)
                record = _run_task(task, algorithm, judge)
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                result = record.get("evaluation", {})
                judgement = record.get("model_judgement") or {}
                print(
                    f"  {record['status']} judge_pass={judgement.get('passed')} "
                    f"judge_score={judgement.get('score')} "
                    f"deterministic_pass={result.get('passed')} "
                    f"coverage={result.get('leaf_coverage')} "
                    f"duration_ms={record['duration_ms']}",
                    flush=True,
                )
                _write_summary(results_path, summary_path, algorithm)
        _write_summary(results_path, summary_path, algorithm)


if __name__ == "__main__":
    main()
