"""Run model configurations against the web-retrieval benchmark.

Edit the constants below, then run::

    python -m benchmark.run_benchmark

Results are appended after every task, so interrupted runs can be resumed.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import signal
from datetime import UTC, datetime
from pathlib import Path
from queue import Empty
from time import monotonic, perf_counter
from typing import Any

from benchmark.algorithms import BenchmarkAlgorithm, TriAgentAlgorithm
from benchmark.evaluate import evaluate_answer
from benchmark.judge import ModelAnswerJudge
from benchmark.ollama_service import OllamaSupervisor


DATASET_PATH = Path(__file__).parent / "tasks" / "web_retrieval_tasks_100_T2.json"
OUTPUT_DIRECTORY = Path("output") / "benchmark"
SCORABLE_ONLY = True
TASK_IDS: set[str] = set()  # Empty means all eligible tasks.
TASK_LIMIT: int | None = None
RESUME = True
MAX_TASK_MICROSECONDS: int | None = 600_000_000  # 10 minutes; None disables it.
JUDGE_MODEL: str | None = "ollama:deepseek-r1:8b"  # Set to None for deterministic scoring only.
OLLAMA_AUTO_RECOVER = True
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_START_TIMEOUT_SECONDS = 30
OLLAMA_CONNECTION_RETRIES = 1


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


def _execute_task(
    task: dict[str, Any], algorithm: BenchmarkAlgorithm,
    judge_model: str | None = None,
) -> dict[str, Any]:
    started = perf_counter()
    try:
        response = algorithm.run(task["instruction"], task["start_url"])
        answer = response.answer
        evaluation = evaluate_answer(answer, task.get("gold_answer"))
        model_judgement = None
        if judge_model is not None:
            try:
                judge = ModelAnswerJudge(judge_model)
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
        duration_microseconds = round((perf_counter() - started) * 1_000_000)
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
            "duration_ms": round(duration_microseconds / 1000, 3),
            "duration_microseconds": duration_microseconds,
            "algorithm_duration_ms": response.duration_ms,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "total_tokens": response.total_tokens,
            "stage_metrics": response.stage_metrics,
            "algorithm_metadata": response.metadata,
        }
    except Exception as exc:
        duration_microseconds = round((perf_counter() - started) * 1_000_000)
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
            "duration_ms": round(duration_microseconds / 1000, 3),
            "duration_microseconds": duration_microseconds,
        }


def _task_worker(result_queue, task, algorithm, judge_model: str | None) -> None:
    if os.name != "nt":
        os.setsid()
    try:
        result_queue.put(_execute_task(task, algorithm, judge_model))
    except BaseException as exc:
        result_queue.put({
            "task_id": task.get("id"),
            "algorithm": algorithm.name,
            "status": "error",
            "error": f"Worker {type(exc).__name__}: {exc}",
        })


def _stop_worker(process: mp.Process) -> None:
    if not process.is_alive():
        return
    if os.name != "nt":
        try:
            process_group = os.getpgid(process.pid)
            if process_group == process.pid:
                os.killpg(process_group, signal.SIGTERM)
            else:
                process.terminate()
        except (ProcessLookupError, PermissionError):
            process.terminate()
    else:
        process.terminate()
    process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)


def _timeout_record(
    task: dict[str, Any], algorithm: BenchmarkAlgorithm, elapsed_microseconds: int,
    limit_microseconds: int | None = None,
) -> dict[str, Any]:
    limit = MAX_TASK_MICROSECONDS if limit_microseconds is None else limit_microseconds
    return {
        "task_id": task["id"],
        "algorithm": algorithm.name,
        "website": task.get("website"),
        "difficulty": task.get("difficulty"),
        "task_type": task.get("task_type"),
        "instruction": task["instruction"],
        "start_url": task["start_url"],
        "status": "timeout",
        "gold_answer": task.get("gold_answer"),
        "error": (
            f"TaskTimeoutError: total task process exceeded "
            f"{limit} microseconds"
        ),
        "duration_ms": round(elapsed_microseconds / 1000, 3),
        "duration_microseconds": elapsed_microseconds,
    }


def _run_task(
    task: dict[str, Any], algorithm: BenchmarkAlgorithm,
    judge_model: str | None = None,
    *, timeout_microseconds: int | None = None,
) -> dict[str, Any]:
    limit = MAX_TASK_MICROSECONDS if timeout_microseconds is None else timeout_microseconds
    if limit is None:
        return _execute_task(task, algorithm, judge_model)
    if limit <= 0:
        raise ValueError("MAX_TASK_MICROSECONDS must be positive or None")

    started = perf_counter()
    timeout_seconds = limit / 1_000_000
    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_task_worker,
        args=(result_queue, task, algorithm, judge_model),
        name=f"benchmark-{algorithm.name}-{task['id']}",
    )
    process.start()
    deadline = monotonic() + timeout_seconds
    while True:
        remaining = deadline - monotonic()
        if remaining <= 0:
            _stop_worker(process)
            result_queue.cancel_join_thread()
            result_queue.close()
            elapsed = round((perf_counter() - started) * 1_000_000)
            return _timeout_record(task, algorithm, elapsed, limit)
        try:
            record = result_queue.get(timeout=min(0.25, remaining))
            break
        except Empty:
            if not process.is_alive():
                record = {
                    "task_id": task["id"],
                    "algorithm": algorithm.name,
                    "status": "error",
                    "error": (
                        f"Algorithm worker exited with code {process.exitcode} "
                        "without returning a result"
                    ),
                    "duration_microseconds": round(
                        (perf_counter() - started) * 1_000_000
                    ),
                }
                record["duration_ms"] = round(
                    record["duration_microseconds"] / 1000, 3
                )
                break
    process.join(timeout=5)
    if process.is_alive():
        _stop_worker(process)
    result_queue.close()
    result_queue.join_thread()
    elapsed_microseconds = round((perf_counter() - started) * 1_000_000)
    record["duration_microseconds"] = elapsed_microseconds
    record["duration_ms"] = round(elapsed_microseconds / 1000, 3)
    return record


def _uses_ollama(algorithm: BenchmarkAlgorithm) -> bool:
    configuration = json.dumps(algorithm.configuration(), default=str).casefold()
    return "ollama:" in configuration or bool(
        JUDGE_MODEL and JUDGE_MODEL.casefold().startswith("ollama:")
    )


def _remaining_microseconds(started: float) -> int | None:
    if MAX_TASK_MICROSECONDS is None:
        return None
    elapsed = round((perf_counter() - started) * 1_000_000)
    return max(0, MAX_TASK_MICROSECONDS - elapsed)


def _service_error_record(
    task: dict[str, Any], algorithm: BenchmarkAlgorithm,
    started: float, exc: Exception,
) -> dict[str, Any]:
    elapsed = round((perf_counter() - started) * 1_000_000)
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
        "error": f"OllamaServiceError: {type(exc).__name__}: {exc}",
        "duration_ms": round(elapsed / 1000, 3),
        "duration_microseconds": elapsed,
    }


def _run_task_with_recovery(
    task: dict[str, Any], algorithm: BenchmarkAlgorithm,
    supervisor: OllamaSupervisor | None,
) -> dict[str, Any]:
    started = perf_counter()
    service_start_count = 0
    connection_retries = 0
    if supervisor is not None:
        try:
            remaining = _remaining_microseconds(started)
            if remaining == 0:
                return _timeout_record(task, algorithm, 0)
            service_start_count += int(supervisor.ensure(
                timeout_seconds=(remaining / 1_000_000 if remaining else None)
            ))
        except Exception as exc:
            if _remaining_microseconds(started) == 0:
                elapsed = round((perf_counter() - started) * 1_000_000)
                return _timeout_record(task, algorithm, elapsed)
            return _service_error_record(task, algorithm, started, exc)

    while True:
        remaining = _remaining_microseconds(started)
        if remaining == 0:
            elapsed = round((perf_counter() - started) * 1_000_000)
            return _timeout_record(task, algorithm, elapsed)
        record = _run_task(
            task, algorithm, JUDGE_MODEL, timeout_microseconds=remaining
        )
        connection_error = record.get("error") or (
            record.get("model_judgement") or {}
        ).get("error")
        if (
            supervisor is None
            or not supervisor.is_connection_refused(connection_error)
            or connection_retries >= OLLAMA_CONNECTION_RETRIES
        ):
            break
        try:
            remaining = _remaining_microseconds(started)
            if remaining == 0:
                elapsed = round((perf_counter() - started) * 1_000_000)
                return _timeout_record(task, algorithm, elapsed)
            supervisor.ensure(
                timeout_seconds=(remaining / 1_000_000 if remaining else None),
            )
            service_start_count += 1
            connection_retries += 1
        except Exception as exc:
            if _remaining_microseconds(started) == 0:
                elapsed = round((perf_counter() - started) * 1_000_000)
                return _timeout_record(task, algorithm, elapsed)
            return _service_error_record(task, algorithm, started, exc)

    elapsed = round((perf_counter() - started) * 1_000_000)
    record["duration_microseconds"] = elapsed
    record["duration_ms"] = round(elapsed / 1000, 3)
    record["ollama_service_start_count"] = service_start_count
    record["ollama_connection_retries"] = connection_retries
    return record


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
        "timeouts": sum(record.get("status") == "timeout" for record in records),
        "max_task_microseconds": MAX_TASK_MICROSECONDS,
        "ollama_service_starts": sum(
            record.get("ollama_service_start_count", 0) for record in records
        ),
        "ollama_connection_retries": sum(
            record.get("ollama_connection_retries", 0) for record in records
        ),
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
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for algorithm in ALGORITHMS:
        supervisor = (
            OllamaSupervisor(
                base_url=OLLAMA_BASE_URL,
                start_timeout_seconds=OLLAMA_START_TIMEOUT_SECONDS,
                log_path=OUTPUT_DIRECTORY / "ollama.log",
            )
            if OLLAMA_AUTO_RECOVER and _uses_ollama(algorithm)
            else None
        )
        safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in algorithm.name)
        results_path = OUTPUT_DIRECTORY / f"{safe_name}.jsonl"
        summary_path = OUTPUT_DIRECTORY / f"{safe_name}.summary.json"
        completed_ids = _load_completed(
            results_path, require_model_judge=JUDGE_MODEL is not None
        )
        pending = [task for task in tasks if task["id"] not in completed_ids]
        print(f"[{algorithm.name}] {len(pending)} pending of {len(tasks)} selected tasks")
        try:
            with results_path.open("a", encoding="utf-8") as output:
                for index, task in enumerate(pending, start=1):
                    print(f"[{algorithm.name}] {index}/{len(pending)} {task['id']}", flush=True)
                    record = _run_task_with_recovery(task, algorithm, supervisor)
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    result = record.get("evaluation", {})
                    judgement = record.get("model_judgement") or {}
                    print(
                        f"  {record['status']} judge_pass={judgement.get('passed')} "
                        f"judge_score={judgement.get('score')} "
                        f"deterministic_pass={result.get('passed')} "
                        f"coverage={result.get('leaf_coverage')} "
                        f"ollama_retries={record.get('ollama_connection_retries', 0)} "
                        f"duration_us={record.get('duration_microseconds')}",
                        flush=True,
                    )
                    _write_summary(results_path, summary_path, algorithm)
            _write_summary(results_path, summary_path, algorithm)
        finally:
            if supervisor is not None:
                supervisor.close()


if __name__ == "__main__":
    main()
