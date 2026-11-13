"""Coordinator-only launcher for one explicitly dispatched existing Sol turn.

This does not select a scientific stage or create an agent. The coordinator must
first finish the Astra handoff and prepare prompt.md plus dispatch_intent.json.
Public output stays in the turn directory; no global configuration is changed.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parent
LOOP = ROOT.parent
REPO = LOOP.parents[2]
THREAD = "01a1d350-d1ec-70f2-9e37-8960b2dfe7a8"
CLI = "/home/liwt/anaconda3/envs/claude-node20/bin/codex"


def save(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("turn_directory", type=Path)
    args = parser.parse_args()
    turn = args.turn_directory.resolve()
    if turn.parent != ROOT or not turn.is_dir():
        parser.error("Use a prepared direct child of sol_fast_cli")
    intent = json.loads((turn / "dispatch_intent.json").read_text())
    state = json.loads((LOOP / "state.json").read_text())
    if state.get("writer_lease") != "implementation":
        parser.error("Implementation does not own the scientific writer lease")
    if state["roles"]["implementation"]["task_id"] != THREAD:
        parser.error("Coordinator role is not the authorized existing CLI UUID")
    if intent.get("task_id") != THREAD:
        parser.error("Dispatch intent names a different task")
    prompt = turn / "prompt.md"
    digest = hashlib.sha256(prompt.read_bytes()).hexdigest()
    if digest != intent["prompt_sha256"]:
        parser.error("Prompt identity differs from the coordinator dispatch")
    request_path = turn / "request.json"
    # An existing request is an ambiguous/already-dispatched operation, even if
    # it says completed. Inspect the actual handle; never launch this turn twice.
    with request_path.open("x") as request_file:
        request_file.write("{}\n")
    argv = [
        CLI, "exec", "--approve-for-me", "--cd", str(REPO),
        "resume", THREAD, "--model", "gpt-6.1-sol",
        "--config", 'model_reasoning_effort="high"',
        "--config", 'service_tier="fast"', "--json",
        "--output-last-message", str(turn / "last_message.txt"), "-",
    ]
    request = {
        "status": "request_prepared", "requested_epoch": time.time(),
        "argv": argv, "requested_thread_id": THREAD,
        "requested_model": "gpt-6.1-sol",
        "requested_reasoning_effort": "high", "requested_service_tier": "fast",
        "prompt": str(prompt), "prompt_sha256": digest,
        "public_events": str(turn / "events.jsonl"),
        "stderr_log": str(turn / "stderr.log"),
        "last_message": str(turn / "last_message.txt"),
        "round": intent["round"], "experiment": state["current_experiment"],
        "scientific_write_lease_granted": True,
        "global_config_modified": False, "backend_effective_service_tier": None,
        "supervisor_pid": os.getpid(), "host_uid": os.getuid(),
    }
    save(request_path, request)
    try:
        with prompt.open("rb") as source, (turn / "events.jsonl").open("xb") as output, (turn / "stderr.log").open("xb") as errors:
            process = subprocess.Popen(argv, stdin=source, stdout=output,
                                       stderr=errors, cwd=REPO)
            request.update(status="CLI_process_running", CLI_pid=process.pid,
                           started_epoch=time.time())
            save(request_path, request)
            print(json.dumps({"CLI_pid": process.pid,
                              "supervisor_pid": os.getpid(),
                              "exact_resume_UUID": THREAD}), flush=True)
            returncode = process.wait()
    except Exception as error:
        request.update(status="launcher_error_inspect_before_any_retry",
                       error_type=type(error).__name__, error=str(error),
                       recorded_epoch=time.time())
        save(request_path, request)
        raise
    request.update(status="CLI_process_exited", returncode=returncode,
                   finished_epoch=time.time())
    save(request_path, request)
    print(json.dumps({"CLI_process_exited": True,
                      "returncode": returncode}), flush=True)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
