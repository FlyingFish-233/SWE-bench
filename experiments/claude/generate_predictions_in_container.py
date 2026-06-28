from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import io
import json
import logging
import tarfile
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Any

import docker
import docker.errors

from swebench.harness.constants import (
    KEY_INSTANCE_ID,
    LATEST,
    RUN_EVALUATION_LOG_DIR,
)
from swebench.harness.docker_build import (
    BuildImageError,
    build_instance_image,
    close_logger,
    setup_logger,
)
from swebench.harness.docker_utils import cleanup_container
from swebench.harness.test_spec.test_spec import make_test_spec, TestSpec
from swebench.harness.utils import load_swebench_dataset, optional_str


MODEL_NAME = "claude-code-qwen3.6-27b-fp8-container"
CONTAINER_WORKDIR = "/testbed"
DEFAULT_ALLOWED_TOOLS = "Bash Edit Read Write Grep Glob LS"
DEFAULT_BASE_URL = "http://127.0.0.1:30010"
DEFAULT_MODEL = "Qwen3.6-27B-FP8"
DEFAULT_CONTAINER_USER = "claude"
DEFAULT_CONTAINER_HOME = "/home/claude"
DEFAULT_TRACE_DIR = "/raid/zwx/SWE-bench/experiments/claude/traces"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate SWE-bench predictions by running Claude Code inside each "
            "SWE-bench instance container."
        )
    )
    parser.add_argument(
        "--dataset_name",
        default="/raid/zwx/datasets/SWE-bench_Lite",
        help="Dataset name or local path accepted by SWE-bench.",
    )
    parser.add_argument("--split", default="test", help="Dataset split to load.")
    parser.add_argument(
        "--instance_ids",
        nargs="+",
        help="Optional space-separated instance IDs to run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of remaining instances to run.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Predictions JSON path to write/update.",
    )
    parser.add_argument(
        "--trace_dir",
        default=DEFAULT_TRACE_DIR,
        help=(
            "Directory for realtime Claude Code traces. Files are written under "
            "<trace_dir>/<run_id>/<instance_id>/."
        ),
    )
    parser.add_argument(
        "--trace_output_format",
        default="stream-json",
        choices=["text", "json", "stream-json"],
        help="Claude Code output format to capture in realtime trace files.",
    )
    parser.add_argument(
        "--include_partial_messages",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Include partial message chunks in Claude Code stream-json output. "
            "Disabled by default so trace logs stay at complete JSON event granularity."
        ),
    )
    parser.add_argument(
        "--include_hook_events",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include hook lifecycle events in Claude Code stream-json output. "
            "Use --no-include_hook_events to disable."
        ),
    )
    parser.add_argument(
        "--run_id",
        required=True,
        help="Run ID used for solve container names and logs.",
    )
    parser.add_argument(
        "--namespace",
        type=optional_str,
        default="swebench",
        help='Docker image namespace. Use "none" for local image builds.',
    )
    parser.add_argument(
        "--instance_image_tag",
        default=LATEST,
        help="Instance image tag.",
    )
    parser.add_argument(
        "--env_image_tag",
        default=LATEST,
        help="Environment image tag.",
    )
    parser.add_argument(
        "--claude_bin",
        default="/raid/zwx/.local/bin/claude",
        help="Host path to the Claude Code binary.",
    )
    parser.add_argument(
        "--claude_version_file",
        default="/raid/zwx/.local/share/claude/versions/2.1.187",
        help="Host path to Claude Code version payload.",
    )
    parser.add_argument(
        "--anthropic_base_url",
        default=DEFAULT_BASE_URL,
        help="Anthropic-compatible API base URL visible from the container.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name to pass to Claude Code.",
    )
    parser.add_argument(
        "--allowed_tools",
        default=DEFAULT_ALLOWED_TOOLS,
        help="Claude Code tools to allow.",
    )
    parser.add_argument(
        "--permission_mode",
        default="bypassPermissions",
        choices=["acceptEdits", "auto", "default", "dontAsk", "plan", "bypassPermissions"],
        help=(
            "Claude Code permission mode. Defaults to bypassPermissions because "
            "Claude runs as a non-root user inside the solve container."
        ),
    )
    parser.add_argument(
        "--claude_timeout",
        type=int,
        default=None,
        help=(
            "Timeout in seconds for each Claude Code solve attempt. "
            "Omit to run without a wall-clock timeout."
        ),
    )
    parser.add_argument(
        "--max_timeout",
        type=int,
        help=(
            "Alias for --claude_timeout. When set, overrides --claude_timeout "
            "for each Claude Code solve attempt."
        ),
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        help=(
            "Maximum number of Claude Code tool-task steps per instance. "
            "Counts stream-json system/task_started events and terminates the "
            "Claude exec when the limit is exceeded."
        ),
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Maximum number of SWE-bench instances to solve concurrently.",
    )
    parser.add_argument(
        "--force_rebuild",
        action="store_true",
        help="Force rebuild local instance images. Ignored for remote namespace images.",
    )
    parser.add_argument(
        "--keep_containers",
        action="store_true",
        help="Leave solve containers running for debugging.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate instances already present in the output file.",
    )
    parser.add_argument(
        "--skip_empty_patches",
        action="store_true",
        help="Do not write predictions whose model_patch is empty.",
    )
    parser.add_argument(
        "--container_user",
        default=DEFAULT_CONTAINER_USER,
        help="Non-root user to create/use for running Claude Code in the container.",
    )
    parser.add_argument(
        "--container_home",
        default=DEFAULT_CONTAINER_HOME,
        help="Home directory for the non-root Claude Code user in the container.",
    )
    parser.add_argument(
        "--no_dangerously_skip_permissions",
        action="store_false",
        dest="dangerously_skip_permissions",
        default=True,
        help="Do not pass Claude Code --dangerously-skip-permissions.",
    )
    args = parser.parse_args()
    if args.max_timeout is not None:
        args.claude_timeout = args.max_timeout
    if args.max_workers < 1:
        parser.error("--max_workers must be at least 1")
    if args.claude_timeout is not None and args.claude_timeout < 1:
        parser.error("--claude_timeout/--max_timeout must be at least 1 when set")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max_steps must be at least 1")
    return args


def load_existing_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {prediction[KEY_INSTANCE_ID]: prediction for prediction in data}


def write_predictions(path: Path, predictions: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(predictions.values(), key=lambda item: item[KEY_INSTANCE_ID])
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def prediction_from_patch(
    instance_id: str,
    patch: str,
) -> dict[str, Any]:
    return {
        KEY_INSTANCE_ID: instance_id,
        "model_patch": patch,
        "model_name_or_path": MODEL_NAME,
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_operation_event(event: dict[str, Any]) -> bool:
    event_type = event.get("type")
    subtype = event.get("subtype")
    if event_type == "result":
        return True
    if event_type == "system" and isinstance(subtype, str):
        return subtype.startswith("task_")
    message = event.get("message")
    if not isinstance(message, dict):
        return False
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(item, dict) and item.get("type") in {"tool_use", "tool_result"}
        for item in content
    )


def make_solve_logger(run_id: str, instance_id: str) -> logging.Logger:
    log_dir = RUN_EVALUATION_LOG_DIR / run_id / MODEL_NAME / instance_id
    return setup_logger(instance_id, log_dir / "solve_container.log")


def ensure_instance_image(
    client: docker.DockerClient,
    test_spec: TestSpec,
    logger: logging.Logger,
    force_rebuild: bool,
) -> None:
    if force_rebuild and test_spec.is_remote_image:
        raise ValueError("Cannot force rebuild a remote namespace image.")

    if force_rebuild and not test_spec.is_remote_image:
        try:
            client.images.remove(test_spec.instance_image_key, force=True)
        except docker.errors.ImageNotFound:
            pass

    try:
        client.images.get(test_spec.instance_image_key)
        logger.info("Instance image exists: %s", test_spec.instance_image_key)
        return
    except docker.errors.ImageNotFound:
        pass

    if test_spec.is_remote_image:
        logger.info("Pulling instance image: %s", test_spec.instance_image_key)
        try:
            client.images.pull(test_spec.instance_image_key)
        except Exception as exc:
            raise BuildImageError(test_spec.instance_id, str(exc), logger) from exc
    else:
        logger.info("Building local instance image: %s", test_spec.instance_image_key)
        build_instance_image(test_spec, client, logger, nocache=False)


def tar_file_bytes(src: Path, dst_name: str) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        tar.add(src, arcname=dst_name)
    buf.seek(0)
    return buf.read()


def copy_host_file_to_container(container: Any, src: Path, dst: PurePosixPath) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"Required host file does not exist: {src}")
    parent = str(dst.parent)
    container.exec_run(f"mkdir -p {parent}", user="root")
    container.put_archive(parent, tar_file_bytes(src, dst.name))


def claude_container_bin(args: argparse.Namespace) -> PurePosixPath:
    return PurePosixPath(args.container_home) / ".local/bin/claude"


def claude_container_version(args: argparse.Namespace) -> PurePosixPath:
    return PurePosixPath(args.container_home) / ".local/share/claude/versions/2.1.187"


def ensure_container_user(container: Any, args: argparse.Namespace) -> None:
    user = args.container_user
    home = args.container_home
    script = f"""
set -eu
if ! id -u {user} >/dev/null 2>&1; then
  if command -v useradd >/dev/null 2>&1; then
    useradd -m -d {home} -s /bin/bash {user}
  elif command -v adduser >/dev/null 2>&1; then
    adduser --disabled-password --gecos '' --home {home} {user}
  else
    mkdir -p {home}
    echo '{user}:x:1000:1000:Claude Code:{home}:/bin/bash' >> /etc/passwd
    echo '{user}:x:1000:' >> /etc/group
  fi
fi
mkdir -p {home}/.local/bin {home}/.local/share/claude/versions
chown -R {user}:{user} {home} /testbed
if command -v sudo >/dev/null 2>&1; then
  if getent group sudo >/dev/null 2>&1; then
    usermod -aG sudo {user} || true
  fi
  mkdir -p /etc/sudoers.d
  printf '%s\\n' '{user} ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/{user}
  chmod 0440 /etc/sudoers.d/{user}
fi
"""
    result = container.exec_run(["/bin/bash", "-lc", script], user="root")
    output = result.output.decode("utf-8", errors="replace")
    if result.exit_code != 0:
        raise RuntimeError(f"Failed to prepare container user {user}: {output}")


def ensure_git_baseline(container: Any, args: argparse.Namespace) -> None:
    script = """
set -eu
cd /testbed
git config --global --add safe.directory /testbed || true
if git -c safe.directory=/testbed rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git -c safe.directory=/testbed config core.fileMode false
  exit 0
fi
git init -q
git config user.email 'claude-code@example.invalid'
git config user.name 'Claude Code Runner'
git config core.fileMode false
git add -A -f .
git commit -q --allow-empty -m 'baseline'
"""
    result = container.exec_run(
        ["/bin/bash", "-c", script],
        user=args.container_user,
        environment={"HOME": args.container_home},
    )
    output = result.output.decode("utf-8", errors="replace")
    if result.exit_code != 0:
        raise RuntimeError(f"Failed to initialize /testbed git baseline: {output}")


def exec_capture(
    client: docker.DockerClient,
    container_id: str,
    cmd: str | list[str],
    *,
    environment: dict[str, str] | None = None,
    workdir: str | None = None,
    user: str = "root",
    timeout: int | None = None,
    trace_stream_path: Path | None = None,
    trace_operations_path: Path | None = None,
    max_steps: int | None = None,
) -> tuple[int | None, str, bool, float, str | None, int]:
    exec_id = client.api.exec_create(
        container=container_id,
        cmd=cmd,
        environment=environment,
        workdir=workdir,
        user=user,
    )["Id"]
    output = bytearray()
    exception: Exception | None = None
    start = time.time()
    timed_out = False
    stop_reason: str | None = None
    step_count = 0
    line_buffer = bytearray()

    trace_stream_file = None
    trace_operations_file = None
    if trace_stream_path is not None:
        trace_stream_path.parent.mkdir(parents=True, exist_ok=True)
        trace_stream_file = trace_stream_path.open("wb")
    if trace_operations_path is not None:
        trace_operations_path.parent.mkdir(parents=True, exist_ok=True)
        trace_operations_file = trace_operations_path.open("w", encoding="utf-8")

    def run() -> None:
        nonlocal exception, step_count, stop_reason
        try:
            stream = client.api.exec_start(exec_id, stream=True)
            for chunk in stream:
                output.extend(chunk)
                if trace_stream_file is not None:
                    trace_stream_file.write(chunk)
                    trace_stream_file.flush()
                if max_steps is not None or trace_operations_file is not None:
                    line_buffer.extend(chunk)
                    while b"\n" in line_buffer:
                        line, _, rest = line_buffer.partition(b"\n")
                        line_buffer[:] = rest
                        if not line.strip():
                            continue
                        try:
                            event = json.loads(line.decode("utf-8", errors="replace"))
                        except json.JSONDecodeError:
                            continue
                        if (
                            trace_operations_file is not None
                            and is_operation_event(event)
                        ):
                            trace_operations_file.write(
                                json.dumps(event, ensure_ascii=False) + "\n"
                            )
                            trace_operations_file.flush()
                        if (
                            event.get("type") == "system"
                            and event.get("subtype") == "task_started"
                        ):
                            step_count += 1
                            if step_count > max_steps and stop_reason is None:
                                stop_reason = "max_steps_exceeded"
                                inspect = client.api.exec_inspect(exec_id)
                                pid = inspect.get("Pid")
                                if pid:
                                    kill_id = client.api.exec_create(
                                        container=container_id,
                                        cmd=["kill", "-TERM", str(pid)],
                                        user="root",
                                    )["Id"]
                                    client.api.exec_start(kill_id)
                                return
        except Exception as exc:
            exception = exc

    try:
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            inspect = client.api.exec_inspect(exec_id)
            pid = inspect.get("Pid")
            if pid:
                kill_id = client.api.exec_create(
                    container=container_id,
                    cmd=["kill", "-TERM", str(pid)],
                    user="root",
                )["Id"]
                client.api.exec_start(kill_id)
            timed_out = stop_reason is None
            thread.join(5)
        if exception is not None:
            raise exception
        elapsed = time.time() - start
        inspect = client.api.exec_inspect(exec_id)
        text = output.decode("utf-8", errors="replace")
        return inspect.get("ExitCode"), text, timed_out, elapsed, stop_reason, step_count
    finally:
        if trace_stream_file is not None:
            trace_stream_file.close()
        if trace_operations_file is not None:
            trace_operations_file.close()


def make_trace_dir(args: argparse.Namespace, instance_id: str) -> Path:
    return Path(args.trace_dir) / args.run_id / instance_id


def collect_patch(
    container: Any,
    args: argparse.Namespace,
    trace_dir: Path,
    logger: logging.Logger,
) -> tuple[str, str | None]:
    """Collect git diff from /testbed and always write trace_dir/patch.diff."""
    try:
        diff_result = container.exec_run(
            ["git", "-C", "/testbed", "-c", "core.fileMode=false", "diff"],
            user=args.container_user,
            environment={"HOME": args.container_home},
        )
        patch = diff_result.output.decode("utf-8", errors="replace")
        (trace_dir / "patch.diff").write_text(patch)
        logger.info("Collected patch bytes: %d", len(patch.encode("utf-8")))
        if diff_result.exit_code != 0:
            return patch, f"git diff failed with exit code {diff_result.exit_code}"
        return patch, None
    except Exception as exc:
        error = f"git diff collection failed: {exc!r}"
        logger.exception(error)
        try:
            (trace_dir / "patch.diff").write_text("")
        except Exception:
            logger.exception("Failed to write empty patch.diff after collection error")
        return "", error


def is_empty_final_response_failure(trace_dir: Path) -> bool:
    stream_path = trace_dir / "claude.stream.jsonl"
    if not stream_path.exists():
        return False
    needle = (
        "[ede_diagnostic] result_type=assistant "
        "last_content_type=none stop_reason=end_turn"
    )
    try:
        with stream_path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if needle in line:
                    return True
    except OSError:
        return False
    return False


def create_solve_container(
    client: docker.DockerClient,
    test_spec: TestSpec,
    run_id: str,
) -> Any:
    name = f"claude.solve.{run_id}.{test_spec.instance_id}".lower()
    try:
        old = client.containers.get(name)
        old.remove(force=True)
    except docker.errors.NotFound:
        pass

    run_args = test_spec.docker_specs.get("run_args", {})
    cap_add = run_args.get("cap_add", [])
    return client.containers.create(
        image=test_spec.instance_image_key,
        name=name,
        user="root",
        detach=True,
        command="tail -f /dev/null",
        platform=test_spec.platform,
        cap_add=cap_add,
        network_mode="host",
        working_dir=CONTAINER_WORKDIR,
    )


def build_prompt(instance: dict[str, Any]) -> str:
    hints = instance.get("hints_text") or ""
    step_limit = ""
    max_steps = instance.get("_claude_max_steps")
    if max_steps is not None:
        step_limit = (
            f"\nYou have a hard budget of at most {max_steps} tool-use steps. "
            "Plan briefly, avoid exploratory loops, and finish before the step budget.\n"
        )
    return f"""We need solve one SWE-bench task in this repository.

You are inside the task repository at /testbed. Modify only the source files
needed to fix the issue. Do not edit tests unless the issue explicitly requires
test-source changes. Keep the patch minimal.
{step_limit}

Instance ID:
{instance[KEY_INSTANCE_ID]}

Problem statement:
{instance.get("problem_statement", "")}

Hints:
{hints}

After making the fix, run a focused relevant test if practical. When finished,
reply with a concise summary. The evaluation script will collect git diff."""


def claude_environment(args: argparse.Namespace) -> dict[str, str]:
    path = (
        f"{args.container_home}/.local/bin:"
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    )
    return {
        "PATH": path,
        "HOME": args.container_home,
        "ANTHROPIC_BASE_URL": args.anthropic_base_url,
        "ANTHROPIC_API_KEY": "dummy",
        "ANTHROPIC_AUTH_TOKEN": "dummy",
        "ANTHROPIC_MODEL": args.model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": args.model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": args.model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": args.model,
        "CLAUDE_CODE_SUBAGENT_MODEL": args.model,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }


def install_claude(container: Any, args: argparse.Namespace) -> str:
    ensure_container_user(container, args)
    bin_path = claude_container_bin(args)
    version_path = claude_container_version(args)
    copy_host_file_to_container(
        container, Path(args.claude_bin), bin_path
    )
    copy_host_file_to_container(
        container, Path(args.claude_version_file), version_path
    )
    container.exec_run(
        (
            f"chmod +x {bin_path} {version_path} && "
            f"chown -R {args.container_user}:{args.container_user} "
            f"{args.container_home}/.local"
        ),
        user="root",
    )
    result = container.exec_run(
        str(bin_path) + " --version",
        user=args.container_user,
        environment=claude_environment(args),
    )
    output = result.output.decode("utf-8", errors="replace").strip()
    if result.exit_code != 0:
        raise RuntimeError(f"Claude Code install check failed: {output}")
    return output


def solve_instance(
    client: docker.DockerClient,
    instance: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    test_spec = make_test_spec(
        instance,
        namespace=args.namespace,
        env_image_tag=args.env_image_tag,
        instance_image_tag=args.instance_image_tag,
    )
    logger = make_solve_logger(args.run_id, test_spec.instance_id)
    trace_dir = make_trace_dir(args, test_spec.instance_id)
    trace_dir.mkdir(parents=True, exist_ok=True)
    status_path = trace_dir / "status.json"
    write_json(
        status_path,
        {
            "instance_id": test_spec.instance_id,
            "run_id": args.run_id,
            "status": "starting",
            "started_at": utc_now(),
        },
    )
    container = None
    try:
        logger.info("Starting solve for %s", test_spec.instance_id)
        ensure_instance_image(client, test_spec, logger, args.force_rebuild)
        container = create_solve_container(client, test_spec, args.run_id)
        container.start()
        logger.info("Solve container started: %s (%s)", container.name, container.id)

        version = install_claude(container, args)
        logger.info("Claude Code version in container: %s", version)
        ensure_git_baseline(container, args)
        logger.info("/testbed git baseline is ready")

        prompt_instance = dict(instance)
        prompt_instance["_claude_max_steps"] = args.max_steps
        prompt = build_prompt(prompt_instance)
        cmd = [
            str(claude_container_bin(args)),
            "--bare",
            "--print",
            "--model",
            args.model,
            "--no-session-persistence",
            "--permission-mode",
            args.permission_mode,
        ]
        if args.dangerously_skip_permissions:
            cmd.append("--dangerously-skip-permissions")
        cmd.extend(
            [
            "--allowedTools",
            args.allowed_tools,
            "--output-format",
            args.trace_output_format,
            ]
        )
        if args.trace_output_format == "stream-json":
            cmd.append("--verbose")
            if args.include_partial_messages:
                cmd.append("--include-partial-messages")
            if args.include_hook_events:
                cmd.append("--include-hook-events")
        cmd.extend(["--", prompt])
        (trace_dir / "prompt.txt").write_text(prompt)
        write_json(
            trace_dir / "command.json",
            {
                "command": cmd,
                "model": args.model,
                "allowed_tools": args.allowed_tools,
                "permission_mode": args.permission_mode,
                "trace_output_format": args.trace_output_format,
                "include_partial_messages": args.include_partial_messages,
                "include_hook_events": args.include_hook_events,
                "timeout_seconds": args.claude_timeout,
                "max_steps": args.max_steps,
                "container_name": container.name,
                "container_id": container.id,
                "claude_version": version,
                "created_at": utc_now(),
            },
        )
        write_json(
            status_path,
            {
                "instance_id": test_spec.instance_id,
                "run_id": args.run_id,
                "status": "running",
                "started_at": utc_now(),
                "trace_dir": str(trace_dir),
            },
        )
        (
            exit_code,
            claude_output,
            timed_out,
            elapsed,
            stop_reason,
            step_count,
        ) = exec_capture(
            client,
            container.id,
            cmd,
            environment=claude_environment(args),
            workdir=CONTAINER_WORKDIR,
            user=args.container_user,
            timeout=args.claude_timeout,
            trace_stream_path=trace_dir / "claude.stream.jsonl",
            trace_operations_path=trace_dir / "claude.operations.jsonl",
            max_steps=args.max_steps,
        )
        logger.info(
            "Claude Code finished: exit_code=%s timed_out=%s elapsed=%.2fs",
            exit_code,
            timed_out,
            elapsed,
        )
        write_json(
            status_path,
            {
                "instance_id": test_spec.instance_id,
                "run_id": args.run_id,
                "status": "claude_finished",
                "exit_code": exit_code,
                "timed_out": timed_out,
                "stop_reason": stop_reason,
                "step_count": step_count,
                "max_steps": args.max_steps,
                "elapsed_seconds": elapsed,
                "finished_at": utc_now(),
                "trace_dir": str(trace_dir),
            },
        )
        logger.info("Claude Code output:\n%s", claude_output)
        patch, patch_error = collect_patch(container, args, trace_dir, logger)
        patch_bytes = len(patch.encode("utf-8"))
        if patch_error is not None:
            raise RuntimeError(patch_error)
        if stop_reason == "max_steps_exceeded":
            raise RuntimeError(
                f"Claude Code exceeded max_steps={args.max_steps} "
                f"(observed {step_count} task_started events)"
            )
        if timed_out:
            raise TimeoutError(
                f"Claude Code timed out after {args.claude_timeout} seconds"
            )
        tolerated_empty_final_response = False
        if exit_code != 0:
            tolerated_empty_final_response = (
                bool(patch.strip()) and is_empty_final_response_failure(trace_dir)
            )
            if not tolerated_empty_final_response:
                raise RuntimeError(f"Claude Code failed with exit code {exit_code}")
            logger.warning(
                "Tolerating Claude Code exit_code=%s because a non-empty patch "
                "was collected and the only terminal failure was an empty final response",
                exit_code,
            )
        write_json(
            status_path,
            {
                "instance_id": test_spec.instance_id,
                "run_id": args.run_id,
                "status": "completed",
                "exit_code": exit_code,
                "timed_out": timed_out,
                "stop_reason": stop_reason,
                "step_count": step_count,
                "max_steps": args.max_steps,
                "elapsed_seconds": elapsed,
                "patch_bytes": patch_bytes,
                "tolerated_empty_final_response": tolerated_empty_final_response,
                "finished_at": utc_now(),
                "trace_dir": str(trace_dir),
            },
        )
        if args.skip_empty_patches and not patch.strip():
            logger.info("Skipping empty patch for %s", test_spec.instance_id)
            return None
        return prediction_from_patch(test_spec.instance_id, patch)
    except Exception as exc:
        failed_status: dict[str, Any] = {}
        if status_path.exists():
            try:
                failed_status = json.loads(status_path.read_text())
            except json.JSONDecodeError:
                failed_status = {}
        if container is not None:
            patch_path = trace_dir / "patch.diff"
            if not patch_path.exists():
                patch, patch_error = collect_patch(container, args, trace_dir, logger)
            else:
                patch = patch_path.read_text(encoding="utf-8", errors="replace")
                patch_error = None
            failed_status["patch_bytes"] = len(patch.encode("utf-8"))
            if patch_error is not None:
                failed_status["patch_collection_error"] = patch_error
        failed_status.update(
            {
                "instance_id": test_spec.instance_id,
                "run_id": args.run_id,
                "status": "failed",
                "error": repr(exc),
                "finished_at": utc_now(),
                "trace_dir": str(trace_dir),
            }
        )
        write_json(
            status_path,
            failed_status,
        )
        raise
    finally:
        if container is not None and not args.keep_containers:
            cleanup_container(client, container, logger)
        close_logger(logger)


def select_instances(
    dataset: list[dict[str, Any]],
    args: argparse.Namespace,
    existing: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if args.instance_ids:
        wanted = set(args.instance_ids)
        dataset = [instance for instance in dataset if instance[KEY_INSTANCE_ID] in wanted]
        found = {instance[KEY_INSTANCE_ID] for instance in dataset}
        missing = sorted(wanted - found)
        if missing:
            raise ValueError(f"Instance IDs not found in dataset: {' '.join(missing)}")
    if not args.overwrite:
        dataset = [
            instance
            for instance in dataset
            if instance[KEY_INSTANCE_ID] not in existing
        ]
    if args.limit is not None:
        dataset = dataset[: args.limit]
    return dataset


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    existing = load_existing_predictions(output)
    dataset = load_swebench_dataset(args.dataset_name, args.split, args.instance_ids)
    selected = select_instances(dataset, args, existing)

    print(f"Loaded {len(dataset)} instances; selected {len(selected)} to solve.")
    if not selected:
        write_predictions(output, existing)
        print(f"No instances to solve. Predictions at {output}")
        return

    predictions = dict(existing)
    errors: list[tuple[str, str]] = []

    def run_one(index: int, instance: dict[str, Any]) -> tuple[int, str, dict[str, Any] | None]:
        instance_id = instance[KEY_INSTANCE_ID]
        client = docker.from_env()
        try:
            print(f"[{index}/{len(selected)}] Solving {instance_id}", flush=True)
            prediction = solve_instance(client, instance, args)
            return index, instance_id, prediction
        finally:
            client.close()

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {}
        for index, instance in enumerate(selected, start=1):
            future = executor.submit(run_one, index, instance)
            futures[future] = instance[KEY_INSTANCE_ID]
        for future in as_completed(futures):
            future_instance_id = futures[future]
            try:
                index, instance_id, prediction = future.result()
            except Exception as exc:
                instance_id = future_instance_id
                errors.append((instance_id, repr(exc)))
                print(f"[error] {instance_id}: {exc!r}", flush=True)
                patch_path = make_trace_dir(args, instance_id) / "patch.diff"
                if patch_path.exists():
                    patch = patch_path.read_text(encoding="utf-8", errors="replace")
                    if patch.strip() or not args.skip_empty_patches:
                        predictions[instance_id] = prediction_from_patch(instance_id, patch)
                        write_predictions(output, predictions)
                        print(
                            f"[{instance_id}] Recovered patch from failed run "
                            f"and wrote prediction ({len(patch)} chars)",
                            flush=True,
                        )
                continue
            if prediction is not None:
                predictions[instance_id] = prediction
                write_predictions(output, predictions)
                print(
                    f"[{index}/{len(selected)}] Wrote patch for {instance_id} "
                    f"({len(prediction['model_patch'])} chars)",
                    flush=True,
                )
            else:
                print(
                    f"[{index}/{len(selected)}] No prediction written for {instance_id}",
                    flush=True,
                )

    write_predictions(output, predictions)
    print(f"Done. Predictions written to {output}")
    if errors:
        formatted = "\n".join(f"- {instance_id}: {error}" for instance_id, error in errors)
        raise RuntimeError(f"{len(errors)} instance(s) failed:\n{formatted}")


if __name__ == "__main__":
    main()
