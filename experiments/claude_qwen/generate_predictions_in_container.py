from __future__ import annotations

import argparse
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
        default=1800,
        help="Timeout in seconds for each Claude Code solve attempt.",
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
    return parser.parse_args()


def load_existing_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {prediction[KEY_INSTANCE_ID]: prediction for prediction in data}


def write_predictions(path: Path, predictions: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(predictions.values(), key=lambda item: item[KEY_INSTANCE_ID])
    path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n")


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


def exec_capture(
    client: docker.DockerClient,
    container_id: str,
    cmd: str | list[str],
    *,
    environment: dict[str, str] | None = None,
    workdir: str | None = None,
    user: str = "root",
    timeout: int | None = None,
) -> tuple[int | None, str, bool, float]:
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

    def run() -> None:
        nonlocal exception
        try:
            stream = client.api.exec_start(exec_id, stream=True)
            for chunk in stream:
                output.extend(chunk)
        except Exception as exc:
            exception = exc

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
        timed_out = True
        thread.join(5)
    if exception is not None:
        raise exception
    elapsed = time.time() - start
    inspect = client.api.exec_inspect(exec_id)
    text = output.decode("utf-8", errors="replace")
    return inspect.get("ExitCode"), text, timed_out, elapsed


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
    return f"""We need solve one SWE-bench task in this repository.

You are inside the task repository at /testbed. Modify only the source files
needed to fix the issue. Do not edit tests unless the issue explicitly requires
test-source changes. Keep the patch minimal.

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
    container = None
    try:
        logger.info("Starting solve for %s", test_spec.instance_id)
        ensure_instance_image(client, test_spec, logger, args.force_rebuild)
        container = create_solve_container(client, test_spec, args.run_id)
        container.start()
        logger.info("Solve container started: %s (%s)", container.name, container.id)

        version = install_claude(container, args)
        logger.info("Claude Code version in container: %s", version)

        cmd = [
            str(claude_container_bin(args)),
            "--bare",
            "--print",
            "--model",
            args.model,
            "--no-session-persistence",
            "--permission-mode",
            args.permission_mode,
            "--allowedTools",
            args.allowed_tools,
            "--",
            build_prompt(instance),
        ]
        if args.dangerously_skip_permissions:
            cmd.insert(-4, "--dangerously-skip-permissions")
        exit_code, claude_output, timed_out, elapsed = exec_capture(
            client,
            container.id,
            cmd,
            environment=claude_environment(args),
            workdir=CONTAINER_WORKDIR,
            user=args.container_user,
            timeout=args.claude_timeout,
        )
        logger.info(
            "Claude Code finished: exit_code=%s timed_out=%s elapsed=%.2fs",
            exit_code,
            timed_out,
            elapsed,
        )
        logger.info("Claude Code output:\n%s", claude_output)
        if timed_out:
            raise TimeoutError(
                f"Claude Code timed out after {args.claude_timeout} seconds"
            )
        if exit_code != 0:
            raise RuntimeError(f"Claude Code failed with exit code {exit_code}")

        diff_result = container.exec_run(
            "git -C /testbed -c core.fileMode=false diff", user="root"
        )
        patch = diff_result.output.decode("utf-8", errors="replace")
        logger.info("Collected patch bytes: %d", len(patch.encode("utf-8")))
        if diff_result.exit_code != 0:
            raise RuntimeError(f"git diff failed: {patch}")
        if args.skip_empty_patches and not patch.strip():
            logger.info("Skipping empty patch for %s", test_spec.instance_id)
            return None
        return {
            KEY_INSTANCE_ID: test_spec.instance_id,
            "model_patch": patch,
            "model_name_or_path": MODEL_NAME,
        }
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

    client = docker.from_env()
    predictions = dict(existing)
    for index, instance in enumerate(selected, start=1):
        instance_id = instance[KEY_INSTANCE_ID]
        print(f"[{index}/{len(selected)}] Solving {instance_id}")
        prediction = solve_instance(client, instance, args)
        if prediction is not None:
            predictions[instance_id] = prediction
            write_predictions(output, predictions)
            print(
                f"[{index}/{len(selected)}] Wrote patch for {instance_id} "
                f"({len(prediction['model_patch'])} chars)"
            )
        else:
            print(f"[{index}/{len(selected)}] No prediction written for {instance_id}")

    write_predictions(output, predictions)
    print(f"Done. Predictions written to {output}")


if __name__ == "__main__":
    main()
