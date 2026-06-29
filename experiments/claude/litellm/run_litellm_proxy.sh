#!/usr/bin/env bash
set -euo pipefail

cd /raid/zwx/SWE-bench
. .venv/bin/activate

HOST="${HOST:-172.17.0.1}"
PORT="${PORT:-30011}"
CONFIG="${CONFIG:-experiments/claude/litellm/litellm_config.yaml}"
QWEN_BASE_URL="${QWEN_BASE_URL:-http://127.0.0.1:30010/v1}"

if ! ip addr show docker0 | grep -q "inet ${HOST}/"; then
  echo "docker0 does not have ${HOST}; set HOST to the docker bridge gateway address." >&2
  exit 1
fi

if ! curl -fsS "${QWEN_BASE_URL%/v1}/v1/models" >/dev/null; then
  echo "Qwen vLLM is not reachable at ${QWEN_BASE_URL}; start experiments/claude/server/run_qwen_server.sh first." >&2
  exit 1
fi

exec litellm --config "$CONFIG" --host "$HOST" --port "$PORT"
