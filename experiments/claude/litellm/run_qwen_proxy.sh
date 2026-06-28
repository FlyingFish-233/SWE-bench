#!/usr/bin/env bash
cd /raid/zwx/SWE-bench
. .venv/bin/activate
exec litellm --config experiments/claude/litellm/qwen_proxy.yaml --host 127.0.0.1 --port 30011
