#!/usr/bin/env bash
set -euo pipefail

cd /raid/zwx/SWE-bench
. .venv/bin/activate

ts=$(date +%Y%m%d-%H%M%S)
dataset_name="${DATASET_NAME:-/raid/zwx/datasets/SWE-bench_Verified}"
split="${SPLIT:-test}"
trace_dir="${TRACE_DIR:-/raid/zwx/SWE-bench/experiments/claude/traces}"
run_id="${RUN_ID:-claude-verified-$ts}"
predictions_path="${PREDICTIONS_PATH:-$trace_dir/$run_id/predictions.json}"
eval_run_id="${EVAL_RUN_ID:-$run_id-eval}"
report_dir="${REPORT_DIR:-experiments/claude/evaluation_reports}"
anthropic_base_url="${ANTHROPIC_BASE_URL:-http://172.17.0.1:30011}"
container_network="${CONTAINER_NETWORK:-bridge}"
max_steps="${MAX_STEPS:-100}"
max_workers="${MAX_WORKERS:-4}"
eval_max_workers="${EVAL_MAX_WORKERS:-$max_workers}"
eval_timeout="${EVAL_TIMEOUT:-1800}"
limit="${LIMIT:-}"
overwrite="${OVERWRITE:-0}"
skip_empty_patches="${SKIP_EMPTY_PATCHES:-0}"

instance_ids=("$@")

prediction_args=(
  --dataset_name "$dataset_name"
  --split "$split"
  --output "$predictions_path"
  --run_id "$run_id"
  --trace_dir "$trace_dir"
  --anthropic_base_url "$anthropic_base_url"
  --container_network "$container_network"
  --max_workers "$max_workers"
)

evaluation_args=(
  --dataset_name "$dataset_name"
  --split "$split"
  --predictions_path "$predictions_path"
  --run_id "$eval_run_id"
  --max_workers "$eval_max_workers"
  --timeout "$eval_timeout"
  --cache_level env
  --clean False
  --report_dir "$report_dir"
)

if ((${#instance_ids[@]})); then
  prediction_args+=(--instance_ids "${instance_ids[@]}")
  evaluation_args+=(--instance_ids "${instance_ids[@]}")
fi

if [[ -n "$max_steps" ]]; then
  prediction_args+=(--max_steps "$max_steps")
fi

if [[ -n "$limit" ]]; then
  prediction_args+=(--limit "$limit")
fi

if [[ "$overwrite" == "1" || "$overwrite" == "true" ]]; then
  prediction_args+=(--overwrite)
fi

if [[ "$skip_empty_patches" == "1" || "$skip_empty_patches" == "true" ]]; then
  prediction_args+=(--skip_empty_patches)
fi

echo "Dataset: $dataset_name"
echo "Split: $split"
echo "Run ID: $run_id"
echo "Predictions: $predictions_path"
echo "Trace dir: $trace_dir/$run_id"
echo "Anthropic base URL: $anthropic_base_url"
echo "Container network: $container_network"
if ((${#instance_ids[@]})); then
  echo "Instances: ${instance_ids[*]}"
else
  echo "Instances: full dataset"
fi

python experiments/claude/generate_predictions_in_container.py "${prediction_args[@]}"

python -m swebench.harness.run_evaluation "${evaluation_args[@]}"
