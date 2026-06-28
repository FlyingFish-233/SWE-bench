#!/usr/bin/env bash
set -euo pipefail

cd /raid/zwx/SWE-bench
. .venv/bin/activate

ts=$(date +%Y%m%d-%H%M%S)
run_id="claude-$ts"
predictions_path="experiments/claude/predictions.$ts.json"
eval_run_id="${run_id}-eval"
report_dir="experiments/claude/evaluation_reports"
instance_ids=(
  astropy__astropy-12907
  astropy__astropy-14182
  astropy__astropy-14365
  astropy__astropy-14995
)

python experiments/claude/generate_predictions_in_container.py \
  --dataset_name /raid/zwx/datasets/SWE-bench_Lite \
  --split test \
  --instance_ids "${instance_ids[@]}" \
  --output "$predictions_path" \
  --run_id "$run_id" \
  --trace_dir "/raid/zwx/SWE-bench/experiments/claude/traces" \
  --anthropic_base_url "http://127.0.0.1:30011" \
  --max_steps 100 \
  --max_workers 2 \
  --overwrite

python -m swebench.harness.run_evaluation \
  --dataset_name /raid/zwx/datasets/SWE-bench_Lite \
  --split test \
  --instance_ids "${instance_ids[@]}" \
  --predictions_path "$predictions_path" \
  --run_id "$eval_run_id" \
  --max_workers 2 \
  --timeout 1800 \
  --cache_level env \
  --clean False \
  --report_dir "$report_dir"

