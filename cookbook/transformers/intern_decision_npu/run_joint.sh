#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python_bin=${PYTHON_BIN:-/workspace/.venv/bin/python}
export PYTHONPATH="$PWD:/workspace/framework/src:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HCCL_CONNECT_TIMEOUT=300 HCCL_EXEC_TIMEOUT=600
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
stage=${1:-smoke}
output=${OUTPUT_ROOT:-/workspace/outputs}
case "$stage" in
  smoke) extra=(--steps 2 --save-every 2 --output "$output/smoke") ;;
  resume) extra=(--steps 3 --save-every 3 --resume "$output/smoke/checkpoint-2" --output "$output/resume") ;;
  train) extra=(--steps 120 --save-every 120 --model-only --validation /data/joint/validation.jsonl --output "$output/train") ;;
  *) exit 2 ;;
esac
"$python_bin" -m torch.distributed.run --nproc-per-node 4 \
  --master-addr 127.0.0.1 --master-port 29661 -m twinkle_adapter.train \
  --model /models/Qwen3.5-4B --data /data/joint/train.jsonl \
  --schedule-steps 120 --global-batch 16 "${extra[@]}"
