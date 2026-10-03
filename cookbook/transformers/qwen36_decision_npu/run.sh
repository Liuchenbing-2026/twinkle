#!/usr/bin/env bash
set -euo pipefail
cd /workspace
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}
export OMP_NUM_THREADS=16 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HCCL_CONNECT_TIMEOUT=300 HCCL_EXEC_TIMEOUT=1800
export PYTHONPATH=/workspace:${PYTHONPATH:-}
exec torchrun --nproc_per_node=4 --master_port=29661 -m twinkle_adapter.train "$@"
