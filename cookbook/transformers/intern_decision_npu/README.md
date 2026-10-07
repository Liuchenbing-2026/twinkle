> 联合字段训练复现使用 [JOINT_REPRODUCTION.md](JOINT_REPRODUCTION.md) 和 `run_joint.sh`。下文保留历史单字段实验说明，不能混用其训练预算或评测口径。

# Qwen3.5-4B text decision training on Ascend

This example updates all language parameters while freezing vision and its projector. It uses hard-label, full-vocabulary CE at decision positions; this is not LoRA and not a reconstruction of an unpublished training mixture. Use one question per example and preserve option order. Gold labels and teacher probabilities never enter the prompt.

## Environment and layout

The tested stack is CANN 9.1.0, torch 2.10.0, torch_npu 2.10.0.post2, Transformers 5.15.1 and Accelerate 1.14.0 on four Ascend NPUs. Twinkle uses PEFT 0.19.0 and NumPy 1.26.4 in its separate environment. Install this repository in the chosen environment and mount this example directory as `/workspace`; mount the original Qwen3.5-4B checkpoint at `/models/Qwen3.5-4B`. Data and checkpoint locations are local mounts, not bundled artifacts. The example is text-only and does not validate image training or Qwen3.6-35B-A3B.

```bash
cd /workspace
python3 download_data.py
python3 prepare_data.py
python3 prepare_inputs.py
mkdir -p results
python3 -m twinkle_adapter.check_contract
bash run.sh --model /models/Qwen3.5-4B --output /workspace/outputs/smoke --steps 3 --save-every 3
bash run.sh --resume /workspace/outputs/smoke/checkpoint-3 --output /workspace/outputs/resume --steps 4 --save-every 4
bash run.sh --model /models/Qwen3.5-4B --output /workspace/outputs/train \
  --steps 600 --save-every 150 --validation /workspace/decision_data/validation.jsonl
```

Training uses four workers, global batch 16, learning rate 2e-6 and a fixed 600-step cosine schedule with 18 warmup steps. Checkpoint selection uses only validation CE. CPU checks, NPU short training, full checkpoint recovery and final independent evaluation are distinct validation stages. Keep accuracy and performance artifacts outside this repository.

Data preparation is deterministic and splits by original case; related questions remain in one split. Preserve published hard labels when rounded teacher probabilities tie. The test split is not used for training or checkpoint selection. The tokenizer encodes each answer symbol as one token; the supported native answer alphabet has 62 symbols.

## Checkpoint handling

The example uses task-local strategy hooks: HF model export normalizes activation-checkpoint wrapper names; optimizer state uses PyTorch DCP shards. Before export, all FSDP modules are resharded to restore optimizer parameter identities after no-grad evaluation. A Gloo CPU group fences checkpoint I/O. Full checkpoints include the scheduler, per-rank RNG and consumed sample count.

The four-rank regression `torchrun --nproc_per_node=4 --master_port=29642 -m twinkle_adapter.check_checkpoint_io` covers update, no-grad evaluation, resharding and exact optimizer-state reload. It uses `/workspace/results`, which must already exist.

This branch lowers the NumPy metadata bound to 1.26.4 for the tested triton-ascend 3.2.2 combination. This does not establish compatibility for every Twinkle feature.
