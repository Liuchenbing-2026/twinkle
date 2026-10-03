# Qwen3.6-35B-A3B text decision training on Ascend

Development recipe for full language-parameter training with the visual encoder frozen. Native chat-template formatting disables thinking. Supervision uses masked causal cross-entropy at decision positions, with labels shifted once for Twinkle's external loss. This is not a LoRA recipe.

Status: small random same-architecture four-NPU training, validation, checkpoint save and resume have passed. CPU-offload norm/clipping/gathering and exact optimizer-state restoration also passed. Full-size model validation is in progress; this branch does not claim completed quality or performance acceptance. Numerical results are stored separately.

Use an isolated container with this directory mounted as `/workspace` and the model mounted as `/models/Qwen3.6-35B-A3B`. The tested stack is PyTorch/torch_npu 2.10, Transformers 5.15.1, Accelerate 1.14.0, CANN 9.1, PEFT 0.19.0 and NumPy 1.26.4. Explicitly allocate four NPUs and provision enough CPU RAM and disk for full parameters, gradients, optimizer state and checkpoint retention.

Run `download_data.py`, `prepare_data.py` and `prepare_inputs.py` to prepare the public case-separated dataset. Training never consumes validation/calibration/test labels. No weights, datasets or private environment information are included.

```bash
bash run.sh --output /workspace/outputs/train --steps 600 --schedule-steps 600 \
  --validation /workspace/decision_data/validation.jsonl
```

FSDP2 uses CPU offload and BF16 computation. Explicit `cpu:gloo,npu:hccl` process-group initialization enables CPU DTensor reductions on the same mesh. Full weights are retained only on rank zero during export. Distributed optimizer checkpoints preserve moments and steps; parameters are resharded before save/load to keep optimizer identities consistent.

`make_tiny_model.py` creates a small random integration model. `check_cpu_offload.py` tests four-rank norms, clipping and CPU gathering against a dense reference. `python -m twinkle_adapter.check_contract` validates encoding and causal-loss semantics. Run `torchrun --nproc_per_node=4 -m twinkle_adapter.check_checkpoint_io` to verify exact CPU-offloaded optimizer-state recovery. Small-model tests do not establish full-model accuracy or memory requirements.
