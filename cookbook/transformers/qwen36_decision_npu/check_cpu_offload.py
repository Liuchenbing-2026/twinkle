"""Four-rank regression: CPU DTensor norms, clipping, and full-weight gather."""
import json
import os
from pathlib import Path
import torch
import torch_npu  # noqa: F401
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard
from twinkle.utils.grad_clip import normalize_and_clip_grad_norm

torch.npu.set_device(int(os.environ['LOCAL_RANK']))
from twinkle.utils.platforms import ensure_hccl_socket_env
ensure_hccl_socket_env(int(os.environ['MASTER_PORT']))
dist.init_process_group('cpu:gloo,npu:hccl')
mesh = init_device_mesh('npu', (dist.get_world_size(),))
original = torch.arange(1, 33, dtype=torch.float32).reshape(8, 4)
parameter = torch.nn.Parameter(distribute_tensor(original.to('npu'), mesh, [Shard(0)]).cpu())
results = []
for norm_type in (2.0, float('inf')):
    gradient = original.cos()
    parameter.grad = distribute_tensor(gradient.to('npu'), mesh, [Shard(0)]).cpu()
    reference = torch.nn.Parameter(original.clone())
    reference.grad = gradient / 7
    expected = torch.nn.utils.clip_grad_norm_([reference], 0.1, norm_type=norm_type)
    actual = normalize_and_clip_grad_norm([parameter], num_tokens=7,
                                         max_grad_norm=0.1, norm_type=norm_type)
    torch.testing.assert_close(torch.tensor(actual), expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(parameter.grad.full_tensor(), reference.grad, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(parameter.full_tensor(), original, rtol=0, atol=0)
    results.append({'norm_type': str(norm_type), 'norm': actual})
if dist.get_rank() == 0:
    report = {'status': 'passed', 'ranks': dist.get_world_size(),
              'norms_and_clipped_gradients_match_dense_reference': True,
              'cpu_parameter_gather_exact': True, 'checks': results}
    Path('/workspace/results/cpu-offload-regression.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
dist.destroy_process_group()
