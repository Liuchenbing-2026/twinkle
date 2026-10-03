"""Tiny random Qwen3.5-MoE for integration tests, never an accuracy result."""
import json
from pathlib import Path
import shutil
import torch
from transformers import Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration

source = Path('/models/Qwen3.6-35B-A3B')
output = Path('/workspace/tiny-model')
config = json.loads((source / 'config.json').read_text())
text = config['text_config']
text.update(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, linear_key_head_dim=16,
            linear_value_head_dim=16, linear_num_key_heads=4, linear_num_value_heads=4,
            moe_intermediate_size=32, shared_expert_intermediate_size=32,
            num_experts=8, num_experts_per_tok=2,
            layer_types=['linear_attention', 'full_attention'])
vision = config['vision_config']
vision.update(hidden_size=32, intermediate_size=64, num_heads=4, depth=1, out_hidden_size=64)
torch.manual_seed(42)
model = Qwen3_5MoeForConditionalGeneration(Qwen3_5MoeConfig.from_dict(config))
output.mkdir(exist_ok=True)
model.save_pretrained(output)
# Tokenizer/processor content is official; small assets may finish before weights.
required = ['tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja']
assert all((source / file).is_file() for file in required), 'Tokenizer download not finished'
for file in source.iterdir():
    if file.suffix in {'.json', '.jinja', '.txt'} and file.name not in {
            'config.json', 'configuration.json', 'generation_config.json', 'model.safetensors.index.json'}:
        shutil.copy2(file, output / file.name)
print(json.dumps({'status': 'created', 'parameters': sum(p.numel() for p in model.parameters()),
                  'purpose': 'synthetic integration only', 'output': str(output)}))
