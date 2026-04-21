import torch
from transformers import AutoModelForCausalLM

base_path = "/data/zyx/model/MiniCPM-2B-sft-bf16-llama-format"
merged_path = "/data/zyx/2026/2D-TPE/output/rel_extraction_2d_merged"

base_model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16)
merged_model = AutoModelForCausalLM.from_pretrained(merged_path, torch_dtype=torch.bfloat16)

diff_count = 0
same_count = 0
for (name1, p1), (name2, p2) in zip(base_model.named_parameters(), merged_model.named_parameters()):
    if not torch.equal(p1, p2):
        diff_count += 1
        print(f"DIFF: {name1}")
    else:
        same_count += 1

print(f"\n总参数: {diff_count + same_count}, 不同: {diff_count}, 相同: {same_count}")
if diff_count == 0:
    print("WARNING: 所有参数完全一致，LoRA 未合并！")
else:
    print("OK: LoRA 已生效。")