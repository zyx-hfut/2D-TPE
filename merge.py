from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base_model_path = "/data/zyx/model/MiniCPM-2B-sft-bf16-llama-format"
lora_path = "/data/zyx/2026/2D-TPE/output/rel_extraction_2d"
output_path = "/data/zyx/2026/2D-TPE/output/rel_extraction_2d_merged"

# 加载基座模型
model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16)

# 加载 LoRA 权重并合并
model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload()

# 保存完整模型
model.save_pretrained(output_path)

# 同时保存 tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(output_path)