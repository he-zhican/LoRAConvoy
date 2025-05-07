import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载基础模型和适配器
base_model_path = "../deepseek8b"
lora_path = "./lora-deepseek-instruct-lora"

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 合并LoRA
merged_model = PeftModel.from_pretrained(base_model, lora_path)
merged_model = merged_model.merge_and_unload()  # 关键合并操作

# 保存完整模型
merged_model.save_pretrained("./merged_model")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained("./merged_model")