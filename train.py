import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# 模型和Tokenizer
base_model_name = "../deepseek8b"
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # 避免 hf 的 warning

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# 设置LoRA配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# 应用LoRA
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# 加载数据集
raw_dataset = load_dataset("json", data_files={"train": "datasets/human_decisions_20250409-120157.json"})["train"]

# 构建 prompt，并进行 tokenizer 编码
delimiter = "###"

def preprocess(example):
    prompt = f"{example['instruction']}{delimiter}{example['input']}\nResponse to user:{delimiter} {example['output'].strip()}"
    tokenized = tokenizer(
        prompt,
        max_length=2048,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return {
        "input_ids": tokenized["input_ids"][0],
        "attention_mask": tokenized["attention_mask"][0]
    }

dataset = raw_dataset.map(preprocess, remove_columns=raw_dataset.column_names)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./lora-deepseek-instruct-output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    num_train_epochs=3,
    bf16=True,
    remove_unused_columns=False,
    save_total_limit=2,
    logging_dir="./logs",
    report_to="none"
)

# 创建Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field=None,  # 不再使用 text 字段
    args=training_args,
    max_seq_length=2048,  # 显式指定
)

# 开始训练
trainer.train()

# 保存LoRA adapter
trainer.model.save_pretrained("./lora-deepseek-instruct-lora")
tokenizer.save_pretrained("./lora-deepseek-instruct-lora")
