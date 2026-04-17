import time

import torch
import os
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

# 设置警告过滤器
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def merge_models():
    print("开始合并模型...")
    
    # 获取当前脚本所在目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 设置模型路径（使用绝对路径）
    base_model_path = "../../modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct"
    lora_path = os.path.join(current_dir, "lora-deepseek-instruct-lora/20251017-182000")
    current_time = time.strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(current_dir, f"merged_models/{current_time}")

    # 验证路径是否存在
    if not os.path.exists(base_model_path):
        raise ValueError(f"基础模型路径不存在: {base_model_path}")
    if not os.path.exists(lora_path):
        raise ValueError(f"LoRA适配器路径不存在: {lora_path}")

    try:
        print("1. 加载基础模型...")
        print(f"基础模型路径: {base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("2. 加载LoRA适配器...")
        print(f"LoRA适配器路径: {lora_path}")
        merged_model = PeftModel.from_pretrained(base_model, lora_path)
        
        print("3. 合并LoRA权重...")
        merged_model = merged_model.merge_and_unload()
        
        print("4. 保存合并后的模型...")
        print(f"输出路径: {output_path}")
        merged_model.save_pretrained(output_path)
        
        print("5. 保存tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        tokenizer.save_pretrained(output_path)
        
        print(f"模型合并完成！已保存到: {output_path}")
        
    except Exception as e:
        print(f"合并过程中出现错误: {str(e)}")
        raise
    
    finally:
        # 清理内存
        if 'base_model' in locals():
            del base_model
        if 'merged_model' in locals():
            del merged_model
        if 'tokenizer' in locals():
            del tokenizer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    merge_models()