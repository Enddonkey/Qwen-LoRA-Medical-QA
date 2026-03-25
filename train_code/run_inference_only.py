# ==============================================================================
# 补跑推理脚本 - 对已训练好的模型生成推理结果
# 用于训练完成但推理中断的情况
# ==============================================================================
import sys, os, torch, json, gc
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_utils import *
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "C:/Users/Administrator/.cache/huggingface/hub/Qwen3-4B-Instruct-2507"
TEST_FILE = "zh_med_500.json"

# 需要补跑推理的实验列表
EXPERIMENTS = [
    {'name': 'lora_7k_qv', 'lora_path': './train_result/lora_7k_qv/lora'},
    {'name': 'lora_7k_qkv', 'lora_path': './train_result/lora_7k_qkv/lora'},
    {'name': 'lora_7k_qkvo', 'lora_path': './train_result/lora_7k_qkvo/lora'},
]


def run_inference(exp):
    name = exp['name']
    lora_path = exp['lora_path']
    saved_data_path = f"./train_result/{name}/saved_data.json"

    # 跳过已完成的
    if os.path.exists(saved_data_path):
        with open(saved_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if len(data) > 0 and len(data[0]) >= 3:
            print(f"[跳过] {name} 推理结果已存在 ({len(data)}条)")
            return

    if not os.path.exists(lora_path):
        print(f"[跳过] {name} LoRA模型不存在: {lora_path}")
        return

    print(f"\n{'='*60}")
    print(f"开始推理: {name}")
    print(f"{'='*60}")

    # 加载测试数据
    assert os.path.exists(TEST_FILE), f"测试文件不存在: {TEST_FILE}"
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # 加载模型（fp16，速度更快）
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    model = load_inference_model(MODEL_ID, lora_path, tokenizer)

    # 逐条推理（max_new_tokens=512）
    answers = generate_answers([d[0] for d in test_data], tokenizer, model)
    for d, a in zip(test_data, answers):
        if len(d) >= 3:
            d[2] = a
        else:
            d.append(a)

    os.makedirs(os.path.dirname(saved_data_path), exist_ok=True)
    with open(saved_data_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)
    print(f"推理结果已保存至 {saved_data_path} ({len(test_data)}条)")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    for exp in EXPERIMENTS:
        run_inference(exp)
    print("\n所有推理任务完成!")
