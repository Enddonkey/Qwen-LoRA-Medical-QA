# ==============================================================================
# 生成测试数据集 - 从Huatuo26M-Lite中随机抽取200条作为测试集
# 输出格式: [[question, reference_answer], ...]
# ==============================================================================
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

import json, random
from datasets import load_dataset

TEST_SIZE = 500  # 测试集大小（500条，fp16推理约1.5小时）
OFFSET = 100000  # 从数据集后部取样，避免与训练集重叠（训练集取前7000条）
OUTPUT_FILE = "zh_med_500.json"

def main():
    print(f"正在加载数据集...")
    dataset = load_dataset("FreedomIntelligence/Huatuo26M-Lite")
    data = dataset['train']
    
    # 从训练集范围之外取样（训练集用前7000条）
    total = len(data)
    print(f"数据集总量: {total}")
    
    # 从offset开始随机抽取
    indices = list(range(OFFSET, min(OFFSET + 50000, total)))
    random.seed(42)
    selected = random.sample(indices, TEST_SIZE)
    
    test_data = []
    for idx in selected:
        item = data[idx]
        test_data.append([item['question'], item['answer']])
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)
    
    print(f"测试集已保存至 {OUTPUT_FILE}，共 {len(test_data)} 条")
    print(f"样本示例:")
    for i in range(min(3, len(test_data))):
        print(f"  Q: {test_data[i][0][:80]}...")
        print(f"  A: {test_data[i][1][:80]}...")
        print()

if __name__ == "__main__":
    main()
