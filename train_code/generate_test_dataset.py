# ==============================================================================
# 从 Huatuo26M-Lite 中抽取1000条未见过的测试数据
# 跳过前7000条（训练数据），从后面随机抽取1000条
# 输出格式与 zh_med.json 一致: [[question, reference_answer], ...]
# ==============================================================================
import os
# 设置代理（解决HuggingFace连接问题）
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

import json
import random
from datasets import load_dataset

TRAIN_SAMPLE_SIZE = 7000  # 训练使用的前7000条
TEST_SAMPLE_SIZE = 1000   # 测试集大小
OUTPUT_PATH = "./zh_med_1000.json"
SEED = 42


def main():
    print("正在加载 Huatuo26M-Lite 数据集...")
    dataset = load_dataset("FreedomIntelligence/Huatuo26M-Lite")
    full_data = dataset['train']
    total_size = len(full_data)
    print(f"数据集总大小: {total_size}")

    # 确保有足够的数据
    assert total_size > TRAIN_SAMPLE_SIZE + TEST_SAMPLE_SIZE, \
        f"数据集太小! 总共{total_size}条，需要{TRAIN_SAMPLE_SIZE + TEST_SAMPLE_SIZE}条"

    # 从训练数据之后的部分随机抽取
    available_indices = list(range(TRAIN_SAMPLE_SIZE, total_size))
    random.seed(SEED)
    selected_indices = random.sample(available_indices, TEST_SAMPLE_SIZE)
    selected_indices.sort()

    print(f"从索引 {TRAIN_SAMPLE_SIZE} 开始，随机抽取 {TEST_SAMPLE_SIZE} 条")
    print(f"索引范围: {min(selected_indices)} ~ {max(selected_indices)}")

    # 构建测试数据
    test_data = []
    for idx in selected_indices:
        item = full_data[idx]
        question = item['question']
        answer = item['answer']
        test_data.append([question, answer])

    # 数据质量检查
    print("\n数据质量检查:")
    empty_q = sum(1 for d in test_data if not d[0] or len(d[0].strip()) == 0)
    empty_a = sum(1 for d in test_data if not d[1] or len(d[1].strip()) == 0)
    avg_q_len = sum(len(d[0]) for d in test_data) / len(test_data)
    avg_a_len = sum(len(d[1]) for d in test_data) / len(test_data)
    print(f"  空问题数: {empty_q}")
    print(f"  空答案数: {empty_a}")
    print(f"  平均问题长度: {avg_q_len:.1f} 字符")
    print(f"  平均答案长度: {avg_a_len:.1f} 字符")

    # 展示前3条样本
    print("\n前3条样本预览:")
    for i in range(3):
        print(f"\n  [{i}] 问题: {test_data[i][0][:80]}...")
        print(f"      答案: {test_data[i][1][:80]}...")

    # 保存
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    print(f"\n测试数据集已保存至: {OUTPUT_PATH}")
    print(f"共 {len(test_data)} 条数据")


if __name__ == "__main__":
    main()
