# ==============================================================================
# 消融实验评估脚本 - 支持大规模测试集的采样评估
# 从1000条测试数据中随机采样进行GPT-4评估，并汇总结果
# ==============================================================================
import json, os, glob, random
from openai import OpenAI
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

EVAL_SAMPLE_SIZE = 50  # 从1000条中采样50条进行GPT-4评估
SEED = 42


def load_api_config():
    try:
        with open('openai_key.txt', 'r', encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
            return lines[0].strip(), lines[1].strip()
    except FileNotFoundError:
        print("Error: openai_key.txt not found!")
        return None, None


def evaluate_response(client, question, baseline_answer, model_answer):
    prompt = f"""We would like to request your feedback on the two AI assistants in response to the user question displayed above.

Please evaluate the helpfulness, relevance, accuracy, level of details of their responses. You should tell me whether Assistant 1 is 'better than', 'worse than', or 'equal to' Assistant 2.

Please first compare their responses and analyze which one is more in line with the given requirements.

In the last line, please output a single line containing only a single label selecting from 'Assistant 1 is better than Assistant 2', 'Assistant 1 is worse than Assistant 2', and 'Assistant 1 is equal to Assistant 2'.

Question: {question}

Assistant 1: {baseline_answer}

Assistant 2: {model_answer}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4", temperature=0.3, max_tokens=3000,
            messages=[
                {"role": "system", "content": "You are an expert evaluator who provides objective, unbiased comparisons between AI assistant responses."},
                {"role": "user", "content": prompt}
            ])
        result_text = response.choices[0].message.content.strip()
        last_line = result_text.split('\n')[-1].strip()
        if 'Assistant 1 is better than Assistant 2' in last_line:
            comparison = 'baseline_better'
        elif 'Assistant 1 is worse than Assistant 2' in last_line:
            comparison = 'model_better'
        elif 'Assistant 1 is equal to Assistant 2' in last_line:
            comparison = 'equal'
        else:
            comparison = None
        return {'comparison': comparison, 'explanation': result_text, 'raw_judgment': last_line}
    except Exception as e:
        print(f"Error: {e}")
        return {'comparison': None, 'explanation': str(e), 'raw_judgment': None}


def evaluate_single_file(filepath, client, sample_size=EVAL_SAMPLE_SIZE):
    """评估单个saved_data.json文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 过滤有效数据（至少3个元素：question, baseline, model_answer）
    valid_data = [d for d in data if isinstance(d, list) and len(d) >= 3]
    print(f"\n处理: {filepath}")
    print(f"总条目: {len(data)}, 有效条目: {len(valid_data)}")

    # 采样
    random.seed(SEED)
    if len(valid_data) > sample_size:
        sampled = random.sample(valid_data, sample_size)
        print(f"采样 {sample_size} 条进行GPT-4评估")
    else:
        sampled = valid_data
        print(f"数据量不足{sample_size}条，使用全部 {len(sampled)} 条")

    evaluations = []
    counts = {'baseline_better': 0, 'model_better': 0, 'equal': 0, 'failed': 0}

    for idx, entry in enumerate(sampled):
        print(f"  评估 {idx+1}/{len(sampled)}...", end=' ')
        result = evaluate_response(client, entry[0], entry[1], entry[2])
        evaluations.append({
            'index': idx, 'question': entry[0][:100],
            'comparison': result['comparison'], 'explanation': result['explanation']
        })
        if result['comparison'] in counts:
            counts[result['comparison']] += 1
            print(result['comparison'])
        else:
            counts['failed'] += 1
            print("failed")

    results = {
        'file': os.path.basename(filepath),
        'total_entries': len(data), 'sampled_entries': len(sampled),
        **counts, 'evaluations': evaluations
    }
    total_valid = counts['baseline_better'] + counts['model_better'] + counts['equal']
    if total_valid > 0:
        results['model_win_rate'] = round(counts['model_better'] / total_valid * 100, 1)
        results['baseline_win_rate'] = round(counts['baseline_better'] / total_valid * 100, 1)
    print(f"  结果: baseline={counts['baseline_better']}, model={counts['model_better']}, equal={counts['equal']}")
    return results


def visualize_ablation(all_results, output_dir='./evaluate/ablation_plots'):
    """为消融实验创建对比可视化"""
    os.makedirs(output_dir, exist_ok=True)

    names = [r['file'].replace('saved_data.json', '').replace('.json', '').strip('_') or r['file'] for r in all_results]
    bb = [r['baseline_better'] for r in all_results]
    mb = [r['model_better'] for r in all_results]
    eq = [r['equal'] for r in all_results]
    win_rates = [r.get('model_win_rate', 0) for r in all_results]

    # 1. 分组柱状图
    fig, ax = plt.subplots(figsize=(14, 7))
    x = range(len(names))
    w = 0.25
    ax.bar([i-w for i in x], bb, w, label='Baseline Better', color='#FF6B6B', edgecolor='black')
    ax.bar(x, mb, w, label='Model Better', color='#4ECDC4', edgecolor='black')
    ax.bar([i+w for i in x], eq, w, label='Equal', color='#95E1D3', edgecolor='black')
    ax.set_xlabel('实验配置', fontsize=12)
    ax.set_ylabel('数量', fontsize=12)
    ax.set_title('消融实验对比 - GPT-4评估结果', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ablation_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 胜率对比图
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(names, win_rates, color='#4ECDC4', edgecolor='black')
    for bar, rate in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{rate}%', ha='center', va='bottom', fontweight='bold')
    ax.set_xlabel('实验配置', fontsize=12)
    ax.set_ylabel('模型胜率 (%)', fontsize=12)
    ax.set_title('消融实验 - 模型胜率对比', fontsize=14, fontweight='bold')
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50%基准线')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ablation_win_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"可视化已保存至: {output_dir}")


def main():
    base_url, api_key = load_api_config()
    if not api_key:
        print("无法获取API key"); return
    client = OpenAI(api_key=api_key, base_url=base_url)
    print("OpenAI client 初始化成功")

    # 查找所有消融实验的saved_data.json
    result_dirs = [
        'train_result/lora_7k_qv',
        'train_result/lora_7k_qkv',
        'train_result/lora_7k_qkvo',
        'train_result/full_finetune_7k',
    ]

    all_results = []
    for d in result_dirs:
        saved_path = os.path.join(d, 'saved_data.json')
        if os.path.exists(saved_path):
            result = evaluate_single_file(saved_path, client)
            if result:
                result['experiment'] = os.path.basename(d)
                all_results.append(result)
        else:
            print(f"跳过: {saved_path} (不存在)")

    if all_results:
        # 保存评估结果
        output_file = 'evaluate/ablation_evaluation_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n评估结果已保存至: {output_file}")

        # 可视化
        visualize_ablation(all_results)

        # 打印汇总
        print("\n" + "=" * 70)
        print("消融实验评估汇总")
        print("=" * 70)
        print(f"{'实验':25s} {'Baseline↑':>10s} {'Model↑':>10s} {'Equal':>8s} {'胜率':>8s}")
        print("-" * 70)
        for r in all_results:
            name = r.get('experiment', r['file'])
            print(f"{name:25s} {r['baseline_better']:>10d} {r['model_better']:>10d} "
                  f"{r['equal']:>8d} {r.get('model_win_rate', 0):>7.1f}%")

    print("\n评估完成!")


if __name__ == "__main__":
    main()
