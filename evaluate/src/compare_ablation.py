# ==============================================================================
# 消融实验训练指标对比分析脚本
# 汇总所有实验的训练时间、Loss、GPU占用、参数量等指标
# ==============================================================================
import json, os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 所有消融实验的metrics文件路径
EXPERIMENTS = {
    'LoRA-QV': 'train_result/lora_7k_qv/training_metrics.json',
    'LoRA-QKV': 'train_result/lora_7k_qkv/training_metrics.json',
    'LoRA-QKVO': 'train_result/lora_7k_qkvo/training_metrics.json',
    'Full-FT': 'train_result/full_finetune_7k/training_metrics.json',
}
OUTPUT_DIR = './evaluate/ablation_plots'


def load_all_metrics():
    """加载所有实验的训练指标"""
    all_metrics = {}
    for name, path in EXPERIMENTS.items():
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                all_metrics[name] = json.load(f)
            print(f"✓ 已加载: {name} ({path})")
        else:
            print(f"✗ 未找到: {name} ({path})")
    return all_metrics


def print_summary_table(all_metrics):
    """打印汇总对比表"""
    print("\n" + "=" * 100)
    print("消融实验训练指标汇总")
    print("=" * 100)
    header = f"{'实验':15s} {'方法':25s} {'Target Modules':20s} {'可训练参数':>15s} {'训练时间(min)':>13s} {'GPU峰值(GB)':>12s} {'Eval Loss':>10s} {'PPL':>8s}"
    print(header)
    print("-" * 100)
    for name, m in all_metrics.items():
        method = m.get('finetune_method', 'N/A')[:24]
        targets = str(m.get('target_modules', 'N/A'))[:19]
        t_params = m.get('trainable_params', 0)
        t_time = m.get('training_time_min', 0)
        gpu = m.get('peak_gpu_memory_gb', 0)
        eloss = m.get('eval_loss', 0)
        ppl = m.get('perplexity', 0)
        if t_params > 1e9:
            param_str = f"{t_params/1e9:.2f}B"
        elif t_params > 1e6:
            param_str = f"{t_params/1e6:.2f}M"
        else:
            param_str = f"{t_params:,}"
        print(f"{name:15s} {method:25s} {targets:20s} {param_str:>15s} {t_time:>13.2f} {gpu:>12.2f} {eloss:>10.4f} {ppl:>8.2f}")
    print("=" * 100)


def plot_trainable_params(all_metrics):
    """可训练参数量对比"""
    names = list(all_metrics.keys())
    params = [m.get('trainable_params', 0) / 1e6 for m in all_metrics.values()]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FF6B6B'][:len(names)]
    bars = ax.bar(names, params, color=colors, edgecolor='black')
    for bar, p in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{p:.1f}M', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax.set_ylabel('可训练参数量 (M)', fontsize=12)
    ax.set_title('消融实验 - 可训练参数量对比', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'params_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_time(all_metrics):
    """训练时间对比"""
    names = list(all_metrics.keys())
    times = [m.get('training_time_min', 0) for m in all_metrics.values()]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FF6B6B'][:len(names)]
    bars = ax.bar(names, times, color=colors, edgecolor='black')
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{t:.1f}min', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax.set_ylabel('训练时间 (分钟)', fontsize=12)
    ax.set_title('消融实验 - 训练时间对比', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'time_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_gpu_memory(all_metrics):
    """GPU显存对比"""
    names = list(all_metrics.keys())
    mems = [m.get('peak_gpu_memory_gb', 0) for m in all_metrics.values()]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FF6B6B'][:len(names)]
    bars = ax.bar(names, mems, color=colors, edgecolor='black')
    for bar, mem in zip(bars, mems):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{mem:.1f}GB', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax.set_ylabel('GPU峰值显存 (GB)', fontsize=12)
    ax.set_title('消融实验 - GPU显存占用对比', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'gpu_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_eval_metrics(all_metrics):
    """Eval Loss和Perplexity对比"""
    names = list(all_metrics.keys())
    losses = [m.get('eval_loss', 0) for m in all_metrics.values()]
    ppls = [m.get('perplexity', 0) for m in all_metrics.values()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FF6B6B'][:len(names)]

    bars1 = ax1.bar(names, losses, color=colors, edgecolor='black')
    for bar, l in zip(bars1, losses):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                 f'{l:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax1.set_ylabel('Eval Loss', fontsize=12)
    ax1.set_title('验证集Loss对比', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    bars2 = ax2.bar(names, ppls, color=colors, edgecolor='black')
    for bar, p in zip(bars2, ppls):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                 f'{p:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax2.set_ylabel('Perplexity', fontsize=12)
    ax2.set_title('困惑度对比', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'eval_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_loss_curves(all_metrics):
    """所有实验的训练Loss曲线叠加对比"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FF6B6B']
    for i, (name, m) in enumerate(all_metrics.items()):
        tl = m.get('train_loss_history', [])
        vl = m.get('val_loss_history', [])
        c = colors[i % len(colors)]
        if tl:
            ax1.plot([x['step'] for x in tl], [x['loss'] for x in tl],
                     label=name, color=c, alpha=0.7)
        if vl:
            ax2.plot([x['step'] for x in vl], [x['eval_loss'] for x in vl],
                     label=name, color=c, marker='o', markersize=4)
    ax1.set_xlabel('Steps'); ax1.set_ylabel('Training Loss')
    ax1.set_title('训练Loss曲线对比', fontsize=14, fontweight='bold')
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.set_xlabel('Steps'); ax2.set_ylabel('Validation Loss')
    ax2.set_title('验证Loss曲线对比', fontsize=14, fontweight='bold')
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss_curves_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_comprehensive_summary(all_metrics):
    """综合雷达图对比"""
    names = list(all_metrics.keys())
    if len(names) < 2:
        return
    # 归一化指标用于雷达图（越小越好的取倒数）
    metrics_raw = {}
    for name, m in all_metrics.items():
        metrics_raw[name] = {
            '参数效率': 1.0 / max(m.get('trainable_params', 1) / 1e6, 0.01),
            '训练速度': 1.0 / max(m.get('training_time_min', 1), 0.01),
            '显存效率': 1.0 / max(m.get('peak_gpu_memory_gb', 1), 0.01),
            '模型质量': 1.0 / max(m.get('perplexity', 1), 0.01),
        }
    # 归一化到0-1
    categories = list(list(metrics_raw.values())[0].keys())
    for cat in categories:
        vals = [metrics_raw[n][cat] for n in names]
        max_v, min_v = max(vals), min(vals)
        rng = max_v - min_v if max_v != min_v else 1
        for n in names:
            metrics_raw[n][cat] = (metrics_raw[n][cat] - min_v) / rng

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FF6B6B']
    for i, name in enumerate(names):
        values = [metrics_raw[name][c] for c in categories]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_title('消融实验综合对比 (归一化)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'radar_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_metrics = load_all_metrics()
    if not all_metrics:
        print("没有找到任何实验指标文件!"); return

    print_summary_table(all_metrics)
    plot_trainable_params(all_metrics)
    plot_training_time(all_metrics)
    plot_gpu_memory(all_metrics)
    plot_eval_metrics(all_metrics)
    plot_loss_curves(all_metrics)
    plot_comprehensive_summary(all_metrics)

    print(f"\n所有对比图表已保存至: {OUTPUT_DIR}")
    print("生成的图表:")
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith('.png'):
            print(f"  - {f}")


if __name__ == "__main__":
    main()
