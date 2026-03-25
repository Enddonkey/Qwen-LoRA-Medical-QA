# Qwen-LoRA-Medical-QA

基于 **Qwen3-4B-Instruct** 的中文医学问答微调项目，系统性地探索 LoRA 参数高效微调与全参数微调的性能差异，并通过消融实验分析不同 LoRA target modules 配置的影响。

## 项目概述

本项目在 Huatuo26M-Lite 中文医学问答数据集上对 Qwen3-4B-Instruct 进行监督微调，核心实验包括：

1. **LoRA 消融实验**：固定 7000 训练样本，对比不同 target modules 配置（QV / QKV / QKVO）
2. **微调方法对比**：LoRA (QLoRA 4-bit) vs 全参数微调 (bf16)，记录训练时间、Loss、GPU 占用等指标
3. **GPT-4 自动评估**：从 500 条测试集中采样 50 条，通过 GPT-4 对比评估模型输出质量

### 为什么选择 7000 条训练样本？

基于前期实验观察：
- **15000 样本 / 5 epoch**：训练时间长，存在过拟合风险
- **25000~50000 样本 / 1 epoch**：数据量大但单 epoch 学习不充分，边际收益递减
- **7000 样本 / 1 epoch**：在消费级 GPU (16GB) 上约 30-35 分钟完成训练，Loss 在 ~200 步内收敛至 1.8 左右，验证集 perplexity 约 6.0，是效率与效果的最佳平衡点。同时该规模便于快速迭代消融实验，在有限算力下完成多组对比。

## 消融实验结果

### 训练指标对比

| 实验 | 方法 | Target Modules | 可训练参数 | 训练时间 | GPU峰值 | Eval Loss | PPL |
|------|------|---------------|-----------|---------|---------|-----------|-----|
| LoRA-QV | QLoRA 4-bit | q_proj, v_proj | 5.90M (0.27%) | 35.0 min | 15.33 GB | 1.7991 | 6.04 |
| LoRA-QKV | QLoRA 4-bit | q_proj, k_proj, v_proj | 7.96M (0.36%) | 30.6 min | 15.35 GB | 1.7817 | 5.94 |
| LoRA-QKVO | QLoRA 4-bit | q_proj, k_proj, v_proj, o_proj | 11.80M (0.53%) | 34.4 min | 13.03 GB | 1.8074 | 6.09 |
| Full-FT | bf16 AdamW | ALL | 3839M (100%) | 待运行 | 待运行 | 待运行 | 待运行 |

### GPT-4 评估结果（模型 vs Baseline）

| 实验 | 采样数 | Baseline胜 | Model胜 | 平局 | 模型胜率 |
|------|--------|-----------|---------|------|---------|
| LoRA-QV | 50 | 16 | 32 | 0 | **66.7%** |
| LoRA-QKV | 50 | 15 | 31 | 4 | **62.0%** |
| LoRA-QKVO | 50 | 16 | 31 | 3 | **62.0%** |

### 关键发现

- **LoRA-QV 配置最优**：仅用 5.9M 参数（0.27%）即达到最高 66.7% 胜率，参数效率最高
- **增加 target modules 未必更好**：QKV 和 QKVO 的 eval loss 和胜率并未显著优于 QV，说明在小数据量下额外参数可能引入噪声
- **训练效率**：三种 LoRA 配置训练时间均在 30-35 分钟，GPU 占用 13-15 GB，适合消费级显卡

## 项目结构

```
├── README.md                          # 项目说明
├── requirements.txt                   # 依赖包
├── run_all.bat                        # 一键运行脚本
├── train_code/                        # 训练脚本
│   ├── train_utils.py                 # 共享工具模块
│   ├── qwen_lora_7k_qv.py            # LoRA-QV 消融实验
│   ├── qwen_lora_7k_qkv.py           # LoRA-QKV 消融实验
│   ├── qwen_lora_7k_qkvo.py          # LoRA-QKVO 消融实验
│   ├── qwen_full_finetune_7k.py       # 全参数微调
│   ├── generate_test_dataset.py       # 测试集生成
│   └── run_inference_only.py          # 独立推理脚本
├── train_result/                      # 训练结果
│   ├── lora_7k_qv/                    # QV 实验结果
│   ├── lora_7k_qkv/                   # QKV 实验结果
│   ├── lora_7k_qkvo/                  # QKVO 实验结果
│   └── full_finetune_7k/              # 全参数微调结果
└── evaluate/                          # 评估模块
    ├── src/
    │   ├── evaluate_ablation.py       # GPT-4 消融评估
    │   └── compare_ablation.py        # 训练指标对比可视化
    ├── ablation_evaluation_results.json
    └── ablation_plots/                # 对比图表
```

## 环境配置

### 硬件要求
- **LoRA 微调**: ≥ 16GB 显存 GPU（如 RTX 4080/4090）
- **全参数微调**: ≥ 16GB 显存 GPU（使用 gradient checkpointing + 8-bit AdamW）

### 安装依赖
```bash
pip install -r requirements.txt
```

主要依赖：
- `torch >= 2.7.0`
- `transformers >= 5.0.0`
- `peft >= 0.18.0`
- `bitsandbytes >= 0.48.0`
- `datasets >= 4.0.0`

## 使用方法

### 1. 生成测试数据集
```bash
python train_code/generate_test_dataset.py
```

### 2. 运行消融实验
```bash
# LoRA-QV (推荐，效果最佳)
python train_code/qwen_lora_7k_qv.py

# LoRA-QKV
python train_code/qwen_lora_7k_qkv.py

# LoRA-QKVO
python train_code/qwen_lora_7k_qkvo.py

# 全参数微调
python train_code/qwen_full_finetune_7k.py
```

### 3. 评估
```bash
# GPT-4 自动评估（需配置 openai_key.txt）
python evaluate/src/evaluate_ablation.py

# 训练指标对比可视化
python evaluate/src/compare_ablation.py
```

## 技术细节

### LoRA 配置
```python
LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],  # 或 QKV / QKVO
    bias="none", task_type="CAUSAL_LM"
)
```

### 量化配置 (QLoRA)
```python
BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)
```

### 数据格式 (ChatML)
```
<|im_start|>system
你是一个专业的医疗健康助手，能够准确回答各类医学问题。<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>
```

## 评估方法

采用 **GPT-4 Pairwise Comparison** 评估：
- 从 500 条测试集中随机采样 50 条
- 将 baseline（原始数据集参考答案）与模型生成答案送入 GPT-4
- GPT-4 从 helpfulness、relevance、accuracy、detail 四个维度判断优劣
- 统计模型胜率作为最终指标

## 参考文献

1. Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685
2. Dettmers, T., et al. (2023). *QLoRA: Efficient Finetuning of Quantized Language Models*. arXiv:2305.14314
3. Qwen Team. (2024). *Qwen Technical Report*.
4. FreedomIntelligence. *Huatuo26M: A Large-scale Chinese Medical QA Dataset*.

## License

MIT License
