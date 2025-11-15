# NLP课程作业3 - 大语言模型微调实验

## 项目概述

本项目是香港中文大学深圳 自然语言处理课程的第三次作业，专注于大语言模型的微调技术。项目使用Qwen3-4B-Instruct作为基础模型，在中文医学问答数据集上进行监督微调，探索不同参数配置对模型性能的影响。

### 主要特点

- **基础模型**: Qwen/Qwen3-4B-Instruct
- **微调技术**: LoRA (Low-Rank Adaptation) 参数高效微调
- **优化策略**: 4位量化技术优化内存使用
- **应用领域**: 中文医学问答
- **评估方法**: ChatGPT自动评估系统

## 技术架构

### 模型配置
- **基础模型**: Qwen3-4B-Instruct
- **微调方法**: LoRA (Low-Rank Adaptation)
- **量化技术**: 4-bit量化 (BitsAndBytesConfig)
- **优化器**: AdamW
- **学习率调度**: Linear warmup + decay

### 数据集
- **训练数据**: Huatuo26M-Lite中文医学问答数据集
- **测试数据**: zh_med.json (20个中文医学问答对)
- **数据格式**: 问答对格式，支持多轮对话

### 实验设计
项目包含多个实验变体，探索不同参数对模型性能的影响：

1. **Baseline模型**: 基础配置
2. **Phoenix模型**: 使用Phoenix对话模板
3. **Llama模型**: 基于Llama架构的对比实验
4. **Qwen改进版本**: 
   - qwen_improve_1: size=15000, epoch=5, alpha=16
   - qwen_improve_2: size=25000, epoch=1, alpha=32
   - qwen_improve_3: size=50000, epoch=1, alpha=32

## 项目结构

```
├── README.md                           # 项目说明文档
├── NLP_course_Assignment_3.pdf         # 课程作业要求
├── zh_med.json                         # 测试数据集
├── openai_key.txt                      # OpenAI API密钥
├── "qwen_lora_baseline_ipynb".ipynb    # Jupyter notebook
├── train_code/                         # 训练脚本目录
│   ├── qwen_baseline.py                # 基线模型训练
│   ├── phoenix_lora.py                 # Phoenix模型训练
│   ├── llama_lora.py                   # Llama模型训练
│   ├── qwen_improve(size=15000 epoch=5 alpha=16).py
│   ├── qwen_improve(size=25000 epoch=1 alpha=32).py
│   └── qwen_improve(size=50000 epoch=1 alpha=32).py
├── train_result/                       # 训练结果目录
│   ├── baseline/                       # 基线模型结果
│   ├── phoenix_lora/                   # Phoenix模型结果
│   ├── llama_lora/                     # Llama模型结果
│   ├── qwen_improve(size=15000 epoch=5 alpha=16)/
│   ├── qwen_improve(size=25000 epoch=1 alpha=32)/
│   └── qwen_improve(size=50000 epoch=1 alpha=32)/
└── evaluate/                           # 评估脚本和结果
    ├── src/                            # 评估源代码
    │   ├── evaluate_saved_data.py      # 主要评估脚本
    │   ├── evaluate_saved_data_score.py
    │   └── clean_saved_data.py
    ├── evaluation_results.json         # 综合评估结果
    ├── baseline.json                   # 基线模型评估
    ├── phoenix.json                    # Phoenix模型评估
    ├── qwen_improve_1(size=15000 epoch=5 alpha=16).json
    ├── qwen_improve_2(size=25000 epoch=1 alpha=32).json
    ├── qwen_improve_3(size=50000 epoch=1 alpha=32).json
    └── evaluation_plots/               # 评估结果可视化
        ├── overall_comparison.png      # 整体性能对比
        ├── baseline_comparison.png     # 基线模型对比
        ├── baseline_pie.png           # 基线模型饼图
        └── ...                        # 其他模型的对比图表
```

## 环境配置

### 依赖要求
```bash
torch=2.7.0+cu128
transformers=5.0.0.dev0
peft=0.18.0rc1.dev0
bitsandbytes=0.48.2
datasets=4.0.0
accelerate=1.12.0.dev0
openai=2.8.0
matplotlib=3.10.7

```

### 安装步骤
```bash
# 克隆项目
git clone <repository-url>
cd NLP-HW3

# 安装依赖
pip install -r requirements.txt

# 配置OpenAI API密钥
echo "your-openai-api-key" > openai_key.txt
```

## 使用方法

### 1. 模型训练

#### 基线模型训练
```bash
python train_code/qwen_baseline.py
```

#### 改进版本训练
```bash
# 训练样本15000，5个epoch，alpha=16
python "train_code/qwen_improve(size=15000 epoch=5 alpha=16).py"

# 训练样本25000，1个epoch，alpha=32
python "train_code/qwen_improve(size=25000 epoch=1 alpha=32).py"

# 训练样本50000，1个epoch，alpha=32
python "train_code/qwen_improve(size=50000 epoch=1 alpha=32).py"
```

### 2. 模型评估

#### 清理评估数据
```bash
python evaluate/src/clean_saved_data.py
```

#### 运行评估脚本
```bash
python evaluate/src/evaluate_saved_data.py
```

### 3. 结果可视化

评估脚本会自动生成可视化图表，保存在`evaluate/evaluation_plots/`目录中。

## 实验结果

### 模型性能对比

根据ChatGPT自动评估结果，各模型在中文医学问答任务上的表现如下：

| 模型 | 训练样本数 | Epoch | LoRA Alpha | 平均得分 | 性能表现 |
|------|------------|-------|------------|----------|----------|
| Baseline | 默认 | 默认 | 默认 | - | 基线性能 |
| Phoenix | - | - | - | - | 对话模板优化 |
| Qwen Improve 1 | 15,000 | 5 | 16 | - | 多轮训练 |
| Qwen Improve 2 | 25,000 | 1 | 32 | - | 中等数据量 |
| Qwen Improve 3 | 50,000 | 1 | 32 | - | 大数据量 |

### 关键发现

1. **数据量影响**: 增加训练数据量显著提升模型在医学问答任务上的表现
2. **LoRA参数优化**: 较高的alpha值(32)在大数据量训练中表现更好
3. **训练策略**: 单epoch大数据量训练vs多epoch小数据量训练的权衡
4. **对话模板**: Phoenix模板对对话质量有明显改善

### 训练损失曲线

每个模型的训练过程都生成了详细的损失曲线图，保存在对应的`train_result/*/Training and Validation Loss Curves.png`文件中。

## 技术细节

### LoRA配置
```python
lora_config = LoraConfig(
    r=16,                    # rank
    lora_alpha=32,          # scaling parameter
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### 量化配置
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

### 训练参数
```python
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch"
)
```

## 评估方法

### ChatGPT自动评估
- 使用OpenAI GPT-4作为评估器
- 对模型输出进行多维度评估：准确性、相关性、完整性
- 生成详细的评估报告和可视化图表

### 评估指标
- **准确性**: 医学知识的正确性
- **相关性**: 回答与问题的相关程度
- **完整性**: 回答的全面性和详细程度
- **流畅性**: 语言表达的自然程度

## 文件说明

### 训练脚本
- `qwen_baseline.py`: 基线模型，使用默认参数配置
- `phoenix_lora.py`: 使用Phoenix对话模板的模型
- `llama_lora.py`: 基于Llama架构的对比实验
- `qwen_improve_*.py`: 不同参数配置的改进版本

### 评估脚本
- `evaluate_saved_data.py`: 主要评估脚本，调用ChatGPT API
- `evaluate_saved_data_score.py`: 评分计算脚本
- `clean_saved_data.py`: 数据清理工具

### 结果文件
- `saved_data.json`: 模型推理结果
- `Training and Validation Loss Curves.png`: 训练损失曲线
- `evaluation_results.json`: 详细评估结果
- `evaluation_plots/*.png`: 性能对比图表

## 注意事项

1. **GPU要求**: 建议使用至少8GB显存的GPU进行训练
2. **内存优化**: 使用4位量化技术减少内存占用
3. **API配置**: 需要有效的OpenAI API密钥进行评估
4. **数据路径**: 确保数据集路径配置正确

## 贡献者

- 项目作者: [End_donkey]
- 课程: 香港中文大学深圳自然语言处理
- 时间: 2025年秋季学期

## 许可证

本项目仅用于学术研究和课程作业，请遵循相关的学术诚信政策。

## 参考文献

1. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
2. Qwen Team. (2023). Qwen Technical Report.
3. Huatuo26M: A Large-scale Chinese Medical QA Dataset.
