# ==============================================================================
# 环境设置与库导入
# ==============================================================================
import torch
import transformers
import math
import os
import json
import gc
from dataclasses import dataclass
from typing import Dict, Sequence
import matplotlib.pyplot as plt
from tqdm import tqdm

from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from fastchat.conversation import get_conv_template
import json, copy
import transformers
from typing import Dict, Sequence, List
from dataclasses import dataclass
from torch.utils.data import Dataset
# ==============================================================================
# 配置常量定义
# ==============================================================================
# 模型路径与训练参数常量
MODEL_ID = "C:/Users/Administrator/.cache/huggingface/hub/Qwen3-4B-Instruct-2507"
OUTPUT_DIR = "./train_result/qwen_improve(size=25000 epoch=1 alpha=32)/checkpoints"
LORA_OUTPUT_PATH = "./train_result/qwen_improve(size=25000 epoch=1 alpha=32)/lora"
TEST_FILE_PATH = "zh_med.json"
SAVED_DATA_PATH = "./train_result/qwen_improve(size=25000 epoch=1 alpha=32)/saved_data.json"
IMAGE_SAVE_PATH = "./train_result/qwen_improve(size=25000 epoch=1 alpha=32)"  # 图片保存目录
IGNORE_INDEX = -100
TRAIN_SAMPLE_SIZE = 25000  # 训练样本数量
TEST_SPLIT_RATIO = 0.1  # 验证集比例
DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"
default_conversation = get_conv_template('phoenix')

# ==============================================================================
# 模型与训练配置
# ==============================================================================
def get_bnb_config():
    """获取4位量化配置"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )


def get_lora_config():
    """获取LoRA微调配置"""
    return LoraConfig(
        r=8,
        lora_alpha=32,  # 通常为r的2倍
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )


def get_training_args():
    """获取训练参数配置"""
    return transformers.TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        optim='paged_adamw_32bit',
        save_strategy="steps",
        eval_strategy="steps",
        save_steps=10,
        eval_steps=10,
        logging_steps=10,
        learning_rate=1e-5,
        weight_decay=0.001,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=True,  # 若GPU不支持可改为fp16=True
        group_by_length=True,
        gradient_checkpointing=True,
        report_to="none",
    )


# ==============================================================================
# 数据处理
# ==============================================================================


class InstructDataset(Dataset):
    def __init__(self, data: Sequence, tokenizer: transformers.PreTrainedTokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        sources = self.data[index]
        if isinstance(index, int):
            sources = [sources]
        data_dict = preprocess([e['conversations'] for e in sources], self.tokenizer)
        if isinstance(index, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])
        return data_dict

def preprocess(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        max_length=1024
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    intermediates = []
    for source in sources:
        header = f"{default_conversation.system_message}"
        conversation, intermediate = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
        intermediates.append(intermediate)

    # tokenize conversations
    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)

    # keep only machine responses as targets
    assert len(targets) == len(intermediates)
    for target, inters in zip(targets, intermediates):
        mask = torch.zeros_like(target, dtype=torch.bool)
        for inter in inters:
            tokenized = _tokenize_fn(inter, tokenizer)
            start_idx = tokenized["input_ids"][0].size(0) - 1
            end_idx = tokenized["input_ids"][1].size(0)
            mask[start_idx:end_idx] = True
        target[~mask] = IGNORE_INDEX

    input_ids = input_ids[:max_length]
    targets = targets[:max_length]
    return dict(input_ids=input_ids, labels=targets)

def _add_speaker_and_signal(header, source, get_conversation=True):
    BEGIN_SIGNAL = DEFAULT_BOS_TOKEN
    END_SIGNAL = DEFAULT_EOS_TOKEN
    conversation = header
    intermediate = []
    for sentence in source:
        from_str = sentence["role"]
        if from_str.lower() == "human":
            from_str = default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = default_conversation.roles[1]
        else:
            from_str = 'unknown'
        # store the string w/o and w/ the response
        value = (from_str + ": " + BEGIN_SIGNAL + sentence["value"] + END_SIGNAL)
        if sentence["role"].lower() == "gpt":
            start = conversation + from_str + ": " + BEGIN_SIGNAL
            end = conversation + value
            intermediate.append([start, end])
        if get_conversation:
            conversation += value
    return conversation, intermediate

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def load_and_preprocess_data(tokenizer):
    """加载并预处理数据集"""
    # 划分训练集和验证集
    dataset = load_dataset("FreedomIntelligence/Huatuo26M-Lite")
    dataset = dataset['train'].map(
        lambda sample: {"conversations": [
            {"role": "human", "value": sample['question']}, 
            {"role": "gpt", "value": sample['answer']}
        ]}, 
        batched=False
    )
    dataset = dataset.select(range(min(TRAIN_SAMPLE_SIZE, len(dataset))))
    dataset_split = dataset.train_test_split(test_size=TEST_SPLIT_RATIO)
    # 使用不同的变量名避免冲突
    raw_train_dataset = dataset_split['train']
    raw_val_dataset = dataset_split['test']
    # 转换为InstructDataset
    instruct_train_dataset = InstructDataset(raw_train_dataset, tokenizer)
    instruct_val_dataset = InstructDataset(raw_val_dataset, tokenizer)
    print(f"训练集样本数: {len(instruct_train_dataset)}")
    print(f"验证集样本数: {len(instruct_val_dataset)}")
    return instruct_train_dataset, instruct_val_dataset


# ==============================================================================
# 模型加载与训练
# ==============================================================================
def load_model_and_tokenizer():
    """加载模型和分词器并配置LoRA"""
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # 设置pad token
    # 加载量化模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=get_bnb_config(),
        device_map={"": "cuda:0"},
        trust_remote_code=True
    )
    
    # 准备模型进行LoRA训练
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, get_lora_config())
    
    return model, tokenizer


def train_model(model, tokenizer, train_dataset, val_dataset):
    """训练模型并返回训练结果"""
    # 初始化训练器
    trainer = transformers.Trainer(
        model=model,
        args=get_training_args(),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    )
    
    # 开始训练
    print("开始训练...")
    trainer.train()
    print("训练完成！")
    
    # 保存LoRA权重
    trainer.save_model(LORA_OUTPUT_PATH)
    
    return trainer


# ==============================================================================
# 结果可视化与评估
# ==============================================================================
def plot_loss_curve(trainer):
    """绘制训练与验证损失曲线"""
    print("正在绘制Loss曲线...")
    log_history = trainer.state.log_history
    train_steps, train_losses = [], []
    val_steps, val_losses = [], []
    
    for log in log_history:
        if 'loss' in log:
            train_steps.append(log['step'])
            train_losses.append(log['loss'])
        if 'eval_loss' in log:
            val_steps.append(log['step'])
            val_losses.append(log['eval_loss'])
    
    plt.figure(figsize=(12, 6))
    plt.plot(train_steps, train_losses, label='Training Loss', color='blue', alpha=0.6)
    plt.plot(val_steps, val_losses, label='Validation Loss', color='red', marker='x')
    plt.title('Training and Validation Loss Curves', fontsize=16)
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(IMAGE_SAVE_PATH, "Training and Validation Loss Curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Loss曲线已保存至：{save_path}")  # 打印保存路径，方便查找


def evaluate_model(trainer):
    """评估模型并计算困惑度"""
    final_eval_metrics = trainer.evaluate()
    print(f"最终评估结果: {final_eval_metrics}")
    
    perplexity = math.exp(final_eval_metrics['eval_loss'])
    print(f"最终模型的困惑度 (Perplexity): {perplexity:.2f}")


# ==============================================================================
# 模型推理与结果保存
# ==============================================================================
def load_inference_model(tokenizer):
    """加载用于推理的合并后模型"""
    # 重新加载量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # 加载基础模型并合并LoRA权重
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        quantization_config=bnb_config, 
        device_map={"": 0}
    )
    model = PeftModel.from_pretrained(model, LORA_OUTPUT_PATH)
    model = model.merge_and_unload()
    model.config.max_length = 512
    model.eval()
    
    return model


@torch.no_grad()
def generate(query_list, return_answer: bool = False,tokenizer=None,model=None):
    def conv_format(query):
        conv = get_conv_template('phoenix')
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()
    query_list = [conv_format(query) for query in query_list]
    input_ids = tokenizer(query_list, padding=True, truncation=True, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    n_input, n_seq = input_ids.shape[0], input_ids.shape[-1]
    output_ids = []
    step = 1
    for index in tqdm(range(0, n_input, step)):
        outputs = model.generate(
            input_ids=input_ids[index: min(n_input, index+step)],
            do_sample=False,
            max_new_tokens=512,
            # temperature=0.7,
            repetition_penalty=1.0,
        )
        output_ids += outputs
    responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    if return_answer:
        return [response[len(query):].strip() for query, response in zip(query_list, responses)]
    return responses


def save_inference_results(test_data, model_answers):
    """保存推理结果到JSON文件"""
    for data, answer in zip(test_data, model_answers):
        data.append(answer)
    os.makedirs(os.path.dirname(SAVED_DATA_PATH), exist_ok=True)
    with open(SAVED_DATA_PATH, 'w', encoding='utf-8') as writer:
        json.dump(test_data, writer, indent=4, ensure_ascii=False)


# ==============================================================================
# 主函数
# ==============================================================================
def main():
    # 1. 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer()
    
    # 2. 准备数据
    train_dataset, val_dataset = load_and_preprocess_data(tokenizer)
    # sample_data = train_dataset[1]
    # print("=" * 80)
    # print("Debuging: ")
    # print(sample_data)
    # print("-" * 80)
    # print(f"input_ids:\n{tokenizer.decode(sample_data['input_ids'])}")
    # # Filter out IGNORE_INDEX before decoding labels
    # z = [token for token in sample_data['labels'] if token != IGNORE_INDEX]
    # print("-" * 80)
    # print(f"labels:\n{tokenizer.decode(z)}")
    # print("=" * 80)

    # 3. 训练模型
    #trainer = train_model(model, tokenizer, train_dataset, val_dataset)
    
    # 4. 可视化与评估
    #plot_loss_curve(trainer)
    #evaluate_model(trainer)
    
    # 5. 清理内存
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # 6. 加载推理模型
    model = load_inference_model(tokenizer)
    
    # 7. 加载测试数据并生成回答
    assert os.path.exists(TEST_FILE_PATH), f"测试文件不存在: {TEST_FILE_PATH}"
    with open(TEST_FILE_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    model_answers = generate([data[0] for data in test_data], return_answer=True, tokenizer=tokenizer,model=model)
    
    # 8. 保存结果
    save_inference_results(test_data, model_answers)
    print(f"推理结果已保存至 {SAVED_DATA_PATH}")


if __name__ == "__main__":
    main()
   