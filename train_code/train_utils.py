# ==============================================================================
# 共享训练工具模块 - 所有消融实验共用
# ==============================================================================
import os
# 设置代理（解决HuggingFace连接问题）
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

import torch, transformers, math, json, gc, time, copy
from dataclasses import dataclass
from typing import Dict, Sequence
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback
from fastchat.conversation import get_conv_template
from torch.utils.data import Dataset

IGNORE_INDEX = -100
DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN = "<s>", "</s>"
default_conversation = get_conv_template('phoenix')


class GPUMemoryCallback(TrainerCallback):
    def __init__(self):
        self.gpu_memory_records = []
    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / (1024**3)
            self.gpu_memory_records.append({'step': state.global_step, 'gpu_memory_gb': round(gpu_mem, 2)})


class InstructDataset(Dataset):
    def __init__(self, data, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        sources = self.data[index]
        if isinstance(index, int):
            sources = [sources]
        data_dict = preprocess([e['conversations'] for e in sources], self.tokenizer)
        if isinstance(index, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])
        return data_dict


def preprocess(sources, tokenizer, max_length=1024):
    conversations, intermediates = [], []
    for source in sources:
        header = f"{default_conversation.system_message}"
        conversation, intermediate = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
        intermediates.append(intermediate)
    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)
    for target, inters in zip(targets, intermediates):
        mask = torch.zeros_like(target, dtype=torch.bool)
        for inter in inters:
            tokenized = _tokenize_fn(inter, tokenizer)
            start_idx = tokenized["input_ids"][0].size(0) - 1
            end_idx = tokenized["input_ids"][1].size(0)
            mask[start_idx:end_idx] = True
        target[~mask] = IGNORE_INDEX
    return dict(input_ids=input_ids[:max_length], labels=targets[:max_length])


def _add_speaker_and_signal(header, source, get_conversation=True):
    conversation, intermediate = header, []
    for sentence in source:
        from_str = sentence["role"]
        if from_str.lower() == "human": from_str = default_conversation.roles[0]
        elif from_str.lower() == "gpt": from_str = default_conversation.roles[1]
        else: from_str = 'unknown'
        value = from_str + ": " + DEFAULT_BOS_TOKEN + sentence["value"] + DEFAULT_EOS_TOKEN
        if sentence["role"].lower() == "gpt":
            intermediate.append([conversation + from_str + ": " + DEFAULT_BOS_TOKEN, conversation + value])
        if get_conversation:
            conversation += value
    return conversation, intermediate


def _tokenize_fn(strings, tokenizer):
    tokenized_list = [tokenizer(text, return_tensors="pt", padding="longest",
                                max_length=tokenizer.model_max_length, truncation=True) for text in strings]
    input_ids = [t.input_ids[0] for t in tokenized_list]
    lens = [t.input_ids.ne(tokenizer.pad_token_id).sum().item() for t in tokenized_list]
    return dict(input_ids=input_ids, labels=input_ids, input_ids_lens=lens, labels_lens=lens)


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))


def load_and_preprocess_data(tokenizer, train_sample_size, test_split_ratio=0.1):
    dataset = load_dataset("FreedomIntelligence/Huatuo26M-Lite")
    dataset = dataset['train'].map(lambda s: {"conversations": [
        {"role": "human", "value": s['question']}, {"role": "gpt", "value": s['answer']}]}, batched=False)
    dataset = dataset.select(range(min(train_sample_size, len(dataset))))
    split = dataset.train_test_split(test_size=test_split_ratio)
    train_ds = InstructDataset(split['train'], tokenizer)
    val_ds = InstructDataset(split['test'], tokenizer)
    print(f"训练集样本数: {len(train_ds)}, 验证集样本数: {len(val_ds)}")
    return train_ds, val_ds


def check_dataset(tokenizer, train_dataset, val_dataset, num_samples=3):
    """训练前检查数据集和token的正确性"""
    print("\n" + "=" * 80)
    print("数据集检查 (Data Sanity Check)")
    print("=" * 80)
    print(f"\n[1] 基本统计: 训练集={len(train_dataset)}, 验证集={len(val_dataset)}")
    for i in range(min(num_samples, len(train_dataset))):
        sample = train_dataset[i]
        ids, labs = sample['input_ids'], sample['labels']
        valid_labs = [t.item() for t in labs if t.item() != IGNORE_INDEX]
        print(f"\n[2] 样本{i}: input_ids长度={len(ids)}, labels长度={len(labs)}, 有效label={len(valid_labs)}")
        if len(valid_labs) == 0:
            print("    ⚠️ 警告: 无有效label!"); continue
        print(f"    input解码(前150字): {tokenizer.decode(ids, skip_special_tokens=False)[:150]}...")
        print(f"    label解码(前150字): {tokenizer.decode(valid_labs, skip_special_tokens=False)[:150]}...")
    total_tok, total_val, empty = 0, 0, 0
    for i in range(min(20, len(train_dataset))):
        labs = train_dataset[i]['labels']
        v = sum(1 for t in labs if t.item() != IGNORE_INDEX)
        total_tok += len(labs); total_val += v
        if v == 0: empty += 1
    ratio = 100 * total_val / max(total_tok, 1)
    print(f"\n[3] Label覆盖率(前20样本): {ratio:.2f}%, 空label样本={empty}")
    if empty > 0: print(f"    ⚠️ 有{empty}个空label样本!")
    if ratio < 5: print(f"    ⚠️ 覆盖率过低!")
    print(f"\n[4] Token检查: pad='{tokenizer.pad_token}'(id={tokenizer.pad_token_id}), "
          f"eos='{tokenizer.eos_token}'(id={tokenizer.eos_token_id}), vocab={tokenizer.vocab_size}")
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    batch = collator([train_dataset[i] for i in range(min(4, len(train_dataset)))])
    print(f"\n[5] Batch检查: input={batch['input_ids'].shape}, labels={batch['labels'].shape}, "
          f"mask非零={batch['attention_mask'].float().mean():.4f}")
    print("=" * 80 + "\n数据集检查完成!\n")
    return True


def do_train(model, tokenizer, train_ds, val_ds, training_args):
    gpu_cb = GPUMemoryCallback()
    trainer = transformers.Trainer(model=model, args=training_args, train_dataset=train_ds,
                                   eval_dataset=val_ds, data_collator=DataCollatorForSupervisedDataset(tokenizer=tokenizer),
                                   callbacks=[gpu_cb])
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    print("开始训练...")
    trainer.train()
    t_total = time.time() - t0
    print(f"训练完成！耗时: {t_total:.2f}秒 ({t_total/60:.2f}分钟)")
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
    print(f"GPU峰值显存: {peak_mem:.2f} GB")
    return trainer, t_total, peak_mem, gpu_cb.gpu_memory_records


def plot_loss(trainer, name, save_dir):
    logs = trainer.state.log_history
    ts, tl, vs, vl = [], [], [], []
    for l in logs:
        if 'loss' in l: ts.append(l['step']); tl.append(l['loss'])
        if 'eval_loss' in l: vs.append(l['step']); vl.append(l['eval_loss'])
    plt.figure(figsize=(12, 6))
    plt.plot(ts, tl, label='Train Loss', color='blue', alpha=0.6)
    plt.plot(vs, vl, label='Val Loss', color='red', marker='x')
    plt.title(f'Loss Curves ({name})'); plt.xlabel('Steps'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "Loss_Curves.png"), dpi=300, bbox_inches="tight"); plt.close()


def eval_model(trainer):
    m = trainer.evaluate()
    ppl = math.exp(m['eval_loss'])
    print(f"eval_loss={m['eval_loss']:.4f}, perplexity={ppl:.2f}")
    return m, ppl


def save_metrics(cfg, t_params, total_params, t_time, peak_mem, gpu_recs, eval_m, ppl, trainer, save_path):
    logs = trainer.state.log_history
    tl = [{'step': l['step'], 'loss': l['loss']} for l in logs if 'loss' in l]
    vl = [{'step': l['step'], 'eval_loss': l['eval_loss']} for l in logs if 'eval_loss' in l]
    metrics = {**cfg, 'trainable_params': t_params, 'total_params': total_params,
               'trainable_ratio': f"{100*t_params/total_params:.4f}%",
               'training_time_sec': round(t_time, 2), 'training_time_min': round(t_time/60, 2),
               'peak_gpu_memory_gb': round(peak_mem, 2), 'eval_loss': eval_m.get('eval_loss'),
               'perplexity': round(ppl, 2), 'train_loss_history': tl, 'val_loss_history': vl,
               'gpu_memory_history': gpu_recs}
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"训练指标已保存至: {save_path}")


@torch.no_grad()
def generate_answers(query_list, tokenizer, model):
    def conv_fmt(q):
        conv = get_conv_template('phoenix')
        conv.append_message(conv.roles[0], q)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()
    formatted = [conv_fmt(q) for q in query_list]
    input_ids = tokenizer(formatted, padding=True, truncation=True, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    output_ids = []
    for i in tqdm(range(len(input_ids))):
        out = model.generate(input_ids=input_ids[i:i+1], do_sample=False, max_new_tokens=512, repetition_penalty=1.0)
        output_ids.append(out[0])
    responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return [r[len(q):].strip() for q, r in zip(formatted, responses)]


def get_bnb_config():
    """获取4-bit量化配置"""
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)


def load_inference_model(model_id, lora_path, tokenizer):
    bnb = get_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map={"": 0})
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()
    model.config.max_length = 512
    model.eval()
    return model
