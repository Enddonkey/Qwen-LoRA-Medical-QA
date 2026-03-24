# ==============================================================================
# Qwen3-4B 全参数微调 (Full Fine-tuning, 无LoRA, bf16)
# 训练样本: 7000, epoch=1
# ==============================================================================
import sys, os, torch, json, gc, time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_utils import *
from transformers import AutoTokenizer, AutoModelForCausalLM

CFG = {
    'experiment_name': 'full_finetune_7k',
    'finetune_method': 'Full Fine-tuning (bf16, 8bit-paged-optimizer)',
    'target_modules': 'ALL',
    'train_sample_size': 7000,
    'learning_rate': 2e-6,  # 全参数微调用更小的学习率
    'num_epochs': 1,
    'batch_size': 1,  # 全参数微调显存极大，必须用最小batch
    'gradient_accumulation_steps': 32,  # 保持等效batch=32
}
MODEL_ID = "C:/Users/Administrator/.cache/huggingface/hub/Qwen3-4B-Instruct-2507"
RESULT_DIR = f"./train_result/{CFG['experiment_name']}"
MODEL_SAVE_PATH = f"{RESULT_DIR}/model"
METRICS_PATH = f"{RESULT_DIR}/training_metrics.json"
SAVED_DATA_PATH = f"{RESULT_DIR}/saved_data.json"
TEST_FILE = "zh_med_1000.json"


def get_args():
    return transformers.TrainingArguments(
        output_dir=f"{RESULT_DIR}/checkpoints", num_train_epochs=CFG['num_epochs'],
        per_device_train_batch_size=CFG['batch_size'], per_device_eval_batch_size=CFG['batch_size'],
        gradient_accumulation_steps=CFG['gradient_accumulation_steps'],
        optim='paged_adamw_8bit',  # 8-bit分页优化器：自动将优化器状态卸载到CPU，大幅节省显存
        save_strategy="steps", eval_strategy="steps",
        save_steps=50, eval_steps=50, logging_steps=10,
        learning_rate=CFG['learning_rate'], weight_decay=0.001, warmup_ratio=0.1,
        lr_scheduler_type="cosine", bf16=True, group_by_length=True,
        gradient_checkpointing=True, report_to="none",
        max_grad_norm=1.0,  # 梯度裁剪，防止显存峰值过高
        save_total_limit=2,  # 全参数模型很大，限制checkpoint数量
        dataloader_num_workers=4, dataloader_pin_memory=True)


def main():
    # 1. 加载模型 - 不使用量化，直接bf16加载
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, trust_remote_code=True)
    model.gradient_checkpointing_enable()
    # 全参数微调：所有参数都可训练
    model.enable_input_require_grads()  # gradient_checkpointing需要
    t_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {t_params:,} / {total_params:,} ({100*t_params/total_params:.4f}%)")

    # 2. 加载数据
    train_ds, val_ds = load_and_preprocess_data(tokenizer, CFG['train_sample_size'])

    # 3. 数据集检查
    check_dataset(tokenizer, train_ds, val_ds)

    # 4. 训练
    trainer, t_time, peak_mem, gpu_recs = do_train(model, tokenizer, train_ds, val_ds, get_args())

    # 5. 保存完整模型
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    trainer.save_model(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)

    # 6. 评估与可视化
    eval_m, ppl = eval_model(trainer)
    plot_loss(trainer, CFG['experiment_name'], RESULT_DIR)
    save_metrics(CFG, t_params, total_params, t_time, peak_mem, gpu_recs, eval_m, ppl, trainer, METRICS_PATH)

    # 7. 推理 - 全参数微调直接加载保存的模型
    del model, trainer; gc.collect(); torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_SAVE_PATH, torch_dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    assert os.path.exists(TEST_FILE), f"测试文件不存在: {TEST_FILE}"
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    answers = generate_answers([d[0] for d in test_data], tokenizer, model)
    for d, a in zip(test_data, answers): d.append(a)
    os.makedirs(os.path.dirname(SAVED_DATA_PATH), exist_ok=True)
    with open(SAVED_DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)
    print(f"推理结果已保存至 {SAVED_DATA_PATH}")


if __name__ == "__main__":
    import transformers
    main()
