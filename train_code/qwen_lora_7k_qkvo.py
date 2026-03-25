# ==============================================================================
# Qwen3-4B LoRA微调 - QKVO模式 (target_modules: q_proj, k_proj, v_proj, o_proj)
# 训练样本: 7000, epoch=1
# ==============================================================================
import sys, os, torch, json, gc
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_utils import *
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM

CFG = {
    'experiment_name': 'lora_7k_qkvo',
    'finetune_method': 'LoRA (QLoRA 4-bit)',
    'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    'train_sample_size': 7000,
    'lora_r': 16, 'lora_alpha': 16, 'lora_dropout': 0.05,
    'learning_rate': 2e-4, 'num_epochs': 1,
    'batch_size': 8, 'gradient_accumulation_steps': 4,
}
MODEL_ID = "C:/Users/Administrator/.cache/huggingface/hub/Qwen3-4B-Instruct-2507"
RESULT_DIR = f"./train_result/{CFG['experiment_name']}"
LORA_PATH = f"{RESULT_DIR}/lora"
METRICS_PATH = f"{RESULT_DIR}/training_metrics.json"
SAVED_DATA_PATH = f"{RESULT_DIR}/saved_data.json"
TEST_FILE = "zh_med_500.json"


def get_args():
    return transformers.TrainingArguments(
        output_dir=f"{RESULT_DIR}/checkpoints", num_train_epochs=CFG['num_epochs'],
        per_device_train_batch_size=CFG['batch_size'], per_device_eval_batch_size=CFG['batch_size'],
        gradient_accumulation_steps=CFG['gradient_accumulation_steps'],
        optim='adamw_8bit', save_strategy="steps", eval_strategy="steps",
        save_steps=100, eval_steps=20, logging_steps=5,
        learning_rate=CFG['learning_rate'], weight_decay=0.001, warmup_ratio=0.03,
        lr_scheduler_type="cosine", bf16=True, group_by_length=False,
        gradient_checkpointing=True, report_to="none",
        dataloader_num_workers=0, dataloader_pin_memory=True)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=get_bnb_config(),
                                                  device_map={"": "cuda:0"}, trust_remote_code=True)
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(r=CFG['lora_r'], lora_alpha=CFG['lora_alpha'],
                          target_modules=CFG['target_modules'], lora_dropout=CFG['lora_dropout'],
                          bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    t_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {t_params:,} / {total_params:,} ({100*t_params/total_params:.4f}%)")

    train_ds, val_ds = load_and_preprocess_data(tokenizer, CFG['train_sample_size'])
    check_dataset(tokenizer, train_ds, val_ds)

    trainer, t_time, peak_mem, gpu_recs = do_train(model, tokenizer, train_ds, val_ds, get_args())
    trainer.save_model(LORA_PATH)

    eval_m, ppl = eval_model(trainer)
    plot_loss(trainer, CFG['experiment_name'], RESULT_DIR)
    save_metrics(CFG, t_params, total_params, t_time, peak_mem, gpu_recs, eval_m, ppl, trainer, METRICS_PATH)

    del model; gc.collect(); torch.cuda.empty_cache()
    model = load_inference_model(MODEL_ID, LORA_PATH, tokenizer)
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
