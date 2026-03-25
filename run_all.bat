@echo off
chcp 65001 >nul
echo ============================================================
echo  消融实验批量训练与评估脚本 (统一1 epoch, 500条测试)
echo  实验: QV, QKV, QKVO
echo ============================================================
echo.

REM ============ 生成测试集 ============

echo [0/3] 生成500条测试集...
python train_code/generate_test_dataset.py
if %errorlevel% neq 0 (
    echo [ERROR] 测试集生成失败!
    pause
    exit /b 1
)
echo.

REM ============ 训练阶段 ============

echo [1/3] 训练 LoRA QV (q_proj, v_proj) - 1 epoch...
echo 开始时间: %date% %time%
python train_code/qwen_lora_7k_qv.py
if %errorlevel% neq 0 (
    echo [ERROR] LoRA QV 失败! 错误码: %errorlevel%
)
echo QV 完成时间: %date% %time%
echo.

echo [2/3] 训练 LoRA QKV (q_proj, k_proj, v_proj) - 1 epoch...
echo 开始时间: %date% %time%
python train_code/qwen_lora_7k_qkv.py
if %errorlevel% neq 0 (
    echo [ERROR] LoRA QKV 失败! 错误码: %errorlevel%
)
echo QKV 完成时间: %date% %time%
echo.

echo [3/3] 训练 LoRA QKVO (q_proj, k_proj, v_proj, o_proj) - 1 epoch...
echo 开始时间: %date% %time%
python train_code/qwen_lora_7k_qkvo.py
if %errorlevel% neq 0 (
    echo [ERROR] LoRA QKVO 失败! 错误码: %errorlevel%
)
echo QKVO 完成时间: %date% %time%
echo.

REM ============ 评估阶段 ============

echo ============================================================
echo  开始GPT-4评估...
echo ============================================================
echo.

echo [评估] 运行消融实验GPT-4评估...
python evaluate/src/evaluate_ablation.py
if %errorlevel% neq 0 (
    echo [ERROR] 评估失败! 错误码: %errorlevel%
)
echo.

echo [对比] 生成训练指标对比图表...
python evaluate/src/compare_ablation.py
if %errorlevel% neq 0 (
    echo [ERROR] 对比分析失败! 错误码: %errorlevel%
)
echo.

echo ============================================================
echo  全部流程完成! %date% %time%
echo ============================================================
pause
