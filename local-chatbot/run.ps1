# ============================================
# local-chatbot - 一键快速训练脚本
# ============================================
# 适用于: GTX 1660 Super 6GB
# ============================================

Write-Host ""
Write-Host "==============================================" -ForegroundColor Magenta
Write-Host "  🚀 Local Chatbot - 一键快速训练" -ForegroundColor Magenta
Write-Host "==============================================" -ForegroundColor Magenta
Write-Host ""

# 切换到脚本所在目录
Set-Location $PSScriptRoot

# 参数
$Action = if ($args.Count -gt 0) { $args[0] } else { "demo" }

switch ($Action) {
    "demo" {
        Write-Host "模式: 快速演示 (小样本，约5分钟可完成)" -ForegroundColor Yellow
        Write-Host ""
        
        # 生成示例数据
        Write-Host "[1/3] 生成示例数据..." -ForegroundColor Cyan
        & python prepare_data.py create-sample --num 50
        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ 数据准备失败" -ForegroundColor Red
            exit 1
        }
        
        # 训练
        Write-Host ""
        Write-Host "[2/3] 开始训练 (50条数据, 1 epoch)..." -ForegroundColor Cyan
        & python train.py --num_samples 50 --epochs 1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ 训练失败" -ForegroundColor Red
            exit 1
        }
        
        # 测试
        Write-Host ""
        Write-Host "[3/3] 快速测试模型..." -ForegroundColor Cyan
        & python inference.py --prompt "你好，介绍一下你自己"
    }
    
    "full" {
        Write-Host "模式: 完整训练 (Alpaca数据集, 约3小时)" -ForegroundColor Yellow
        Write-Host ""
        
        # 下载数据
        Write-Host "[1/3] 下载 Alpaca 数据集..." -ForegroundColor Cyan
        & python prepare_data.py download-alpaca
        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ 数据下载失败" -ForegroundColor Red
            exit 1
        }
        
        # 训练
        Write-Host ""
        Write-Host "[2/3] 开始训练 (Alpaca, 3 epochs)..." -ForegroundColor Cyan
        & python train.py
        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ 训练失败" -ForegroundColor Red
            exit 1
        }
        
        Write-Host ""
        Write-Host "[3/3] 训练完成! 运行 .\run.ps1 chat 开始对话" -ForegroundColor Green
    }
    
    "chat" {
        Write-Host "模式: 交互对话" -ForegroundColor Yellow
        & python inference.py
    }
    
    "web" {
        Write-Host "模式: Web 界面" -ForegroundColor Yellow
        & python inference.py --web
    }
    
    default {
        Write-Host "用法: .\run.ps1 [command]" -ForegroundColor White
        Write-Host ""
        Write-Host "命令:" -ForegroundColor White
        Write-Host "  demo    快速演示 (50条样本, 约5分钟)" -ForegroundColor Gray
        Write-Host "  full    完整训练 (Alpaca数据集, 约3小时)" -ForegroundColor Gray
        Write-Host "  chat    交互对话模式" -ForegroundColor Gray
        Write-Host "  web     Web界面模式" -ForegroundColor Gray
    }
}

Write-Host ""
