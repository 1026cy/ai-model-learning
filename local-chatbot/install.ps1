# ============================================
# local-chatbot - 本地对话模型训练项目
# ============================================
# 一键安装脚本 (PowerShell)
# ============================================

Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  Local Chatbot - 环境安装" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""

# 检查 Python
Write-Host "[1/4] 检查 Python..." -ForegroundColor Yellow
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "❌ Python 未安装！请先安装 Python 3.10+" -ForegroundColor Red
    exit 1
}
$pyVersion = & python --version
Write-Host "   ✅ $pyVersion" -ForegroundColor Green

# 检查 CUDA
Write-Host ""
Write-Host "[2/4] 检查 CUDA..." -ForegroundColor Yellow
$cudaCmd = Get-Command nvcc -ErrorAction SilentlyContinue
if ($cudaCmd) {
    Write-Host "   ✅ CUDA 已安装" -ForegroundColor Green
} else {
    Write-Host "   ⚠️ 未检测到 CUDA，将使用 CPU (速度较慢)" -ForegroundColor Yellow
}

# 升级 pip
Write-Host ""
Write-Host "[3/4] 升级 pip..." -ForegroundColor Yellow
& python -m pip install --upgrade pip
Write-Host "   ✅ pip 已升级" -ForegroundColor Green

# 安装 PyTorch (CUDA 11.8 版本)
Write-Host ""
Write-Host "[4/4] 安装依赖包..." -ForegroundColor Yellow
Write-Host "   安装 PyTorch (CUDA 11.8)..." -ForegroundColor DarkGray

$cudaCheck = & nvidia-smi 2>$null
if ($LASTEXITCODE -eq 0) {
    # 有 NVIDIA GPU，安装 CUDA 版本
    & pip install torch==2.3.0+cu118 torchvision==0.18.0+cu118 torchaudio==2.3.0+cu118 `
        --index-url https://download.pytorch.org/whl/cu118
} else {
    # 无 GPU，安装 CPU 版本
    & pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
}

# 安装 bitsandbytes (Windows 需要特殊处理)
Write-Host "   安装 bitsandbytes..." -ForegroundColor DarkGray
& pip install bitsandbytes>=0.41.0

# 安装其他依赖
Write-Host "   安装 HuggingFace 生态..." -ForegroundColor DarkGray
& pip install transformers>=4.40.0 datasets accelerate peft trl>=0.8.0 evaluate tokenizers

Write-Host "   安装其他依赖..." -ForegroundColor DarkGray
& pip install pyyaml jsonlines pandas tqdm sentencepiece psutil safetensors

# 可选: 安装推理依赖
Write-Host "   安装推理依赖 (gradio)..." -ForegroundColor DarkGray
& pip install gradio

Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  ✅ 安装完成！" -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "快速开始:" -ForegroundColor White
Write-Host "  # 1. 生成示例数据 (快速测试)" -ForegroundColor Gray
Write-Host "  python prepare_data.py create-sample --num 50"
Write-Host ""
Write-Host "  # 2. 开始训练" -ForegroundColor Gray
Write-Host "  python train.py --num_samples 50"
Write-Host ""
Write-Host "  # 3. 与模型对话" -ForegroundColor Gray
Write-Host "  python inference.py"
Write-Host ""
Write-Host "  # 4. 启动 Web 界面" -ForegroundColor Gray
Write-Host "  python inference.py --web"
Write-Host ""
