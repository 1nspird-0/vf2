# ViralFlip Quick Start Script for Windows
# =========================================
# Optimized for RTX 5070 (12GB VRAM) on Vast.ai
#
# Usage:
#   .\scripts\quick_start.ps1              # Full training
#   .\scripts\quick_start.ps1 -SkipData    # Skip downloads
#   .\scripts\quick_start.ps1 -Resume      # Resume training

param(
    [switch]$SkipData,
    [switch]$Resume,
    [switch]$QuickTest,
    [int]$ParallelDownloads = 4
)

$ErrorActionPreference = "Stop"

Write-Host @"

╔══════════════════════════════════════════════════════════════╗
║          VIRALFLIP - MAXIMUM ACCURACY TRAINING               ║
║                  Optimized for RTX 5070                      ║
╚══════════════════════════════════════════════════════════════╝

"@ -ForegroundColor Cyan

# Check Python
Write-Host "[CHECK] Python version..." -ForegroundColor Yellow
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Python not found" -ForegroundColor Red
    exit 1
}

# Check GPU
Write-Host "`n[CHECK] GPU availability..." -ForegroundColor Yellow
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Install package
Write-Host "`n[SETUP] Installing ViralFlip..." -ForegroundColor Yellow
pip install -e . -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Installation failed" -ForegroundColor Red
    exit 1
}

# Download data
if (-not $SkipData) {
    Write-Host "`n[DATA] Downloading datasets (with resume support)..." -ForegroundColor Yellow
    python scripts/download_more_data.py --full --parallel $ParallelDownloads
} else {
    Write-Host "`n[SKIP] Data download skipped" -ForegroundColor Gray
}

# Generate synthetic data
if ($QuickTest) {
    Write-Host "`n[DATA] Generating quick test data (100 users)..." -ForegroundColor Yellow
    python scripts/make_synthetic_episodes.py --output data/synthetic --n-users 100 --days 90
} else {
    Write-Host "`n[DATA] Generating large-scale synthetic data (500 users x 180 days)..." -ForegroundColor Yellow
    python scripts/make_synthetic_episodes.py --max-accuracy --output data/synthetic_large
}

# Training
$dataPath = if ($QuickTest) { "data/synthetic" } else { "data/synthetic_large" }
$config = if ($QuickTest) { "configs/default.yaml" } else { "configs/high_performance.yaml" }

$trainCmd = "python scripts/train.py --config $config --data $dataPath"
if ($Resume) {
    $trainCmd += " --resume runs/latest/checkpoint.pt"
}

Write-Host "`n[TRAIN] Starting training..." -ForegroundColor Yellow
Write-Host "Command: $trainCmd" -ForegroundColor Gray
Invoke-Expression $trainCmd

Write-Host @"

╔══════════════════════════════════════════════════════════════╗
║                    TRAINING COMPLETE!                        ║
╚══════════════════════════════════════════════════════════════╝

Next steps:
  - Check runs/ folder for results
  - Run: python scripts/evaluate.py --run_dir runs/<your_run>
  - Run: python scripts/run_ablations.py --run_dir runs/<your_run>

"@ -ForegroundColor Green

