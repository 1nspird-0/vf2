# ViralFlip Essential Data Downloads
# Run: .\scripts\download_essential.ps1

$ErrorActionPreference = "Continue"
$ProgressPreference = 'SilentlyContinue'  # Speeds up downloads

$dataDir = "data\real"
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null

Write-Host "`n======================================================" -ForegroundColor Cyan
Write-Host "DOWNLOADING ESSENTIAL DATASETS FOR VIRALFLIP" -ForegroundColor Cyan  
Write-Host "======================================================`n" -ForegroundColor Cyan

# -----------------------------------------------------------------------------
# 1. COUGHVID - 25,000 cough samples (LARGEST COUGH DATASET)
# -----------------------------------------------------------------------------
Write-Host "[1/5] COUGHVID (25k coughs) - from Zenodo" -ForegroundColor Yellow

$coughvidDir = "$dataDir\coughvid"
New-Item -ItemType Directory -Force -Path $coughvidDir | Out-Null

# Zenodo record 7024894 - latest version
$coughvidUrl = "https://zenodo.org/api/records/7024894/files-archive"
$coughvidZip = "$coughvidDir\coughvid.zip"

if (!(Test-Path $coughvidZip)) {
    Write-Host "  Downloading from Zenodo (5GB, be patient)..."
    try {
        Invoke-WebRequest -Uri $coughvidUrl -OutFile $coughvidZip -TimeoutSec 3600
        Write-Host "  SUCCESS!" -ForegroundColor Green
    } catch {
        Write-Host "  Direct failed. Trying alternative..." -ForegroundColor Red
        # Alternative: direct file link
        $altUrl = "https://zenodo.org/records/7024894/files/public_dataset_v5.zip?download=1"
        try {
            Invoke-WebRequest -Uri $altUrl -OutFile $coughvidZip -TimeoutSec 3600
            Write-Host "  SUCCESS with alt URL!" -ForegroundColor Green
        } catch {
            Write-Host "  FAILED: Download manually from https://zenodo.org/records/7024894" -ForegroundColor Red
        }
    }
} else {
    Write-Host "  Already downloaded" -ForegroundColor Green
}

# -----------------------------------------------------------------------------
# 2. FluSense - 120k respiratory events
# -----------------------------------------------------------------------------
Write-Host "`n[2/5] FluSense (120k events) - from GitHub" -ForegroundColor Yellow

$flusenseDir = "$dataDir\flusense"
New-Item -ItemType Directory -Force -Path $flusenseDir | Out-Null

$flusenseUrl = "https://github.com/Forsad/FluSense-data/archive/refs/heads/master.zip"
$flusenseZip = "$flusenseDir\flusense.zip"

if (!(Test-Path $flusenseZip)) {
    Write-Host "  Downloading from GitHub..."
    try {
        Invoke-WebRequest -Uri $flusenseUrl -OutFile $flusenseZip -TimeoutSec 600
        Write-Host "  SUCCESS!" -ForegroundColor Green
        Write-Host "  Extracting..."
        Expand-Archive -Path $flusenseZip -DestinationPath $flusenseDir -Force
    } catch {
        Write-Host "  FAILED: $($_.Exception.Message)" -ForegroundColor Red
    }
} else {
    Write-Host "  Already downloaded" -ForegroundColor Green
}

# -----------------------------------------------------------------------------
# 3. RAVDESS - Voice/speech baseline
# -----------------------------------------------------------------------------
Write-Host "`n[3/5] RAVDESS (Voice quality) - from Zenodo" -ForegroundColor Yellow

$ravdessDir = "$dataDir\ravdess"
New-Item -ItemType Directory -Force -Path $ravdessDir | Out-Null

$ravdessUrl = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
$ravdessZip = "$ravdessDir\ravdess.zip"

if (!(Test-Path $ravdessZip)) {
    Write-Host "  Downloading from Zenodo (215MB)..."
    try {
        Invoke-WebRequest -Uri $ravdessUrl -OutFile $ravdessZip -TimeoutSec 600
        Write-Host "  SUCCESS!" -ForegroundColor Green
        Write-Host "  Extracting..."
        Expand-Archive -Path $ravdessZip -DestinationPath $ravdessDir -Force
    } catch {
        Write-Host "  FAILED: $($_.Exception.Message)" -ForegroundColor Red
    }
} else {
    Write-Host "  Already downloaded" -ForegroundColor Green
}

# -----------------------------------------------------------------------------
# 4. MHEALTH - Multi-modal health sensing
# -----------------------------------------------------------------------------
Write-Host "`n[4/5] MHEALTH (Activity + HR) - from UCI" -ForegroundColor Yellow

$mhealthDir = "$dataDir\mhealth"
New-Item -ItemType Directory -Force -Path $mhealthDir | Out-Null

$mhealthUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases/00319/MHEALTHDATASET.zip"
$mhealthZip = "$mhealthDir\mhealth.zip"

if (!(Test-Path $mhealthZip)) {
    Write-Host "  Downloading from UCI..."
    try {
        Invoke-WebRequest -Uri $mhealthUrl -OutFile $mhealthZip -TimeoutSec 300
        Write-Host "  SUCCESS!" -ForegroundColor Green
        Write-Host "  Extracting..."
        Expand-Archive -Path $mhealthZip -DestinationPath $mhealthDir -Force
    } catch {
        Write-Host "  FAILED: $($_.Exception.Message)" -ForegroundColor Red
    }
} else {
    Write-Host "  Already downloaded" -ForegroundColor Green
}

# -----------------------------------------------------------------------------
# 5. PAMAP2 - Physical Activity Monitoring
# -----------------------------------------------------------------------------
Write-Host "`n[5/5] PAMAP2 (Activity + HR) - from UCI" -ForegroundColor Yellow

$pamap2Dir = "$dataDir\pamap2"
New-Item -ItemType Directory -Force -Path $pamap2Dir | Out-Null

$pamap2Url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip"
$pamap2Zip = "$pamap2Dir\pamap2.zip"

if (!(Test-Path $pamap2Zip)) {
    Write-Host "  Downloading from UCI..."
    try {
        Invoke-WebRequest -Uri $pamap2Url -OutFile $pamap2Zip -TimeoutSec 600
        Write-Host "  SUCCESS!" -ForegroundColor Green
        Write-Host "  Extracting..."
        Expand-Archive -Path $pamap2Zip -DestinationPath $pamap2Dir -Force
    } catch {
        Write-Host "  FAILED: $($_.Exception.Message)" -ForegroundColor Red
    }
} else {
    Write-Host "  Already downloaded" -ForegroundColor Green
}

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
Write-Host "`n======================================================" -ForegroundColor Cyan
Write-Host "DOWNLOAD SUMMARY" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan

$datasets = @(
    @{Name="UCI HAR"; Path="$dataDir\uci_har"; Samples="10k"; Use="Activity recognition"},
    @{Name="ESC-50"; Path="$dataDir\esc50"; Samples="40 coughs"; Use="Cough detection"},
    @{Name="COUGHVID"; Path="$dataDir\coughvid"; Samples="25k coughs"; Use="Cough detection"},
    @{Name="FluSense"; Path="$dataDir\flusense"; Samples="120k events"; Use="Respiratory events"},
    @{Name="RAVDESS"; Path="$dataDir\ravdess"; Samples="7k speech"; Use="Voice quality"},
    @{Name="MHEALTH"; Path="$dataDir\mhealth"; Samples="Multi-modal"; Use="Activity + HR"},
    @{Name="PAMAP2"; Path="$dataDir\pamap2"; Samples="Multi-modal"; Use="Activity + HR"}
)

foreach ($ds in $datasets) {
    if (Test-Path $ds.Path) {
        $size = (Get-ChildItem $ds.Path -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
        Write-Host "[OK] $($ds.Name): $($ds.Samples) - $([math]::Round($size, 1)) MB" -ForegroundColor Green
    } else {
        Write-Host "[XX] $($ds.Name): Not downloaded" -ForegroundColor Red
    }
}

Write-Host "`n------------------------------------------------------" -ForegroundColor White
Write-Host "MANUAL DOWNLOADS NEEDED (registration required):" -ForegroundColor Yellow
Write-Host "  - ExtraSensory: http://extrasensory.ucsd.edu/" -ForegroundColor White
Write-Host "  - StudentLife: https://studentlife.cs.dartmouth.edu/" -ForegroundColor White
Write-Host "  - UBFC-rPPG: https://sites.google.com/view/yaboromance/ubfc-rppg" -ForegroundColor White
Write-Host "======================================================`n" -ForegroundColor Cyan

