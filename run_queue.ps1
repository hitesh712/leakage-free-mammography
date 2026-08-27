# Runs the remaining revision experiments back-to-back once the 10-seed main run
# finishes, so the GPU is never idle. Each stage logs to revision\logs\.
$ErrorActionPreference = 'Continue'
$root = 'D:\pipeline2_mammogram\pipeline2_mammogram'
$rev  = Join-Path $root 'revision'
$tf   = Join-Path $root '.venv_tfdml\Scripts\python.exe'
$pt   = Join-Path $root 'mammo_clip_dlora\.venv\Scripts\python.exe'
Set-Location $rev

function Wait-ForPid([int]$procId) {
    while ($true) {
        $p = Get-Process -Id $procId -ErrorAction SilentlyContinue
        if (-not $p) { return }
        Start-Sleep -Seconds 20
    }
}

function Stage([string]$name, [string]$exe, [string[]]$argv) {
    $log = Join-Path $rev "logs\$name.log"
    "=== $name starting $(Get-Date -Format 'HH:mm:ss') ===" | Tee-Object -FilePath $log
    & $exe @argv 2>&1 | Tee-Object -FilePath $log -Append
    "=== $name finished $(Get-Date -Format 'HH:mm:ss') exit=$LASTEXITCODE ===" | Tee-Object -FilePath $log -Append
}

# The main run owns the GPU until it exits.
if ($args.Count -ge 1) { Wait-ForPid ([int]$args[0]) }

$env:TF_CPP_MIN_LOG_LEVEL = '3'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'

Stage 'rev03_corrected' $tf @('rev03_corrected_split.py', '--seeds', '10')
Stage 'rev06_perturbation' $tf @('rev06_perturbation.py', '--seeds', '3', '--repeats', '3')
Stage 'rev07_gradcam' $tf @('rev07_gradcam.py', '--seeds', '3', '--n', '400')
Stage 'rev05_wholeimage' $pt @('rev05_wholeimage.py', '--seeds', '3', '--sizes', '224', '512')
Stage 'rev04_arch' $pt @('rev04_arch_ablation.py', '--seeds', '3')

"ALL STAGES DONE $(Get-Date -Format 'HH:mm:ss')" | Tee-Object -FilePath (Join-Path $rev 'logs\queue_done.log')
