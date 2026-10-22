# One lane of the final protocol on Windows: for each seed -> train (resumable) -> evaluate -> CDM (if docker works).
#   . C:\Coding\Python\envs\mathocr\activate.ps1
#   scripts\ops\final_lane.ps1 -Seeds 1,2,3 [-Config configs\final_v2.yaml] [-Workers 4]
# Run dirs: runs\<config name>_s<seed> in the repo (checkpoints are git-ignored); run.json records the git commit,
# so run it from a clean checkout of the commit to report.  Re-running continues an interrupted seed (--resume).
# Before a long run: keep the display on while plugged in (the GPU power limit drops 175 -> 55 W when it turns off;
# HANDOFF section 7) -- `powercfg /change monitor-timeout-ac 0`.
param(
    [int[]]$Seeds = @(1),
    [string]$Config = "configs\final_v2.yaml",
    [int]$Workers = 4
)
$ErrorActionPreference = "Stop"
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
if (-not $env:VIRTUAL_ENV) { . C:\Coding\Python\envs\mathocr\activate.ps1 | Out-Null }
$env:PYTHONPATH = "$PWD\src"  # this checkout's code, ahead of the editable install of the main repo
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
# Expandable segments for training and evaluation: without them the cache fragments over the many batch
# shapes and, with the driver's sysmem fallback, spills into system RAM (evaluation 3x slower).
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$name = [IO.Path]::GetFileNameWithoutExtension($Config)
foreach ($seed in $Seeds) {
    $run = "runs\${name}_s$seed"
    New-Item -ItemType Directory -Force $run | Out-Null
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') seed ${seed}: train"
    python -m mathocr.train --config $Config --seed $seed --run-dir $run --resume --set train.workers=$Workers
    if ($LASTEXITCODE -ne 0) { throw "train failed seed $seed" }
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') seed ${seed}: evaluate"
    python -m mathocr.evaluate --run-dir $run
    if ($LASTEXITCODE -ne 0) { throw "evaluate failed seed $seed" }
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') seed ${seed}: cdm"
    docker image inspect mathocr-cdm *> $null
    if ($LASTEXITCODE -eq 0) {
        python -m mathocr.cdm --run-dir $run --cpus 4
        if ($LASTEXITCODE -ne 0) { Write-Warning "cdm failed seed $seed" }
    } else {
        Write-Warning "docker image mathocr-cdm missing: CDM skipped (docker build -f docker/cdm.Dockerfile -t mathocr-cdm docker/)"
    }
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') seed ${seed}: done"
}
