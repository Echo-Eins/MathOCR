# Stage 3 probe queue on base B2 (B1 + TriScan hv), one GPU job at a time.  Reads the queue file (first argument,
# default queue3.txt; lines name|overrides[|PROBE_ROOT], B2 base overrides implied; '#' lines ignored) on every round,
# runs the first job not yet started, exits when none is left -- jobs can be appended while it runs.
param([string]$QueueFile = "queue_micro.txt")
$env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')
. C:\Coding\Python\envs\mathocr\activate.ps1 | Out-Null
Set-Location C:\Coding\Python\MathOCR_queue  # a pinned worktree: edits to main never reach a running job
$log = Split-Path -Parent $MyInvocation.MyCommand.Path
$f = "$log\stage2.log"
$queue = "$log\$QueueFile"
$env:PROBE_ROOT = 'D:\MathOCR\probe_fonts70'
$base = @('train.optimizer=muon', 'train.lr=6.0e-4', 'model.enc_stride=8x4', 'image.target_glyph_px=18.0',
          'model.attn_kernel=triton', 'image.max_h=384', 'image.max_w=1536', 'model.max_len=512',
          'augment.p_stack=0.1', 'augment.stack_style=rl',
          'model.encoder=triscan', 'model.triscan_streams=hv', 'model.triscan_exchange=false')
while ($true) {
    $started = @(Get-Content $f | ForEach-Object { if ($_ -match ' start (\S+)$') { $Matches[1] } })
    $next = Get-Content $queue | Where-Object { $_ -and -not $_.StartsWith('#') } |
        Where-Object { $started -notcontains $_.Split('|')[0].Trim() } | Select-Object -First 1
    if (-not $next) { break }
    $parts = $next.Split('|')
    $name = $parts[0].Trim()
    $over = @($parts[1].Trim() -split '\s+' | Where-Object { $_ })
    $env:PROBE_ROOT = if ($parts.Count -gt 2 -and $parts[2].Trim()) { $parts[2].Trim() } else { 'D:\MathOCR\probe_fonts70' }
    $t0 = Get-Date
    "$(Get-Date -Format 'HH:mm:ss') start $name" | Out-File -Append -Encoding utf8 $f
    if ($name.StartsWith('cmd_')) {  # a GPU script in the same one-job-at-a-time lane: the field is its argument list
        python @over *> "$log\$name.log"
    } else {
        $seed = if ($name -match '_s(\d+)$') { $Matches[1] } else { '1' }  # the seed is the name's _sN suffix
        python scripts/probe.py --name $name --seed $seed --epochs 1 --set @($base + $over) *> "$log\probe_$name.log"
    }
    "$(Get-Date -Format 'HH:mm:ss') done $name exit $LASTEXITCODE in $([math]::Round(((Get-Date) - $t0).TotalMinutes, 1)) min" |
        Out-File -Append -Encoding utf8 $f
    if ($name.StartsWith('cmd_pytest') -and $LASTEXITCODE -ne 0) {  # never train on code whose tests fail
        "$(Get-Date -Format 'HH:mm:ss') tests failed: queue stopped" | Out-File -Append -Encoding utf8 $f
        break
    }
}
"$(Get-Date -Format 'HH:mm:ss') queue empty" | Out-File -Append -Encoding utf8 $f
