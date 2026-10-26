# Quick multi-row check (~15-20 min): 500-step fine-tunes of the stand's final config (win_b3_final6_s1) -- the
# same data (control of the fine-tune itself), + rotation 0.3 (the vertical path), + the TeX long blocks (data) --
# each scored on the full-CPE references with >= 6 rows (multirow_eval.py).
$env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')
. C:\Coding\Python\envs\mathocr\activate.ps1 | Out-Null
Set-Location C:\Coding\Python\MathOCR_queue
$env:PYTHONPATH = "$PWD\src"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$log = Split-Path -Parent $MyInvocation.MyCommand.Path
$f = "$log\stage2.log"
$init = 'D:\MathOCR\probe_fonts_many_b\runs\win_b3_final6_s1\checkpoints\best.pt'
$common = @('train.optimizer=muon', 'train.lr=3.0e-4', 'model.enc_stride=8x4', 'image.target_glyph_px=18.0',
            'model.attn_kernel=triton', 'image.max_h=384', 'image.max_w=1536', 'model.max_len=768',
            'augment.p_stack=0.1', 'augment.stack_style=rl', 'model.encoder=triscan', 'model.triscan_streams=hv',
            'model.triscan_exchange=false', 'train.muon_polar=true', 'train.muon_neuron_norm=true',
            "train.init_from=$init", 'train.epochs=1', 'train.max_steps_per_epoch=500', 'train.warmup_steps=50',
            'train.val_decode_samples=50', 'train.workers=2', 'train.log_every=50')
$runs = @(@{ name = 'ft_ctrl'; data = 'D:\MathOCR\probe_fonts_many_b\data20k'; extra = @() },
          @{ name = 'ft_rot'; data = 'D:\MathOCR\probe_fonts_many_b\data20k'; extra = @('augment.p_rotate=0.3') },
          @{ name = 'ft_long'; data = 'D:\MathOCR\probe_long_b\data20k'; extra = @() })
foreach ($r in $runs) {
    $dir = "D:\MathOCR\ft_runs\$($r.name)"
    "$(Get-Date -Format 'HH:mm:ss') start $($r.name)" | Out-File -Append -Encoding utf8 $f
    python -m mathocr.train --config configs\base.yaml --seed 1 --run-dir $dir --set @($common + @("data.out_dir=$($r.data)") + $r.extra) *> "$log\$($r.name)_train.log"
    python scripts\diagnostics\multirow_eval.py --name $($r.name) --ckpt "$dir\checkpoints\last.pt" --min-rows 6 *> "$log\$($r.name)_eval.log"
    "$(Get-Date -Format 'HH:mm:ss') done $($r.name) exit $LASTEXITCODE" | Out-File -Append -Encoding utf8 $f
}
"$(Get-Date -Format 'HH:mm:ss') queue empty (ft multirow)" | Out-File -Append -Encoding utf8 $f
