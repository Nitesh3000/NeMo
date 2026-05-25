# ============================================================================
# Voice Assistant — one-shot setup  (Windows / PowerShell)
#
# Run from the cloned repo root:   .\setup.ps1
#
# What it does:
#   1. Verifies Python 3.13 is installed
#   2. Creates virtual environment (venv313/)
#   3. Installs PyTorch CPU + all dependencies
#   4. Installs editdistance + nv_one_logger stubs (C-ext / NVIDIA-internal pkgs)
#   5. Patches NeMo source for Windows / lightning-2.6 compatibility
#   6. Downloads Piper voice model
#   7. Creates .env file (if missing)
# ============================================================================

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT

Write-Host "`n=== 1/7  Checking prerequisites ===" -ForegroundColor Cyan
$py = (py -3.13 -c "import sys; print(sys.executable)") 2>$null
if (-not $py) { throw "Python 3.13 not found. Install from python.org (https://www.python.org/downloads/release/python-3135/)" }
Write-Host "  Python 3.13: $py"

Write-Host "`n=== 2/7  Creating virtual environment ===" -ForegroundColor Cyan
if (-not (Test-Path "venv313")) {
    py -3.13 -m venv venv313
    Write-Host "  Created venv313/"
} else {
    Write-Host "  venv313/ already exists"
}
$PIP    = "$ROOT\venv313\Scripts\pip.exe"
$PYTHON = "$ROOT\venv313\Scripts\python.exe"

Write-Host "`n=== 3/7  Installing PyTorch CPU + dependencies ===" -ForegroundColor Cyan
& $PIP install --upgrade pip | Out-Null
& $PIP install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
& $PIP install -r requirements.txt
& $PIP install -e . --no-deps    # local NeMo source as editable package

Write-Host "`n=== 4/7  Installing stubs for unavailable packages ===" -ForegroundColor Cyan
$SP = "$ROOT\venv313\Lib\site-packages"

# editdistance — pure-Python (C build fails without MSVC on Windows)
@'
def eval(a, b):
    """Levenshtein distance (pure-Python stub)."""
    if len(a) < len(b): return eval(b, a)
    if len(b) == 0: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(ca != cb)))
        prev = curr
    return prev[-1]
'@ | Out-File -Encoding utf8 "$SP\editdistance.py" -Force
Write-Host "  editdistance stub: OK"

# nv_one_logger — NVIDIA-internal telemetry (training-only, safe to stub for inference)
$NV = "$SP\nv_one_logger"
New-Item -ItemType Directory -Force "$NV\api" | Out-Null
New-Item -ItemType Directory -Force "$NV\training_telemetry\api" | Out-Null
New-Item -ItemType Directory -Force "$NV\training_telemetry\integration" | Out-Null
foreach ($p in @("$NV\__init__.py","$NV\api\__init__.py","$NV\training_telemetry\__init__.py","$NV\training_telemetry\api\__init__.py","$NV\training_telemetry\integration\__init__.py")) {
    '' | Out-File -Encoding utf8 $p -Force
}
"class OneLoggerConfig:`n    def __init__(self, **kw): pass" | Out-File -Encoding utf8 "$NV\api\config.py" -Force
"def on_app_start(*a, **kw): pass" | Out-File -Encoding utf8 "$NV\training_telemetry\api\callbacks.py" -Force
"class TrainingTelemetryConfig:`n    def __init__(self, **kw): pass" | Out-File -Encoding utf8 "$NV\training_telemetry\api\config.py" -Force
@'
class _P:
    _inst = None
    def with_base_config(self, *a, **kw): return self
    def with_export_config(self, *a, **kw): return self
    def configure_provider(self, *a, **kw): return self
    def set_training_telemetry_config(self, *a, **kw): pass
    class _C: telemetry_config = None
    config = _C()

class TrainingTelemetryProvider:
    @staticmethod
    def instance():
        if _P._inst is None: _P._inst = _P()
        return _P._inst
'@ | Out-File -Encoding utf8 "$NV\training_telemetry\api\training_telemetry_provider.py" -Force
"class TimeEventCallback:`n    def __init__(self, *a, **kw): pass" | Out-File -Encoding utf8 "$NV\training_telemetry\integration\pytorch_lightning.py" -Force
Write-Host "  nv_one_logger stub: OK"

Write-Host "`n=== 5/7  Patching NeMo source for Windows + lightning-2.6 ===" -ForegroundColor Cyan
function Patch-File($path, $old, $new) {
    if (-not (Test-Path $path)) { return }
    $content = Get-Content $path -Raw
    if ($content -match [regex]::Escape($old)) {
        $content = $content -replace [regex]::Escape($old), $new
        Set-Content -Path $path -Value $content -NoNewline
        Write-Host "  Patched: $path"
    } else {
        Write-Host "  Already patched: $path"
    }
}

Patch-File "nemo\utils\exp_manager.py" `
    "from lightning.pytorch.loggers import MLFlowLogger, NeptuneLogger, TensorBoardLogger, WandbLogger" `
    "from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger, WandbLogger`r`ntry:`r`n    from lightning.pytorch.loggers import NeptuneLogger`r`nexcept ImportError:`r`n    NeptuneLogger = None"

Patch-File "nemo\collections\asr\parts\preprocessing\perturb.py" `
    "from nemo.utils import webdataset as wds" "import webdataset as wds"
Patch-File "nemo\collections\asr\data\audio_to_text.py" `
    "from nemo.utils import webdataset as wds" "import webdataset as wds"
Patch-File "nemo\collections\asr\data\audio_to_label.py" `
    "from nemo.utils import webdataset as wds" "import webdataset as wds"

Write-Host "`n=== 6/7  Downloading Piper voice model ===" -ForegroundColor Cyan
New-Item -ItemType Directory -Force "voices" | Out-Null
$voiceURL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"
foreach ($f in @("en_US-lessac-medium.onnx", "en_US-lessac-medium.onnx.json")) {
    $dst = "voices\$f"
    if (-not (Test-Path $dst)) {
        Write-Host "  Downloading $f ..."
        Invoke-WebRequest "$voiceURL/$f" -OutFile $dst -UseBasicParsing
    } else {
        Write-Host "  Already exists: $f"
    }
}

Write-Host "`n=== 7/7  Setting up .env ===" -ForegroundColor Cyan
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "  Created .env from template." -ForegroundColor Yellow
    Write-Host "  >>> EDIT .env and set GEMINI_API_KEY <<<" -ForegroundColor Yellow
} else {
    Write-Host "  .env already exists"
}

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Edit .env  -> set GEMINI_API_KEY (get one at https://aistudio.google.com/apikey)"
Write-Host "  2. Run server: .\venv313\Scripts\python.exe server.py"
Write-Host "  3. Open browser at: http://127.0.0.1:8000"
Write-Host ""
