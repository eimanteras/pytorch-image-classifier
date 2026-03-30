Param(
    [switch]$Gpu
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "[INFO] Kuriama virtuali aplinka (.venv)..."
    py -3.12 -m venv .venv
}

if (-not (Test-Path $venvPython)) {
    throw "Nepavyko rasti .venv\\Scripts\\python.exe po venv sukurimo. Patikrinkite, ar veikia 'py -3.12'."
}

Write-Host "[INFO] Atnaujinamas pip..."
& $venvPython -m pip install --upgrade pip

if ($Gpu) {
    Write-Host "[INFO] Diegiami PyTorch GPU paketai (cu126)..."
    & $venvPython -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
}

Write-Host "[INFO] Diegiamos projekto priklausomybės..."
& $venvPython -m pip install -r requirements.txt

Write-Host "[INFO] Paleidziamas modelis..."
& $venvPython image_classifier.py
