$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    Write-Host 'First run: python -m venv --system-site-packages .venv'
    Write-Host 'Then: .\.venv\Scripts\python.exe -m pip install -r requirements-backend.txt'
    exit 1
}
& $pythonPath tools/serve.py
