param([string]$Python = '')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not $Python) {
    $Python = (Get-Command python -ErrorAction Stop).Source
}
& $Python -c 'import sys; assert sys.version_info >= (3, 12), "Python 3.12 or later is required"'
if ($LASTEXITCODE -ne 0) { throw 'Pass a Python 3.12 executable using -Python.' }
New-Item -ItemType Directory -Force -Path 'tmp' | Out-Null
$env:TEMP = Join-Path $PSScriptRoot 'tmp'
$env:TMP = $env:TEMP
$env:PYTHONIOENCODING = 'utf-8'
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & $Python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed.' }
}
& '.\.venv\Scripts\python.exe' -m pip install --timeout 120 --cache-dir '.\tmp\pip-cache' -r requirements-lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
Write-Output 'Ready. Run .\run_reproduction.ps1'
