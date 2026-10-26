param([switch]$SkipDemo, [int]$Epochs = 100)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Run setup.ps1 first, or create a Python 3.12 environment using requirements.txt.'
}
$env:PYTHONIOENCODING = 'utf-8'
$taskArgs = @('run_reproduction.py', '--epochs', "$Epochs")
if ($SkipDemo) { $taskArgs += '--skip-demo' }
& $taskPython @taskArgs
if ($LASTEXITCODE -ne 0) { throw 'Reproduction failed. See results/logs.' }
