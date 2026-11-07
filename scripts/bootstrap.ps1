# One-time project setup for Windows PowerShell: Git + DVC + DagsHub remote. Reads .env
#   powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path .env)) { Copy-Item .env.example .env; Write-Host ">> Created .env - fill in your DagsHub details, then re-run."; exit 1 }
Get-Content .env | Where-Object { $_ -match '^\s*[^#].*=' } | ForEach-Object {
    $k, $v = $_ -split '=', 2; [Environment]::SetEnvironmentVariable($k.Trim(), $v.Trim(), "Process") }

if (-not (Test-Path .git)) { git init -b main }
if (-not (Test-Path .dvc)) { dvc init }

if ($env:DAGSHUB_USERNAME -and $env:DAGSHUB_USERNAME -ne "your-dagshub-username") {
    $remote = "https://dagshub.com/$($env:DAGSHUB_USERNAME)/$($env:DAGSHUB_REPO).dvc"
    dvc remote add -d -f origin $remote
    dvc remote modify origin --local auth basic
    dvc remote modify origin --local user $env:DAGSHUB_USERNAME
    dvc remote modify origin --local password $env:DAGSHUB_TOKEN
    Write-Host ">> DVC remote 'origin' -> $remote"
} else { Write-Host ">> DagsHub not configured in .env - skipping DVC remote." }
Write-Host ">> Done. Next: dvc repro   (see docs\SETUP.md)"
