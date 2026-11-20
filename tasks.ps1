<#
.SYNOPSIS
  Windows equivalent of the Makefile.  Usage:  ./tasks.ps1 <target>
  Targets: setup, scrape, normalize, index, run, test, lint, eval, docker, docker-run, deploy
#>
param([Parameter(Position = 0)][string]$Target = "help", [string]$BaseUrl = "http://localhost:7860")
$ErrorActionPreference = "Stop"
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"

function Invoke-Step([string]$exe, [string[]]$argList) {
    & $exe @argList
    if ($LASTEXITCODE -ne 0) { throw "'$exe $($argList -join ' ')' failed with exit code $LASTEXITCODE" }
}

switch ($Target) {
    "setup" {
        Invoke-Step "python" @("-m", "venv", ".venv")
        Invoke-Step $py @("-m", "pip", "install", "-r", "requirements.txt", "-r", "requirements-scrape.txt", "-r", "requirements-dev.txt")
    }
    "scrape" {
        Invoke-Step $py @("-m", "scraper.wasalt", "--sample", "4000")
        Invoke-Step $py @("-m", "scraper.darglobal")
    }
    "normalize" { Invoke-Step $py @("-m", "scraper.normalize") }
    "index" { Invoke-Step $py @("-m", "app.build_index", "--write-cache") }
    "run" { Invoke-Step $py @("-m", "uvicorn", "app.main:app", "--port", "7860", "--reload") }
    "test" { Invoke-Step $py @("-m", "pytest", "-q") }
    "lint" { Invoke-Step $py @("-m", "ruff", "check", ".") }
    "eval" { Invoke-Step $py @("eval/run_eval.py", "--base-url", $BaseUrl) }
    "docker" { Invoke-Step "docker" @("build", "-t", "realestate-chatbot", ".") }
    "docker-run" { Invoke-Step "docker" @("run", "--rm", "-p", "7860:7860", "--env-file", ".env", "realestate-chatbot") }
    "deploy" { Invoke-Step $py @("scripts/deploy_hf.py") }
    default {
        Write-Output "Usage: ./tasks.ps1 <setup|scrape|normalize|index|run|test|lint|eval|docker|docker-run|deploy> [-BaseUrl url]"
    }
}
