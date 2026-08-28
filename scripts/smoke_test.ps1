$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Missing .venv. Create it and install the project first."
}
& $python -m asgcn_recon.smoke --workspace (Join-Path $projectRoot "data\smoke")
& $python -m pytest -q
