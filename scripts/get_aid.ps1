param(
    [string]$Destination = ".\data\EventAid-R",
    [string[]]$Scenes = @()
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$manifestPath = Join-Path $projectRoot "manifests\eventaid_r.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$destinationPath = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $Destination))
New-Item -ItemType Directory -Path $destinationPath -Force | Out-Null

$selected = $manifest.files
if ($Scenes.Count -gt 0) {
    $selected = $manifest.files | Where-Object { $Scenes -contains $_.scene }
    $missing = $Scenes | Where-Object { $_ -notin $selected.scene }
    if ($missing.Count -gt 0) {
        throw "Unknown scene(s): $($missing -join ', ')"
    }
}

Write-Host "EventAid-R destination: $destinationPath"
Write-Host "Selected scenes: $($selected.scene -join ', ')"
foreach ($item in $selected) {
    $target = Join-Path $destinationPath ($item.scene + ".zip")
    Write-Host "Downloading $($item.scene) ($($item.size))"
    & curl.exe -L --fail --retry 5 --retry-delay 3 --continue-at - --output $target $item.url
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed: $($item.scene)"
    }
}
Write-Host "Done. ZIP files are read directly; do not extract them."
