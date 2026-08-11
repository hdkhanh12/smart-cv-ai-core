param(
    [Parameter(Mandatory = $true)]
    [string[]]$Path
)

$ErrorActionPreference = "Stop"

foreach ($resultPath in $Path) {
    if (-not (Test-Path -LiteralPath $resultPath -PathType Leaf)) {
        Write-Error "Không tìm thấy file: $resultPath"
        continue
    }

    # ReadAllText uses UTF-8 for these generated JSON files and avoids differences
    # between the Encoding enum values in Windows PowerShell versions.
    $resolvedPath = (Resolve-Path -LiteralPath $resultPath).Path
    $result = [System.IO.File]::ReadAllText($resolvedPath) | ConvertFrom-Json

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor DarkGray
    Write-Host "File:       $resolvedPath" -ForegroundColor Cyan
    Write-Host "Candidate:  $($result.profile.candidateName)"
    Write-Host "Headline:   $($result.profile.headline)"
    Write-Host "Experience: $($result.profile.totalExperienceYears) years"
    Write-Host "Summary:    $($result.summary.text)"

    foreach ($score in $result.scores) {
        Write-Host ""
        Write-Host "------------------------------------------------------------" -ForegroundColor DarkGray
        Write-Host "$($score.name)" -ForegroundColor Green
        Write-Host "Status:     $($score.status)"
        if ($null -eq $score.score) {
            Write-Host "Score:      suppressed (manual review / insufficient data)"
        }
        else {
            Write-Host "Score:      $($score.score) / 100"
        }
        Write-Host "Confidence: $($score.confidence)"
        Write-Host ""
        Write-Host "Breakdown:" -ForegroundColor Cyan
        $score.breakdown | Format-List

        $warnings = @($score.warnings)
        Write-Host "Warnings:" -ForegroundColor Yellow
        if ($warnings.Count) {
            $warnings |
                Select-Object code, message |
                Format-Table -Wrap -AutoSize
        }
        else {
            Write-Host "Không có warning."
        }
    }

    $pipelineWarnings = @($result.warnings)
    Write-Host ""
    Write-Host "Pipeline warnings: $($pipelineWarnings.Count)" -ForegroundColor Yellow
    if ($pipelineWarnings.Count) {
        $pipelineWarnings |
            Select-Object code, stage, message |
            Format-Table -Wrap -AutoSize
    }
}
