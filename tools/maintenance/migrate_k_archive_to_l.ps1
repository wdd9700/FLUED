param(
    [string]$SourceRoot = 'K:\FLUED_archive',
    [string]$DestinationRoot = 'L:\FLUED_archive\migrated_from_K_20260712',
    [string]$LogPath = 'L:\FLUED_archive\migration_20260712.log'
)

$ErrorActionPreference = 'Stop'
$source = (Resolve-Path -LiteralPath $SourceRoot).Path
New-Item -ItemType Directory -Path $DestinationRoot -Force | Out-Null
$destination = (Resolve-Path -LiteralPath $DestinationRoot).Path
if (-not $source.StartsWith('K:\') -or -not $destination.StartsWith('L:\')) {
    throw "Unsafe migration roots: source=$source destination=$destination"
}

$names = @(
    'F_checkpoints',
    'ab1_weight_0.3',
    'E_checkpoints',
    'v3_full_model_5080_20260626',
    'cloud_5090_D1_20260610'
)

function Write-MigrationLog([string]$Message) {
    $line = "[{0:yyyy-MM-dd HH:mm:ss}] {1}" -f (Get-Date), $Message
    $line | Tee-Object -FilePath $LogPath -Append
}

function Get-Sha256Hex([string]$Path) {
    $stream = [System.IO.File]::OpenRead($Path)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($sha.ComputeHash($stream))).Replace('-', '')
    } finally {
        $sha.Dispose()
        $stream.Dispose()
    }
}

foreach ($name in $names) {
    $src = Join-Path $source $name
    $dst = Join-Path $destination $name
    if (-not (Test-Path -LiteralPath $src)) {
        if (Test-Path -LiteralPath $dst) {
            Write-MigrationLog "skip-complete $name"
            continue
        }
        throw "Missing both source and destination: $name"
    }
    New-Item -ItemType Directory -Path $dst -Force | Out-Null
    $manifestPath = Join-Path $destination ("_{0}_manifest.json" -f $name)
    if (Test-Path -LiteralPath $manifestPath) {
        $parsedManifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
        $manifest = @()
        foreach ($parsedEntry in $parsedManifest) {
            $manifest += $parsedEntry
        }
    } else {
        $manifest = @(
            Get-ChildItem -LiteralPath $src -Recurse -File | ForEach-Object {
                [pscustomobject]@{
                    relative = $_.FullName.Substring($src.Length).TrimStart('\')
                    length = $_.Length
                    sha256 = ''
                    complete = $false
                }
            }
        )
        $manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
    }
    $sourceBytes = ($manifest | Measure-Object length -Sum).Sum
    Write-MigrationLog ("start {0} files={1} size={2:N2}GB" -f $name, $manifest.Count, ($sourceBytes / 1GB))

    $index = 0
    foreach ($entry in $manifest) {
        $relative = [string]$entry.relative
        $sourceFile = Join-Path $src $relative
        $target = Join-Path $dst $relative
        $parent = Split-Path $target -Parent
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
        if ($entry.complete -and -not (Test-Path -LiteralPath $sourceFile)) {
            $index++
            continue
        }
        if (-not (Test-Path -LiteralPath $sourceFile)) {
            throw "Missing source before verified completion: $sourceFile"
        }
        $partial = "$target.partial"
        $sourceHash = ''
        if ((Test-Path -LiteralPath $partial) -and (Get-Item -LiteralPath $partial).Length -eq [long]$entry.length) {
            Write-MigrationLog "verify-existing-partial $relative"
            $sourceHash = Get-Sha256Hex $sourceFile
            $destinationHash = Get-Sha256Hex $partial
            if ($sourceHash -ne $destinationHash) {
                Remove-Item -LiteralPath $partial -Force
                $sourceHash = ''
            }
        } elseif (Test-Path -LiteralPath $partial) {
            Remove-Item -LiteralPath $partial -Force
        }
        if (-not $sourceHash) {
            $hasher = [System.Security.Cryptography.IncrementalHash]::CreateHash(
                [System.Security.Cryptography.HashAlgorithmName]::SHA256
            )
            $buffer = New-Object byte[] (16MB)
            $input = [System.IO.File]::OpenRead($sourceFile)
            try {
                $output = [System.IO.File]::Create($partial)
                try {
                    while (($read = $input.Read($buffer, 0, $buffer.Length)) -gt 0) {
                        $output.Write($buffer, 0, $read)
                        $hasher.AppendData($buffer, 0, $read)
                    }
                    $output.Flush($true)
                } finally {
                    $output.Dispose()
                }
            } finally {
                $input.Dispose()
            }
            $sourceHash = ([BitConverter]::ToString($hasher.GetHashAndReset())).Replace('-', '')
            $hasher.Dispose()
            if ((Get-Item -LiteralPath $partial).Length -ne [long]$entry.length) {
                throw "Length mismatch after copy: $sourceFile"
            }
            $destinationHash = Get-Sha256Hex $partial
            if ($sourceHash -ne $destinationHash) {
                throw "SHA256 mismatch: $sourceFile"
            }
        }
        Move-Item -LiteralPath $partial -Destination $target -Force
        $entry.sha256 = $sourceHash
        $entry.complete = $true
        $manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
        $resolvedFile = (Resolve-Path -LiteralPath $sourceFile).Path
        if (-not $resolvedFile.StartsWith($src + '\')) {
            throw "Unsafe source file deletion: $resolvedFile"
        }
        Remove-Item -LiteralPath $resolvedFile -Force
        $index++
        Write-MigrationLog ("file {0} {1}/{2} size={3:N2}GB sha256={4}" -f $name, $index, $manifest.Count, ([long]$entry.length / 1GB), $sourceHash)
    }

    $destinationFiles = @(Get-ChildItem -LiteralPath $dst -Recurse -File | Where-Object { $_.Extension -ne '.partial' })
    $destinationBytes = ($destinationFiles | Measure-Object Length -Sum).Sum
    if ($destinationFiles.Count -ne $manifest.Count -or $destinationBytes -ne $sourceBytes) {
        throw "Directory verification failed: $name"
    }
    $resolvedSource = (Resolve-Path -LiteralPath $src).Path
    if (-not $resolvedSource.StartsWith($source + '\')) {
        throw "Unsafe source deletion: $resolvedSource"
    }
    Remove-Item -LiteralPath $resolvedSource -Recurse -Force
    Write-MigrationLog ("complete {0} size={1:N2}GB" -f $name, ($destinationBytes / 1GB))
}

Write-MigrationLog 'all-complete'
