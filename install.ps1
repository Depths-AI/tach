& {
    Set-StrictMode -Version Latest
    $ErrorActionPreference = "Stop"
    $ProgressPreference = "SilentlyContinue"

    $Repository = "Depths-AI/tach"
    $RequestedVersion = if ([string]::IsNullOrWhiteSpace($env:TACH_VERSION)) { "latest" } else { $env:TACH_VERSION.Trim() }
    if ($RequestedVersion -ne "latest" -and -not $RequestedVersion.StartsWith("v")) {
        $RequestedVersion = "v$RequestedVersion"
    }

    if (-not [string]::IsNullOrWhiteSpace($env:TACH_INSTALL_DIR)) {
        $InstallDir = $env:TACH_INSTALL_DIR
    } elseif (-not [string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
        $InstallDir = Join-Path $env:LOCALAPPDATA "Tach\bin"
    } else {
        $InstallDir = Join-Path $env:USERPROFILE ".tach\bin"
    }

    $NativeArchitecture = if (-not [string]::IsNullOrWhiteSpace($env:PROCESSOR_ARCHITEW6432)) {
        $env:PROCESSOR_ARCHITEW6432
    } else {
        $env:PROCESSOR_ARCHITECTURE
    }
    $TargetArchitecture = switch ($NativeArchitecture.ToUpperInvariant()) {
        "AMD64" { "amd64"; break }
        "X86_64" { "amd64"; break }
        "ARM64" { "arm64"; break }
        default { throw "tach installer: unsupported Windows architecture: $NativeArchitecture" }
    }

    $ReleaseBase = if ($RequestedVersion -eq "latest") {
        "https://github.com/$Repository/releases/latest/download"
    } else {
        "https://github.com/$Repository/releases/download/$RequestedVersion"
    }
    $Asset = "tach-windows-$TargetArchitecture.zip"
    $TempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("tach-install-" + [guid]::NewGuid().ToString("N"))
    $ArchivePath = Join-Path $TempDir $Asset
    $ChecksumsPath = Join-Path $TempDir "checksums.txt"

    New-Item -ItemType Directory -Path $TempDir | Out-Null
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -UseBasicParsing -Uri "$ReleaseBase/$Asset" -OutFile $ArchivePath
        Invoke-WebRequest -UseBasicParsing -Uri "$ReleaseBase/checksums.txt" -OutFile $ChecksumsPath

        $ChecksumLine = Get-Content $ChecksumsPath | Where-Object {
            $Fields = $_.Trim() -split '\s+'
            $Fields.Count -ge 2 -and $Fields[1] -eq $Asset
        } | Select-Object -First 1
        if ([string]::IsNullOrWhiteSpace($ChecksumLine)) {
            throw "tach installer: $Asset is missing from checksums.txt"
        }
        $ExpectedHash = ($ChecksumLine.Trim() -split '\s+')[0].ToLowerInvariant()
        $ActualHash = (Get-FileHash -Algorithm SHA256 -Path $ArchivePath).Hash.ToLowerInvariant()
        if ($ActualHash -ne $ExpectedHash) {
            throw "tach installer: checksum verification failed for $Asset"
        }

        Expand-Archive -Path $ArchivePath -DestinationPath $TempDir -Force
        New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
        Copy-Item -Force -Path (Join-Path $TempDir "tach.exe") -Destination (Join-Path $InstallDir "tach.exe")

        $UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
        $PathEntries = if ([string]::IsNullOrWhiteSpace($UserPath)) { @() } else { @($UserPath -split ";") }
        $AlreadyPresent = $PathEntries | Where-Object { $_.TrimEnd("\") -ieq $InstallDir.TrimEnd("\") }
        if (-not $AlreadyPresent) {
            $NewUserPath = if ([string]::IsNullOrWhiteSpace($UserPath)) { $InstallDir } else { "$UserPath;$InstallDir" }
            [Environment]::SetEnvironmentVariable("Path", $NewUserPath, "User")
        }
        if (-not (($env:Path -split ";") | Where-Object { $_.TrimEnd("\") -ieq $InstallDir.TrimEnd("\") })) {
            $env:Path = "$InstallDir;$env:Path"
        }

        Write-Host "Installed tach to $(Join-Path $InstallDir 'tach.exe')"
    } finally {
        Remove-Item -LiteralPath $TempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}
