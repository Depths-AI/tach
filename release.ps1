[CmdletBinding(DefaultParameterSetName = "Dry")]
param(
  [Parameter(Mandatory = $true, Position = 0)]
  [string]$Version,
  [Parameter(Mandatory = $true)]
  [string]$Notes,
  [Parameter(Mandatory = $true, ParameterSetName = "Dry")]
  [switch]$Dry,
  [Parameter(Mandatory = $true, ParameterSetName = "Publish")]
  [switch]$Publish
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$root = $PSScriptRoot
$repository = "Depths-AI/tach"
$packageName = "@depths/tach"
$registry = "https://registry.npmjs.org/"
$semver = '^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)(?:\.(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*))*)?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$'
if ($Version -notmatch $semver) {
  throw "VERSION must be semantic and prefixed with v, for example v0.1.0"
}
$packageVersion = $Version.Substring(1)
$Notes = $Notes.Trim()
if (-not $Notes) {
  throw "release requires meaningful -Notes text"
}
$releaseDir = Join-Path $root "dist\releases\$Version"
$workDir = Join-Path $releaseDir ".work"
$targets = @(
  [pscustomobject]@{ OS = "windows"; Arch = "amd64"; File = "tach-windows-amd64.exe" },
  [pscustomobject]@{ OS = "windows"; Arch = "arm64"; File = "tach-windows-arm64.exe" },
  [pscustomobject]@{ OS = "linux"; Arch = "amd64"; File = "tach-linux-amd64" },
  [pscustomobject]@{ OS = "linux"; Arch = "arm64"; File = "tach-linux-arm64" },
  [pscustomobject]@{ OS = "darwin"; Arch = "arm64"; File = "tach-darwin-arm64" }
)
$runtimeFiles = @(
  "tach-vulkan.windows.x86_64.dll",
  "tach-vulkan.linux.x86_64.so"
)

function Assert-Command([string]$Name) {
  if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
    throw "required command not found: $Name"
  }
}

function Invoke-Checked(
  [string]$Command,
  [string[]]$Arguments,
  [string]$WorkingDirectory = $root
) {
  Push-Location $WorkingDirectory
  try {
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
      throw "$Command exited $LASTEXITCODE"
    }
  } finally {
    Pop-Location
  }
}

function Read-Checked(
  [string]$Command,
  [string[]]$Arguments,
  [string]$WorkingDirectory = $root
) {
  Push-Location $WorkingDirectory
  try {
    $output = & $Command @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
      throw "$Command exited $LASTEXITCODE $([Environment]::NewLine)$($output -join [Environment]::NewLine)"
    }
    return ($output -join [Environment]::NewLine).Trim()
  } finally {
    Pop-Location
  }
}

function Write-Utf8([string]$Path, [string]$Text) {
  [IO.File]::WriteAllText(
    $Path,
    $Text,
    (New-Object Text.UTF8Encoding($false))
  )
}

function Get-Sha256([string]$Path) {
  return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Assert-VersionCoherence {
  $package = Get-Content -LiteralPath (Join-Path $root "tach-ts\package.json") -Raw |
    ConvertFrom-Json
  if ($package.version -ne $packageVersion) {
    throw "tach-ts package version is $($package.version), expected $packageVersion"
  }
  foreach ($path in @(
    "browser-test\package.json",
    "deno-test\package.json",
    "showcase-ts\package.json"
  )) {
    $consumer = Get-Content -LiteralPath (Join-Path $root $path) -Raw |
      ConvertFrom-Json
    if ($consumer.dependencies.$packageName -ne $packageVersion) {
      throw "$path does not depend on $packageName@$packageVersion"
    }
  }
  $extension = Get-Content -LiteralPath (Join-Path $root "vscode\package.json") -Raw |
    ConvertFrom-Json
  if ($extension.version -ne $packageVersion) {
    throw "VS Code extension version is $($extension.version), expected $packageVersion"
  }
  $main = Get-Content -LiteralPath (Join-Path $root "main.go") -Raw
  if ($main -notmatch "var version = `"$([regex]::Escape($packageVersion))`"") {
    throw "main.go fallback version does not match $packageVersion"
  }
}

function Assert-PublishCommit {
  if ((Read-Checked "git" @("branch", "--show-current")) -ne "master") {
    throw "publish requires the master branch"
  }
  $head = Read-Checked "git" @("rev-parse", "HEAD")
  $remote = ((Read-Checked "git" @("ls-remote", "origin", "refs/heads/master")) -split '\s+')[0]
  if ($head -ne $remote) {
    throw "HEAD is not origin/master"
  }
}

function Get-HttpStatus($ErrorRecord) {
  if ($null -eq $ErrorRecord.Exception.Response) {
    return 0
  }
  return [int]$ErrorRecord.Exception.Response.StatusCode
}

function Read-Secrets {
  $path = Join-Path $root ".env"
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    throw "publish requires $path"
  }
  $values = @{}
  foreach ($source in [IO.File]::ReadAllLines($path)) {
    $line = $source.Trim()
    if (-not $line -or $line.StartsWith("#")) {
      continue
    }
    $separator = $line.IndexOf("=")
    if ($separator -lt 1) {
      throw "invalid .env line: $source"
    }
    $name = $line.Substring(0, $separator).Trim()
    $value = $line.Substring($separator + 1).Trim()
    if ($value.Length -ge 2 -and
        (($value[0] -eq '"' -and $value[$value.Length - 1] -eq '"') -or
         ($value[0] -eq "'" -and $value[$value.Length - 1] -eq "'"))) {
      $value = $value.Substring(1, $value.Length - 2)
    }
    if ($values.ContainsKey($name) -or -not $value) {
      throw "invalid .env value for $name"
    }
    $values[$name] = $value
  }
  foreach ($name in @("GITHUB_TOKEN", "NPM_TOKEN")) {
    if (-not $values.ContainsKey($name)) {
      throw ".env is missing $name"
    }
  }
  return $values
}

function Get-PublishedNpmVersion {
  $encoded = [Uri]::EscapeDataString($packageName)
  try {
    $document = Invoke-RestMethod -Uri "$registry$encoded/$packageVersion" -Method Get
    return [string]$document.version
  } catch {
    if ((Get-HttpStatus $_) -eq 404) {
      return $null
    }
    throw
  }
}

function Invoke-GitHub(
  [string]$Method,
  [string]$Path,
  $Body = $null
) {
  $parameters = @{
    Uri = "https://api.github.com$Path"
    Method = $Method
    Headers = $script:githubHeaders
  }
  if ($null -ne $Body) {
    $parameters.ContentType = "application/json"
    $parameters.Body = ConvertTo-Json $Body -Depth 8 -Compress
  }
  return Invoke-RestMethod @parameters
}

function Get-GitHubRelease {
  $releases = Invoke-GitHub "Get" "/repos/$repository/releases?per_page=100"
  $matches = @($releases | Where-Object tag_name -eq $Version)
  if ($matches.Count -gt 1) {
    throw "GitHub contains duplicate releases for $Version"
  }
  return $matches | Select-Object -First 1
}

function Assert-Stage {
  $manifestPath = Join-Path $releaseDir "release.json"
  $checksumsPath = Join-Path $releaseDir "checksums.txt"
  if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf) -or
      -not (Test-Path -LiteralPath $checksumsPath -PathType Leaf)) {
    throw "run .\release.ps1 $Version before publishing"
  }
  $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
  $head = Read-Checked "git" @("rev-parse", "HEAD")
  if ($manifest.schema -ne 2 -or $manifest.tag -ne $Version -or
      $manifest.version -ne $packageVersion -or $manifest.commit -ne $head) {
    throw "release stage does not describe $Version at HEAD"
  }
  if ([string]::IsNullOrWhiteSpace($manifest.notes)) {
    throw "release stage has no human-authored notes"
  }
  if ($manifest.dirty) {
    throw "release stage was built from a dirty tree; commit and rebuild it"
  }
  $payload = @($targets.File) + "depths-tach-$packageVersion.tgz"
  if ((Compare-Object ($payload | Sort-Object) (@($manifest.artifacts.name) | Sort-Object))) {
    throw "release manifest inventory differs from the intended artifacts"
  }
  $expected = $payload + @("release.json", "checksums.txt")
  $actual = @(Get-ChildItem -LiteralPath $releaseDir -File | ForEach-Object Name)
  if ((Compare-Object ($expected | Sort-Object) ($actual | Sort-Object))) {
    throw "release stage inventory differs from the intended artifacts"
  }
  foreach ($artifact in $manifest.artifacts) {
    $path = Join-Path $releaseDir $artifact.name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or
        (Get-Item -LiteralPath $path).Length -ne $artifact.bytes -or
        (Get-Sha256 $path) -ne $artifact.sha256) {
      throw "release artifact failed verification: $($artifact.name)"
    }
  }
  foreach ($line in [IO.File]::ReadAllLines($checksumsPath)) {
    $fields = $line.Trim() -split '\s+', 2
    if ($fields.Count -ne 2) {
      throw "invalid checksums.txt line"
    }
    $path = Join-Path $releaseDir $fields[1]
    if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or
        (Get-Sha256 $path) -ne $fields[0]) {
      throw "checksum failed: $($fields[1])"
    }
  }
  return $manifest
}

function New-ReleaseStage {
  Assert-VersionCoherence
  foreach ($command in @("deno", "git", "go", "npm", "spirv-val", "zig")) {
    Assert-Command $command
  }
  if (Test-Path -LiteralPath $releaseDir) {
    Remove-Item -LiteralPath $releaseDir -Recurse -Force
  }
  New-Item -ItemType Directory -Path $workDir | Out-Null
  $head = Read-Checked "git" @("rev-parse", "HEAD")
  $dirty = -not [string]::IsNullOrWhiteSpace(
    (Read-Checked "git" @("status", "--porcelain", "--untracked-files=all"))
  )
  try {
    Write-Host "Running complete release validation"
    Invoke-Checked "npm" @("ci", "--ignore-scripts")
    Invoke-Checked "npm" @("test")

    $hostCompiler = Join-Path $workDir "tach.exe"
    Invoke-Checked "go" @(
      "build",
      "-trimpath",
      "-ldflags=-s -w -X main.version=$packageVersion",
      "-o",
      $hostCompiler,
      "."
    )
    Invoke-Checked $hostCompiler @("_version")

    $savedOS = $env:GOOS
    $savedArch = $env:GOARCH
    $savedCgo = $env:CGO_ENABLED
    try {
      $env:CGO_ENABLED = "0"
      foreach ($target in $targets) {
        $env:GOOS = $target.OS
        $env:GOARCH = $target.Arch
        Write-Host "Building $($target.OS)/$($target.Arch)"
        Invoke-Checked "go" @(
          "build",
          "-trimpath",
          "-ldflags=-s -w -X main.version=$packageVersion",
          "-o",
          (Join-Path $releaseDir $target.File),
          "."
        )
      }
    } finally {
      $env:GOOS = $savedOS
      $env:GOARCH = $savedArch
      $env:CGO_ENABLED = $savedCgo
    }

    $packageDir = Join-Path $workDir "npm-package"
    $packageNative = Join-Path $packageDir "native"
    New-Item -ItemType Directory -Path $packageNative | Out-Null
    foreach ($file in @("package.json", "README.md")) {
      Copy-Item -LiteralPath (Join-Path $root "tach-ts\$file") -Destination $packageDir
    }
    Copy-Item -LiteralPath (Join-Path $root "LICENSE") -Destination $packageDir
    Copy-Item -LiteralPath (Join-Path $root "tach-ts\dist") -Destination $packageDir -Recurse
    $windowsRuntime = Join-Path $root "tach-ts\native\$($runtimeFiles[0])"
    if (-not (Test-Path -LiteralPath $windowsRuntime -PathType Leaf)) {
      throw "native validation did not produce $windowsRuntime"
    }
    Copy-Item -LiteralPath $windowsRuntime -Destination $packageNative

    if (-not $env:VULKAN_SDK) {
      throw "VULKAN_SDK must name the official Vulkan SDK"
    }
    $linuxRuntime = Join-Path $packageNative $runtimeFiles[1]
    $savedOS = $env:GOOS
    $savedArch = $env:GOARCH
    $savedCgo = $env:CGO_ENABLED
    $savedCC = $env:CC
    $savedCgoFlags = $env:CGO_CFLAGS
    try {
      $env:GOOS = "linux"
      $env:GOARCH = "amd64"
      $env:CGO_ENABLED = "1"
      $env:CC = "zig cc -target x86_64-linux-gnu"
      $env:CGO_CFLAGS = "-I$($env:VULKAN_SDK.Replace('\', '/'))/Include"
      Invoke-Checked "go" @(
        "build",
        "-tags",
        "tachvulkan",
        "-buildmode=c-shared",
        "-trimpath",
        "-o",
        $linuxRuntime,
        "./native"
      )
    } finally {
      $env:GOOS = $savedOS
      $env:GOARCH = $savedArch
      $env:CGO_ENABLED = $savedCgo
      $env:CC = $savedCC
      $env:CGO_CFLAGS = $savedCgoFlags
    }
    $linuxHeader = [IO.Path]::ChangeExtension($linuxRuntime, ".h")
    if (Test-Path -LiteralPath $linuxHeader) {
      Remove-Item -LiteralPath $linuxHeader
    }
    $elf = [IO.File]::ReadAllBytes($linuxRuntime)
    if ($elf.Length -lt 20 -or $elf[0] -ne 0x7f -or
        $elf[1] -ne 0x45 -or $elf[2] -ne 0x4c -or $elf[3] -ne 0x46 -or
        $elf[4] -ne 2 -or $elf[5] -ne 1 -or
        ($elf[18] -bor ($elf[19] -shl 8)) -ne 62) {
      throw "Zig did not produce an x86-64 Linux ELF runtime"
    }
    if ((Get-Content -LiteralPath (Join-Path $packageDir "package.json") -Raw |
        ConvertFrom-Json).version -ne $packageVersion) {
      throw "@depths/tach version does not match $Version"
    }
    Invoke-Checked "npm" @(
      "pack",
      $packageDir,
      "--pack-destination",
      $releaseDir,
      "--ignore-scripts"
    )
    $archive = Join-Path $releaseDir "depths-tach-$packageVersion.tgz"
    if (-not (Test-Path -LiteralPath $archive -PathType Leaf)) {
      throw "npm pack did not produce $archive"
    }

    $installDir = Join-Path $workDir "install-test"
    New-Item -ItemType Directory -Path $installDir | Out-Null
    Invoke-Checked "npm" @("init", "--yes") $installDir
    Invoke-Checked "npm" @("install", $archive, "--ignore-scripts") $installDir
    $installedNative = Join-Path $installDir "node_modules\@depths\tach\native"
    $nativeNames = @(Get-ChildItem -LiteralPath $installedNative -File | ForEach-Object Name)
    if ((Compare-Object ($runtimeFiles | Sort-Object) ($nativeNames | Sort-Object))) {
      throw "packed native inventory contains missing or unintended files"
    }
    $savedCompiler = $env:TACH_BIN
    try {
      $env:TACH_BIN = $hostCompiler
      $cli = Join-Path $installDir "node_modules\.bin\tach.cmd"
      Invoke-Checked $cli @("version") $installDir
      Read-Checked $cli @("instructions") $installDir | Out-Null
      Read-Checked $cli @("instructions", "--details", "1", "85") $installDir | Out-Null
      $verify = Join-Path $installDir "verify.ts"
      Write-Utf8 $verify @'
const api = await import("@depths/tach");
if (Object.keys(api).sort().join() !== "TachError,tach") Deno.exit(1);
const compiler = await import("@depths/tach/compiler");
await compiler.compilerPath();
'@
      Invoke-Checked "deno" @(
        "run", "--allow-env", "--allow-read", "--allow-run",
        "--node-modules-dir=manual", $verify
      ) $installDir
    } finally {
      $env:TACH_BIN = $savedCompiler
    }

    $artifacts = @(
      Get-ChildItem -LiteralPath $releaseDir -File |
        Sort-Object Name |
        ForEach-Object {
          [pscustomobject]@{
            name = $_.Name
            bytes = $_.Length
            sha256 = Get-Sha256 $_.FullName
          }
        }
    )
    $manifest = [ordered]@{
      schema = 2
      tag = $Version
      version = $packageVersion
      commit = $head
      dirty = $dirty
      notes = $Notes
      generatedAt = [DateTime]::UtcNow.ToString("o")
      tools = [ordered]@{
        go = Read-Checked "go" @("version")
        npm = Read-Checked "npm" @("--version")
        deno = ((Read-Checked "deno" @("--version")) -split [Environment]::NewLine)[0]
        zig = Read-Checked "zig" @("version")
      }
      artifacts = $artifacts
    }
    $manifestPath = Join-Path $releaseDir "release.json"
    Write-Utf8 $manifestPath ((ConvertTo-Json $manifest -Depth 6) + [Environment]::NewLine)
    $checksumFiles = @($artifacts.name) + "release.json"
    $checksums = $checksumFiles |
      Sort-Object |
      ForEach-Object { "$(Get-Sha256 (Join-Path $releaseDir $_))  $_" }
    Write-Utf8 (Join-Path $releaseDir "checksums.txt") (($checksums -join [Environment]::NewLine) + [Environment]::NewLine)
  } finally {
    if (Test-Path -LiteralPath $workDir) {
      Remove-Item -LiteralPath $workDir -Recurse -Force
    }
  }
  Write-Host "Verified release artifacts written to $releaseDir"
  if ($dirty) {
    Write-Warning "The tree was dirty. Commit the changes before publishing."
  }
}

function Publish-Npm([string]$Archive, [string]$Npmrc) {
  Invoke-Checked "npm" @(
    "publish", $Archive, "--access", "public",
    "--registry", $registry, "--userconfig", $Npmrc
  )
}

function Publish-Release {
  foreach ($command in @("git", "npm")) {
    Assert-Command $command
  }
  Assert-PublishCommit
  $head = Read-Checked "git" @("rev-parse", "HEAD")
  $manifest = Assert-Stage
  $secrets = Read-Secrets
  $script:githubHeaders = @{
    Accept = "application/vnd.github+json"
    Authorization = "Bearer $($secrets.GITHUB_TOKEN)"
    "X-GitHub-Api-Version" = "2026-03-10"
  }
  $npmrc = [IO.Path]::GetTempFileName()
  try {
    Write-Utf8 $npmrc @"
registry=$registry
//registry.npmjs.org/:_authToken=$($secrets.NPM_TOKEN)
"@
    $githubUser = Invoke-GitHub "Get" "/user"
    $npmUser = Read-Checked "npm" @(
      "whoami",
      "--registry",
      $registry,
      "--userconfig",
      $npmrc
    )
    Write-Host "Authenticated GitHub as $($githubUser.login)"
    Write-Host "Authenticated npm as $npmUser"

    $published = Get-PublishedNpmVersion
    if ($null -eq $published) {
      $archive = Join-Path $releaseDir "depths-tach-$packageVersion.tgz"
      Publish-Npm $archive $npmrc
      for ($attempt = 0; $attempt -lt 300 -and $null -eq $published; $attempt++) {
        $published = Get-PublishedNpmVersion
        if ($null -eq $published) {
          Start-Sleep -Seconds 2
        }
      }
    }
    if ($published -ne $packageVersion) {
      throw "npm did not publish $packageName@$packageVersion"
    }

    $release = Get-GitHubRelease
    if ($null -eq $release) {
      $release = Invoke-GitHub "Post" "/repos/$repository/releases" @{
        tag_name = $Version
        target_commitish = $head
        name = "Tach $Version"
        body = [string]$manifest.notes
        draft = $false
        prerelease = $packageVersion.Contains("-")
      }
      Write-Host "Published GitHub release $Version"
    } elseif ($release.draft -or $release.target_commitish -ne $head) {
      throw "existing GitHub release is not the published release for $head"
    }
    if ($release.name -ne "Tach $Version" -or
        $release.body -ne [string]$manifest.notes) {
      $release = Invoke-GitHub "Patch" "/repos/$repository/releases/$($release.id)" @{
        name = "Tach $Version"
        body = [string]$manifest.notes
      }
      Write-Host "Updated human-authored GitHub release notes"
    }

    $publicFiles = @($manifest.artifacts.name) + @("release.json", "checksums.txt")
    $release = Invoke-GitHub "Get" "/repos/$repository/releases/$($release.id)"
    foreach ($name in $publicFiles) {
      $path = Join-Path $releaseDir $name
      $hash = Get-Sha256 $path
      $asset = @($release.assets | Where-Object name -eq $name)
      if ($asset.Count -gt 1) {
        throw "GitHub release contains duplicate asset $name"
      }
      if ($asset.Count -eq 1) {
        if ($asset[0].size -ne (Get-Item -LiteralPath $path).Length -or
            $asset[0].digest -ne "sha256:$hash") {
          throw "GitHub asset differs from local stage: $name"
        }
        Write-Host "Verified existing GitHub asset $name"
        continue
      }
      $upload = $release.upload_url -replace '\{\?name,label\}$', ""
      $uri = "${upload}?name=$([Uri]::EscapeDataString($name))"
      Invoke-WebRequest -Uri $uri -Method Post -Headers $script:githubHeaders `
        -ContentType "application/octet-stream" -InFile $path -UseBasicParsing | Out-Null
      Write-Host "Uploaded $name"
      $release = Invoke-GitHub "Get" "/repos/$repository/releases/$($release.id)"
    }
    Write-Host "Released $packageName@$packageVersion from commit $head"
  } finally {
    if (Test-Path -LiteralPath $npmrc) {
      Remove-Item -LiteralPath $npmrc -Force
    }
    $script:githubHeaders = $null
  }
}

Set-Location $root
if ($Publish) {
  Assert-PublishCommit
  $manifest = try { Assert-Stage } catch { $null }
  if ($null -eq $manifest -or [string]$manifest.notes -ne $Notes) {
    if (-not [string]::IsNullOrWhiteSpace(
        (Read-Checked "git" @("status", "--porcelain", "--untracked-files=all")))) {
      throw "publish requires a clean tree when building release artifacts"
    }
    New-ReleaseStage
  } else {
    Write-Host "Reusing the verified release artifacts in $releaseDir"
  }
  Publish-Release
} else {
  New-ReleaseStage
}
