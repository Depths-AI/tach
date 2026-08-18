# Releasing Tach

Run releases locally from the repository root in a visible PowerShell window. The script has exactly two modes and reads `GITHUB_TOKEN` and `NPM_TOKEN` from the ignored `.env` file.

## Dry mode

Dry mode runs every validation gate, builds all release artifacts, verifies the packed npm package, and writes the result under `dist/releases/<version>`. It does not change npm or GitHub.

```powershell
.\release.ps1 v0.1.2 -Dry -Notes "Meaningful release notes"
```

## Actual mode

Actual mode requires `master` equal to `origin/master`. It performs the complete dry-mode work itself, or reuses the same verified clean artifacts when resuming an interrupted release. It publishes the npm package, waits for npm publication, then creates the published GitHub release and uploads its verified artifacts. It never creates a GitHub draft.

```powershell
.\release.ps1 v0.1.2 -Publish -Notes "Meaningful release notes"
```

Near the end, npm requests browser approval through its interactive CLI. Run actual mode in a visible PowerShell window; npm opens the one-time approval page and waits for approval before completing publication. The script then verifies the public npm version, creates the published GitHub release, uploads its artifacts, and reports completion.

The npm package version, compiler version, VS Code extension version, and local harness dependencies must already match the requested release version. A published npm name/version pair cannot be reused, even after unpublishing.

References: [npm publish](https://docs.npmjs.com/cli/v11/commands/npm-publish), [npm publishing and 2FA](https://docs.npmjs.com/requiring-2fa-for-package-publishing-and-settings-modification/), and [GitHub release API](https://docs.github.com/en/rest/releases/releases).
