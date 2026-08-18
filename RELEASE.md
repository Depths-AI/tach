# Releasing Tach

Run releases locally from the repository root in a visible PowerShell window. The script has exactly two modes and reads `GITHUB_TOKEN` and `NPM_TOKEN` from the ignored `.env` file.

## Dry mode

Dry mode runs every validation gate, builds all release artifacts, verifies the packed npm package, and writes the result under `dist/releases/<version>`. It does not change npm or GitHub.

```powershell
.\release.ps1 v0.1.2 -Dry -Notes "Meaningful release notes"
```

## Actual mode

Actual mode requires a clean `master` equal to `origin/master`. It performs the complete dry-mode work itself, creates the draft GitHub release and uploads its verified artifacts, then submits the npm package.

```powershell
.\release.ps1 v0.1.2 -Publish -Notes "Meaningful release notes"
```

npm prints a browser approval URL near the end. The script displays the full URL, copies it to the clipboard, and waits. Open the URL in the intended npm browser profile and approve the publication. Once npm reports the version publicly, the script publishes the GitHub draft and reports completion.

The npm package version, compiler version, VS Code extension version, and local harness dependencies must already match the requested release version. A published npm name/version pair cannot be reused, even after unpublishing.

References: [npm publish](https://docs.npmjs.com/cli/v11/commands/npm-publish), [npm publishing and 2FA](https://docs.npmjs.com/requiring-2fa-for-package-publishing-and-settings-modification/), and [GitHub release API](https://docs.github.com/en/rest/releases/releases).
