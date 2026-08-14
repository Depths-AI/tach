#!/bin/sh
set -eu

usage() {
  echo "usage: ./release.sh VERSION [--publish]" >&2
  echo "  VERSION must be a semantic version prefixed with v, for example v0.1.0" >&2
  exit 2
}

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  usage
fi

version=$1
package_version=${version#v}
publish=false
if [ "$#" -eq 2 ]; then
  if [ "$2" != "--publish" ]; then
    usage
  fi
  publish=true
fi

if [ "$package_version" = "$version" ]; then
  usage
fi
case $version in
  *[!A-Za-z0-9._-]*) usage ;;
esac

for command in cc git go node npm sha256sum spirv-val; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "tach release: required command not found: $command" >&2
    exit 1
  fi
done
if ! node -e 'process.exit(/^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)(?:\.(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*))*)?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$/.test(process.argv[1]) ? 0 : 1)' "$package_version"; then
  usage
fi

repository=${TACH_GITHUB_REPOSITORY:-Depths-AI/tach}
workspace=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
cd "$workspace"
dist_dir=$workspace/dist
release_dir=$dist_dir/$version

if [ -e "$release_dir" ]; then
  echo "tach release: $release_dir already exists" >&2
  exit 1
fi

if [ "$publish" = true ]; then
  if ! command -v gh >/dev/null 2>&1; then
    echo "tach release: gh is required with --publish" >&2
    exit 1
  fi
  if [ -n "$(git status --porcelain)" ]; then
    echo "tach release: commit all changes before publishing" >&2
    exit 1
  fi
  gh auth status >/dev/null
  npm whoami >/dev/null
fi

mkdir -p "$dist_dir"
stage_dir=$(mktemp -d "$dist_dir/.tach-release.XXXXXX")
artifacts_dir=$stage_dir/artifacts
package_dir=$stage_dir/npm-package
mkdir "$artifacts_dir" "$package_dir"

cleanup() {
  if [ -d "$stage_dir" ]; then
    find "$stage_dir" -depth -delete
  fi
}
trap cleanup EXIT
trap 'exit 1' HUP INT TERM

echo "Running local release checks"
go test -count=1 ./...
go vet ./...
npm run check:duplicates
npm ci --ignore-scripts
npm test
host_compiler=$stage_dir/tach-host
go build -trimpath -ldflags="-s -w -X main.version=$package_version" -o "$host_compiler" .

for target in linux/amd64 linux/arm64 darwin/amd64 darwin/arm64 windows/amd64 windows/arm64; do
  target_os=${target%/*}
  target_arch=${target#*/}
  extension=
  if [ "$target_os" = windows ]; then
    extension=.exe
  fi
  asset=$artifacts_dir/tach-$target_os-$target_arch$extension

  echo "Building $target_os/$target_arch"
  CGO_ENABLED=0 GOOS=$target_os GOARCH=$target_arch \
    go build -trimpath -ldflags="-s -w -X main.version=$package_version" -o "$asset" .
done

cp tach-ts/package.json tach-ts/README.md tach-ts/cli.mjs LICENSE "$package_dir/"
cp -R tach-ts/dist tach-ts/scripts "$package_dir/"
(cd "$package_dir" && \
  npm version "$package_version" --no-git-tag-version --ignore-scripts >/dev/null)
npm pack "$package_dir" \
  --pack-destination "$artifacts_dir" \
  --ignore-scripts >/dev/null

package_archive=$artifacts_dir/depths-tach-$package_version.tgz
install_test_dir=$stage_dir/install-test
mkdir "$install_test_dir"
(
  cd "$install_test_dir"
  npm init --yes >/dev/null
  TACH_BIN="$host_compiler" npm install "$package_archive" >/dev/null
  TACH_BIN="$host_compiler" node_modules/.bin/tach version >/dev/null
  node_modules/.bin/tach instructions >/dev/null
  node_modules/.bin/tach instructions --details 1 85 >/dev/null
  node --input-type=module -e 'const m = await import("@depths/tach"); if (Object.keys(m).sort().join() !== "TachError,tach") process.exit(1)'
)

(
  cd "$artifacts_dir"
  sha256sum tach-* depths-tach-*.tgz > checksums.txt
)
mv "$artifacts_dir" "$release_dir"

echo "Release artifacts written to $release_dir"

if [ "$publish" = true ]; then
  commit=$(git rev-parse HEAD)
  echo "Publishing native binaries for $version to github.com/$repository"
  gh release create "$version" \
    "$release_dir"/tach-* \
    "$release_dir/checksums.txt" \
    --repo "$repository" \
    --target "$commit" \
    --title "Tach $version" \
    --generate-notes

  echo "Publishing @depths/tach@$package_version to npm"
  npm publish "$release_dir"/depths-tach-"$package_version".tgz --access public
fi
