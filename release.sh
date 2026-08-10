#!/bin/sh
set -eu

usage() {
  echo "usage: ./release.sh VERSION [--publish]" >&2
  echo "  VERSION must start with v, for example v0.1.0" >&2
  exit 2
}

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  usage
fi

version=$1
publish=false
if [ "$#" -eq 2 ]; then
  if [ "$2" != "--publish" ]; then
    usage
  fi
  publish=true
fi

case $version in
  v[0-9]*) ;;
  *) usage ;;
esac
case $version in
  *[!A-Za-z0-9._-]*) usage ;;
esac

for command in go tar sha256sum python3; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "tach release: required command not found: $command" >&2
    exit 1
  fi
done

repository="${TACH_GITHUB_REPOSITORY:-Depths-AI/tach}"
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
fi

mkdir -p "$dist_dir"
stage_dir=$(mktemp -d "$dist_dir/.tach-release.XXXXXX")
artifacts_dir=$stage_dir/artifacts
mkdir "$artifacts_dir"

cleanup() {
  if [ -d "$stage_dir" ]; then
    find "$stage_dir" -depth -delete
  fi
}
trap cleanup EXIT
trap 'exit 1' HUP INT TERM

echo "Running release checks"
go test -count=1 ./...
go vet ./...

for target in linux/amd64 linux/arm64 darwin/amd64 darwin/arm64 windows/amd64 windows/arm64; do
  target_os=${target%/*}
  target_arch=${target#*/}
  package_dir=$stage_dir/package-$target_os-$target_arch
  mkdir "$package_dir"

  if [ "$target_os" = windows ]; then
    binary=$package_dir/tach.exe
  else
    binary=$package_dir/tach
  fi

  echo "Building $target_os/$target_arch"
  CGO_ENABLED=0 GOOS=$target_os GOARCH=$target_arch \
    go build -trimpath -ldflags="-s -w -X main.version=$version" -o "$binary" .

  if [ "$target_os" = windows ]; then
    asset=$artifacts_dir/tach-$target_os-$target_arch.zip
    (cd "$package_dir" && python3 -m zipfile -c "$asset" tach.exe)
  else
    asset=$artifacts_dir/tach-$target_os-$target_arch.tar.gz
    tar -czf "$asset" -C "$package_dir" tach
  fi
done

(cd "$artifacts_dir" && sha256sum tach-* > checksums.txt)
mv "$artifacts_dir" "$release_dir"

echo "Release artifacts written to $release_dir"

if [ "$publish" = true ]; then
  commit=$(git rev-parse HEAD)
  echo "Publishing $version to github.com/$repository"
  gh release create "$version" \
    "$release_dir"/tach-* \
    "$release_dir/checksums.txt" \
    --repo "$repository" \
    --target "$commit" \
    --title "Tach $version" \
    --generate-notes
fi
