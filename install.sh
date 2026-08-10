#!/bin/sh
set -eu

repository="Depths-AI/tach"
requested_version="${TACH_VERSION:-latest}"

if [ -n "${TACH_INSTALL_DIR:-}" ]; then
  install_dir=$TACH_INSTALL_DIR
elif [ -n "${HOME:-}" ]; then
  install_dir=$HOME/.local/bin
else
  echo "tach installer: HOME is unset; set TACH_INSTALL_DIR" >&2
  exit 1
fi

case $(uname -s) in
  Linux) target_os=linux ;;
  Darwin) target_os=darwin ;;
  *)
    echo "tach installer: unsupported operating system: $(uname -s)" >&2
    exit 1
    ;;
esac

case $(uname -m) in
  x86_64 | amd64) target_arch=amd64 ;;
  arm64 | aarch64) target_arch=arm64 ;;
  *)
    echo "tach installer: unsupported architecture: $(uname -m)" >&2
    exit 1
    ;;
esac

if [ "$requested_version" = latest ]; then
  release_base="https://github.com/$repository/releases/latest/download"
else
  case $requested_version in
    v*) ;;
    *) requested_version="v$requested_version" ;;
  esac
  release_base="https://github.com/$repository/releases/download/$requested_version"
fi

asset="tach-$target_os-$target_arch.tar.gz"
temp_dir=$(mktemp -d "${TMPDIR:-/tmp}/tach-install.XXXXXX")
archive_path=$temp_dir/$asset
checksums_path=$temp_dir/checksums.txt

cleanup() {
  rm -f "$archive_path" "$checksums_path" "$temp_dir/tach"
  rmdir "$temp_dir" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 1' HUP INT TERM

curl --proto '=https' --tlsv1.2 --fail --location --retry 3 --silent --show-error \
  --output "$archive_path" "$release_base/$asset"
curl --proto '=https' --tlsv1.2 --fail --location --retry 3 --silent --show-error \
  --output "$checksums_path" "$release_base/checksums.txt"

expected_hash=$(awk -v asset="$asset" '$2 == asset { print $1 }' "$checksums_path")
if [ -z "$expected_hash" ]; then
  echo "tach installer: $asset is missing from checksums.txt" >&2
  exit 1
fi

if command -v sha256sum >/dev/null 2>&1; then
  actual_hash=$(sha256sum "$archive_path" | awk '{ print $1 }')
elif command -v shasum >/dev/null 2>&1; then
  actual_hash=$(shasum -a 256 "$archive_path" | awk '{ print $1 }')
else
  echo "tach installer: sha256sum or shasum is required" >&2
  exit 1
fi

if [ "$actual_hash" != "$expected_hash" ]; then
  echo "tach installer: checksum verification failed for $asset" >&2
  exit 1
fi

tar -xzf "$archive_path" -C "$temp_dir" tach
mkdir -p "$install_dir"
install -m 0755 "$temp_dir/tach" "$install_dir/tach"

echo "Installed tach to $install_dir/tach"
case :$PATH: in
  *:"$install_dir":*) ;;
  *) echo "Add $install_dir to PATH to run tach from a new shell." ;;
esac
