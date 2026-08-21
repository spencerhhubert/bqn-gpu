#!/usr/bin/env sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
revision=$(tr -d '[:space:]' < "$repo_root/deps/cbqn.rev")
source_dir="$repo_root/.build/cbqn-src"
output_dir="$repo_root/.build/cbqn"

mkdir -p "$repo_root/.build" "$output_dir"

if [ ! -d "$source_dir/.git" ]; then
  git clone --quiet https://github.com/dzaima/CBQN.git "$source_dir"
fi

git -C "$source_dir" fetch --quiet --depth=1 origin "$revision"
git -C "$source_dir" checkout --quiet --detach "$revision"
git -C "$source_dir" submodule update --quiet --init --recursive --depth=1

make -C "$source_dir" shared-o3 REPLXX=0 FFI=0 CC=cc notui=1
cp "$source_dir/libcbqn.so" "$output_dir/libcbqn.so"

printf '%s\n' "$output_dir/libcbqn.so"
