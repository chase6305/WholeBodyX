#!/usr/bin/env bash
set -euo pipefail
wbx_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$wbx_root"
wbx_python=OFF
if [[ "${1:-}" == "--python" && $# == 1 ]]; then
  wbx_python=ON
elif [[ $# != 0 ]]; then
  echo "Usage: bash scripts/build.sh [--python]" >&2
  exit 2
fi
conan install . -pr:h profiles/linux-gcc12-release -pr:b profiles/linux-gcc12-release \
  --output-folder=build/conan --build=missing --lockfile=conan.lock
wbx_args=()
if [[ "$wbx_python" == ON ]]; then
  if [[ ! -x .venv/bin/python ]]; then
    echo "Run uv sync before building Python bindings" >&2
    exit 2
  fi
  wbx_pybind_dir="$(.venv/bin/python -m pybind11 --cmakedir)"
  wbx_args+=("-DPython_EXECUTABLE=$wbx_root/.venv/bin/python" "-Dpybind11_DIR=$wbx_pybind_dir")
fi
cmake -S . -B build/native \
  "-DCMAKE_TOOLCHAIN_FILE=$wbx_root/build/conan/build/Release/generators/conan_toolchain.cmake" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON "-DWBX_BUILD_PYTHON=$wbx_python" "${wbx_args[@]}"
cmake --build build/native -j "${WBX_JOBS:-2}"
ctest --test-dir build/native --output-on-failure
