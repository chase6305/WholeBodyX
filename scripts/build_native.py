"""Stage the native distribution, then use the standard Python build frontend."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--toolchain", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("dist"))
    args = parser.parse_args()
    toolchain = args.toolchain.resolve(strict=True)
    root = Path(__file__).resolve().parents[1]
    output = args.out.resolve()
    with tempfile.TemporaryDirectory(prefix="wholebodyx-native-") as temporary:
        stage = Path(temporary)
        for folder in ("cpp", "include", "bindings", "cmake"):
            shutil.copytree(root / folder, stage / folder)
        shutil.copy2(root / "CMakeLists.txt", stage)
        shutil.copy2(root / "native/pyproject.toml", stage / "pyproject.toml")
        environment = os.environ.copy()
        environment.setdefault("CMAKE_GENERATOR", "Unix Makefiles")
        environment.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", "2")
        subprocess.run([sys.executable, "-m", "build", "--no-isolation", "--wheel", "--sdist",
                        str(stage), "--outdir", str(output),
                        f"--config-setting=cmake.define.CMAKE_TOOLCHAIN_FILE={toolchain}"], check=True, env=environment)


if __name__ == "__main__":
    main()
