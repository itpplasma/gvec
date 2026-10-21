#!/bin/bash
# The same native test selection runs in a build container or a heavy Slurm tier.
set -euo pipefail
src=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$src"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    root=$(dirname "$src")
    tier=$(cat "$root/tier.txt")
    . "$root/task.env"
    . /home/ert/opt/kin6d-env.sh
    export PATH=/home/ert/.local/bin:$PATH
else
    tier=${1:?usage: ci/run.sh TIER}
    root=$(mktemp -d "${RUNNER_TEMP:-/tmp}/native-ci.XXXXXX")
    export FC=/usr/bin/gfortran CC=/usr/bin/gcc CXX=/usr/bin/g++
    export OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
fi
unset DISPLAY
export CMAKE_TOOLCHAIN_FILE=$src/ci/cluster-toolchain.cmake
if [ -d "$root/mirrors" ]; then
    export GIT_CONFIG_GLOBAL=$root/gitconfig
    : > "$GIT_CONFIG_GLOBAL"
    git config --global protocol.file.allow always
    shopt -s nullglob
    for mirror in "$root"/mirrors/*/*.git; do
        slug=${mirror#"$root/mirrors/"}; slug=${slug%.git}
        git config --global --add "url.file://$mirror.insteadOf" "https://github.com/$slug.git"
        git config --global --add "url.file://$mirror.insteadOf" "https://github.com/$slug"
        git config --global --add "url.file://$mirror.insteadOf" "git@github.com:$slug.git"
    done
fi
export GITHUB_WORKSPACE=$src
export GITHUB_ENV=$root/github-env GITHUB_PATH=$root/github-path GITHUB_OUTPUT=$root/github-output
: > "$GITHUB_ENV"; : > "$GITHUB_PATH"; : > "$GITHUB_OUTPUT"
export SCI_CPUS=${SCI_CPUS:-8}
export CMAKE_BUILD_PARALLEL_LEVEL=$SCI_CPUS CTEST_PARALLEL_LEVEL=$SCI_CPUS
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
# Registered pytest tiers share their case output directories, as upstream CI.
export CTEST_PARALLEL_LEVEL=1
export RUNNER_TEMP=$root/tmp
mkdir -p "$RUNNER_TEMP" ci-out
# Compiler probes and isolated Python builds must execute outside noexec /tmp.
export TMPDIR=$RUNNER_TEMP
{ hostname; /usr/bin/gfortran --version | head -1; git rev-parse HEAD; } | tee ci-out/toolchain.txt
load_job_environment() {
    while IFS='=' read -r key value; do
        [[ $key =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] && export "$key=$value"
    done < "$GITHUB_ENV"
    while IFS= read -r item; do [ -z "$item" ] || export PATH="$item:$PATH"; done < "$GITHUB_PATH"
}
case "$tier" in
  ci-fortran)
    python3 -m venv "$root/venv"
    . "$root/venv/bin/activate"
    export MPLBACKEND="Agg"
    export PYTHON_VERSION="3.12"
    export FBTOOLS_REV="14a5c4a7c3ff7458780edf7ebb0bd76712b44fbf"
    echo 'Python test dependencies (used by the registered ctest tests)'
    (
python -m pip install pytest numpy pyyaml f90nml netcdf4 xarray scipy \
  matplotlib tabulate tomlkit tqdm pyevtk requests f90wrap

    )
    load_job_environment
    echo Configure
    (
cmake --preset ${SCI_MATRIX_PRESET} -B build -G Ninja -DPython_EXECUTABLE="$(which python)" -DCMAKE_Fortran_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache
    )
    load_job_environment
    echo Build
    (
cmake --build build --parallel "${SCI_CPUS:-8}"
    )
    load_job_environment
    echo 'Test (registered ctest tests)'
    (
ctest --test-dir build --output-on-failure --no-tests=error
    )
    load_job_environment
    ;;
  ci-python)
    python3 -m venv "$root/venv"
    . "$root/venv/bin/activate"
    export MPLBACKEND="Agg"
    export PYTHON_VERSION="3.12"
    export FBTOOLS_REV="14a5c4a7c3ff7458780edf7ebb0bd76712b44fbf"
    export SKBUILD_CMAKE_BUILD_TYPE="${SCI_MATRIX_BUILD_TYPE}"
    echo 'Build wheel (scikit-build-core, as upstream .script_build_py)'
    (
export SKBUILD_CMAKE_DEFINE="USE_OPENMP=On;USE_MPI=Off"
export CMAKE_Fortran_COMPILER_LAUNCHER="ccache"
export CMAKE_C_COMPILER_LAUNCHER="ccache"
python -m pip install build
python -m build --wheel
ls dist/

    )
    load_job_environment
    echo 'Install wheel'
    (
python -m pip install dist/*.whl pytest
pygvec --version

    )
    load_job_environment
    echo 'Unit tests (upstream .script_unit)'
    (
python -m pytest -m unit -v -rA --require-pygvec --junitxml=pytest.xml
    )
    load_job_environment
    ;;
  *) echo "unknown CI tier: $tier" >&2; exit 2 ;;
esac
