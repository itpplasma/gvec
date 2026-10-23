#!/usr/bin/env bash
# Native and Python tests with the compiler selected by FC.
set -euo pipefail
export MPLBACKEND=Agg
unset DISPLAY
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CMAKE_BUILD_PARALLEL_LEVEL=2
: "${FC:?Select a Fortran compiler}"
"$FC" --version
python3 -m venv .compiler-venv
# shellcheck source=/dev/null
source .compiler-venv/bin/activate
python -m pip install -r requirements.txt pytest pyyaml f90nml netcdf4 xarray \
  scipy matplotlib tabulate tomlkit tqdm pyevtk requests
# Build compiler-specific modules against the system netCDF-C library.
curl --fail --location --retry 3 \
  https://github.com/Unidata/netcdf-fortran/archive/refs/tags/v4.6.1.tar.gz \
  --output netcdf-fortran.tar.gz
mkdir .compiler-netcdf-src
tar -xzf netcdf-fortran.tar.gz --strip-components=1 -C .compiler-netcdf-src
cmake -S .compiler-netcdf-src -B .compiler-netcdf-build -G Ninja \
  -DCMAKE_Fortran_COMPILER="$FC" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/.compiler-netcdf" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=ON -DENABLE_TESTS=OFF
cmake --build .compiler-netcdf-build --parallel 2
cmake --install .compiler-netcdf-build
export PKG_CONFIG_PATH="$PWD/.compiler-netcdf/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="$PWD/.compiler-netcdf/lib:${LD_LIBRARY_PATH:-}"
cmake --preset gvec_config_release -B .compiler-native -G Ninja \
  -DCMAKE_Fortran_COMPILER="$FC" -DPython_EXECUTABLE="$(command -v python)"
cmake --build .compiler-native --parallel 2
ctest --test-dir .compiler-native --output-on-failure --no-tests=error
python -m pip install --no-build-isolation '.[dev]'
python -m pytest --require-pygvec -m unit -q --tb=short
