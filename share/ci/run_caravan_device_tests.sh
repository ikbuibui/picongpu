#!/usr/bin/env bash
# Build and run Caravan's accelerator and MPI tests on this dual-GPU host.
set -euo pipefail

usage()
{
    echo "Usage: $0 <cuda|hip>" >&2
    echo "Environment overrides: BUILD_ROOT, BUILD_TYPE, CUDA_ARCH, HIP_ARCH," >&2
    echo "  HIP_VERSION, PIC_PROFILE, SPACK_SETUP, PARALLEL, CLEAN (default: 1)" >&2
    exit 2
}

[[ $# -eq 1 ]] || usage
backend=$1
[[ $backend == cuda || $backend == hip ]] || usage

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
spack_setup=${SPACK_SETUP:-/home/spack/share/spack/setup-env.sh}
pic_profile=${PIC_PROFILE:-$HOME/pixi-envs/pic/hal_picongpu_profile.sh}
build_root=${BUILD_ROOT:-$HOME/build/picongpu-device-tests}
build_type=${BUILD_TYPE:-Debug} # Caravan tests use assert(), so keep assertions enabled.
parallel=${PARALLEL:-$(nproc)}
clean=${CLEAN:-1}
build_dir="$build_root/caravan-$backend"

[[ -f $spack_setup ]] || { echo "Missing Spack setup: $spack_setup" >&2; exit 1; }
[[ -f $pic_profile ]] || { echo "Missing PIConGPU profile: $pic_profile" >&2; exit 1; }
# shellcheck disable=SC1090
source "$spack_setup"
# shellcheck disable=SC1090
source "$pic_profile"

cmake_args=(
    -S "$repo/share/ci/caravan-device-tests"
    -B "$build_dir"
    -DCMAKE_BUILD_TYPE="$build_type"
    -DPICONGPU_SOURCE_DIR="$repo"
)

if [[ $backend == cuda ]]; then
    cuda_arch=${CUDA_ARCH:-80}
    command -v nvcc >/dev/null || { echo "The profile did not provide nvcc" >&2; exit 1; }
    cuda_root=$(dirname "$(dirname "$(command -v nvcc)")")
    export LD_LIBRARY_PATH="$cuda_root/lib64:$cuda_root/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
    cmake_args+=(
        -Dalpaka_ACC_GPU_CUDA_ENABLE=ON
        -Dalpaka_ACC_GPU_CUDA_ONLY_MODE=ON
        -Dalpaka_CUDA_EXPT_EXTENDED_LAMBDA=ON
        -DCMAKE_CUDA_ARCHITECTURES="$cuda_arch"
        -DCMAKE_CUDA_COMPILER="$(command -v nvcc)"
        -DCMAKE_CUDA_HOST_COMPILER="$(command -v g++)"
    )
else
    hip_version=${HIP_VERSION:-7.0.2}
    hip_arch=${HIP_ARCH:-gfx1100}
    spack unload cuda >/dev/null 2>&1 || true
    spack load "hip@$hip_version"
    hip_compiler="$(hipconfig -l)/clang++"
    [[ -x $hip_compiler ]] || { echo "Missing HIP compiler: $hip_compiler" >&2; exit 1; }
    gcc_root=$(dirname "$(dirname "$(command -v g++)")")
    export LD_LIBRARY_PATH="$gcc_root/lib64:${LD_LIBRARY_PATH:-}"
    cmake_args+=(
        -Dalpaka_ACC_GPU_HIP_ENABLE=ON
        -Dalpaka_ACC_GPU_HIP_ONLY_MODE=ON
        -Dalpaka_DISABLE_VENDOR_RNG=ON
        -DCMAKE_HIP_ARCHITECTURES="$hip_arch"
        -DCMAKE_HIP_COMPILER="$hip_compiler"
        -DCMAKE_CXX_COMPILER="$(command -v g++)"
    )
fi

if [[ $clean == 1 ]]; then
    rm -rf "$build_dir"
fi
mkdir -p "$build_dir"
printf 'Configuring Caravan %s tests in %s\n' "$backend" "$build_dir"
cmake "${cmake_args[@]}" 2>&1 | tee "$build_dir/configure.log"
cmake --build "$build_dir" --parallel "$parallel" 2>&1 | tee "$build_dir/build.log"
ctest --test-dir "$build_dir" --output-on-failure 2>&1 | tee "$build_dir/test.log"
