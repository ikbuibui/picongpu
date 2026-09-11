#!/usr/bin/env bash
# Build and run the PMacc target-example regressions on CUDA or HIP.
set -euo pipefail

usage()
{
    echo "Usage: $0 <cuda|hip> [all|gameOfLife2D|heatEquation2D]" >&2
    echo "Environment overrides: BUILD_ROOT, CUDA_ARCH, HIP_ARCH, HIP_VERSION," >&2
    echo "  PIC_PROFILE, SPACK_SETUP, PARALLEL, CLEAN (default: 1)" >&2
    exit 2
}

[[ $# -ge 1 && $# -le 2 ]] || usage
backend=$1
selection=${2:-all}
[[ $backend == cuda || $backend == hip ]] || usage
[[ $selection == all || $selection == gameOfLife2D || $selection == heatEquation2D ]] || usage

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
spack_setup=${SPACK_SETUP:-/home/spack/share/spack/setup-env.sh}
pic_profile=${PIC_PROFILE:-$HOME/pixi-envs/pic/hal_picongpu_profile.sh}
build_root=${BUILD_ROOT:-$HOME/build/picongpu-device-tests}
parallel=${PARALLEL:-$(nproc)}
clean=${CLEAN:-1}

[[ -f $spack_setup ]] || { echo "Missing Spack setup: $spack_setup" >&2; exit 1; }
[[ -f $pic_profile ]] || { echo "Missing PIConGPU profile: $pic_profile" >&2; exit 1; }
# shellcheck disable=SC1090
source "$spack_setup"
# shellcheck disable=SC1090
source "$pic_profile"

backend_args=()
if [[ $backend == cuda ]]; then
    cuda_arch=${CUDA_ARCH:-80}
    command -v nvcc >/dev/null || { echo "The profile did not provide nvcc" >&2; exit 1; }
    cuda_root=$(dirname "$(dirname "$(command -v nvcc)")")
    export LD_LIBRARY_PATH="$cuda_root/lib64:$cuda_root/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
    backend_args+=(
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
    # PMacc's Game of Life uses XorMin, so unlike the standalone Caravan
    # tests these vendor RNG packages are required.
    spack load "hip@$hip_version" "rocrand@$hip_version" "hiprand@$hip_version"
    hip_compiler="$(hipconfig -l)/clang++"
    [[ -x $hip_compiler ]] || { echo "Missing HIP compiler: $hip_compiler" >&2; exit 1; }
    gcc_root=$(dirname "$(dirname "$(command -v g++)")")
    export LD_LIBRARY_PATH="$gcc_root/lib64:${LD_LIBRARY_PATH:-}"
    # Do not force CMAKE_CXX_COMPILER here. The loaded HIP package selects a
    # compatible Clang host compiler, and changing it in an existing CMake cache
    # can make CMake discard the alpaka HIP options (CLEAN=1 avoids that cache too).
    backend_args+=(
        -Dalpaka_ACC_GPU_HIP_ENABLE=ON
        -Dalpaka_ACC_GPU_HIP_ONLY_MODE=ON
        -DCMAKE_HIP_ARCHITECTURES="$hip_arch"
        -DCMAKE_HIP_COMPILER="$hip_compiler"
    )
fi

run_example()
{
    local name=$1
    local source_dir release_option build_dir
    case $name in
        gameOfLife2D)
            source_dir="$repo/share/pmacc/examples/gameOfLife2D"
            release_option=-DGOL_RELEASE=ON
            ;;
        heatEquation2D)
            source_dir="$repo/share/pmacc/examples/heatEquation2D"
            release_option=-DHEATEQ_RELEASE=ON
            ;;
        *) usage ;;
    esac

    build_dir="$build_root/${name}-${backend}"
    if [[ $clean == 1 ]]; then
        rm -rf "$build_dir"
    fi
    mkdir -p "$build_dir"

    printf 'Configuring %s for %s in %s\n' "$name" "$backend" "$build_dir"
    cmake -S "$source_dir" -B "$build_dir" "$release_option" \
        "${backend_args[@]}" 2>&1 | tee "$build_dir/configure.log"
    cmake --build "$build_dir" --parallel "$parallel" 2>&1 | tee "$build_dir/build.log"
    ctest --test-dir "$build_dir" --output-on-failure 2>&1 | tee "$build_dir/test.log"
}

if [[ $selection == all ]]; then
    run_example gameOfLife2D
    run_example heatEquation2D
else
    run_example "$selection"
fi
