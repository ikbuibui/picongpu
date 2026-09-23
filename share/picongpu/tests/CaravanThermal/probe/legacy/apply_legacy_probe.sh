#!/usr/bin/env bash
# Apply the Thermal correctness-probe instrumentation to a legacy PIConGPU
# checkout (candidate: commit 7964ed0823) so it can serve as the WP0 oracle.
#
# Usage:
#   apply_legacy_probe.sh [LEGACY_SOURCE_ROOT]
#
# The source root must already contain include/picongpu. The script is
# idempotent for the header copy but applies the Simulation.hpp patch once.
set -euo pipefail

legacy_root="${1:-/tmp/picongpu-legacy-7964ed0823}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
picongpu_root="$(cd "${here}/../../../.." && pwd)"

if [[ ! -d "${legacy_root}/include/picongpu" ]]; then
    echo "error: ${legacy_root}/include/picongpu not found" >&2
    exit 1
fi

mkdir -p "${legacy_root}/include/picongpu/thermalProbe"
cp "${picongpu_root}/include/picongpu/thermalProbe/ThermalProbe.hpp" \
   "${legacy_root}/include/picongpu/thermalProbe/ThermalProbe.hpp"

if git -C "${legacy_root}" apply --check "${here}/0001-legacy-probe-hooks.patch" 2>/dev/null; then
    git -C "${legacy_root}" apply "${here}/0001-legacy-probe-hooks.patch"
    echo "applied ${here}/0001-legacy-probe-hooks.patch"
else
    echo "note: patch already applied or does not apply cleanly; verify Simulation.hpp hooks manually"
fi

cat <<EOF
Instrumentation written to ${legacy_root}.
Configure a probe-enabled legacy build with:

  cmake -S ${legacy_root}/include/picongpu -B <build-dir> \\
    -DCMAKE_BUILD_TYPE=Debug \\
    -DPIC_EXTENSION_PATH=${legacy_root}/share/picongpu/benchmarks/Thermal \\
    -DPARAM_OVERWRITES:STRING='-DPARAM_CURRENTSOLVER=EmZ;-DPARAM_PARTICLESHAPE=CIC;-DPARAM_PPC=25u' \\
    -Dalpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE=ON \\
    -Dalpaka_ACC_GPU_CUDA_ENABLE=OFF -Dalpaka_ACC_GPU_HIP_ENABLE=OFF \\
    -DCMAKE_CXX_FLAGS='-DPICONGPU_THERMAL_PROBE -DPICONGPU_THERMAL_PROBE_LEGACY'
  cmake --build <build-dir> --target picongpu -j
EOF
