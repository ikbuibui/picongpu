#!/usr/bin/env bash
# Run the complete local CUDA/HIP Caravan and PMacc example matrix.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

"$script_dir/run_caravan_device_tests.sh" cuda
"$script_dir/run_caravan_device_tests.sh" hip
"$script_dir/run_pmacc_device_examples.sh" cuda all
"$script_dir/run_pmacc_device_examples.sh" hip all
