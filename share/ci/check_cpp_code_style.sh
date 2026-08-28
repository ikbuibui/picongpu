#!/bin/bash

set -e
set -o pipefail


#############################################################################
# Keep native MPI calls inside the approved PMacc integration boundary      #
#############################################################################
directMpiCalls=$(grep -RInE \
    --include='*.hpp' --include='*.tpp' --include='*.cpp' --include='*.cu' \
    '(^|[^[:alnum:]_])MPI_[[:alnum:]_]+[[:space:]]*[(]' \
    include/pmacc share/pmacc/examples || true)
unauthorizedMpiCalls=$(printf '%s\n' "$directMpiCalls" \
    | grep -Ev '^include/pmacc/mpi/(MPIReduce|MPI_StructAsArray)\.hpp:' || true)
if [[ -n "$unauthorizedMpiCalls" ]]; then
    echo "Native MPI calls outside the approved PMacc integration boundary:" >&2
    echo "$unauthorizedMpiCalls" >&2
    exit 1
fi

#############################################################################
# Conformance with Alpaka: Do not write __global__ CUDA kernels directly    #
#############################################################################
test/hasCudaGlobalKeyword include/pmacc
test/hasCudaGlobalKeyword share/pmacc/examples
test/hasCudaGlobalKeyword include/picongpu
test/hasCudaGlobalKeyword share/picongpu/examples

#############################################################################
# Enforce angle brackets <...> for includes of external library files       #
#############################################################################
test/hasExtLibIncludeBrackets include boost
test/hasExtLibIncludeBrackets include alpaka
test/hasExtLibIncludeBrackets include mallocMC
test/hasExtLibIncludeBrackets include/picongpu pmacc
test/hasExtLibIncludeBrackets share/picongpu/examples pmacc
test/hasExtLibIncludeBrackets share/picongpu/examples boost
test/hasExtLibIncludeBrackets share/picongpu/examples alpaka
test/hasExtLibIncludeBrackets share/picongpu/examples mallocMC
test/hasExtLibIncludeBrackets share/pmacc/examples pmacc

#############################################################################
# Disallow doxygen with \                                                   #
#############################################################################
test/hasWrongDoxygenStyle include param
test/hasWrongDoxygenStyle include tparam
test/hasWrongDoxygenStyle include see
test/hasWrongDoxygenStyle include return
test/hasWrongDoxygenStyle include treturn
test/hasWrongDoxygenStyle share param
test/hasWrongDoxygenStyle share tparam
test/hasWrongDoxygenStyle share see
test/hasWrongDoxygenStyle share return
test/hasWrongDoxygenStyle share treturn
