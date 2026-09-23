# Standalone tests for the PIConGPU field-absorber selection policy.
#
# absorberPolicyTest.cpp exercises the dependency-light policy helper and is built
# twice (minimal + ordinary) without PMacc/alpaka/MPI.
# absorberFactoryTest.cpp exercises the production AbsorberFactory and therefore
# links PMacc (which provides alpaka and Caravan); it is also built twice.
cmake_minimum_required(VERSION 3.28)
project(PIConGPUAbsorberPolicyTest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

get_filename_component(PICONGPU_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../.." ABSOLUTE)

# Default to the CPU serial backend when the caller did not select one.
if(
    NOT DEFINED alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE
    AND NOT DEFINED alpaka_ACC_GPU_CUDA_ENABLE
    AND NOT DEFINED alpaka_ACC_GPU_HIP_ENABLE
)
    set(alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE ON CACHE BOOL "" FORCE)
    set(alpaka_ACC_GPU_CUDA_ENABLE OFF CACHE BOOL "" FORCE)
    set(alpaka_ACC_GPU_HIP_ENABLE OFF CACHE BOOL "" FORCE)
endif()

list(APPEND CMAKE_PREFIX_PATH "$ENV{MPI_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{BOOST_ROOT}")

find_package(PMacc REQUIRED CONFIG PATHS "${PICONGPU_REPO_ROOT}/include/pmacc" NO_DEFAULT_PATH)
find_package(MPI REQUIRED COMPONENTS CXX)

enable_testing()

# Pure policy helper, built minimal and ordinary without PMacc.
foreach(variant IN ITEMS minimal ordinary)
    set(target "absorber-policy-${variant}-test")
    add_executable(${target} absorberPolicyTest.cpp)
    target_include_directories(${target} PRIVATE "${PICONGPU_REPO_ROOT}/include")
    if(variant STREQUAL "minimal")
        target_compile_definitions(${target} PRIVATE PICONGPU_MINIMAL_CARAVAN_THERMAL)
    endif()
    add_test(NAME absorber-policy-${variant} COMMAND ${target})
endforeach()

# Production factory integration, built minimal and ordinary with PMacc.
foreach(variant IN ITEMS minimal ordinary)
    set(target "absorber-factory-${variant}-test")
    alpaka_add_executable(${target} absorberFactoryTest.cpp)
    target_include_directories(${target} PRIVATE "${PICONGPU_REPO_ROOT}/include")
    if(variant STREQUAL "minimal")
        target_compile_definitions(${target} PRIVATE PICONGPU_MINIMAL_CARAVAN_THERMAL)
    endif()
    target_link_libraries(${target} PRIVATE pmacc::pmacc)
    add_test(NAME absorber-factory-${variant} COMMAND ${target})
endforeach()
