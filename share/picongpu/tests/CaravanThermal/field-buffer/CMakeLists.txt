# Standalone focused test for the PIConGPU field-buffer storage operations.
#
# This project deliberately does not build the PIConGPU application. It links
# PMacc (which provides alpaka and Caravan) and the repository Catch2 so that
# `picongpu/fields/detail/FieldBufferOperations.hpp` can be exercised with a real
# GridBuffer, independently of the still-unmigrated application translation units.
cmake_minimum_required(VERSION 3.28)
project(PIConGPUFieldBufferTest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

get_filename_component(
    PICONGPU_REPO_ROOT
    "${CMAKE_CURRENT_LIST_DIR}/../../../../.."
    ABSOLUTE
)

# Default to the CPU serial backend when the caller did not select one.
if(NOT DEFINED alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE
   AND NOT DEFINED alpaka_ACC_GPU_CUDA_ENABLE
   AND NOT DEFINED alpaka_ACC_GPU_HIP_ENABLE)
    set(alpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE ON CACHE BOOL "" FORCE)
    set(alpaka_ACC_GPU_CUDA_ENABLE OFF CACHE BOOL "" FORCE)
    set(alpaka_ACC_GPU_HIP_ENABLE OFF CACHE BOOL "" FORCE)
endif()

list(APPEND CMAKE_PREFIX_PATH "$ENV{MPI_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{BOOST_ROOT}")

# PMaccConfig provides pmacc::pmacc together with alpaka and Caravan.
find_package(
    PMacc
    REQUIRED CONFIG
    PATHS "${PICONGPU_REPO_ROOT}/include/pmacc"
    NO_DEFAULT_PATH
)
find_package(MPI REQUIRED COMPONENTS CXX)

# Repository-provided Catch2, matching PMacc's own test setup.
add_subdirectory(
    "${PICONGPU_REPO_ROOT}/thirdParty/catch2"
    "${CMAKE_BINARY_DIR}/catch2"
)

enable_testing()

alpaka_add_executable(
    picongpu-field-buffer-test
    fieldBufferTest.cpp
    fieldCommunicationTest.cpp
    "${PICONGPU_REPO_ROOT}/include/pmacc/test/TestMain.cpp"
)
target_compile_definitions(picongpu-field-buffer-test PRIVATE TEST_DIM=3)
target_include_directories(
    picongpu-field-buffer-test
    PRIVATE "${PICONGPU_REPO_ROOT}/include"
)
target_link_libraries(picongpu-field-buffer-test PRIVATE pmacc::pmacc Catch2)

add_test(
    NAME field-buffer-test
    COMMAND
        "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} 1
        "$<TARGET_FILE:picongpu-field-buffer-test>"
        ${MPIEXEC_POSTFLAGS}
)
set_tests_properties(field-buffer-test PROPERTIES TIMEOUT 120)

# Exercise the periodic field communication on two ranks.
add_test(
    NAME field-buffer-test-2-ranks
    COMMAND
        "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} 2
        "$<TARGET_FILE:picongpu-field-buffer-test>"
        ${MPIEXEC_POSTFLAGS}
)
set_tests_properties(field-buffer-test-2-ranks PROPERTIES TIMEOUT 120)
