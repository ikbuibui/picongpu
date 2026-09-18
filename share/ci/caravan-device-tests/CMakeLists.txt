# Standalone driver for compiling and running Caravan's tests with an alpaka
# accelerator backend. Use ../run_caravan_device_tests.sh rather than invoking
# this file directly so the matching Spack environment is active.
cmake_minimum_required(VERSION 3.28)
project(CaravanDeviceTests LANGUAGES CXX)

enable_testing()

if(NOT DEFINED PICONGPU_SOURCE_DIR)
    get_filename_component(PICONGPU_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
endif()

set(alpaka_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(alpaka_BUILD_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory("${PICONGPU_SOURCE_DIR}/thirdParty/alpaka" alpaka)

set(CARAVAN_BUILD_TESTING ON CACHE BOOL "" FORCE)
add_subdirectory("${PICONGPU_SOURCE_DIR}/include/caravan" caravan)
