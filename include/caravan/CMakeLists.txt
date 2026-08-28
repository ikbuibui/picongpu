# This file is part of PIConGPU.
# SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
cmake_minimum_required(VERSION 3.28)
project(Caravan LANGUAGES CXX)

add_library(caravan-core INTERFACE)
add_library(caravan::core ALIAS caravan-core)
target_compile_features(caravan-core INTERFACE cxx_std_20)
target_include_directories(
    caravan-core
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..> $<INSTALL_INTERFACE:include>
)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(Threads REQUIRED)
add_library(caravan-mpi STATIC mpi/communicator.cpp mpi/context.cpp mpi/operations.cpp)
add_library(caravan::mpi ALIAS caravan-mpi)
target_link_libraries(caravan-mpi PUBLIC caravan::core MPI::MPI_CXX PRIVATE Threads::Threads)

if(TARGET alpaka::alpaka)
    set(_caravan_alpaka_default ON)
else()
    set(_caravan_alpaka_default OFF)
endif()
option(CARAVAN_BUILD_ALPAKA "Build the Caravan alpaka adapter" ${_caravan_alpaka_default})
unset(_caravan_alpaka_default)
if(CARAVAN_BUILD_ALPAKA)
    if(NOT TARGET alpaka::alpaka)
        find_package(alpaka REQUIRED)
    endif()
    add_library(caravan-alpaka INTERFACE)
    add_library(caravan::alpaka ALIAS caravan-alpaka)
    target_link_libraries(caravan-alpaka INTERFACE caravan::core alpaka::alpaka)
endif()

option(CARAVAN_BUILD_TESTING "Build Caravan tests" ${PROJECT_IS_TOP_LEVEL})
if(CARAVAN_BUILD_TESTING)
    enable_testing()
    add_executable(caravan-core-test test/core.cpp)
    target_link_libraries(caravan-core-test PRIVATE caravan::core Threads::Threads)
    add_test(NAME caravan-core COMMAND caravan-core-test)

    if(TARGET caravan::alpaka)
        alpaka_add_executable(caravan-alpaka-test test/alpaka.cpp)
        target_link_libraries(caravan-alpaka-test PRIVATE caravan::alpaka)
        add_test(NAME caravan-alpaka COMMAND caravan-alpaka-test)

        alpaka_add_executable(caravan-alpaka-mpi-test test/alpaka_mpi.cpp)
        target_link_libraries(caravan-alpaka-mpi-test PRIVATE caravan::alpaka caravan::mpi)
        add_test(
            NAME caravan-alpaka-mpi
            COMMAND
                ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:caravan-alpaka-mpi-test> ${MPIEXEC_POSTFLAGS}
        )
        set_tests_properties(caravan-alpaka-mpi PROPERTIES PROCESSORS 1 TIMEOUT 30)
    endif()

    add_executable(caravan-mpi-test test/mpi.cpp)
    target_link_libraries(caravan-mpi-test PRIVATE caravan::mpi)
    foreach(ranks IN ITEMS 1 2 4)
        add_test(
            NAME caravan-mpi-${ranks}-ranks
            COMMAND
                ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:caravan-mpi-test> ${MPIEXEC_POSTFLAGS}
        )
        set_tests_properties(caravan-mpi-${ranks}-ranks PROPERTIES PROCESSORS ${ranks} TIMEOUT 30)
    endforeach()
endif()
