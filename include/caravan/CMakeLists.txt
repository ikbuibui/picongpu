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
add_library(caravan-mpi STATIC mpi.cpp)
add_library(caravan::mpi ALIAS caravan-mpi)
target_link_libraries(caravan-mpi PUBLIC caravan::core MPI::MPI_CXX PRIVATE Threads::Threads)

option(CARAVAN_BUILD_TESTING "Build Caravan tests" ${PROJECT_IS_TOP_LEVEL})
if(CARAVAN_BUILD_TESTING)
    enable_testing()
    add_executable(caravan-core-test test/core.cpp)
    target_link_libraries(caravan-core-test PRIVATE caravan::core Threads::Threads)
    add_test(NAME caravan-core COMMAND caravan-core-test)

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
