#
# Copyright 2023 Benjamin Worpitz, Jan Stephan, Jiří Vyskočil
# SPDX-License-Identifier: ISC
#

################################################################################
# Required CMake version.

cmake_minimum_required(VERSION 3.28)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

################################################################################
# Project.

set(_TARGET_NAME randomCells2D)

project(${_TARGET_NAME} LANGUAGES CXX)

if(PROJECT_IS_TOP_LEVEL)
    enable_testing()
endif()

#-------------------------------------------------------------------------------
# Find alpaka.

if(NOT TARGET alpaka::alpaka)
    option(alpaka_USE_SOURCE_TREE "Use alpaka's source tree instead of an alpaka installation" OFF)

    if(alpaka_USE_SOURCE_TREE)
        # Don't build the examples recursively
        set(alpaka_BUILD_EXAMPLES OFF)
        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../.." "${CMAKE_BINARY_DIR}/alpaka")
    else()
        find_package(alpaka REQUIRED)
    endif()
endif()

#-------------------------------------------------------------------------------
# Add Caravan's alpaka adapter.

if(NOT TARGET caravan::alpaka)
    if(TARGET caravan::core)
        message(FATAL_ERROR "randomCells2D requires the Caravan alpaka adapter")
    endif()

    set(
        randomCells2D_CARAVAN_SOURCE_DIR
        "${CMAKE_CURRENT_LIST_DIR}/../../../../include/caravan"
        CACHE PATH "Path to the Caravan source tree")
    set(CARAVAN_BUILD_MPI OFF)
    set(CARAVAN_BUILD_TESTING OFF)
    set(CARAVAN_BUILD_ALPAKA ON)
    add_subdirectory(
        "${randomCells2D_CARAVAN_SOURCE_DIR}"
        "${CMAKE_BINARY_DIR}/caravan"
        EXCLUDE_FROM_ALL)
endif()

#-------------------------------------------------------------------------------
# Add executable.

alpaka_add_executable(
    ${_TARGET_NAME}
    src/randomCells2D.cpp)
target_link_libraries(
    ${_TARGET_NAME}
    PUBLIC alpaka::alpaka caravan::alpaka)

set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER example)

add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME})
