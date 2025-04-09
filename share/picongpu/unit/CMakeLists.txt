# Copyright 2023-2024 Rene Widera
#
# This file is part of PIConGPU.
#
# PIConGPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PIConGPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIConGPU.
# If not, see <http://www.gnu.org/licenses/>.
#

cmake_minimum_required(VERSION 3.25.0)
project("UnitTest")

###############################################################################
# Language Flags
###############################################################################

# enforce C++20
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)


################################################################################
# CMake policies
#
# Search in <PackageName>_ROOT:
#   https://cmake.org/cmake/help/v3.12/policy/CMP0074.html
################################################################################
if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

################################################################################
# PMacc
################################################################################
find_package(PMacc REQUIRED CONFIG PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../../../include/pmacc")

###############################################################################
# Catch2
###############################################################################

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../thirdParty/catch2 ${CMAKE_BINARY_DIR}/catch2)

###############################################################################
# JSON
###############################################################################

 set(JSON_BuildTests OFF CACHE INTERNAL "")
 set(JSON_MultipleHeaders OFF CACHE INTERNAL "")
 set(JSON_ImplicitConversions OFF CACHE INTERNAL "")
 set(JSON_Install OFF CACHE INTERNAL "")  # only used PRIVATE

add_subdirectory(
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../thirdParty/nlohmann_json"
    "${CMAKE_CURRENT_BINARY_DIR}/build_nlohmann_json")

################################################################################
# Find Boost
################################################################################

list(APPEND CMAKE_PREFIX_PATH "$ENV{BOOST_ROOT}")

find_package(Boost 1.74.0 REQUIRED COMPONENTS program_options)
if(TARGET Boost::boost)
    list(PREPEND HOST_LIBS Boost::boost Boost::program_options)
else()
    include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
    list(PREPEND HOST_LIBS ${Boost_LIBRARIES})
endif()

################################################################################
# MPI
################################################################################

# MPI is provided by pmacc but to execute the binaries via root additional flags must be given to the execution command
option(USE_MPI_AS_ROOT_USER "add --allow-run-as-root mpiexec used by ctest" OFF)

if(USE_MPI_AS_ROOT_USER)
    set(MPI_RUNTIME_FLAGS "--allow-run-as-root")
endif()

# PIConGPU

include_directories(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}/../../../include")


################################################################################
# OpenPMD
################################################################################

# find openPMD installation
find_package(openPMD 0.15 CONFIG COMPONENTS MPI)
if(openPMD_FOUND)
    # HDF5 and ADIOS require linking against C libraries but this is not correctly
    # exposed.
    list(PREPEND HOST_LIBS MPI::MPI_C)

    if(openPMD_HAVE_ADIOS2 OR openPMD_HAVE_HDF5)
        message(STATUS "Found openPMD: ${openPMD_DIR}")
        add_definitions(-DENABLE_OPENPMD=1)
        # TOML
        set(
            PIC_toml11_PROVIDER "intern" CACHE
            STRING "Use internally shipped or external toml11 library.")
        set_property(
            CACHE PIC_toml11_PROVIDER
            PROPERTY STRINGS "intern;extern")
        mark_as_advanced(PIC_toml11_PROVIDER)
        if(${PIC_toml11_PROVIDER} STREQUAL "intern")
            add_subdirectory(
                "${CMAKE_CURRENT_SOURCE_DIR}/../../../thirdParty/toml11"
                "${CMAKE_CURRENT_BINARY_DIR}/build_toml11"
                # EXCLUDE_FROM_ALL ensures that toml11 is not part of
                # make install.
                # The library is header-only and linking it against
                # PIConGPU targets is sufficient.
                # It needs not be part of any build or install targets
                # explicitly.
                EXCLUDE_FROM_ALL)
        else()
            # Since toml11 4.0 was a breaking change,
            # it needs separate find_package() calls
            find_package(toml11 3.7.0 CONFIG QUIET)
            if(NOT toml11_FOUND)
                find_package(toml11 4.0 CONFIG REQUIRED)
            endif()
            message(STATUS "toml11: Found version '${toml11_VERSION}'")
        endif()
        list(PREPEND HOST_LIBS openPMD::openPMD)
    else()
        message(STATUS "Found openPMD at ${openPMD_DIR}, but PIConGPU requires"
                        " availability of either its ADIOS2 or HDF5 backend - "
                        "NOT BUILDING the openPMD plugin")
        PIC_dependency_set_status(openPMD FALSE)
    endif()
    OPTION(ENABLE_OPENPMD ON)
else(openPMD_FOUND)
    message(STATUS "Could NOT find openPMD - set openPMD_DIR or check your CMAKE_PREFIX_PATH")
endif(openPMD_FOUND)

###############################################################################
# Targets
###############################################################################

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/test)
include_directories(BEFORE ./include)

# CTest
enable_testing()

# We need to build and link this to check CLI aspects.
add_library(argparser
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../include/picongpu/ArgsParser.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../include/picongpu/versionFormat.x.cpp"
)

target_link_libraries(argparser PRIVATE ${HOST_LIBS})
target_link_libraries(argparser PRIVATE pmacc::pmacc)

# Add subdirectories for tests
add_subdirectory(dimensional_tests)
add_subdirectory(non_dimensional_tests)
