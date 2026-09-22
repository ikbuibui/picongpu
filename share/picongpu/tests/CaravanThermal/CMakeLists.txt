# Thermal smoke-test driver for the PIConGPU Caravan migration slice.
#
# This project intentionally does not build PIConGPU. Point it at the dedicated
# minimal Thermal executable once that target exists, then run its CTest suite.
cmake_minimum_required(VERSION 3.28)
project(PIConGPUCaravanThermalSmoke LANGUAGES CXX)

include(CTest)
find_package(MPI REQUIRED COMPONENTS CXX)

set(
    PICONGPU_THERMAL_EXECUTABLE
    ""
    CACHE FILEPATH
    "Path to the minimal PIConGPU Thermal executable under test"
)

if(BUILD_TESTING)
    if(NOT PICONGPU_THERMAL_EXECUTABLE)
        message(FATAL_ERROR
            "Set PICONGPU_THERMAL_EXECUTABLE to the minimal Thermal executable. "
            "This driver does not build the executable."
        )
    endif()

    foreach(test_name IN ITEMS initialized-zero-step smoke-two-steps)
        if(test_name STREQUAL "initialized-zero-step")
            set(steps 0)
        else()
            set(steps 2)
        endif()
        set(work_dir "${CMAKE_BINARY_DIR}/${test_name}")
        file(MAKE_DIRECTORY "${work_dir}")
        add_test(
            NAME thermal-caravan-${test_name}
            COMMAND
                "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} 1
                "${PICONGPU_THERMAL_EXECUTABLE}"
                -d 1 1 1
                -g 32 32 32
                -s ${steps}
                --periodic 1 1 1
                ${MPIEXEC_POSTFLAGS}
        )
        set_tests_properties(
            thermal-caravan-${test_name}
            PROPERTIES
                LABELS "caravan;thermal;smoke;single-rank"
                TIMEOUT 120
                WORKING_DIRECTORY "${work_dir}"
        )
    endforeach()
endif()
