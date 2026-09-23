# Initial cache for the Thermal Caravan migration smoke target.
#
# Use with the PIConGPU application source directory, not this test-driver
# directory. It selects the opt-in plugin-free migration target.
get_filename_component(_thermal_extension "${CMAKE_CURRENT_LIST_DIR}/../../benchmarks/Thermal" ABSOLUTE)

set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "" FORCE)
set(PICONGPU_MINIMAL_CARAVAN_THERMAL ON CACHE BOOL "" FORCE)
set(PIC_EXTENSION_PATH "${_thermal_extension}" CACHE PATH "" FORCE)
set(PARAM_OVERWRITES "-DPARAM_CURRENTSOLVER=EmZ;-DPARAM_PARTICLESHAPE=CIC;-DPARAM_PPC=25u" CACHE STRING "" FORCE)
