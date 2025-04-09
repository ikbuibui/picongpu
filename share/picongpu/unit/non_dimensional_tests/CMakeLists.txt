# Handle tests without dimensionality
file(GLOB_RECURSE NON_DIMENSIONAL_TESTS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
foreach(testCaseFilepath ${NON_DIMENSIONAL_TESTS})
    get_filename_component(testCaseFilename ${testCaseFilepath} NAME)
    string(REPLACE "UT.cpp" "" testCase ${testCaseFilename})
    set(testExe "${PROJECT_NAME}-${testCase}")
    alpaka_add_executable(${testExe} ${testCaseFilepath})
    target_link_libraries(${testExe} PUBLIC Catch2 Catch2WithMain)
    target_link_libraries(${testExe} PRIVATE pmacc::pmacc)
    if(${testExe} MATCHES ".*CLI.*")
      target_link_libraries(${testExe} PUBLIC argparser)
    endif()
    target_include_directories(
        ${testExe}
        SYSTEM PRIVATE
        $<TARGET_PROPERTY:nlohmann_json::nlohmann_json,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(${testExe} PRIVATE nlohmann_json::nlohmann_json)
    target_link_libraries(${testExe} PRIVATE openPMD::openPMD)
    target_compile_definitions(${testExe} PUBLIC ENABLE_OPENPMD=1)
    add_test(NAME "${testCase}" COMMAND mpiexec ${MPI_RUNTIME_FLAGS} -n 1 ./${testExe})
endforeach()
