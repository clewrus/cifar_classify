cmake_minimum_required(VERSION 3.16)   # use a modern version (4.1 doesn’t exist)

project(CifarClassify LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# --- External deps ---
set(ORT_DIR "${CMAKE_SOURCE_DIR}/ext/onnxruntime-win-x64-1.22.1")
set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/ext/opencv")

# --- Utils library ---
add_library(utils src/utils.cpp)

# ONNX Runtime
target_include_directories(utils PUBLIC "${ORT_DIR}/include")
target_link_libraries(utils PUBLIC "${ORT_DIR}/lib/onnxruntime.lib")

# OpenCV
target_include_directories(utils PUBLIC "${OpenCV_DIR}/include")
target_link_libraries(utils PUBLIC
    debug "${OpenCV_DIR}/lib/opencv_world4120d.lib"
    optimized "${OpenCV_DIR}/lib/opencv_world4120.lib"
)

# Copy DLL next to the executable
function(copy_runtime_dlls target_name)
    add_custom_command(TARGET ${target_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ORT_DIR}/lib/onnxruntime.dll"
            $<TARGET_FILE_DIR:${target_name}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<$<CONFIG:Debug>:"${OpenCV_DIR}/lib/opencv_world4120d.dll">
            $<$<CONFIG:Release>:"${OpenCV_DIR}/lib/opencv_world4120.dll">
            $<TARGET_FILE_DIR:${target_name}>
    )
endfunction()

# --- Main executable ---
add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE utils)
copy_runtime_dlls(${PROJECT_NAME})


# --- Tests ---
enable_testing()
add_subdirectory(ext/googletest-1.17.0)

add_executable(Tests src/test.cpp)
target_link_libraries(Tests PRIVATE utils GTest::gtest_main)
copy_runtime_dlls(Tests)

# Enable path to the sample data inside test binary
target_compile_definitions(Tests PRIVATE
    SAMPLES_DIR="${CMAKE_SOURCE_DIR}/../samples"
    MODELS_DIR="${CMAKE_SOURCE_DIR}/../models"
)

include(GoogleTest)
gtest_discover_tests(Tests)
