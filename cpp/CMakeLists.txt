cmake_minimum_required(VERSION 4.1)

# Project name and language
project(CifarClassify LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add executable from source file
add_executable(${PROJECT_NAME} src/main.cpp)

# ONNX Runtime
SET(ORT_DIR "${CMAKE_SOURCE_DIR}/ext/onnxruntime-win-x64-1.22.1")
target_include_directories(${PROJECT_NAME} PRIVATE "${ORT_DIR}/include")
target_link_libraries(${PROJECT_NAME} PRIVATE "${ORT_DIR}/lib/onnxruntime.lib")

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ORT_DIR}/lib/onnxruntime.dll"
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
)

# OpenCV
SET(OpenCV_DIR "${CMAKE_SOURCE_DIR}/ext/opencv")
target_include_directories(${PROJECT_NAME} PRIVATE "${OpenCV_DIR}/include")
target_link_libraries(${PROJECT_NAME} PRIVATE
    debug "${OpenCV_DIR}/lib/opencv_world4120d.lib"
    optimized "${OpenCV_DIR}/lib/opencv_world4120.lib"
)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<$<CONFIG:Debug>:"${OpenCV_DIR}/lib/opencv_world4120d.dll">
        $<$<CONFIG:Release>:"${OpenCV_DIR}/lib/opencv_world4120.dll">
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
)