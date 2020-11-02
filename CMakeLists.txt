cmake_minimum_required(VERSION 3.17)
project(cudaopengl LANGUAGES C CUDA CXX )

set(CMAKE_CXX_STANDARD 14)

include(cmake/lidanBase.cmake)
include(cmake/LidanBuild.cmake)

ADD_SUBSRC(cudadir cudalib)
ADD_SUBSRC(gladdir externel/)
ADD_SUBSRC(shaderdirs shader_tools)

Lidan_ADD_LIB(PATH "${CMAKE_CURRENT_SOURCE_DIR}/externel/glfw/" INCLUDEDIRS includedir MSVC "TRUE" SOURCEDIR glfwsrc COMPONENTS "glfw3")

include_directories(C:\\Users\\lidan\\CLionProjects\\cudaopengl\\externel\\glad\\include
                    C:\\Users\\lidan\\CLionProjects\\cudaopengl\\cudalib)
add_executable(cudaopengl main.cu cudalib/help_gl.h cudalib/helper_string.h
        ${cudadir} ${gladdir} ${glfwsrc} ${shaderdirs} shader_tools/gl_tools.h shader_tools/kernel.cu shader_tools/Shader.cpp
        shader_tools/Shader.h shader_tools/GLSL_Program.cpp shader_tools/GLSL_Program.h shader_tools/shader_common.h)

target_link_libraries(${PROJECT_NAME} glfw3 )