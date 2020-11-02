//
// Created by lidan on 22/10/2020.
//

#pragma once
#include "shader_common.h"
#include "Shader.h"
#include <glad/glad.h>

class GLSLProgram {
public:
    GLuint program;
    bool linked;
private:
    Shader* vertex_shader;
    Shader* fragment_shader;
public:
    GLSLProgram::GLSLProgram();
    GLSLProgram::GLSLProgram(Shader* vertex, Shader* fragment);
    void GLSLProgram::compile();
    void GLSLProgram::use();
private:
    void GLSLProgram::printLinkError(GLuint program);
};