//
// Created by lidan on 22/10/2020.
//

#ifndef CUDAOPENGL_SHADER_H
#define CUDAOPENGL_SHADER_H

#include "shader_common.h"

class Shader {
public:
    GLuint shader ;
    GLint compiled ;
    GLenum shadertype ;
    std::string shader_name ;

private:
    std::string shader_src ;


public:
    Shader::Shader() ;
    Shader::Shader(const std::string &shader_name, const char *shader_text, GLenum shadertype);
    Shader::Shader(const std::string &shader_name, const std::string &shader_text, GLenum shadertype);
    std::string Shader::getsrc() const ;
    void Shader::setSrc(const std::string &new_source);
    void Shader::setSrc(const char* new_source);
    void Shader::compile();
private:
    void Shader::getCompilationError(GLuint shader);
};


#endif //CUDAOPENGL_SHADER_H
