//
// Created by lidan on 22/10/2020.
//

#ifndef CUDAOPENGL_SHADER_COMMON_H
#define CUDAOPENGL_SHADER_COMMON_H
#include <glad/glad.h>
#include <cstdio>
#include <string>
#include <fstream>
#include <exception>

using namespace std ;

struct ShaderStringHelper{
    const char *p ;
    ShaderStringHelper(const string& s):p(s.c_str()) {};
    operator  const char**(){ return &p ;}
};

inline static std::string loadFileToString(const char* filename)
{
    std::fstream file(filename,std::ios::in) ;
    std::string text ;
    if(file)
    {
        file.seekg(0,std::ios::end) ;
        text.resize(file.tellg()) ;
        file.seekg(0,std::ios::beg) ;
        file.read(&text[0],text.size()) ;
        file.close() ;
    }else{
        std::string error_message = std::string("File not found: ") + filename;
        fprintf(stderr, error_message.c_str());
        throw std::runtime_error(error_message);
    }
    return text ;
}

#endif //CUDAOPENGL_SHADER_COMMON_H
