#pragma once
#include <GLFW/glfw3.h>
#include <string>
#include <iostream>

using namespace std ;

void printGLFWInfo(GLFWwindow* window)
{
    int p = glfwGetWindowAttrib(window,GLFW_OPENGL_PROFILE) ;
    string version = glfwGetVersionString();
    string opengl_profile = "";
    if(p == GLFW_OPENGL_COMPAT_PROFILE)
    {
        opengl_profile = "Use compatibility profile" ;

    }else if(p == GLFW_OPENGL_CORE_PROFILE)
    {
        opengl_profile = "Using Core profile" ;
    }

    std::cout<<"version of GLFW is:"<<version<<std::endl ;
    std::cout<<"profile of GLFW is"<<opengl_profile<<std::endl ;
}