#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "cudalib/helper_cuda.h"
#include "cudalib/helper_string.h"
#include "cudalib/help_gl.h"

#include <string>
#include <filesystem>
#include "shader_tools/gl_tools.h"
#include "shader_tools/GLSL_Program.h"
#include "shader_tools/Shader.h"

#include "shader_tools/gl_tools.h"
#include "shader_tools/glfw_tools.h"


using namespace std;

GLFWwindow* window = NULL ;
int Width = 256 ;
int Height = 256 ;

GLuint VBO,VAO,EBO ;
Shader drawtex_f; // GLSL fragment shader
Shader drawtex_v; // GLSL fragment shader
GLSLProgram shdrawtex; // GLSLS program for textured draw

// Cuda <-> OpenGl interop resources

void* cuda_dev_render_buffer; // Cuda buffer for initial render
struct cudaGraphicsResource* cuda_tex_resource;
GLuint opengl_tex_cuda;  // OpenGL Texture for cuda result


extern "C" void
// Forward declaration of CUDA render
launch_cudaRender(dim3 grid, dim3 block, int sbytes, unsigned char *g_odata, int imgw);


//CUDA

size_t size_tex_data;
unsigned int num_texels;
unsigned int num_values;

static const char *glsl_drawtex_vertshader_src =
        "#version 330 core\n"
        "layout (location = 0) in vec3 position;\n"
        "layout (location = 1) in vec3 color;\n"
        "layout (location = 2) in vec2 texCoord;\n"
        "\n"
        "out vec3 ourColor;\n"
        "out vec2 ourTexCoord;\n"
        "\n"
        "void main()\n"
        "{\n"
        "	gl_Position = vec4(position, 1.0f);\n"
        "	ourColor = color;\n"
        "	ourTexCoord = texCoord;\n"
        "}\n";

static const char *glsl_drawtex_fragshader_src =
        "#version 330 core\n"
        "uniform usampler2D tex;\n"
        "in vec3 ourColor;\n"
        "in vec2 ourTexCoord;\n"
        "out vec4 color;\n"
        "void main()\n"
        "{\n"
        "   	vec4 c = texture(tex, ourTexCoord);\n"
        "   	color = c / 255.0;\n"
        "}\n";



// QUAD GEOMETRY
GLfloat vertices[] = {
        // Positions          // Colors           // Texture Coords
        1.0f, 1.0f, 0.5f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f,  // Top Right
        1.0f, -1.0f, 0.5f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,  // Bottom Right
        -1.0f, -1.0f, 0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,  // Bottom Left
        -1.0f, 1.0f, 0.5f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f // Top Left
};
// you can also put positions, colors and coordinates in seperate VBO's
GLuint indices[] = {  // Note that we start from 0!
        0, 1, 3,  // First Triangle
        1, 2, 3   // Second Triangle
};


// Create 2D OpenGL texture in gl_tex and bind it to CUDA in cuda_tex
void createGLTextureForCUDA(GLuint* gl_tex, cudaGraphicsResource** cuda_tex, unsigned int size_x, unsigned int size_y)
{
    // create an OpenGL texture
    glGenTextures(1, gl_tex); // generate 1 texture
    glBindTexture(GL_TEXTURE_2D, *gl_tex); // set it as current target
    // set basic texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // clamp s coordinate
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // clamp t coordinate
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    // Specify 2D texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI, size_x, size_y, 0, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, NULL);
    // Register this texture with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterImage(cuda_tex, *gl_tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
    SDK_CHECK_ERROR_GL();
}


void createGLTextureForGpuProcess(GLuint* gl_tex, cudaGraphicsResource** cuda_tex,unsigned int size_x,unsigned int size_y)
{
    glGenTextures(1,gl_tex) ;
    glBindTexture(GL_TEXTURE_2D,*gl_tex) ;

    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE) ;
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE) ;
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST) ;
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST) ;

    // Specify 2D texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI, size_x, size_y, 0, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, NULL);
    // Register this texture with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterImage(cuda_tex, *gl_tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
    SDK_CHECK_ERROR_GL();

}


void initGLBuffers()
{
    // create texture that will receive the result of cuda kernel
    createGLTextureForCUDA(&opengl_tex_cuda, &cuda_tex_resource, Width, Height);
    // create shader program
    drawtex_v = Shader("Textured draw vertex shader", glsl_drawtex_vertshader_src, GL_VERTEX_SHADER);
    drawtex_f = Shader("Textured draw fragment shader", glsl_drawtex_fragshader_src, GL_FRAGMENT_SHADER);
    shdrawtex = GLSLProgram(&drawtex_v, &drawtex_f);
    shdrawtex.compile();
    SDK_CHECK_ERROR_GL();
}

// Keyboard
void keyboardfunc(GLFWwindow* window, int key, int scancode, int action, int mods){
}


bool initGL(){


    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        glfwTerminate() ;
        return 0;
    }


    glViewport(0, 0, Width, Height); // viewport for x,y to normalized device coordinates transformation
    SDK_CHECK_ERROR_GL();



    return true;
}


void initCUDABuffers()
{
    // set up vertex data parameters
    num_texels = Width * Width;
    num_values = num_texels * 4;
    size_tex_data = sizeof(GLubyte) * num_values;
    // We don't want to use cudaMallocManaged here - since we definitely want
    checkCudaErrors(cudaMalloc(&cuda_dev_render_buffer, size_tex_data)); // Allocate CUDA memory for color output
}
bool initGLFW(){
    if (!glfwInit()) exit(EXIT_FAILURE);
    // These hints switch the OpenGL profile to core
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    window = glfwCreateWindow(Width, Width, "SimpleCUDA2GL Modern OpenGL", NULL, NULL);
    if (!window){ glfwTerminate(); exit(EXIT_FAILURE); }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetKeyCallback(window, keyboardfunc);
    return true;
}


void generateCUDAImage()
{
    // calculate grid size
    dim3 block(16, 16, 1);
    dim3 grid(Width / block.x, Height / block.y, 1); // 2D grid, every thread will compute a pixel
    launch_cudaRender(grid, block, 0, (unsigned char *) cuda_dev_render_buffer, Width); // launch with 0 additional shared memory allocated

    // We want to copy cuda_dev_render_buffer data to the texture
    // Map buffer objects to get CUDA device pointers
    cudaArray *texture_ptr;
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex_resource, 0));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_resource, 0, 0));

    int num_texels = Width * Height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;
    checkCudaErrors(cudaMemcpyToArray(texture_ptr, 0, 0, cuda_dev_render_buffer, size_tex_data, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_tex_resource, 0));
}

void display(void)
{
    generateCUDAImage() ;
    glfwPollEvents() ;

    glClearColor(0.2,0.3,0.3,1.0) ;
    glClear(GL_COLOR_BUFFER_BIT) ;

    glActiveTexture(GL_TEXTURE0) ;
    glBindTexture(GL_TEXTURE_2D,opengl_tex_cuda) ;

    shdrawtex.use() ;
    glUniform1i(glGetUniformLocation(shdrawtex.program,"tex"),0) ;

    glBindVertexArray(VAO) ;
    glDrawElements(GL_TRIANGLES,6,GL_UNSIGNED_INT,0) ;
    glBindVertexArray(0) ;

    SDK_CHECK_ERROR_GL() ;

    glfwSwapBuffers(window) ;
}



int main(int argc,char** argv) {
    initGLFW() ;
    initGL() ;

    printGLFWInfo(window) ;
    printGLInfo() ;

    findCudaDevice(argc, (const char **)argv);
    initGLBuffers();
    initCUDABuffers();

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);


    glBindVertexArray(VAO) ;
    glBindBuffer(GL_ARRAY_BUFFER,VBO) ;
    glBufferData(GL_ARRAY_BUFFER,sizeof(vertices),vertices,GL_STATIC_DRAW) ;

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // Position attribute (3 floats)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(0);
    // Color attribute (3 floats)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
    glEnableVertexAttribArray(1);
    // Texture attribute (2 floats)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)(6 * sizeof(GLfloat)));
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    // Note that this is allowed, the call to glVertexAttribPointer registered VBO as the currently bound
    // vertex buffer object so afterwards we can safely unbind
    glBindVertexArray(0);

    while (!glfwWindowShouldClose(window))
    {
        display();
        glfwWaitEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    exit(EXIT_SUCCESS);

}
