//
// Created by lidan on 22/10/2020.
//

#include <cuda_runtime.h>
#include <string>
#include <iostream>
#include <stdlib.h>

cudaError_t cuda();

__global__ void kernel(){

}

template<typename op>
__device__ op _clamp(op x, op a, op b)
{
    return max(a,min(x,b)) ;
}

__device__ int rgbToInt(float r, float g, float b)
{
    r = _clamp<float>(r, 0.0f, 255.0f);
    g = _clamp<float>(g, 0.0f, 255.0f);
    b = _clamp<float>(b, 0.0f, 255.0f);
    return (int(b) << 16) | (int(g) << 8) | int(r);
}

// make uchar4 for write char4

__global__ void
cudaRender(unsigned int *g_odata,int imgw)
{
    extern __shared__ uchar4 sdata[] ;

    int tx = threadIdx.x ;
    int ty = threadIdx.y ;
    int bx = blockDim.x ;
    int by = blockDim.y ;

    int x = tx + blockIdx.x*bx ;
    int y = ty + blockIdx.y*by ;

    uchar4 c4 = make_uchar4((x & 0x20) ? 100 : 0, 0, (y & 0x20) ? 100 : 0, 0);
    g_odata[y*imgw+x]  = rgbToInt(c4.z, c4.y, c4.x);
}


struct cuComplex{
    float r;
    float i;

    __device__  cuComplex(float x,float y) : r(x),i(y){}

    __device__ float manitude2(void)
    {
        return r*r + i* i ;
    }

    __device__ cuComplex operator*(const cuComplex& a)
    {
        return cuComplex(r*a.r-i*a.i ,i*a.r + r* a.i ) ;
    }

    __device__ cuComplex operator+(const cuComplex& a)
    {
        return cuComplex(r+a.r,i+a.i ) ;
    }
};

__device__ int julia(int src_width , int src_height, int x,int y,float scale)
{
    float jx = scale * (float) (src_width/2- x)/(src_width/2) ;
    float jy = scale * (float) (src_height/2 - y)/(src_height/2) ;

    cuComplex c(-0.8, 0.156) ;
    cuComplex d(jx,jy) ;

//    for(int i = 0 ;i <200 ;i++)
//    {
//        d = d*d + c;
//        if(d.manitude2() > 1000)
//        {
//            return 0 ;
//        }
//    }
//
//    return 1 ;


    int iterations = 0;

    while (true) {
        iterations++;
        if (iterations >1000) return 0;
        d = d*d + c;
        if (d.i > 150) return iterations;
        if (d.r > 150) return iterations;
    }
    return iterations ;

}


__global__ void kernel(const unsigned int width,const unsigned int height ,unsigned char* ptr,float scale )
{
    const unsigned int  idx = (blockIdx.x* blockDim.x) + threadIdx.x ;
    const unsigned int idy = (blockIdx.y*blockDim.y) + threadIdx.y ;

    const unsigned int tid = idy*gridDim.x*blockDim.x + idx ;

    int x = tid / width ;
    int y = tid % width ;

    if(x < height)
    {
        int juliaValue = julia(width,height,x,y,scale) ;

        ptr[(x*width+y)*4 + 0] = 190+120*juliaValue;
        ptr[(x*width+y)*4 + 1] = 40+35*round(cos(juliaValue/5.0));
        ptr[(x*width+y)*4 + 2] =  18+6*(juliaValue%10);
        ptr[(x*width+y)*4 + 3] = 255;
    }
}


extern "C" void
launch_cudaRender(dim3 grid, dim3 block, int sbytes, unsigned char *g_odata, int imgw)
{
//    cudaRender <<< grid, block, sbytes >>>(g_odata, imgw);
    kernel<<< grid, block, sbytes >>>(imgw,imgw,g_odata, 1.2);
}





