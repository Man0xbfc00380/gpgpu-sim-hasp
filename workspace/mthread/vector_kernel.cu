#include  <stdio.h>
#include  <stdlib.h>
#include  <cuda_runtime.h>
#include  "vector_kernel.h"


__global__  void  vectorAddKernel( float   * a,  float   * b,  float   * c,  int  n)
{
    int  tid;
    tid  =  threadIdx.x;
    if (tid  <  n) c[tid]  =  a[tid]  +  b[tid];
}

__global__  void  vectorSubKernel( float   * a,  float   * b,  float   * c,  int  n)
{
    int  tid;
    tid  =  threadIdx.x;
    if (tid  <  n) c[tid]  =  a[tid]  -  b[tid];
}

__global__  void  vectorMulKernel( float   * a,  float   * b,  float   * c,  int  n)
{
    int  tid;
    tid  =  threadIdx.x;
    if (tid  <  n) c[tid]  =  a[tid]  *  b[tid];
}
