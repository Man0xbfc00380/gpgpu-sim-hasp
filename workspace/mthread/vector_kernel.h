#include  <stdio.h>
#include  <cuda_runtime.h>
#include  <stdlib.h>

__global__  void  vectorAddKernel(float* a, float* b, float* c,  int  n);
__global__  void  vectorSubKernel(float* a, float* b, float* c,  int  n);
__global__  void  vectorMulKernel(float* a, float* b, float* c,  int  n);