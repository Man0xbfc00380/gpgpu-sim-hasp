#ifndef __VECTOR_KERNEL_H__
#define __VECTOR_KERNEL_H__

#include  <stdio.h>
#include  "cuda_runtime.h"
#include  <stdlib.h>

__global__  void  vectorAddKernel(float* a, float* b, float* c,  int  n);
__global__  void  vectorSubKernel(float* a, float* b, float* c,  int  n);
__global__  void  vectorMulKernel(float* a, float* b, float* c,  int  n);

__global__  void  haspSet_vectorAddKernel_th0_sh10_mem8();
__global__  void  haspSet_vectorSubKernel_th1_sh10_mem8();
__global__  void  haspSet_vectorMulKernel_th1_sh10_mem8();
#endif