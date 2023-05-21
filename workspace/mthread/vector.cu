#include  "vector.h"
#include  "vector_kernel.h"

void  vectorAdd( float   * a,  float   * b,  float   * c,  int  n) {
    float   * d_a,  * d_b,  * d_c;
    haspSet_vectorAddKernel_th0_sh10_mem8<<< 1, 1 >>> ();
    cudaMalloc(( void   ** ) & d_a, n  *   sizeof ( float ));
    cudaMemcpy(d_a, a, n  *   sizeof ( float ), cudaMemcpyHostToDevice);
    cudaMalloc(( void   ** ) & d_b, n  *   sizeof ( float ));
    cudaMemcpy(d_b, b, n  *   sizeof ( float ), cudaMemcpyHostToDevice);
    cudaMalloc(( void   ** ) & d_c, n  *   sizeof ( float ));
    cudaMemcpy(d_b, b, n  *   sizeof ( float ), cudaMemcpyHostToDevice);
    vectorAddKernel <<< 1 , n >>> (d_a, d_b, d_c, n);
    cudaMemcpy(c, d_c, n  *   sizeof ( float ), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

void  vectorSub( float   * a,  float   * b,  float   * c,  int  n) {
    float   * d_a,  * d_b,  * d_c;
    haspSet_vectorSubKernel_th1_sh10_mem8 <<< 1, 1 >>> ();
    cudaMalloc(( void   ** ) & d_a, n  *   sizeof ( float ));
    cudaMemcpy(d_a, a, n  *   sizeof ( float ), cudaMemcpyHostToDevice);
    cudaMalloc(( void   ** ) & d_b, n  *   sizeof ( float ));
    cudaMemcpy(d_b, b, n  *   sizeof ( float ), cudaMemcpyHostToDevice);
    cudaMalloc(( void   ** ) & d_c, n  *   sizeof ( float ));
    cudaMemcpy(d_b, b, n  *   sizeof ( float ), cudaMemcpyHostToDevice);
    vectorSubKernel <<< 1 , n >>> (d_a, d_b, d_c, n);
    cudaMemcpy(c, d_c, n  *   sizeof ( float ), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

void  vectorMul( float   * a,  float   * b,  float   * c,  int  n) {
    float   * d_a,  * d_b,  * d_c;
    haspSet_vectorMulKernel_th1_sh10_mem8 <<< 1, 1 >>> ();
    cudaMalloc(( void   ** ) & d_a, n  *   sizeof ( float ));
    cudaMemcpy(d_a, a, n  *   sizeof ( float ), cudaMemcpyHostToDevice);
    cudaMalloc(( void   ** ) & d_b, n  *   sizeof ( float ));
    cudaMemcpy(d_b, b, n  *   sizeof ( float ), cudaMemcpyHostToDevice);
    cudaMalloc(( void   ** ) & d_c, n  *   sizeof ( float ));
    cudaMemcpy(d_b, b, n  *   sizeof ( float ), cudaMemcpyHostToDevice);
    vectorMulKernel <<< 1 , n >>> (d_a, d_b, d_c, n);
    cudaMemcpy(c, d_c, n  *   sizeof ( float ), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}