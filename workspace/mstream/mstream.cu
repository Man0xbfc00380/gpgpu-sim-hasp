#include <stdio.h>
#include <cuda_runtime.h>

__global__  void  haspSet_vectorAddKernel_th1_sh5_mem4(){}
__global__  void  haspSet_vectorMulKernel_th2_sh25_mem20(){}
__global__  void  vectorAddKernel(int* a,  int* b,  int* c,  int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) c[tid] = a[tid] + b[tid];
}
__global__  void  vectorMulKernel(int* a,  int* b,  int* c,  int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) c[tid] = a[tid] * b[tid];
}

int main()
{
    int N  = 500000;
    int N1 = 100000;
    int N2 = 500000;

    int* h_a = new int[N];
    int* h_b = new int[N];
    int* h_c = new int[N];
    int* h_d = new int[N];

    for (int i = 0; i < N; i++) {
        h_a[i] = 1 + i;
        h_b[i] = 2;
    }

    int *d_a, *d_b, *d_c, *d_d;
    
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    cudaMalloc(&d_c, N * sizeof(int));
    cudaMalloc(&d_d, N * sizeof(int));

    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 64;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    printf("[Grid: %d, Block: %d]\n", blocksPerGrid, threadsPerBlock);
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    haspSet_vectorAddKernel_th1_sh5_mem4<<<1, 1, 0, stream1>>> ();
    haspSet_vectorMulKernel_th2_sh25_mem20<<<1, 1, 0, stream2>>> ();

    vectorAddKernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_a, d_b, d_c, N1);
    cudaMemcpyAsync(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost, stream1);

    vectorMulKernel<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_a, d_b, d_d, N2);
    cudaMemcpyAsync(h_d, d_d, N * sizeof(int), cudaMemcpyDeviceToHost, stream2);

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);

    for (int i = 0; i < 5; i++) {
        printf("id: %d, (i+1) + 2 = %d, (i+1) * 2 = %d\n", i, h_c[i], h_d[i]);
    }
    printf("... ...\n");
    for (int i = 995; i < 1000; i++) {
        printf("id: %d, (i+1) + 2 = %d, (i+1) * 2 = %d\n", i, h_c[i], h_d[i]);
    }

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] h_d;

    return 0;
}

