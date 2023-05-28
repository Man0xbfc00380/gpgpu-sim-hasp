#include <stdio.h>
#include <cuda_runtime.h>

// User-Level Specific Function Interface
__global__  void  haspSet_th1_sh5_mem4(){}
__global__  void  haspSet_th2_sh24_mem20(){}
__global__  void  haspMalloc_th1(){}
__global__  void  haspMalloc_th2(){}
__global__  void  haspUnset_th1(){}
__global__  void  haspUnset_th2(){}

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
    int N  = 50000;
    int N1 = 10000;
    int N2 = 50000;

    int* h_a = new int[N];
    int* h_b = new int[N];
    int* h_c = new int[N];
    int* h_d = new int[N];

    for (int i = 0; i < N; i++) {
        h_a[i] = 1 + i;
        h_b[i] = 2;
    }

    int *d_a1, *d_b1, *d_a2, *d_b2, *d_c, *d_d;
    
    int threadsPerBlock = 64;
    int blocksPerGrid1 = (N1 + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid2 = (N2 + threadsPerBlock - 1) / threadsPerBlock;

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    haspSet_th1_sh5_mem4<<<1, 1, 0, stream1>>> ();
    haspSet_th2_sh24_mem20<<<1, 1, 0, stream2>>> ();

    haspMalloc_th1<<<1, 1, 0, stream1>>>(); cudaMalloc(&d_a1, N1 * sizeof(int));
    haspMalloc_th1<<<1, 1, 0, stream1>>>(); cudaMalloc(&d_b1, N1 * sizeof(int));
    haspMalloc_th2<<<1, 1, 0, stream2>>>(); cudaMalloc(&d_a2, N2 * sizeof(int));
    haspMalloc_th2<<<1, 1, 0, stream2>>>(); cudaMalloc(&d_b2, N2 * sizeof(int));
    haspMalloc_th1<<<1, 1, 0, stream1>>>(); cudaMalloc(&d_c, N1 * sizeof(int));
    haspMalloc_th2<<<1, 1, 0, stream2>>>(); cudaMalloc(&d_d, N2 * sizeof(int));

    printf("cudaMalloc Addr: %p-%p-%p-%p-%p-%p\n", d_a1, d_b1, d_a2, d_b2, d_c, d_d);

    cudaMemcpy(d_a1, h_a, N1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, h_b, N1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a2, h_a, N2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, h_b, N2 * sizeof(int), cudaMemcpyHostToDevice);

    vectorAddKernel<<<blocksPerGrid1, threadsPerBlock, 0, stream1>>>(d_a1, d_b1, d_c, N1);
    cudaMemcpyAsync(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost, stream1);
    haspUnset_th1<<<1, 1, 0, stream1>>> ();

    vectorMulKernel<<<blocksPerGrid2, threadsPerBlock, 0, stream2>>>(d_a2, d_b2, d_a2, N2);
    vectorMulKernel<<<blocksPerGrid2, threadsPerBlock, 0, stream2>>>(d_a2, d_b2, d_d, N2);
    cudaMemcpyAsync(h_d, d_d, N * sizeof(int), cudaMemcpyDeviceToHost, stream2);
    haspUnset_th2<<<1, 1, 0, stream2>>> ();

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    cudaFree(d_a1);
    cudaFree(d_b1);
    cudaFree(d_a2);
    cudaFree(d_b2);
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

