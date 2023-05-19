/* file: hello.cu */
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void add(int a, int b, int *c)
{
    *c = a + b;
}

int main()
{
    int *c;
    c = (int*)malloc(sizeof(int));
    int *dev_c;
    cudaMalloc((void **)&dev_c, sizeof(int));
    add<<<1, 1>>>(2, 7, dev_c);
    cudaMemcpy(c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
    printf("[cuda] 2 + 7 = %d\n",*c);
    return 0;
}