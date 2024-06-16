
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__global__ void matrixMultiply(int *c, const int *a, const int *b, int m , int n , int k)
{

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    long tempSum= 0;
    int i = threadIdx.x;
    for (int i = 0; i < n; i++) {
      tempSum += a[ row * n + i ] + b[i * k + col ];

    }
    c[row  * k + col] = tempSum;
   
}

int main()
{

    int m = 1 << 7; // 128;
    int n = 1 << 6; // 64
    int k = 1 << 5; // 32


    int l = 1 << 4; // 16





    return 0;
}

