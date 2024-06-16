

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <assert.h>

#include <math.h>
#include <iostream>

#define TILE_WIDTH 16

__global__ void matrixMultiply(int *c, const int *a, const int *b, int m , int n , int k )
{
    //DEVICE - GPU

    __shared__ int ds_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ int ds_b[TILE_WIDTH][TILE_WIDTH];

    int Row = blockIdx.y * blockDim.y  + threadIdx.y ;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    long cValue = 0.0;
    for (int t = 0; t < n / TILE_WIDTH ; t++) {

        //Loads one cell in a tile for A and B. So for t = 0,1,2,3 it is loading 
        // So for a it loads  a[Row * 64 + 0 * TILE_WIDTH + tx]  , a[Row * 64 + 1 * TILE_WIDTH + tx]   ,  a[Row * 64 + 2 * TILE_WIDTH + tx]   , a[Row * 64 + 3 * TILE_WIDTH + tx]  
        // So for b it loads  b[  (0 * TILE_WIDTH + ty) * 32 + Col ]  , b[(1 * TILE_WIDTH + ty) * 32 + Col]   ,  b[(0 * TILE_WIDTH + ty) * 32 + Col]   , b[(0 * TILE_WIDTH + ty) * 32 + Col]  


        ds_a[ty][tx] = a[ Row  * n + t * TILE_WIDTH + tx ] ;
        ds_b[ty][tx] = b[ (t * TILE_WIDTH + ty) * k + Col];

        __syncthreads();

        //Multiply to column of A by row of B of tx, ty output cell

        for (int i = 0; i < TILE_WIDTH; i++) {
            cValue += ds_a[tx][i] * ds_b[i][ty];
        }

        __syncthread();
    
    }

    c[Row * k + Col ] = cValue





}

int main()
{
    //HOST - DESKTOP/ CPU

    int m = 1 << 7; // 128;
    int n = 1 << 6; // 64
    int k = 1 << 5; // 32

    
    size_t a_bytes = m * n * sizeof(int);
    size_t b_bytes = n * k * sizeof(int);
    size_t c_bytes = m * k * sizeof(long);

    int* h_a, * h_b;
    long* h_c, * v_c;

    h_a = (int*)malloc(a_bytes);
    h_b = (int*)malloc(b_bytes);
    h_c = (long*)malloc(c_bytes);
    v_c = (long*)malloc(c_bytes);



    int* dev_a, * dev_b;
    long* dev_c;

    cudaError_t cudaStatus;


    cudaStatus = cudaMalloc((void**)&dev_a, a_bytes);
    cudaStatus = cudaMalloc((void**)&dev_b, b_bytes);
    cudaStatus = cudaMalloc((void**)&dev_c, c_bytes);

    cudaStatus = cudaMemcpy(dev_a, h_a, a_bytes, cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_b, h_b, b_bytes, cudaMemcpyHostToDevice);



    int y = m / TILE_WIDTH;
    int x = k / TILE_WIDTH; 

    dim3 grd(x, y);
    dim3 blk(TILE_WIDTH , TILE_WIDTH);


    matrixMultiply <<< grd, blk >>> (dev_c, dev_a, dev_b, n, m , n  , k);

    cudaDeviceSynchronize();

    cudaStatus = cudaMemcpy(h_c, dev_c, c_bytes, cudaMemcpyDeviceToHost);


    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    free(h_a);
    free(h_b);
    free(h_c);
    free(v_c);


    return 0;
}
