
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include <stdio.h>

//#include <helper_cuda.h>
//#include <helper_functions.h>

#include <assert.h>
#include <stdint.h>

#include <math.h>
#include <iostream>

using namespace std;

cudaError_t multiplyMatrixWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void multiplyMatrixKernel(int *c, const int *a, const int *b, int n , clock_t *timer )
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;


    int bid = blockIdx.y * gridDim.x + blockIdx.x ;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int gridSize = gridDim.x * gridDim.y;

    int temp_sum = 0;

    if(tid == 0 ) timer[bid] = clock();

    if (row < n && col < n) {
        for (int k = 0; k < n; k++) {
            temp_sum += a[row * n + k] * b[ k * n + col ]; 
        }
        c[row * n + col] = temp_sum;
    
    }
    __syncthreads();

    if (tid == 0) timer[bid ] = clock();

}

void init_matrices(int * a, int * b , int n) {

    for (int i = 0; i < n * n ; i++) {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }

}

int main()
{
    const int n = 1 << 10;
    size_t bytes = n * n * sizeof(int);

    int *h_a, *h_b , *h_c , *v_c;
    
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);
    v_c = (int*)malloc(bytes);

    init_matrices(h_a , h_b , n);


    clock_t startTimer = clock();
    for (int i = 0; i < n ; i++) {
    
        for (int j = 0; j < n; j++) {
        
            int temp_sum = 0;
            for (int k = 0; k < n; k++) {
                temp_sum += h_a[i * n  + k] * h_b[ k * n + j ];
            }
            v_c[i * n + j] = temp_sum;
        }
    }
    clock_t endTimer = clock();

    cout << "Elapsed time : " << (  endTimer - startTimer) << endl;



    // Add vectors in parallel.
    cudaError_t cudaStatus = multiplyMatrixWithCuda(h_c, h_a, h_b, n );
    free(h_a);
    free(h_b);
    free(h_c);
    free(v_c);

    if (cudaStatus != cudaSuccess) {
        cout << "multiplyMatrixWithCuda failed!" << endl;
        return 1;
    }



    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        //fprintf(stderr, "cudaDeviceReset failed!");
        cout << "cudaDeviceReset failed!" << endl;
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t multiplyMatrixWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        //fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        cout << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?" << endl;

    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        //fprintf(stderr, "cudaMalloc failed!");
        cout << "cudaMalloc failed!" << endl;

    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        //fprintf(stderr, "cudaMalloc failed!");
        cout << "cudaMalloc failed!" << endl;

    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        //fprintf(stderr, "cudaMalloc failed!");
        cout << "cudaMalloc failed!" << endl;

    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        //fprintf(stderr, "cudaMemcpy failed!");
        cout << "cudaMalloc failed!" << endl;

    }

    cudaStatus = cudaMemcpy(dev_b, b, size * size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        //fprintf(stderr, "cudaMemcpy failed!");
        cout << "cudaMalloc failed!" << endl;

    }

    int BLOCK_SIZE = 16;
    int GRID_SIZE = (int)ceil(size / BLOCK_SIZE);
    int NUM_BLOCKS = GRID_SIZE * GRID_SIZE ;

    clock_t *dtimer = NULL;

    clock_t *timer = (clock_t *) malloc(2 * NUM_BLOCKS * sizeof(clock_t));

    dim3 grid(GRID_SIZE, GRID_SIZE);
    dim3 block(BLOCK_SIZE , BLOCK_SIZE);

    // Launch a kernel on the GPU with one thread for each element.
    multiplyMatrixKernel <<< grid, block >>>(dev_c, dev_a, dev_b, size , dtimer);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        //fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cout << "addKernel launch failed: " << cudaGetErrorString(cudaStatus) << endl;

    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        //fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        cout <<  "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching addKernel! " << endl;

    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(dev_c , c,  size * size *  sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        //fprintf(stderr, "cudaMemcpy failed!");
        cout << "cudaMemcpy failed!! "  << endl;

    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(dtimer , timer  , size * size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        //fprintf(stderr, "cudaMemcpy failed!");
        cout << "cudaMemcpy failed!! " << endl;

    }

    long double avgElapsedClocks = 0;

    for (int i = 0; i < NUM_BLOCKS; i++) {
        avgElapsedClocks += (long double)(timer[i + NUM_BLOCKS] - timer[i]);
    }


    avgElapsedClocks = avgElapsedClocks / NUM_BLOCKS;
    printf("Average clocks/block = %Lf\n", avgElapsedClocks);

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
