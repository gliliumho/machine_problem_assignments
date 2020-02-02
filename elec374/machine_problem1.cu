#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <Windows.h>

#define N 1024			//Width of matrices
#define BLOCK_WIDTH 4	//Block width for GPU kernel call
#define RANGE 10.0	//Range of random floating point numbers

__global__ void matrixMult_GPU(float *dA, float *dB, float *dC)
{
	// int i = blockIdx.x * blockDim.x + threadIdx.x;
    float product = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N){
		for (int i=0; i<N; i++)
		//product += dA[blockIdx.x*blockDim.x + i] * dB[i* blockDim.x+threadIdx.x];
		product += dA[row*N + i] * dB[i* N + col];

		dC[row*N + col] = product;
	}
}


__host__ void matrixMult_CPU(float A[N*N], float B[N*N], float C[N*N])
{

    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            float product = 0;
			for (int k=0; k<N; k++)
                product += A[i*N + k] * B[k*N + j];

            C[i*N + j] = product;
        }
    }
}


//refer CUDA programming guide pg21
int main()
{
	// Initialize seeder for rand. num. generator
	srand((unsigned)time(NULL));

	// Allocate memory for matrices
	size_t size =  N*N*sizeof(float);
    float *hA = (float*)malloc(size);
    float *hB = (float*)malloc(size);
    float *cpu_result = (float*)malloc(size);
    float *cuda_result = (float*)malloc(size);

	// Initialize hA and hB with random floating-point values
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            hA[i*N + j] = (float)rand()/(float)(RAND_MAX/RANGE);
            hB[i*N + j] = (float)rand()/(float)(RAND_MAX/RANGE);
        }
    }

    // CPU TIMING.....
    LARGE_INTEGER freq;
    LARGE_INTEGER t1, t2, tDiff;
    double elapsedtime_CPU = 0.0;
    float elapsedtime_GPU = 0.0;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t1);

	//Your CPU function(s) goes here...
    matrixMult_CPU(hA, hB, cpu_result);

    QueryPerformanceCounter(&t2);
    tDiff.QuadPart = t2.QuadPart -t1.QuadPart;
    double freq_ms = (double) freq.QuadPart / 1000.0;
    elapsedtime_CPU = tDiff.QuadPart / freq_ms;
    printf("Elapsed time on the CPU function(s) is %lf ms \n", elapsedtime_CPU);


	/********************************************************************************
		GPU part
	********************************************************************************/

	// Declare and allocate GPU memory
	float *dev_a = 0;
	float *dev_b = 0;
	float *dev_c = 0;
	cudaMalloc(&dev_a, size);
    cudaMalloc(&dev_b, size);
    cudaMalloc(&dev_c, size);

	// Initialize result memory (to make sure we're not getting old values
	// if my code isn't working properly
	cudaMemset(dev_c, 0, N * N * sizeof(float));

	// GPU Timing
	cudaEvent_t t_start, t_stop;
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_stop);
	cudaEventRecord(t_start,0);

	// Copy matrices from host memory to GPU memory
    cudaMemcpy(dev_a, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, hB, size, cudaMemcpyHostToDevice);

    //cudaEventRecord(t_stop,0);
    //cudaEventSynchronize(t_stop);
    //cudaEventElapsedTime(&elapsedtime_GPU, t_start, t_stop);
    //printf("Elapsed time for cudaMemcpy() is %f ms\n",elapsedtime_GPU);


	// Define configuration for kernel
	int numBlocks = N/BLOCK_WIDTH;
	if (N % BLOCK_WIDTH) numBlocks++;

	dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
	dim3 blockGrid(numBlocks, numBlocks);


	// Call GPU kernel & synchronize (to wait until all computation done)
    matrixMult_GPU<<< blockGrid, dimBlock >>>(dev_a, dev_b, dev_c);
    cudaDeviceSynchronize();

	// Copy result from GPU memory to host memory
	cudaMemcpy(cuda_result, dev_c, size, cudaMemcpyDeviceToHost);


    cudaEventRecord(t_stop,0);
    cudaEventSynchronize(t_stop);
    cudaEventElapsedTime(&elapsedtime_GPU, t_start, t_stop);
    printf("Elapsed time on the GPU function(s) is %f ms\n",elapsedtime_GPU);


	// Free allocated GPU memory
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    // Verify GPU result with CPU result
	int correct = 1;
    for(int i=0; i<N; i++)
        for(int j=0; j<N; j++)
            if(cpu_result[i*N + j] != cuda_result[i*N + j])
                correct=0;
			// else
			// 	printf("i=%d,j=%d,a=%1f,b=%1f cpu=%1f, gpu=%1f\n", i, j, hA[i*N + j], hB[i*N + j], cpu_result[i*N + j], cuda_result[i*N + j]);

	// Free allocated host memory
	free(hA);
	free(hB);
	free(cpu_result);
	free(cuda_result);

    if(correct == 1)
        printf("Verification passed!\n\n");
    else
        printf("Verification failed!\n\n");

	return 0;
}
