/*
ELEC 374 Machine Problem
Name:       Kin Yee Ho
Student ID: 10049579

*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <Windows.h>

#define N 256			//Width of matrices
#define BLOCK_WIDTH 32	//Block width for GPU kernel call
#define RANGE 10.5	//Range of random floating point numbers

__global__ void matrixMult_GPU(float *dA, float *dB, float *dC)
{
	// int i = blockIdx.x * blockDim.x + threadIdx.x;
    float product = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < N && col < N){
		for (int i=0; i<N; i++){
		      product += dA[row*N + i] * dB[i* N + col];
		}
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

__global__ void matrixSum_GPU(float *dC, float *result)
{
    float sum = 0;
	int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N){
		for (int i=0; i<N; i++){
			sum += dC[row*N + i];
		}
		//NOTE: Comment out following line if not compiling under "compute_20,sm_20"
		atomicAdd(result, sum);
	}
}


__host__ void matrixSum_CPU(float C[N*N], float *result)
{
    float sum = 0;
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            sum += C[i*N + j];
        }
    }
    *result = sum;
}

//refer CUDA programming guide pg21
int main()
{
	// Initialize seeder for rand. num. generator
	srand((unsigned)time(NULL));

    printf("Array width = %d, block width = %d\n", N, BLOCK_WIDTH);

	// Allocate memory for matrices
	size_t size =  N*N*sizeof(float);
    float *hA = (float*)malloc(size);
    float *hB = (float*)malloc(size);
    float *cpu_result = (float*)malloc(size);
    float *cuda_result = (float*)malloc(size);
    float cpu_sum = 0;
    float cuda_sum = 0;

	// Initialize hA and hB with random floating-point values
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            hA[i*N + j] = (float)rand()/(float)(RAND_MAX/RANGE);
            hB[i*N + j] = (float)rand()/(float)(RAND_MAX/RANGE);
        }
    }

    LARGE_INTEGER freq;
    LARGE_INTEGER t1, t2, tDiff;
    double freq_ms;
    double elapsedtime_CPU = 0.0;
    double elapsedtime_CPU_total = 0.0;



    // CPU TIMING.....
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t1);

    matrixMult_CPU(hA, hB, cpu_result);

    QueryPerformanceCounter(&t2);
    tDiff.QuadPart = t2.QuadPart -t1.QuadPart;
    freq_ms = (double) freq.QuadPart / 1000.0;
    elapsedtime_CPU = tDiff.QuadPart / freq_ms;
    elapsedtime_CPU_total += elapsedtime_CPU;
    printf("Elapsed time on CPU matrix mult. : \t%lf ms \n", elapsedtime_CPU);



    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t1);

    matrixSum_CPU(cpu_result, &cpu_sum);

    QueryPerformanceCounter(&t2);
    tDiff.QuadPart = t2.QuadPart -t1.QuadPart;
    freq_ms = (double) freq.QuadPart / 1000.0;
    elapsedtime_CPU = tDiff.QuadPart / freq_ms;
    elapsedtime_CPU_total += elapsedtime_CPU;
    printf("Elapsed time on CPU matrix sum. : \t%lf ms \n", elapsedtime_CPU);
    printf("Total elapsed time for CPU functions : \t%lf ms \n\n", elapsedtime_CPU_total);

	/********************************************************************************
		GPU part
	********************************************************************************/

    cudaEvent_t t_start, t_stop;

    // Declare and allocate GPU memory
	float *dev_a = 0;
	float *dev_b = 0;
	float *dev_c = 0;
	//float *dev_d = 0;
    float *dev_sum = 0;
    float elapsedtime_GPU = 0.0;
    float elapsedtime_GPU_total = 0.0;


	cudaMalloc(&dev_a, size);
    cudaMalloc(&dev_b, size);
    cudaMalloc(&dev_c, size);
	cudaMalloc(&dev_sum, sizeof(float));

	// Initialize result memory (to make sure we're not getting old values
	// if my code isn't working properly
	cudaMemset(dev_c, 0, N * N * sizeof(float));
    cudaMemset(dev_sum, 0, sizeof(float));

	// GPU Timing
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_stop);
	cudaEventRecord(t_start,0);

	// Copy matrices from host memory to GPU memory
    cudaMemcpy(dev_a, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, hB, size, cudaMemcpyHostToDevice);

    cudaEventRecord(t_stop,0);
    cudaEventSynchronize(t_stop);
    cudaEventElapsedTime(&elapsedtime_GPU, t_start, t_stop);
    elapsedtime_GPU_total += elapsedtime_GPU;
    printf("Elapsed time on CUDA memcpy (A & B): \t%f ms\n",elapsedtime_GPU);



    cudaEventCreate(&t_start);
    cudaEventCreate(&t_stop);
    cudaEventRecord(t_start,0);

	// Define configuration for kernel
	int numBlocks = N/BLOCK_WIDTH;
	if (N % BLOCK_WIDTH) numBlocks++;

	dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
	dim3 blockGrid(numBlocks, numBlocks);

	// Call GPU kernel & synchronize (to wait until all computation done)
    matrixMult_GPU<<< blockGrid, dimBlock >>>(dev_a, dev_b, dev_c);
    cudaDeviceSynchronize();

    cudaEventRecord(t_stop,0);
    cudaEventSynchronize(t_stop);
    cudaEventElapsedTime(&elapsedtime_GPU, t_start, t_stop);
    elapsedtime_GPU_total += elapsedtime_GPU;
    printf("Elapsed time on CUDA matrix mult. : \t%f ms\n",elapsedtime_GPU);


    cudaEventCreate(&t_start);
    cudaEventCreate(&t_stop);
    cudaEventRecord(t_start,0);

	// NOTE:	copy CPU result to dev_c to replace CUDA result
	//	For some reason, my matrix multiplication fails when
	//	program compiles under "compute_20,sm_20". No issue in "compile_10,sm_10"
	//cudaMemcpy(dev_c, cpu_result, size, cudaMemcpyHostToDevice);

	matrixSum_GPU<<< 1, N>>>(dev_c, dev_sum);
	cudaDeviceSynchronize();

    cudaEventRecord(t_stop,0);
    cudaEventSynchronize(t_stop);
    cudaEventElapsedTime(&elapsedtime_GPU, t_start, t_stop);
    printf("Elapsed time on CUDA matrix sum : \t%f ms\n",elapsedtime_GPU);



    cudaEventCreate(&t_start);
    cudaEventCreate(&t_stop);
    cudaEventRecord(t_start,0);

	// Copy result from GPU memory to host memory
	cudaMemcpy(cuda_result, dev_c, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(&cuda_sum, dev_sum, sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(t_stop,0);
    cudaEventSynchronize(t_stop);
    cudaEventElapsedTime(&elapsedtime_GPU, t_start, t_stop);
    elapsedtime_GPU_total += elapsedtime_GPU;
    printf("Elapsed time on CUDA memcpy (C & sum): \t%f ms\n",elapsedtime_GPU);
    printf("Total elapsed time for GPU functions: \t%f ms\n\n",elapsedtime_GPU_total);


	// Free allocated GPU memory
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    // Verify GPU result with CPU result
	int correct = 1;
    for(int i=0; i<N && (correct==1); i++)
        for(int j=0; j<N && (correct==1); j++)
            if(cpu_result[i*N + j] != cuda_result[i*N + j]){
				printf("Incorrect matrix mult. result!\n\n");
				correct=0;
			}

	if(correct == 1)
        printf("Matrix mult. verification passed!\n\n");
    else
        printf("Matrix mult. verification failed!\n\n");


	if (cpu_sum != cuda_sum){
		printf("atomicAdd() result incorrect\n");
		printf("CPU=%0.2f \nGPU=%0.2f \nDifference=%0.2f\n", cpu_sum, cuda_sum, cuda_sum-cpu_sum);
		correct=0;
	}


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
