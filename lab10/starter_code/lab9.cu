/* ------------
 * This code is provided solely for the personal and private use of
 * students taking the CSC367H5 course at the University of Toronto.
 * Copying for purposes other than this use is expressly prohibited.
 * All forms of distribution of this code, whether as given or with
 * any changes, are expressly prohibited.
 *
 * Authors: Bogdan Simion, Felipe de Azevedo Piovezan
 *
 * All of the files in this directory and all subdirectories are:
 * Copyright (c) 2019 Bogdan Simion
 * -------------
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define M 1024 * 1024
#define threads_block 512
#define MAX_ARR_SIZE 32

#define RUN_SIMPLE 1
#define RUN_THREADS 2
#define RUN_BLOCKS 3
#define RUN_TIMES 4

// Simple kernel: blocks, threads = 1, 1
__global__ void array_add_simple(float *a, float *b, int N) {
  for(int i = 0; i<N;i++){
    a[i] += b[i];
  }
}

// Simple kernel: blocks, threads = 1, 512
__global__ void array_add_threads_only(float *a, float *b, int N) {
  // Edit me!
}

// Complex kernel, utilize both blocks and threads
__global__ void array_add_threads_blocks(float *a, float *b, int N) {
  // Edit me!
}

// Complex kernel, utilize both blocks and threads
// Add b elements 'times' number of times
__global__ void array_add_times(float *a, float *b, int N, int times) {
  // Edit me!
}

/*Initialize the device arrays, timing variables, call kernels
  with the right number of threads and blocks
 */
void run_test(int arrsize, int times, int type) {
  float *a_h, *b_h,*a_d,*b_d;
  cudaEvent_t start, stop;
  float transfer_in, computation_time, transfer_out; // timing values
  int N = arrsize * M;

  //dim3 threads(threads_block, 1);
	dim3 blocks((N+threads_block-1)/threads_block);

  size_t SIZE = N * sizeof(float);
  a_h = (float *)malloc(SIZE);
  b_h = (float *)malloc(SIZE);
  cudaMalloc((void **)&a_d, SIZE);
  cudaMalloc((void **)&b_d, SIZE);

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    a_h[i] = (rand() % 10000) / 100.0f;
    b_h[i] = (rand() % 10000) / 100.0f;
  }
  cudaEventCreate(&start);
	cudaEventCreate(&stop);
  cudaEventRecord(start);
  cudaMemcpy(a_d, a_h, SIZE, cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b_h, SIZE, cudaMemcpyHostToDevice);
  cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&transfer_in, start, stop);



  if (type == RUN_SIMPLE) {
    cudaEventRecord(start);
    array_add_simple <<<1,1>>>(a_d,b_d,N);
  } else if (type == RUN_THREADS) {
    cudaEventRecord(start);
    
  } else if (type == RUN_BLOCKS) {
    cudaEventRecord(start);
  } else if (type == RUN_TIMES) {
    cudaEventRecord(start);
  } else {
    printf("Unknown run type\n");
    goto transfer_out;
  }

  cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&computation_time, start, stop);
transfer_out:
  cudaEventRecord(start);
	cudaMemcpy(a_h, a_d, SIZE, cudaMemcpyDeviceToHost);
  cudaMemcpy(b_h, b_d, SIZE, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&transfer_out, start, stop);
  // print timing results. Do not change this printf.
  printf("%5d %5d %15.2f %15.2f %15.2f\n", times, arrsize, transfer_in,
         computation_time, transfer_out);
  cudaFree(a_d);
  cudaFree(b_d);
  free(a_h);
  free(b_h);
}

int main(int argc, char *argv[]) {
  int arrsize = 1;
  int i;

  // Run with blocks, threads = 1,1
  // Number of times is constant(once), array size varies
  // Do not change this printf.
  printf("Times Size(M) TransferIn(ms) Computation(ms) TransferOut(ms)\n");
  for (arrsize = 1; arrsize <= MAX_ARR_SIZE; i++, arrsize *= 2) {
    run_test(arrsize, 1, RUN_SIMPLE);
  }

  // Run with several blocks and threads
  // Number of times is constant(once), array size varies
  // Do not change this printf.
  printf("\nTimes Size(M) TransferIn(ms) Computation(ms) TransferOut(ms)\n");
  for (arrsize = 1; arrsize <= MAX_ARR_SIZE; i++, arrsize *= 2) {
    run_test(arrsize, 1, RUN_THREADS);
  }

  // Run with several blocks and threads
  // Number of times is constant(once), array size varies
  // Do not change this printf.
  printf("\nTimes Size(M) TransferIn(ms) Computation(ms) TransferOut(ms)\n");
  for (arrsize = 1; arrsize <= MAX_ARR_SIZE; i++, arrsize *= 2) {
    run_test(arrsize, 1, RUN_BLOCKS);
  }

  // Number of times varies, array size is constant (maximum number of elem)
  // Do not change this printf.
  printf("\nTimes Size(M) TransferIn(ms) Computation(ms) TransferOut(ms)\n");
  int times = 1;
  arrsize = MAX_ARR_SIZE;
  for (i = 0; i < 10; i++, times *= 2) {
    run_test(arrsize, times, RUN_TIMES);
  }
}
