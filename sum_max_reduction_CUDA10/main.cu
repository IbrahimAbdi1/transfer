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

#include <cuda.h>
#include <stdio.h>
#include <limits.h>

#include "kernels.h"
#include "cpu_reductions.h"

//-----------------------------------------
//--- constants, macros, definitions ------
//-----------------------------------------
#define M (1024*1024)
#define M2      (2*M)
#define M8      (8*M)
#define M32    (32*M)

#define NUM_ITERATIONS 100 // Change this to run several iterations and get average
#define MAX_BLOCK_SIZE 65535
#define TO_SECONDS 1000

#define REDUCE_SUM 1
#define REDUCE_MAX 2

template <unsigned int blockSize>
void call_add_kernel(int kernel, const dim3 &dimGrid, const dim3 &dimBlock,
        int shMemSize, int *d_idata, int *d_odata, int size)
{
  switch (kernel)
  {
  case 1:
    {
      reduce1<<< dimGrid, dimBlock, shMemSize >>>(d_idata, d_odata);
      break;
    }
  case 2:
    {
      reduce2<<< dimGrid, dimBlock, shMemSize >>>(d_idata, d_odata);
      break;
    }
  case 3:
    {
      reduce3<<< dimGrid, dimBlock, shMemSize >>>(d_idata, d_odata);
      break;
    }
  case 4:
    {
      reduce4<<< dimGrid, dimBlock, shMemSize >>>(d_idata, d_odata);
      break;
    }
  case 5:
    {
      reduce5<<< dimGrid, dimBlock, shMemSize >>>(d_idata, d_odata);
      break;
    }
  case 6:
    {
      reduce6<blockSize><<< dimGrid, dimBlock, shMemSize >>>(d_idata,
                                                             d_odata);
      break;
    }
  case 7:
    {
      reduce7<blockSize><<< dimGrid, dimBlock, shMemSize >>>(d_idata,
                                                             d_odata, size);
      break;
    }
  case 8:
    {
      reduce8<<< dimGrid, dimBlock >>>(d_idata, d_odata, size);
      break;
    }
  case 9:
    {
      reduce9<<< dimGrid, dimBlock >>>(d_idata, d_odata, size);
      break;
    }
  case 10:
    {
      reduce10<<< dimGrid, dimBlock >>>(d_idata, d_odata, size);
      break;
    }
  }
}

void reduce(int kernel, int size, int threads, int blocks, int *d_idata,
            int *d_odata, int aggregate)
{
  dim3 dimBlock(threads,1,1);
  dim3 dimGrid (blocks, 1,1);
  int shMemSize = (threads <= 32) ? 2 * threads * sizeof(int) : threads * sizeof(int);

  if (aggregate == REDUCE_MAX)
  {
    switch (threads)
    {
    case 512:
      reduce_max<512><<< dimGrid, dimBlock, shMemSize >>>(d_idata,
                                                          d_odata, size);
      break;
    case 256:
      reduce_max<256><<< dimGrid, dimBlock, shMemSize >>>(d_idata,
                                                          d_odata, size);
      break;
    case 128:
      reduce_max<128><<< dimGrid, dimBlock, shMemSize >>>(d_idata,
                                                          d_odata, size);
      break;
    case 64:
      reduce_max< 64><<< dimGrid, dimBlock, shMemSize >>>(d_idata,
                                                          d_odata, size);
      break;
    case 32:
      reduce_max< 32><<< dimGrid, dimBlock, shMemSize >>>(d_idata,
                                                          d_odata, size);
      break;
    case 16:
      reduce_max< 16><<< dimGrid, dimBlock, shMemSize >>>(d_idata,
                                                          d_odata, size);
      break;
    case  8:
      reduce_max<  8><<< dimGrid, dimBlock, shMemSize >>>(d_idata,
                                                          d_odata, size);
      break;
    case  4:
      reduce_max<  4><<< dimGrid, dimBlock, shMemSize >>>(d_idata,
                                                          d_odata, size);
      break;
    case  2:
      reduce_max<  2><<< dimGrid, dimBlock, shMemSize >>>(d_idata,
                                                          d_odata, size);
      break;
    case  1:
      reduce_max<  1><<< dimGrid, dimBlock, shMemSize >>>(d_idata,
                                                          d_odata, size);
      break;
    default:
      printf("invalid number of threads, exiting...\n");
      exit(1);
    }
  }
  else if (aggregate == REDUCE_SUM)
  {
    switch (threads)
    {
    case 512:
      call_add_kernel<512>(kernel, dimGrid, dimBlock, shMemSize,
                           d_idata, d_odata, size);
      break;
    case 256:
      call_add_kernel<256>(kernel, dimGrid, dimBlock, shMemSize,
                           d_idata, d_odata, size);
      break;
    case 128:
      call_add_kernel<128>(kernel, dimGrid, dimBlock, shMemSize,
                           d_idata, d_odata, size);
      break;
    case 64:
      call_add_kernel<64 >(kernel, dimGrid, dimBlock, shMemSize,
                           d_idata, d_odata, size);
      break;
    case 32:
      call_add_kernel<32 >(kernel, dimGrid, dimBlock, shMemSize,
                           d_idata, d_odata, size);
      break;
    case 16:
      call_add_kernel<16 >(kernel, dimGrid, dimBlock, shMemSize,
                           d_idata, d_odata, size);
      break;
    case  8:
      call_add_kernel<8  >(kernel, dimGrid, dimBlock, shMemSize,
                           d_idata, d_odata, size);
      break;
    case  4:
      call_add_kernel<4  >(kernel, dimGrid, dimBlock, shMemSize,
                           d_idata, d_odata, size);
      break;
    case  2:
      call_add_kernel<2  >(kernel, dimGrid, dimBlock, shMemSize,
                           d_idata, d_odata, size);
      break;
    case  1:
      call_add_kernel<1  >(kernel, dimGrid, dimBlock, shMemSize,
                           d_idata, d_odata, size);
      break;
    default:
      printf("invalid number of threads, exiting...\n");
      exit(1);
    }
  }
  else
  {
    printf("Error: unknown aggregate function for reduction!");
  }
}

int CPU_reduction(int* data, int num_elem, float &cpu_time, int aggregate_type)
{
  int result = 0;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cpu_time = 0;
  float this_it;

  if(aggregate_type == REDUCE_MAX)
  {
    for(int j = 0; j < NUM_ITERATIONS; j++)
    {
      cudaEventRecord(start);
      result = reduceCPU_max(data, num_elem);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&this_it, start, stop);
      cpu_time += this_it;
    }
  }
  else if(aggregate_type == REDUCE_SUM)
  {
    for(int j = 0; j < NUM_ITERATIONS; j++)
    {
      cudaEventRecord(start);
      result = reduceCPU_sum(data, num_elem);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&this_it, start, stop);
      cpu_time += this_it;
    }
  }
  else
  {
    printf("Error: Unknown reduction function!");
  }

  cpu_time /= NUM_ITERATIONS;
  return result;
}

/*Returns true if the kernel should be repeated afterwards*/
bool calculate_blocks_and_threads(int kernel, int invocation, int n,
                                  int &blocks, int &threads)
{
  switch (kernel)
  {
  case 1:
  case 2:
  case 3:
    {
      threads = (n < maxThreads) ? (n) : maxThreads;
      blocks = (n + (threads - 1)) / threads;
      return blocks != 1;
    }
  case 4:
  case 5:
  case 6:
    {
      threads = (n < maxThreads*2) ? (n/2) : maxThreads;
      blocks = (n + (threads * 2 - 1)) / (threads * 2);
      return blocks != 1;
    }
  case 7:
  case 11:
    {
      threads = (n < maxThreads*2) ? (n/2) : maxThreads;
      blocks = (n + (threads * 2 - 1)) / (threads * 2);
      blocks = MY_MIN(maxBlocks, blocks);
      return blocks != 1;
    }
  case 8:
    {
      if (invocation == 0)
      {
        threads = 512;
        blocks = (n + (threads - 1)) / threads;
        blocks = MY_MIN(512, blocks);
        return blocks != 1;
      }
      else
      {
        threads = 512;
        blocks = 1;
        return false;
      }
    }

  case 9:
  case 10:
    {
      threads = 512;
      blocks = (n + (threads - 1)) / threads;
      return false;
    }
  default:
    {
      printf("invalid kernel number, exiting...\n");
      exit(1);
    }
  }
}

void GPU_reducer(int kernel, int* d_idata, int* d_odata, int n,
                 int aggregate_type, float &gpu_time)
{

  if (aggregate_type != REDUCE_MAX &&
      aggregate_type != REDUCE_SUM)
  {
    printf("Error: Unknown reduction function!");
    exit(1);
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  gpu_time = 0;

  for(int i = 0; i < NUM_ITERATIONS; i++)
  {
    /* The following line is only needed for kernels 8, 9, 10.
       It is only needed because we are reusing the output array
       for multiple test cases*/
    cudaMemsetAsync(d_odata,0,sizeof(int));
    int numThreads, numBlocks;
    int iteration_n = n;
    bool should_repeat = calculate_blocks_and_threads(kernel, 0,
                                                      iteration_n, numBlocks, numThreads);

    cudaEventRecord(start);
    reduce(kernel, iteration_n, numThreads, numBlocks, d_idata, d_odata,
           aggregate_type);

    while(should_repeat)
    {
      iteration_n = numBlocks;
      should_repeat = calculate_blocks_and_threads(kernel, 1,
                                                   iteration_n, numBlocks, numThreads);
      reduce(kernel, iteration_n, numThreads, numBlocks, d_odata,
             d_odata, aggregate_type);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float this_it;
    cudaEventElapsedTime(&this_it, start, stop);
    gpu_time += this_it;
  }

  gpu_time /= NUM_ITERATIONS;
}

int transfer_result(int *d_odata, float &gpu_time)
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  int result = 0;

  cudaEventRecord(start);
  cudaMemcpy(&result, d_odata, sizeof(int), cudaMemcpyDeviceToHost);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&gpu_time, start, stop);
  return result;
}

  template<int kernel>
void run_arrays()
{
  unsigned int array_size = M32 * sizeof(int);

  // allocate memory on the host
  int maxNumBlocks = MY_MIN(M32 / maxThreads, MAX_BLOCK_SIZE);
  int *h_idata = (int*) malloc(array_size);
  int* h_odata = (int*) malloc(maxNumBlocks*2*sizeof(int));

  // allocate memory on the device
  int *d_idata = NULL;
  int *d_odata = NULL;
  cudaMalloc((void**) &d_idata, array_size);
  cudaMalloc((void**) &d_odata, maxNumBlocks*2*sizeof(int));

  if (!h_idata || !h_odata || !d_idata || !d_odata)
  {
    printf("Cannot allocate memory\n");
    exit(1);
  }

  srand(17);

  // create random input data on CPU
  for(int i = 0; i < M32; i++)
  {
    h_idata[i] = (int)(rand() % 50);
  }

  // initialize result array
  for(int i = 0; i < maxNumBlocks*2; i++)
  {
    h_odata[i] = 0;
  }

  //timing measurements
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // copy data to device memory
  cudaEventRecord(start);
  cudaMemcpy(d_idata, h_idata, array_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_odata, h_odata, maxNumBlocks*2*sizeof(int),
             cudaMemcpyHostToDevice);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float transfer_in;
  cudaEventElapsedTime(&transfer_in, start, stop);

  printf("\n=============\n");

  int type = REDUCE_SUM;
  if(kernel == NUM_KERNELS + 1)
  {
    type = REDUCE_MAX;
    printf("Reduce max results:\n");
  }

  printf("kernel = %d \n", kernel);
  printf("Size GPU_results CPU_results CPU_time(ms) GPU_time(ms) "
         "TransferIn(ms) TransferOut(ms) Speedup_noTrf Speedup\n");

  // iterate over array sizes 2M, 8M, 32M (as required)
  for (int i = M2; i <= M32; i*=4)
  {
    float GPUtime=0;
    float CPUtime=0;
    float transfer_out=0;
    int CPU_result=0;
    int GPU_result=0;

    GPU_reducer(kernel, d_idata, d_odata, i, type, GPUtime);
    GPU_result = transfer_result(d_odata, transfer_out);
    CPU_result = CPU_reduction(h_idata, i, CPUtime, type);

    (i == M2)?printf("%4.4s ", "2M"):((i==M32)?printf("%4.4s ", "32M"):
                                      printf("%4.4s ", "8M"));
    printf("%11d ", GPU_result);
    printf("%11d ", CPU_result);
    printf("%12.6f ", CPUtime/TO_SECONDS);
    printf("%12.6f ", GPUtime/TO_SECONDS);
    printf("%14.6f ", transfer_in/TO_SECONDS);
    printf("%15.6f ", transfer_out/TO_SECONDS);
    printf("%13.2f ", CPUtime/GPUtime);
    printf("%7.2f\n", CPUtime / (GPUtime+transfer_in+transfer_out));
  }
  printf("\n");

  cudaFree(d_idata);
  cudaFree(d_odata);
  free(h_idata);
  free(h_odata);
}

int main(int argc, char** argv)
{
  printf("Times below are reported in seconds\n");
  run_arrays<1>();
  run_arrays<2>();
  run_arrays<3>();
  run_arrays<4>();
  run_arrays<5>();
  run_arrays<6>();
  run_arrays<7>();
  run_arrays<8>();
  run_arrays<9>();
  run_arrays<10>();
  //"kernel 11" = max reduction"
  run_arrays<11>();
  return 0;
}
