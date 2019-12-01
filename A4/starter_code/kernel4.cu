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

#include "kernels.h"

void run_kernel4(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height) {
  // Figure out how to split the work into threads and call the kernel below.
  int colCount = width;
  int pixelCount = height*width;
  int32_t *g_min_max;
  int32_t *deviceMatrix_IN,*deviceMatrix_OUT;
  int8_t *deviceFilter;
  int size = height*width*sizeof(int32_t);
  int numBlocks = pixelCount / 1024;
  int first = 1;
  int numThreads, nblocks;
  int iteration_n = pixelCount;
  printf("colCount %d numBlocks %d\n",colCount,numBlocks);

  cudaMalloc((void**)&deviceMatrix_IN,size);
  cudaMalloc((void**)&deviceMatrix_OUT,size);
  cudaMalloc((void**)&deviceFilter,dimension*dimension*sizeof(int8_t));
  cudaMalloc((void**)&g_min_max,2*(numBlocks+1)*sizeof(int32_t));
  
  cudaMemcpy(deviceMatrix_IN,input,size, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceMatrix_OUT,output,size, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceFilter,filter,dimension*dimension*sizeof(int8_t),cudaMemcpyHostToDevice);

  kernel1<<<colCount/1024+1,1024>>>(deviceFilter,dimension,deviceMatrix_IN,deviceMatrix_OUT,width,height); 

  int32_t *max = g_min_max;
  int32_t *min = g_min_max + (numBlocks +1);
  bool should_repeat = calculate_blocks_and_threads(iteration_n, nblocks, numThreads);
  printf("pixels %d blocks %d threads %d\n",iteration_n, nblocks, numThreads);
   gpu_min_max_switch_threads(iteration_n, numThreads, nblocks, deviceMatrix_OUT, max, min, first);

   first = 0;

    while(should_repeat)
    {
      iteration_n = nblocks;
      printf("HERE: %d blocks \n", nblocks);
      should_repeat = calculate_blocks_and_threads(iteration_n, nblocks, numThreads);
      printf("pixels %d blocks %d threads %d\n",iteration_n, nblocks, numThreads);
      gpu_min_max_switch_threads(iteration_n, numThreads, nblocks, g_min_max, max, min, first);
    }
  
  normalize1<<<numBlocks + 1,1024>>>(deviceMatrix_OUT,width,height,g_min_max);

   cudaMemcpy(output,deviceMatrix_OUT,size, cudaMemcpyDeviceToHost);
   
   cudaFree(deviceMatrix_IN);
   cudaFree(deviceMatrix_OUT);
   cudaFree(deviceFilter);
   cudaFree(g_min_max);
}

__global__ void kernel4(const int8_t *filter, int32_t dimension,const int32_t *input, int32_t *output, int32_t width,int32_t height) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < height*width){
    int filter_centre = dimension/2;
    for(int i=0;i<height;i++){
      int row = width*i;
      int column = idx;
      int sum = 0;
      int s_row = row - filter_centre;
      int s_column = column - filter_centre;
      for(int r = 0;r<dimension;r++){
          int n_row = s_row + r;
          for(int c = 0;c<dimension;c++){
              int n_column = s_column + c;
              if((n_row >= 0) && (n_column >= 0) && (n_column < width) && (n_row < height)){
                  sum += (filter[r*dimension + c]) * (input[n_row*width + n_column]);
                  
              }
          }
      }
      output[idx +(width*i)];
    }
  }

}

__global__ void normalize4(int32_t *image, int32_t width, int32_t height,int32_t *smallest_biggest) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int start_row = idx*width;
  

  for(int i = 0;i<height;i++){
    if(smallest_biggest[0] != smallest_biggest[1] && idx < width * height){
      image[idx+(i*width)] = ((image[idx+(i*width)] - smallest_biggest[1]) * 255) / (smallest_biggest[0] - smallest_biggest[1]);
    }
  }
  
}
