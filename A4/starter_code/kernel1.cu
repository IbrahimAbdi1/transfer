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
 #include <stdio.h>
 #include <string>
 #include <unistd.h>
 #include <math.h>


 #define MY_MIN(x,y) ((x < y) ? x : y)


 
 void run_kernel1(const int8_t *filter, int32_t dimension, const int32_t *input,
                  int32_t *output, int32_t width, int32_t height) {
   // Figure out how to split the work into threads and call the kernel below.
   int pixelCount = width*height;
   int32_t *g_min_max;
   int32_t *deviceMatrix_IN,*deviceMatrix_OUT;
   int8_t *deviceFilter;
   int size = height*width*sizeof(int32_t);
   int numBlocks = pixelCount / 1024;
   int first = 1;
   int numThreads, nblocks;
   int iteration_n = pixelCount;
   printf("pixelCount %d numBlocks %d\n",pixelCount,numBlocks);
 
   cudaMalloc((void**)&deviceMatrix_IN,size);
   cudaMalloc((void**)&deviceMatrix_OUT,size);
   cudaMalloc((void**)&deviceFilter,dimension*dimension*sizeof(int8_t));
   cudaMalloc((void**)&g_min_max,2*(numBlocks+1)*sizeof(int32_t));
   
   cudaMemcpy(deviceMatrix_IN,input,size, cudaMemcpyHostToDevice);
   cudaMemcpy(deviceMatrix_OUT,output,size, cudaMemcpyHostToDevice);
   cudaMemcpy(deviceFilter,filter,dimension*dimension*sizeof(int8_t),cudaMemcpyHostToDevice);
 
   kernel1<<<numBlocks+1,1024>>>(deviceFilter,dimension,deviceMatrix_IN,deviceMatrix_OUT,width,height); 

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
 
 
 
 __global__ void kernel1(const int8_t *filter, int32_t dimension, const int32_t *input, 
 int32_t *output, int32_t width,int32_t height) {
 
   // shared memory   
  
   // get index given tid
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   
   // call apply2d on input @ index and store it  on output @ index
   if(idx < height*width){
     int row = idx%width;
     int column = idx/width; // expensive 
    
     // apply2d function (really bad need fix)
     int32_t sum = 0;
     int filter_centre = dimension/2;
     
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
     
     output[idx] = sum;
     
   }
 
   
                           
 }
 
 __global__ void normalize1(int32_t *image, int32_t width, int32_t height, int32_t *smallest_biggest) {
     
    
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   //int idx = tidx/width + width*(tidx%width);
   if(smallest_biggest[0] != smallest_biggest[1] && idx < width * height){
     image[idx] = ((image[idx] - smallest_biggest[1]) * 255) / (smallest_biggest[0] - smallest_biggest[1]);
   }
 }
 
 

