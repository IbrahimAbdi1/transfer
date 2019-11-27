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





void run_kernel1(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height) {
  // Figure out how to split the work into threads and call the kernel below.
  int pixelCount = width*height;
  int32_t *d_min_max;
  int32_t *g_min_max;
  int32_t *deviceMatrix_IN,*deviceMatrix_OUT;
  int8_t *deviceFilter;
  int size = height*width*sizeof(int32_t);
  int numBlocks = pixelCount/1024;
  

  cudaMalloc((void**)&deviceMatrix_IN,size);
  cudaMalloc((void**)&deviceMatrix_OUT,size);
  cudaMalloc((void**)&deviceFilter,dimension*dimension*sizeof(int8_t));
  cudaMalloc((void**)&d_min_max,2*sizeof(int32_t));
  cudaMalloc((void**)&g_min_max,(numBlocks + 1)*sizeof(int32_t));


  
  cudaMemcpy(deviceMatrix_IN,input,size, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceMatrix_OUT,output,size, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceFilter,filter,dimension*dimension*sizeof(int8_t),cudaMemcpyHostToDevice);
  kernel1<<<pixelCount/1024 + 1,1024>>>(deviceFilter,dimension,deviceMatrix_IN,deviceMatrix_OUT,width,height); 

  if(pixelCount < 1024){
      kernel1<<<pixelCount/1024 + 1,1024>>>(deviceFilter,dimension,deviceMatrix_IN,deviceMatrix_OUT,width,height);
      find_min_max<<<1,pixelCount,2*1024*sizeof(int32_t)>>>(deviceMatrix_OUT,d_min_max,pixelCount);
      normalize1<<<pixelCount/1024 + 1,1024>>>(deviceMatrix_OUT,width,height,d_min_max);
  }
  else{
      kernel1<<<numBlocks + 1,1024>>>(deviceFilter,dimension,deviceMatrix_IN,deviceMatrix_OUT,width,height);
      find_min_max<<<numBlocks + 1,1024,2*1024*sizeof(int32_t)>>>(deviceMatrix_OUT,g_min_max,pixelCount);
      while(numBlocks > 0){
          find_min_max<<<numBlocks + 1,1024,2*1024*sizeof(int32_t)>>>(deviceMatrix_OUT,d_min_max,pixelCount);
          numBlocks = numBlocks / 1024;
      }

  }


   cudaMemcpy(output,deviceMatrix_OUT,size, cudaMemcpyDeviceToHost);
   
   cudaFree(deviceMatrix_IN);
   cudaFree(deviceMatrix_OUT);
   cudaFree(deviceFilter);
   cudaFree(d_min_max);
}



__global__ void kernel1(const int8_t *filter, int32_t dimension, const int32_t *input, 
int32_t *output, int32_t width,int32_t height) {

  // shared memory   
 
  // get index given tid
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  // call apply2d on input @ index and store it  on output @ index
  if(idx < height*width){
    int row = idx/width;
    int column = idx%width; // expensive 
   
    // apply2d function (really bad need fix)
    int32_t sum = 0;
    int filter_centre = dimension/2;
    
    int s_row = row - filter_centre;
    int s_column = column - filter_centre;
    for(int r = 0;r<dimension;r++){`
        int n_row = s_row + r;
        for(int c = 0;c<dimension;c++){
            int n_column = s_column + c;
            if((n_row >= 0) && (n_column >= 0) && (n_column < width) && (n_row < height)){
                sum += (filter[r*dimension + c]) * (input[n_row*width + n_column]);
                
            }
        }
    }
    
    output[idx] = sum;
    // store it in shared 
    // sync 

    // 
    
  }

  
                          
}

__global__ void normalize1(int32_t *image, int32_t width, int32_t height, int32_t *smallest_biggest) {

   
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(smallest_biggest[0] != smallest_biggest[1] && idx < width * height){
    image[idx] = ((image[idx] - smallest_biggest[0]) * 255) / (smallest_biggest[1] - smallest_biggest[0]);
    printf("normalized %d\n",image[idx]);
  }
}


// need to account for pixels < 1024 
// tid switch to threadId 
__global__ void find_min_max(int32_t *arr,int32_t *max_min,int32_t pixelCount){
    // index 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int threadID = threadIdx.x;

    //one big array first half for fiding max secound for finding min
    extern __shared__ int32_t max_min_data[blockDim.x];
    
    // load only max no need for both less time i think
    if(tid < pixelCount){
        max_min_data[tid] = arr[tid];
    }
    __syncthreads();
    
    // need seperate first stride for filling in min
     // fix this
     
    int first_stride = blockDim.x/2;
     

    if( < first_stride){
        
        if(max_min_data[threadID] < max_min_data[threadID + first_stride]){
            int32_t temp = max_min_data[threadID];
            max_min_data[threadID] = max_min_data[threadID + first_stride];
            max_min_data[blockDim.x+threadID] = temp;
        }
        else{
            max_min_data[blockDim.x+threadID] = max_min_data[threadID + first_stride];
        }
    }
    __syncthreads();
   

    for(int stride = first_stride/2;stride > 0; stride>>= 1){
        if(threadID < stride){
            // cheack max
            if(max_min_data[threadID] < max_min_data[threadID + stride]){
                max_min_data[threadID] = max_min_data[threadID + stride];
                
            }
            // check min 
            if(max_min_data[blockDim.x+threadID] > max_min_data[blockDim.x+threadID + stride]){
               
                max_min_data[blockDim.x+threadID] = max_min_data[blockDim.x+threadID + stride];
            }
        }

       __syncthreads();

    }
    
    
    if(threadID == 0){
        printf("\nMin %d Max %d\n", max_min_data[blockDim.x],max_min_data[0]);
        max_min[0] = max_min_data[blockDim.x];
        max_min[1] = max_min_data[0]; 
    }

}