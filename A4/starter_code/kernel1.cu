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

  kernel1<<<numBlocks+1,1024>>>(deviceFilter,dimension,deviceMatrix_IN,deviceMatrix_OUT,width,height); 

  find_min_max<<<numBlocks+1,1024,2048*sizeof(double)>>>(deviceMatrix_OUT,d_min_max,pixelCount);

  normalize1<<<numBlocks + 1,1024>>>(deviceMatrix_OUT,width,height,d_min_max);

   cudaMemcpy(output,deviceMatrix_OUT,size, cudaMemcpyDeviceToHost);
   
   cudaFree(deviceMatrix_IN);
   cudaFree(deviceMatrix_OUT);
   cudaFree(deviceFilter);
   cudaFree(d_min_max);
   cudaFree(g_min_max);
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
    int blockSize = blockDim.x;
    int threadID = threadIdx.x;

    extern __shared__ double max_min_data[];

    // either load data or pad
    // max is 0 min is 1
    if(tid < pixelCount){
        int32_t g_pixel = arr[tid];
        max_min_data[threadID*2] = (double)g_pixel; // max is even 
        max_min_data[threadID*2+1] = (double)g_pixel; // min is odd
    }
    else{
        max_min_data[threadID*2] = -INFINITY;
        max_min_data[threadID*2 + 1] = INFINITY;
    }
    __syncthreads();

    // complete unroll 

        if(threadID < 512){
            if(max_min_data[threadID*2] < max_min_data[threadID*2+512]){max_min_data[threadID*2] = max_min_data[threadID*2+512];}
            if(max_min_data[threadID*2+1] > max_min_data[threadID*2+1+512]){max_min_data[threadID*2+1] = max_min_data[threadID*2+1+512];}
        }
        __syncthreads();
        if(threadID < 256){
            if(max_min_data[threadID*2] < max_min_data[threadID*2+256]){max_min_data[threadID*2] = max_min_data[threadID*2+256];}
            if(max_min_data[threadID*2+1] > max_min_data[threadID*2+1+256]){max_min_data[threadID*2+1] = max_min_data[threadID*2+1+256];}

        }
        __syncthreads();
        if(threadID < 128){
            if(max_min_data[threadID*2] < max_min_data[threadID*2+128]){max_min_data[threadID*2] = max_min_data[threadID*2+128];}
            if(max_min_data[threadID*2+1] > max_min_data[threadID*2+1+128]){max_min_data[threadID*2+1] = max_min_data[threadID*2+1+128];}

        }
        __syncthreads();
        if(threadID < 64){
            if(max_min_data[threadID*2] < max_min_data[threadID*2+64]){max_min_data[threadID*2] = max_min_data[threadID*2+64];}
            if(max_min_data[threadID*2+1] > max_min_data[threadID*2+1+64]){max_min_data[threadID*2+1] = max_min_data[threadID*2+1+64];}
        }
        __syncthreads();

        // wrap size issues
        if(threadID < 32){
            if(max_min_data[threadID*2] < max_min_data[threadID*2+32]){max_min_data[threadID*2] = max_min_data[threadID*2+32];}
            if(max_min_data[threadID*2+1] > max_min_data[threadID*2+1+32]){max_min_data[threadID*2+1] = max_min_data[threadID*2+1+32];}

        }
        
        if(threadID < 16){
            if(max_min_data[threadID*2] < max_min_data[threadID*2+16]){max_min_data[threadID*2] = max_min_data[threadID*2+16];}
            if(max_min_data[threadID*2+1] > max_min_data[threadID*2+1+16]){max_min_data[threadID*2+1] = max_min_data[threadID*2+1+16];}

        }
        
        if(threadID < 8){
            if(max_min_data[threadID*2] < max_min_data[threadID*2+8]){max_min_data[threadID*2] = max_min_data[threadID*2+8];}
            if(max_min_data[threadID*2+1] > max_min_data[threadID*2+1+8]){max_min_data[threadID*2+1] = max_min_data[threadID*2+1+8];}

        }
        
        if(threadID < 4){
            if(max_min_data[threadID*2] < max_min_data[threadID*2+4]){max_min_data[threadID*2] = max_min_data[threadID*2+4];}
            if(max_min_data[threadID*2+1] > max_min_data[threadID*2+1+4]){max_min_data[threadID*2+1] = max_min_data[threadID*2+1+4];}

        }
        
        if(threadID < 2){
            if(max_min_data[threadID*2] < max_min_data[threadID*2+2]){max_min_data[threadID*2] = max_min_data[threadID*2+2];}
            if(max_min_data[threadID*2+1] > max_min_data[threadID*2+1+2]){max_min_data[threadID*2+1] = max_min_data[threadID*2+1+2];}

        }

        if(tid == 0){
            printf("max %d min %d\n", (int)max_min_data[0],(int)max_min_data[1]);
            max_min[1] = max_min_data[0];
            max_min[0] = max_min_data[1];
        }

}