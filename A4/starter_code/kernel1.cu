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


int32_t *d_min_max;


void run_kernel1(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height) {
  // Figure out how to split the work into threads and call the kernel below.
  cudaMalloc(&d_min_max,sizeof(int32_t)*2);
  int pixelCount = width*height;
  kernel1<<<pixelCount/1024 + 1,1024>>>(filter,dimension,input,output,width,height);
  printf("hello\n");
  // reduction memes until finnito
  find_min_max<<<1,pixelCount>>>(output,&d_min_max[1],&d_min_max[0]);
  printf("hello2\n");
  normalize1<<<pixelCount/1024 + 1,1024>>>(output,width,height,d_min_max[0],d_min_max[1]); // dont know 
   
}



__global__ void kernel1(const int8_t *filter, int32_t dimension, const int32_t *input, 
int32_t *output, int32_t width,int32_t height) {

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

__global__ void normalize1(int32_t *image, int32_t width, int32_t height, int32_t smallest, int32_t biggest) {

  // reduction needs to happen maybe 
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(smallest != biggest && idx < width * height){
    image[idx] = ((image[idx] - smallest) * 255) / (biggest - smallest);
  }
}


// problem ony works with one block
__global__ void find_min_max(int32_t *arr,int32_t *max,int32_t *min){
    // index 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    //load to share data 2 share datas one for min and max 
    extern __shared__ int32_t max_data[];
    extern __shared__ int32_t min_data[];

    max_data[tid] = arr[tid];
    __syncthreads();

    // need first stride for filling in min
    int first_stride = blockDim.x/2;
    if(tid < first_stride){
        
        if(max_data[tid] < max_data[tid + first_stride]){
            int32_t temp = max_data[tid];
            max_data[tid] = max_data[tid + first_stride];
            min_data[tid] = temp;
        }
        else{
            min_data[tid] = max_data[tid + first_stride];
        }
    }
    __syncthreads();
    
    for(int stride = first_stride/2;stride > 0; stride>>= 1){
        if(tid < stride){
            
            if(max_data[tid] < max_data[tid + stride]){
                int32_t temp = max_data[tid];
                max_data[tid] = max_data[tid + stride];
                if(min_data[tid] > temp){
                    min_data[tid] = temp;
                }
            }
            else if(max_data[tid] >= max_data[tid + stride]){
                if(min_data[tid] > max_data[tid + stride]){
                    min_data[tid] = max_data[tid + stride];
                }
            }
        }

       __syncthreads();

    }
    
    if(tid == 0){
        min[0] = min_data[0];
        max[0] = max_data[0]; 
    }

}