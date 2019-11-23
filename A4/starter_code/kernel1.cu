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
  int32_t *deviceMatrix_IN,*deviceMatrix_OUT;
  int8_t *deviceFilter;
  int size = height*width*sizeof(int32_t);

  cudaMalloc((void**)&deviceMatrix_IN,size);
  cudaMalloc((void**)&deviceMatrix_OUT,size);
  cudaMalloc((void**)&deviceFilter,dimension*dimension*sizeof(int8_t));
  cudaMalloc((void**)&d_min_max,2*sizeof(int));


  
  cudaMemcpy(deviceMatrix_IN,input,size, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceMatrix_OUT,output,size, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceFilter,filter,dimension*dimension*sizeof(int8_t),cudaMemcpyHostToDevice);
  
  printf("hehe %d %d %d %d\n",input[0],input[1],input[2],input[3]);
  kernel1<<<pixelCount/1024 + 1,pixelCount>>>(deviceFilter,dimension,deviceMatrix_IN,deviceMatrix_OUT,width,height);
  
  
   
   //printf("hehe2 %d %d %d %d\n",output[0],output[1],output[2],output[3]);
  
  // reduction memes until finnito
  find_min_max<<<1,pixelCount,2*size>>>(deviceMatrix_OUT,d_min_max);
  
   normalize1<<<pixelCount/1024 + 1,pixelCount>>>(deviceMatrix_OUT,width,height,d_min_max); // dont know 
   cudaMemcpy(output,deviceMatrix_OUT,size, cudaMemcpyHostToDevice);
   printf("Device Variable Copying:\t%s\n", cudaGetErrorString(cudaGetLastError()));
   cudaFree(deviceMatrix_IN);
   cudaFree(deviceMatrix_OUT);
   cudaFree(deviceFilter);
   cudaFree(d_min_max);
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
    //printf("id %d has sum %d\n",idx,sum);
    output[idx] = sum;
    printf("output at %d has sum %d\n",idx,output[idx]);
    
  }

  
                          
}

__global__ void normalize1(int32_t *image, int32_t width, int32_t height, int32_t *smallest_biggest) {

  // reduction needs to happen maybe 
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(smallest_biggest[0] != smallest_biggest[1] && idx < width * height){
    image[idx] = ((image[idx] - smallest_biggest[0]) * 255) / (smallest_biggest[1] - smallest_biggest[0]);
    printf("normalized %d\n",image[idx]);
  }
}


// problem ony works with one block
__global__ void find_min_max(int32_t *arr,int32_t *max_min){
    // index 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    //load to share data 2 share datas one for min and max 
    extern __shared__ int32_t max_min_data[];
    

    max_min_data[tid] = arr[tid];
    __syncthreads();
    
    // need first stride for filling in min
    int first_stride = blockDim.x/2;
    if(tid < first_stride){
        
        if(max_min_data[tid] < max_min_data[tid + first_stride]){
            int32_t temp = max_min_data[tid];
            max_min_data[tid] = max_min_data[tid + first_stride];
            max_min_data[blockDim.x+tid] = temp;
        }
        else{
            max_min_data[blockDim.x+tid] = max_min_data[tid + first_stride];
        }
    }
    __syncthreads();
    printf("first tid 0 stride Min %d Max %d\n", max_min_data[blockDim.x],max_min_data[0]);
    printf("first tid 1 stride Min %d Max %d\n", max_min_data[blockDim.x+1],max_min_data[1]);
    for(int stride = first_stride/2;stride > 0; stride>>= 1){
        if(tid < stride){
            
            if(max_min_data[tid] < max_min_data[tid + stride]){
                int32_t temp = max_min_data[tid];
                max_min_data[tid] = max_min_data[tid + stride];
                if(max_min_data[blockDim.x+tid] > temp){
                    max_min_data[blockDim.x+tid] = temp;
                }
            }
            if(max_min_data[tid] >= max_min_data[tid + stride]){
                if(max_min_data[blockDim.x+tid] > max_min_data[tid + stride]){
                    max_min_data[blockDim.x+tid] = max_min_data[tid + stride];
                }
            }
            if(max_min_data[blockDim.x+tid] > max_min_data[blockDim.x+tid + stride]){
               
                max_min_data[blockDim.x+tid] = max_min_data[blockDim.x+tid + stride];
            }
        }

       __syncthreads();

    }
    
    if(tid == 0){
        printf("\nMin %d Max %d\n", max_min_data[blockDim.x],max_min_data[0]);
        max_min[0] = max_min_data[blockDim.x];
        max_min[1] = max_min_data[0]; 
    }

}