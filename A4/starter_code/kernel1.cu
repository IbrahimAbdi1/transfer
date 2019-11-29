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
  printf("pixelCount %d numBlocks %d\n",pixelCount,numBlocks);

  cudaMalloc((void**)&deviceMatrix_IN,size);
  cudaMalloc((void**)&deviceMatrix_OUT,size);
  cudaMalloc((void**)&deviceFilter,dimension*dimension*sizeof(int8_t));
  cudaMalloc((void**)&d_min_max,2*sizeof(int32_t));
  cudaMalloc((void**)&g_min_max,2*(numBlocks + 1)*sizeof(int32_t));


  
  cudaMemcpy(deviceMatrix_IN,input,size, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceMatrix_OUT,output,size, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceFilter,filter,dimension*dimension*sizeof(int8_t),cudaMemcpyHostToDevice);

  kernel1<<<numBlocks+1,1024>>>(deviceFilter,dimension,deviceMatrix_IN,deviceMatrix_OUT,width,height); 
  printf("pixelCount %d numBlocks %d\n",pixelCount,numBlocks);
  find_min_max<<<numBlocks+1,1024,2048*sizeof(double)>>>(deviceMatrix_OUT,g_min_max,pixelCount,2*pixelCount);
  if(pixelCount > 1024){
    pixelCount = numBlocks+1;
    numBlocks = numBlocks / 1024;
    while(numBlocks > 0){
        find_min_max<<<numBlocks+1,1024,2048*sizeof(double)>>>(g_min_max,g_min_max,pixelCount,2*pixelCount);
        pixelCount = numBlocks+1;
        numBlocks = numBlocks / 1024;
        printf("loop pixelCount %d numBlocks %d\n",pixelCount,numBlocks);
    }
    printf("out loop pixelCount %d numBlocks %d\n",pixelCount,numBlocks);
    find_min_max<<<numBlocks+1,1024,2048*sizeof(double)>>>(g_min_max,g_min_max,pixelCount,2*pixelCount);

  }
  
  normalize1<<<numBlocks + 1,1024>>>(deviceMatrix_OUT,width,height,g_min_max);

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
    image[idx] = ((image[idx] - smallest_biggest[1]) * 255) / (smallest_biggest[0] - smallest_biggest[1]);
  }
}


// need to account for pixels < 1024 
// tid switch to threadId 
__global__ void find_min_max(int32_t *arr,int32_t *max_min,int32_t pixelCount,int pixelBlock){
    // index 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int threadID = threadIdx.x;

    __shared__ double max_min_data[2][1024];

    // either load data or pad
    // max is 0 min is 1
    if(tid < pixelCount){
        int32_t g_pixel = arr[tid];
        max_min_data[0][threadID] = (double)g_pixel;
        max_min_data[1][threadID] = (double)g_pixel;
    }
    else{
        max_min_data[0][threadID] = -INFINITY;
        max_min_data[1][threadID] = INFINITY;
    }
    __syncthreads();

    // complete unroll 

    
        if(threadID < 512){
            if(max_min_data[0][threadID] < max_min_data[0][threadID+512]){
                max_min_data[0][threadID] = max_min_data[0][threadID+512];
            }
            if(max_min_data[1][threadID] > max_min_data[1][threadID+512]){
                max_min_data[1][threadID] = max_min_data[1][threadID+512];
            }

        }
        __syncthreads();
        if(threadID < 256){
            if(max_min_data[0][threadID] < max_min_data[0][threadID+256]){
                max_min_data[0][threadID] = max_min_data[0][threadID+256];
            }
            if(max_min_data[1][threadID] > max_min_data[1][threadID+256]){
                max_min_data[1][threadID] = max_min_data[1][threadID+256];
            }

        }
        __syncthreads();
        if(threadID < 128){
            if(max_min_data[0][threadID] < max_min_data[0][threadID+128]){
                max_min_data[0][threadID] = max_min_data[0][threadID+128];
            }
            if(max_min_data[1][threadID] > max_min_data[1][threadID+128]){
                max_min_data[1][threadID] = max_min_data[1][threadID+128];
            }

        }
        __syncthreads();
        if(threadID < 64){
            if(max_min_data[0][threadID] < max_min_data[0][threadID+64]){
                max_min_data[0][threadID] = max_min_data[0][threadID+64];
            }
            if(max_min_data[1][threadID] > max_min_data[1][threadID+64]){
                max_min_data[1][threadID] = max_min_data[1][threadID+64];
            }

        }
        __syncthreads();

        // we dont need to sync threads after this point (usless)
        if(threadID < 32){
            if(max_min_data[0][threadID] < max_min_data[0][threadID+32]){max_min_data[0][threadID] = max_min_data[0][threadID+32];}
            if(max_min_data[1][threadID] > max_min_data[1][threadID+32]){max_min_data[1][threadID] = max_min_data[1][threadID+32];}

        }
        
        if(threadID < 16){
            if(max_min_data[0][threadID] < max_min_data[0][threadID+16]){max_min_data[0][threadID] = max_min_data[0][threadID+16];}
            if(max_min_data[1][threadID] > max_min_data[1][threadID+16]){max_min_data[1][threadID] = max_min_data[1][threadID+16];}

        }
        
        if(threadID < 8){
            if(max_min_data[0][threadID] < max_min_data[0][threadID+8]){max_min_data[0][threadID] = max_min_data[0][threadID+8];}
            if(max_min_data[1][threadID] > max_min_data[1][threadID+8]){max_min_data[1][threadID] = max_min_data[1][threadID+8];}

        }
        
        if(threadID < 4){
            if(max_min_data[0][threadID] < max_min_data[0][threadID+4]){max_min_data[0][threadID] = max_min_data[0][threadID+4];}
            if(max_min_data[1][threadID] > max_min_data[1][threadID+4]){max_min_data[1][threadID] = max_min_data[1][threadID+4];}

        }
        
        if(threadID < 2){
            if(max_min_data[0][threadID] < max_min_data[0][threadID+2]){max_min_data[0][threadID] = max_min_data[0][threadID+2];}
            if(max_min_data[1][threadID] > max_min_data[1][threadID+2]){max_min_data[1][threadID] = max_min_data[1][threadID+2];}

        }
        
        if(threadID < 1){
            if(max_min_data[0][threadID] < max_min_data[0][threadID+1]){max_min_data[0][threadID] = max_min_data[0][threadID+1];}
            if(max_min_data[1][threadID] > max_min_data[1][threadID+1]){max_min_data[1][threadID] = max_min_data[1][threadID+1];}

        }
        

        if(threadID == 0){
           // printf("vlock %d max %d min %d\n", blockIdx.x,(int)max_min_data[0][0],(int)max_min_data[1][0]);
            max_min[blockIdx.x*2] = max_min_data[0][0];
            max_min[blockIdx.x*2+1] = max_min_data[1][0];
        }
    

}