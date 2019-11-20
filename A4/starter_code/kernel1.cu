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

void run_kernel1(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height) {
  // Figure out how to split the work into threads and call the kernel below.

  int pixelCount = height*width;
  kernel1<<<pixelCount/1024 + 1,1024>>>(filter,dimension,input,output,width,height);
  
  // get total available threads
   

  // if threadCount < pixelcount call kernel1 with appropriate params

  // else call kernel1 multple times ig

  
  // wait for all threads

   
  
  // then call normalize 
}

__global__ void kernel1(const int8_t *filter, int32_t dimension, const int32_t *input, 
int32_t *output, int32_t width,int32_t height) {

  // get index given tid
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // call apply2d on input @ index and store it  on output @ index
  if(idx < height*width){
    int row = idx/width;
    int column = idx%width;
   
   // apply2d function
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

__global__ void normalize1(int32_t *image, int32_t width, int32_t height,
                           int32_t smallest, int32_t biggest) {

  // reductions memes 
    
}



