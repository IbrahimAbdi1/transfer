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

void run_kernel3(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height) {
  // Figure out how to split the work into threads and call the kernel below.


}

__global__ void kernel3(const int8_t *filter, int32_t dimension,const int32_t *input, int32_t *output, int32_t width,int32_t height){

  // 

  // thread will be height 
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < height*width){

    int start_row = idx*width;
    int end_row = start_row + width;
    int filter_centre = dimension/2;
    
    for(int i = start_row;i<end_row;i++){
      int row = start_row;
      int column = i;
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
      
      output[idx+i] = sum;
    }

  }

}

__global__ void normalize3(int32_t *image, int32_t width, int32_t height,int32_t *smallest_biggest) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int start_row = idx*width;
  int end_row = start_row + width;

  for(int i = start_row;i<end_row;i++){
    if(smallest_biggest[0] != smallest_biggest[1] && idx < width * height){
      image[idx+i] = ((image[idx+i] - smallest_biggest[1]) * 255) / (smallest_biggest[0] - smallest_biggest[1]);
    }
  }


}
