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
  
  // get total available threads
  int threadCount = 

  // if threadCount < pixelcount call kernel1 with appropriate params

  // else call kernel1 multple times ig

  
  // wait for all threads 
  
  // then call normalize 
}

__global__ void kernel1(const int8_t *filter, int32_t dimension, const int32_t *input, 
int32_t *output, int32_t width,int32_t height) {

  // get index given tid

  // call apply2d on input @ index and store it  on output @ index

                          
}

__global__ void normalize1(int32_t *image, int32_t width, int32_t height,
                           int32_t smallest, int32_t biggest) {

  // reductions memes 
    
}
