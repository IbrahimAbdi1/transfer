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

#include <stdio.h>
#include <string>
#include <unistd.h>

#include "pgm.h"

/* Use this function to print the time of each of your kernels.
 * The parameter names are intuitive, but don't hesitate to ask
 * for clarifications.
 * DO NOT modify this function.*/
void print_run(float time_cpu, int kernel, float time_gpu_computation,
               float time_gpu_transfer_in, float time_gpu_transfer_out) {
  printf("%12.6f ", time_cpu);
  printf("%5d ", kernel);
  printf("%12.6f ", time_gpu_computation);
  printf("%14.6f ", time_gpu_transfer_in);
  printf("%15.6f ", time_gpu_transfer_out);
  printf("%13.2f ", time_cpu / time_gpu_computation);
  printf("%7.2f\n", time_cpu / (time_gpu_computation + time_gpu_transfer_in +
                                time_gpu_transfer_out));
}

int main(int argc, char **argv) {
  int c;
  std::string input_filename, cpu_output_filename, base_gpu_output_filename;
  if (argc < 3) {
    printf("Wrong usage. Expected -i <input_file> -o <output_file>\n");
    return 0;
  }

  while ((c = getopt(argc, argv, "i:o:")) != -1) {
    switch (c) {
    case 'i':
      input_filename = std::string(optarg);
      break;
    case 'o':
      cpu_output_filename = std::string(optarg);
      base_gpu_output_filename = std::string(optarg);
      break;
    default:
      return 0;
    }
  }

  pgm_image source_img;
  init_pgm_image(&source_img);

  if (load_pgm_from_file(input_filename.c_str(), &source_img) != NO_ERR) {
    printf("Error loading source image.\n");
    return 0;
  }

  /* Do not modify this printf */
  printf("CPU_time(ms) Kernel GPU_time(ms) TransferIn(ms) TransferOut(ms) "
         "Speedup_noTrf Speedup\n");

  /* TODO: run your CPU implementation here and get its time. Don't include
   * file IO in your measurement.*/
  /* For example: */
  {
    std::string cpu_file = cpu_output_filename;
    pgm_image cpu_output_img;
    copy_pgm_image_size(&source_img, &cpu_output_img);
    // Start time
    // run_best_cpu(args...);  // From kernels.h
    // End time
    // print_run(args...)      // Defined on the top of this file
    save_pgm_to_file(cpu_file.c_str(), &cpu_output_img);
    destroy_pgm_image(&cpu_output_img);
  }

  /* TODO:
   * run each of your gpu implementations here,
   * get their time,
   * and save the output image to a file.
   * Don't forget to add the number of the kernel
   * as a prefix to the output filename:
   * Print the execution times by calling print_run().
   */

  /* For example: */
  {
    std::string gpu_file = "1" + base_gpu_output_filename;
    pgm_image gpu_output_img;
    pgm_image *gpu_output_img_d;
    copy_pgm_image_size(&source_img, &gpu_output_img);

    cudaMalloc(&gpu_output_img_d,sizeof(pgm_image));
    int size = gpu_output_img.width*gpu_output_img.height;
    float transfer_in;
    // cuda memes
    cudaEvent_t start, stop;

    // cuda creats 
    cudaEventCreate(&start);
	  cudaEventCreate(&stop);

    // time for transfer in 
    cudaEventRecord(start);
    cudaMemcpy(gpu_output_img_d, size*sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
	  cudaEventSynchronize(stop);
	  cudaEventElapsedTime(&transfer_in, start, stop);

    // Start time
    // run_kernel1(args...);  // From kernels.h
    // End time
    // print_run(args...)     // Defined on the top of this file
    //printf("Memcopy time "%14.6f \n",transfer_in);
    save_pgm_to_file(gpu_file.c_str(), &gpu_output_img);
    destroy_pgm_image(&gpu_output_img);
  }

  /* Repeat that for all 5 kernels. Don't hesitate to ask if you don't
   * understand the idea. */
}
