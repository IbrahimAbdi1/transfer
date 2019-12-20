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
#include <time.h>

#include "pgm.h"
#include "kernels.h"

/* Use this function to print the time of each of your kernels.
 * The parameter names are intuitive, but don't hesitate to ask
 * for clarifications.
 * DO NOT modify this function.*/
void print_run(float time_cpu, int kernel, float time_gpu_computation,
               float time_gpu_transfer_in, float time_gpu_transfer_out)
{
  printf("%12.6f ", time_cpu);
  printf("%5d ", kernel);
  printf("%12.6f ", time_gpu_computation);
  printf("%14.6f ", time_gpu_transfer_in);
  printf("%15.6f ", time_gpu_transfer_out);
  printf("%13.2f ", time_cpu / time_gpu_computation);
  printf("%7.2f\n", time_cpu / (time_gpu_computation + time_gpu_transfer_in +
                                time_gpu_transfer_out));
}

int main(int argc, char **argv)
{
  int c;
  std::string input_filename, cpu_output_filename, base_gpu_output_filename;
  if (argc < 3)
  {
    printf("Wrong usage. Expected -i <input_file> -o <output_file>\n");
    return 0;
  }

  while ((c = getopt(argc, argv, "i:o:")) != -1)
  {
    switch (c)
    {
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
  const int8_t FILTER[] = {
    0, 1, 1, 2, 2, 2, 1, 1, 0,
    1, 2, 4, 5, 5, 5, 4, 2, 1,
    1, 4, 5, 3, 0, 3, 5, 4, 1,
    2, 5, 3, -12, -24, -12, 3, 5, 2,
    2, 5, 0, -24, -40, -24, 0, 5, 2,
    2, 5, 3, -12, -24, -12, 3, 5, 2,
    1, 4, 5, 3, 0, 3, 5, 4, 1,
    1, 2, 4, 5, 5, 5, 4, 2, 1,
    0, 1, 1, 2, 2, 2, 1, 1, 0,
  };
  pgm_image source_img;
  init_pgm_image(&source_img);

  if (load_pgm_from_file(input_filename.c_str(), &source_img) != NO_ERR)
  {
    printf("Error loading source image.\n");
    return 0;
  }

  /* Do not modify this printf */
  printf("CPU_time(ms) Kernel GPU_time(ms) TransferIn(ms) TransferOut(ms) "
         "Speedup_noTrf Speedup\n");

  /* TODO: run your CPU implementation here and get its time. Don't include
   * file IO in your measurement.*/
  /* For example: */
  float time_cpu;
  {
    std::string cpu_file = cpu_output_filename;
    pgm_image cpu_output_img;
    copy_pgm_image_size(&source_img, &cpu_output_img);

    // Start time
    struct timespec start, stop;
    clock_gettime(CLOCK_MONOTONIC, &start);
    run_best_cpu(FILTER, 9, source_img.matrix, cpu_output_img.matrix, cpu_output_img.width, cpu_output_img.height); // From kernels.h
    // End time
    clock_gettime(CLOCK_MONOTONIC, &stop);
    time_cpu = ((stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / 1000000000) * 1000;
    // print_run(args...)      // Defined on the top of this file
    print_run(time_cpu, 0, 0, 0, 0);
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
     // Cold Access Run (Kernel 1)
  {
    pgm_image gpu_output_img;
    copy_pgm_image_size(&source_img, &gpu_output_img);
    int32_t *g_min_max;
    int32_t *deviceMatrix_IN,*deviceMatrix_OUT;
    int8_t *deviceFilter;
    int pixelCount = gpu_output_img.width*gpu_output_img.height;
    int numBlocks = pixelCount/1024;
    int size = pixelCount*sizeof(int32_t);
    cudaMalloc((void**)&deviceMatrix_IN,size);
    cudaMalloc((void**)&deviceMatrix_OUT,size);
    cudaMalloc((void**)&deviceFilter,9*9*sizeof(int8_t));
    cudaMalloc((void**)&g_min_max,2*(numBlocks+1)*sizeof(int32_t));

    cudaMemcpy(deviceMatrix_IN,source_img.matrix,size, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatrix_OUT,source_img.matrix,size, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceFilter,FILTER,9*9*sizeof(int8_t),cudaMemcpyHostToDevice);
    run_kernel1(deviceFilter,9,deviceMatrix_IN,deviceMatrix_OUT,gpu_output_img.width,gpu_output_img.height,g_min_max);
    cudaMemcpy(gpu_output_img.matrix,deviceMatrix_OUT,size, cudaMemcpyDeviceToHost);

    cudaFree(deviceMatrix_IN);
    cudaFree(deviceMatrix_OUT);
    cudaFree(deviceFilter);
    cudaFree(g_min_max);

    destroy_pgm_image(&gpu_output_img);
  }
  {
    std::string gpu_file = "1" + base_gpu_output_filename;
    pgm_image gpu_output_img;
    copy_pgm_image_size(&source_img, &gpu_output_img);
    int32_t *g_min_max;
    int32_t *deviceMatrix_IN, *deviceMatrix_OUT;
    int8_t *deviceFilter;
    int pixelCount = gpu_output_img.width * gpu_output_img.height;
    int numBlocks = pixelCount / 1024;
    int size = pixelCount * sizeof(int32_t);
    cudaMalloc((void **)&deviceMatrix_IN, size);
    cudaMalloc((void **)&deviceMatrix_OUT, size);
    cudaMalloc((void **)&deviceFilter, 9 * 9 * sizeof(int8_t));
    cudaMalloc((void **)&g_min_max, 2 * (numBlocks + 1) * sizeof(int32_t));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start time
    cudaEventRecord(start);

    cudaMemcpy(deviceMatrix_IN, source_img.matrix, size, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatrix_OUT, source_img.matrix, size, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceFilter, FILTER, 9 * 9 * sizeof(int8_t), cudaMemcpyHostToDevice);

    // End time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float transfer_in;
    cudaEventElapsedTime(&transfer_in, start, stop);

    // Start time
    cudaEventRecord(start);
    run_kernel1(deviceFilter, 9, deviceMatrix_IN, deviceMatrix_OUT, gpu_output_img.width, gpu_output_img.height, g_min_max);
    cudaDeviceSynchronize();
    // End time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_gpu;
    cudaEventElapsedTime(&time_gpu, start, stop);

    // Start time
    cudaEventRecord(start);
    cudaMemcpy(gpu_output_img.matrix, deviceMatrix_OUT, size, cudaMemcpyDeviceToHost);
    // End time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float transfer_out;
    cudaEventElapsedTime(&transfer_out, start, stop);

    cudaFree(deviceMatrix_IN);
    cudaFree(deviceMatrix_OUT);
    cudaFree(deviceFilter);
    cudaFree(g_min_max);

    // print_run(args...)     // Defined on the top of this file
    print_run(time_cpu, 1, time_gpu, transfer_in, transfer_out);
    save_pgm_to_file(gpu_file.c_str(), &gpu_output_img);
    destroy_pgm_image(&gpu_output_img);
  }

  /* Repeat that for all 5 kernels. Don't hesitate to ask if you don't
   * understand the idea. */

  {
    std::string gpu_file = "2" + base_gpu_output_filename;
    pgm_image gpu_output_img;
    copy_pgm_image_size(&source_img, &gpu_output_img);
    int32_t *g_min_max;
    int32_t *deviceMatrix_IN, *deviceMatrix_OUT;
    int8_t *deviceFilter;
    int pixelCount = gpu_output_img.width * gpu_output_img.height;
    int numBlocks = pixelCount / 1024;
    int size = pixelCount * sizeof(int32_t);
    cudaMalloc((void **)&deviceMatrix_IN, size);
    cudaMalloc((void **)&deviceMatrix_OUT, size);
    cudaMalloc((void **)&deviceFilter, 9 * 9 * sizeof(int8_t));
    cudaMalloc((void **)&g_min_max, 2 * (numBlocks + 1) * sizeof(int32_t));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start time
    cudaEventRecord(start);
    cudaMemcpy(deviceMatrix_IN, source_img.matrix, size, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatrix_OUT, source_img.matrix, size, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceFilter, FILTER, 9 * 9 * sizeof(int8_t), cudaMemcpyHostToDevice);
    // End time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float transfer_in;
    cudaEventElapsedTime(&transfer_in, start, stop);

    // Start time
    cudaEventRecord(start);
    run_kernel2(deviceFilter, 9, deviceMatrix_IN, deviceMatrix_OUT, gpu_output_img.width, gpu_output_img.height, g_min_max);
    // End time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_gpu;
    cudaEventElapsedTime(&time_gpu, start, stop);

    // Start time
    cudaEventRecord(start);
    cudaMemcpy(gpu_output_img.matrix, deviceMatrix_OUT, size, cudaMemcpyDeviceToHost);
    // End time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float transfer_out;
    cudaEventElapsedTime(&transfer_out, start, stop);

    cudaFree(deviceMatrix_IN);
    cudaFree(deviceMatrix_OUT);
    cudaFree(deviceFilter);
    cudaFree(g_min_max);

    // print_run(args...)     // Defined on the top of this file
    print_run(time_cpu, 2, time_gpu, transfer_in, transfer_out);
    save_pgm_to_file(gpu_file.c_str(), &gpu_output_img);
    destroy_pgm_image(&gpu_output_img);
  }

  {
    std::string gpu_file = "3" + base_gpu_output_filename;
    pgm_image gpu_output_img;
    copy_pgm_image_size(&source_img, &gpu_output_img);
    int32_t *g_min_max;
    int32_t *deviceMatrix_IN, *deviceMatrix_OUT;
    int8_t *deviceFilter;
    int pixelCount = gpu_output_img.width * gpu_output_img.height;
    int numBlocks = pixelCount / 1024;
    int size = pixelCount * sizeof(int32_t);
    cudaMalloc((void **)&deviceMatrix_IN, size);
    cudaMalloc((void **)&deviceMatrix_OUT, size);
    cudaMalloc((void **)&deviceFilter, 9 * 9 * sizeof(int8_t));
    cudaMalloc((void **)&g_min_max, 2 * (numBlocks + 1) * sizeof(int32_t));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start time
    cudaEventRecord(start);
    cudaMemcpy(deviceMatrix_IN, source_img.matrix, size, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatrix_OUT, source_img.matrix, size, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceFilter, FILTER, 9 * 9 * sizeof(int8_t), cudaMemcpyHostToDevice);
    // End time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float transfer_in;
    cudaEventElapsedTime(&transfer_in, start, stop);

    // Start time
    cudaEventRecord(start);
    run_kernel3(deviceFilter, 9, deviceMatrix_IN, deviceMatrix_OUT, gpu_output_img.width, gpu_output_img.height, g_min_max);
    // End time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_gpu;
    cudaEventElapsedTime(&time_gpu, start, stop);

    // Start time
    cudaEventRecord(start);
    cudaMemcpy(gpu_output_img.matrix, deviceMatrix_OUT, size, cudaMemcpyDeviceToHost);
    // End time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float transfer_out;
    cudaEventElapsedTime(&transfer_out, start, stop);

    cudaFree(deviceMatrix_IN);
    cudaFree(deviceMatrix_OUT);
    cudaFree(deviceFilter);
    cudaFree(g_min_max);

    // print_run(args...)     // Defined on the top of this file
    print_run(time_cpu, 3, time_gpu, transfer_in, transfer_out);
    save_pgm_to_file(gpu_file.c_str(), &gpu_output_img);
    destroy_pgm_image(&gpu_output_img);
  }

  {
    std::string gpu_file = "4" + base_gpu_output_filename;
    pgm_image gpu_output_img;
    copy_pgm_image_size(&source_img, &gpu_output_img);
    int32_t *g_min_max;
    int32_t *deviceMatrix_IN, *deviceMatrix_OUT;
    int8_t *deviceFilter;
    int pixelCount = gpu_output_img.width * gpu_output_img.height;
    int numBlocks = pixelCount / 1024;
    int size = pixelCount * sizeof(int32_t);
    cudaMalloc((void **)&deviceMatrix_IN, size);
    cudaMalloc((void **)&deviceMatrix_OUT, size);
    cudaMalloc((void **)&deviceFilter, 9 * 9 * sizeof(int8_t));
    cudaMalloc((void **)&g_min_max, 2 * (numBlocks + 1) * sizeof(int32_t));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start time
    cudaEventRecord(start);
    cudaMemcpy(deviceMatrix_IN, source_img.matrix, size, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatrix_OUT, source_img.matrix, size, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceFilter, FILTER, 9 * 9 * sizeof(int8_t), cudaMemcpyHostToDevice);
    // End time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float transfer_in;
    cudaEventElapsedTime(&transfer_in, start, stop);

    // Start time
    cudaEventRecord(start);
    run_kernel4(deviceFilter, 9, deviceMatrix_IN, deviceMatrix_OUT, gpu_output_img.width, gpu_output_img.height, g_min_max);
    // End time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_gpu;
    cudaEventElapsedTime(&time_gpu, start, stop);

    // Start time
    cudaEventRecord(start);
    cudaMemcpy(gpu_output_img.matrix, deviceMatrix_OUT, size, cudaMemcpyDeviceToHost);
    // End time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float transfer_out;
    cudaEventElapsedTime(&transfer_out, start, stop);

    cudaFree(deviceMatrix_IN);
    cudaFree(deviceMatrix_OUT);
    cudaFree(deviceFilter);
    cudaFree(g_min_max);

    // print_run(args...)     // Defined on the top of this file
    save_pgm_to_file(gpu_file.c_str(), &gpu_output_img);
    print_run(time_cpu, 4, time_gpu, transfer_in, transfer_out);
    destroy_pgm_image(&gpu_output_img);
  }
  {
    std::string gpu_file = "5" + base_gpu_output_filename;
    pgm_image gpu_output_img;
    copy_pgm_image_size(&source_img, &gpu_output_img);
    void *universal;
    int32_t *g_min_max;
    int32_t *deviceMatrix_IN, *deviceMatrix_OUT;
    int8_t *deviceFilter;
    int pixelCount = gpu_output_img.width * gpu_output_img.height;
    int numBlocks = pixelCount / 1024;
    int size = pixelCount * sizeof(int32_t);

    // Pinned memory allocation for images

    int32_t *new_source = NULL;
    cudaMallocHost((void **)&new_source, size);
    memcpy(new_source, source_img.matrix, size);

    cudaMalloc((void **)&deviceMatrix_IN, size);
    cudaMalloc((void **)&deviceMatrix_OUT, size);
    cudaMalloc((void **)&deviceFilter, 9 * 9 * sizeof(int8_t));
    cudaMalloc((void **)&g_min_max, 2 * (numBlocks + 1) * sizeof(int32_t));
    cudaMalloc((void **)&universal, size + size + 9 * 9 * sizeof(int8_t) + 2 * (numBlocks + 1) * sizeof(int32_t));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start time
    cudaEventRecord(start);
    cudaMemcpy(deviceMatrix_IN, new_source, size, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatrix_OUT, new_source, size, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceFilter, FILTER, 9 * 9 * sizeof(int8_t), cudaMemcpyHostToDevice);
    // End time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float transfer_in;
    cudaEventElapsedTime(&transfer_in, start, stop);

    // Start time
    cudaEventRecord(start);
    run_kernel5(deviceFilter, 9, deviceMatrix_IN, deviceMatrix_OUT, gpu_output_img.width, gpu_output_img.height, g_min_max);
    // End time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_gpu;
    cudaEventElapsedTime(&time_gpu, start, stop);

    // Start time
    cudaEventRecord(start);
    cudaMemcpy(gpu_output_img.matrix, deviceMatrix_OUT, size, cudaMemcpyDeviceToHost);
    // End time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float transfer_out;
    cudaEventElapsedTime(&transfer_out, start, stop);

    cudaFree(new_source);
    cudaFree(deviceMatrix_IN);
    cudaFree(deviceMatrix_OUT);
    cudaFree(deviceFilter);
    cudaFree(g_min_max);

    // print_run(args...)     // Defined on the top of this file
    print_run(time_cpu, 5, time_gpu, transfer_in, transfer_out);
    save_pgm_to_file(gpu_file.c_str(), &gpu_output_img);
    destroy_pgm_image(&gpu_output_img);
  }
}
