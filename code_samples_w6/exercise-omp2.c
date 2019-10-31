// ------------
// This code is provided solely for the personal and private use of
// students taking the CSC367H5 course at the University of Toronto.
// Copying for purposes other than this use is expressly prohibited.
// All forms of distribution of this code, whether as given or with
// any changes, are expressly prohibited.
//
// Authors: Bogdan Simion
//
// All of the files in this directory and all subdirectories are:
// Copyright (c) 2019 Bogdan Simion
// -------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

int* generate_data(int n) {
  int *a = malloc(n * sizeof(int));
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    a[i] = i*i + n;
  }
  return a;
}

void compute(int *a, int *b, int *weight, int *result, int n) {
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    result[i] += a[i] * weight[i] + b[i];
  }
}

int main(int argc, char *argv[]) {
  int n = 10000000;
  int *a1 = generate_data(n);
  int *a2 = generate_data(n);
  int *w = generate_data(n);
  int *res = calloc(n, sizeof(int));

  double start_time = omp_get_wtime(); 
  for(int i = 0; i < 100; i++) {
    compute(a1, a2, w, res, n);
  }
  double elapsed_time = omp_get_wtime() - start_time; 
  printf("Elapsed time: %lf\n", elapsed_time);
  printf("Result[0]: %d\n", res[0]);
	
  free(a1); free(a2); free(res);
  return 0;
}
