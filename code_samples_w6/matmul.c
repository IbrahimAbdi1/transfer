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
#include <omp.h>


int main(int argc, char *argv[]) {

	int n = 8;
	int **a = malloc(n*sizeof(int*));
	int **b = malloc(n*sizeof(int*));
	int **c = malloc(n*sizeof(int*));
	
	for(int i = 0; i < n; i++) {
		a[i] = malloc(n*sizeof(int));
		b[i] = malloc(n*sizeof(int));
		c[i] = malloc(n*sizeof(int));
		for(int j = 0; j < n; j++) {
			a[i][j] = 1; //i+j+1;
			b[i][j] = 1; //i+j+1;
		}
	}

	#pragma omp parallel \
        shared(a, b, c, n) \
        num_threads(8) 
  {
		#pragma omp for schedule(static)
		for(int i = 0; i < n; i++) {
			for(int j = 0; j < n; j++) {
				c[i][j] = 0;
				for(int k = 0; k < n; k++) {		
					c[i][j] += a[i][k] * b[k][j];
				}
			}
		}
	}

	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			printf("%d ", c[i][j]);
		}
		printf("\n");
	}

	return 0;
}
