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
	int *array = malloc(n*sizeof(int));
	int *factor = malloc(n*sizeof(int));
	int *prefix_sums = malloc(n*sizeof(int));
	
	for(int i = 0; i < n; i++) {
		array[i] = i+1;
		factor[i] = 1;
	}

	array[0] *= factor[0];
	prefix_sums[0] = array[0];
	#pragma omp parallel for ordered shared(prefix_sums, array, factor, n) 
	for(int i = 1; i < n; i++) {

		// Process loop iterations in parallel
		array[i] *= factor[i];

		// Only one thread executes this, item depends on previous iteration
		#pragma omp ordered 
		{ 
			prefix_sums[i] = prefix_sums[i-1] + array[i]; 
		}	
	}
	
	for(int i = 0; i < n; i++) {
		printf("Prefix sum[0-%d] = %d\n", i, prefix_sums[i]);
	}

	return 0;
}



 
