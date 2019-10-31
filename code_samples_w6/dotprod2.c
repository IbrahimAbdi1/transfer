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

	long size = 32*1024*1024;
	int *a = malloc(size*sizeof(int));
	int *b = malloc(size*sizeof(int));
	
	for(int i = 0; i < size; i++) {
		a[i] = 1;
		b[i] = 2;
	}

	long dotprod = 0;
	double start, end;	

	start = omp_get_wtime();
	#pragma omp parallel shared(size, a, b) \
	            reduction(+: dotprod) num_threads(8) 
	{
		#pragma omp for
		for(int i = 0; i < size; i++) {
			dotprod += a[i] * b[i];
		}
	}
	end = omp_get_wtime();

	printf("Elapsed time = %lf\n", end - start);
	printf("Dot product = %ld\n", dotprod);

	return 0;
}



 
