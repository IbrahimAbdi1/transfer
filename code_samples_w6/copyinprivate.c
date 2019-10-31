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


short cut, term;
#pragma omp threadprivate(cut, term)

int main(int argc, char *argv[]){

	cut = 1, term = 3;

	#pragma omp parallel num_threads(8) copyin(cut, term)
	{
		int tid = omp_get_thread_num();
		cut += tid * 10; 
		term += tid * 100; 	
		printf("Thread %d – cut=%d term=%d\n", tid, cut, term);
	}

	printf("--------------------------\n");

	#pragma omp parallel num_threads(8) 
	{
		int tid = omp_get_thread_num();

		#pragma omp single copyprivate(cut, term) 
		{
			scanf("%hd %hd", &cut, &term); 
		}

		printf("Thread %d – cut=%d term=%d\n", tid, cut, term);
	}

	return 0;
}


