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
#include <omp.h>

int main(int argc, char *argv[]) {
	printf("Starting a parallel region, spawning threads\n");

	#pragma omp parallel 
	{
		printf("Hello world\n");
	}
	printf("Bye world\n");

	return 0;
}

