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

#define MY_MAX(x,y) ((x > y) ? x : y)
#define MY_MIN(x,y) ((x < y) ? x : y)

int reduceCPU_max(int *array, int num_elem) {
	int max = INT_MIN;
	for(int i = 0; i < num_elem; i++) {
		max = MY_MAX(max, array[i]);
	}
	return max;
}

int reduceCPU_sum(int *array, int num_elem) {
	int sum = 0;
	for(int i = 0; i < num_elem; i++) {
		sum += array[i];
	}
	return sum;
}
