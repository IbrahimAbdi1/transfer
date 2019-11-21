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

#ifndef __ERROR_HANDLING_H__
#define __ERROR_HANDLING_H__ 

#define ERROR_CHECK_ENABLED

inline void 
__checkSafeCall(cudaError_t error, const char *filename, int line) {
	#ifdef ERROR_CHECK_ENABLED
	if (error != cudaSuccess) {
		fprintf(stderr, "Error: %s in %s:%d\n", cudaGetErrorString(error), filename, line);
		exit(EXIT_FAILURE);
	}
	#endif
}
#define checkSafeCall(error) (__checkSafeCall(error, __FILE__, __LINE__))

inline void 
__checkKernelError(const char *filename, int line) {
	#ifdef ERROR_CHECK_ENABLED
	cudaError error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "Error: %s in %s:%d\n", cudaGetErrorString(error), filename, line);
		exit(EXIT_FAILURE);
	}
	#endif
}
#define checkKernelError() (__checkKernelError(__FILE__, __LINE__))

inline void 
__checkKernelErrorSync(const char *filename, int line) {
	#ifdef ERROR_CHECK_ENABLED
	cudaError error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "Error: %s in %s:%d\n", cudaGetErrorString(error), filename, line);
		exit(EXIT_FAILURE);
	}

	// Sync device, just in case. Caveat: this takes a performance hit.
	error = cudaDeviceSynchronize();
	if(cudaSuccess != error) {
		fprintf(stderr, "Sync error: %s in %s:%d\n", cudaGetErrorString(error), filename, line);
		exit(EXIT_FAILURE);
	}
	#endif
}
#define checkKernelErrorSync() (__checkKernelErrorSync(__FILE__, __LINE__))

// Does not work in earlier capabilities
#define CudaAssert(expr) \
	if (!(expr)) { \
		printf( “Cuda assert failed block:thread - %d:%d\n”, blockIdx.x, threadIdx.x ); \
	}

#endif
