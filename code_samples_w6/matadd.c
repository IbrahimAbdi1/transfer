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


void matadd_outer(int **a, int **b, int **c, int n) {
	#pragma omp parallel for shared(a, b, c, n) num_threads(8)
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			c[i][j] = a[i][j] + b[i][j];
		}
	}
}

void matadd_outer_swapped(int **a, int **b, int **c, int n) {
	#pragma omp parallel for shared(a, b, c, n) num_threads(8)
	for(int j = 0; j < n; j++) {
		for(int i = 0; i < n; i++) {
			c[i][j] = a[i][j] + b[i][j];
		}
	}
}

void matadd_inner(int **a, int **b, int **c, int n) {
	for(int i = 0; i < n; i++) {
		#pragma omp parallel for shared(a, b, c, n) num_threads(8)
		for(int j = 0; j < n; j++) {
			c[i][j] = a[i][j] + b[i][j];
		}
	}
}

void matadd_inner_swapped(int **a, int **b, int **c, int n) {
	for(int j = 0; j < n; j++) {
		#pragma omp parallel for shared(a, b, c, n) num_threads(8)
		for(int i = 0; i < n; i++) {
			c[i][j] = a[i][j] + b[i][j];
		}
	}
}

void matadd_nested(int **a, int **b, int **c, int n) {
	#pragma omp parallel for shared(a, b, c, n) num_threads(4)
	for(int i = 0; i < n; i++) {
		
		#pragma omp parallel for shared(a, b, c, n) num_threads(2)
		for(int j = 0; j < n; j++) {
			c[i][j] = a[i][j] + b[i][j];
		}
	}
}

void matadd_collapse(int **a, int **b, int **c, int n) {
	#pragma omp parallel for shared(a, b, c, n) collapse(2) num_threads(8)
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			c[i][j] = a[i][j] + b[i][j];
		}
	}
}

//-----------------------------------
//----- TEST SCHEDULING METHODS -----
//-----------------------------------

void matadd_outer_static(int **a, int **b, int **c, int n) {
	#pragma omp parallel for shared(a, b, c, n) num_threads(8) schedule(static)
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			c[i][j] = a[i][j] + b[i][j];
		}
	}
}

void matadd_outer_dynamic(int **a, int **b, int **c, int n) {
	#pragma omp parallel for shared(a, b, c, n) num_threads(8) schedule(dynamic)
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			c[i][j] = a[i][j] + b[i][j];
		}
	}
}

void matadd_outer_guided(int **a, int **b, int **c, int n) {
	#pragma omp parallel for shared(a, b, c, n) num_threads(8) schedule(guided)
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			c[i][j] = a[i][j] + b[i][j];
		}
	}
}

void matadd_collapse_static(int **a, int **b, int **c, int n) {
	#pragma omp parallel for shared(a, b, c, n) collapse(2) num_threads(8) schedule(static)
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			c[i][j] = a[i][j] + b[i][j];
		}
	}
}

void matadd_collapse_dynamic(int **a, int **b, int **c, int n) {
	#pragma omp parallel for shared(a, b, c, n) collapse(2) num_threads(8) schedule(dynamic)
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			c[i][j] = a[i][j] + b[i][j];
		}
	}
}

void matadd_collapse_guided(int **a, int **b, int **c, int n) {
	#pragma omp parallel for shared(a, b, c, n) collapse(2) num_threads(8) schedule(guided)
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			c[i][j] = a[i][j] + b[i][j];
		}
	}
}

void check_correctness(int **a, int **b, int **c, int n) {
	int broken = 0;
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			//printf("%d ", c[i][j]);
			if(c[i][j] != (a[i][j] + b[i][j])) {
				broken = 1;
				break;
			}
		}
		//printf("\n");
	}
	if (broken) {
		printf("===== BROKEN =====\n");
	}
}

int main(int argc, char *argv[]) {

	if(argc != 3) {
		printf("Usage: %s <matrix-size> <outer|outer-sw|inner|inner-sw|nested|collapse|all>\n", argv[0]);
		exit(1);
	}
	char *mode = argv[2];

	int n = atoi(argv[1]);
	int **a = malloc(n*sizeof(int*));
	int **b = malloc(n*sizeof(int*));
	int **c = malloc(n*sizeof(int*));
	
	for(int i = 0; i < n; i++) {
		a[i] = malloc(n*sizeof(int));
		b[i] = malloc(n*sizeof(int));
		c[i] = malloc(n*sizeof(int));
		for(int j = 0; j < n; j++) {
			a[i][j] = 1; 
			b[i][j] = 1;
		}
	}

	omp_set_nested(1);

	double start, end;

	if (!strcmp(mode, "outer") || !strcmp(mode, "all")) {
		start = omp_get_wtime();
		matadd_outer(a, b, c, n);
		end = omp_get_wtime();
		printf("Elapsed (outer)    =%lf\n", end - start);
		check_correctness(a, b, c, n);
	}

	if (!strcmp(mode, "outer-sw") || !strcmp(mode, "all")) {
		start = omp_get_wtime();
		matadd_outer_swapped(a, b, c, n);
		end = omp_get_wtime();
		printf("Elapsed (outer-sw) =%lf\n", end - start);
		check_correctness(a, b, c, n);
	}

	if (!strcmp(mode, "inner") || !strcmp(mode, "all")) {
		start = omp_get_wtime();	
		matadd_inner(a, b, c, n);
		end = omp_get_wtime();
		printf("Elapsed (inner)    =%lf\n", end - start);
		check_correctness(a, b, c, n);
	}

	if (!strcmp(mode, "inner-sw") || !strcmp(mode, "all")) {
		start = omp_get_wtime();	
		matadd_inner_swapped(a, b, c, n);
		end = omp_get_wtime();
		printf("Elapsed (inner-sw) =%lf\n", end - start);
		check_correctness(a, b, c, n);
	}

	if (!strcmp(mode, "nested") || !strcmp(mode, "all")) {
		start = omp_get_wtime();
		matadd_nested(a, b, c, n);
		end = omp_get_wtime();
		printf("Elapsed (nested)   =%lf\n", end - start);
		check_correctness(a, b, c, n);
	}
	
	if (!strcmp(mode, "collapse") || !strcmp(mode, "all")) {
		start = omp_get_wtime();
		matadd_collapse(a, b, c, n);
		end = omp_get_wtime();
		printf("Elapsed (collapse) =%lf\n", end - start);
		check_correctness(a, b, c, n);
	}

	// SCHEDULING POLICIES 
	if (!strcmp(mode, "outer-sta") || !strcmp(mode, "all")) {
		start = omp_get_wtime();
		matadd_outer_static(a, b, c, n);
		end = omp_get_wtime();
		printf("Elapsed (outer-sts)=%lf\n", end - start);
		check_correctness(a, b, c, n);
	}	

	if (!strcmp(mode, "outer-dyn") || !strcmp(mode, "all")) {
		start = omp_get_wtime();
		matadd_outer_dynamic(a, b, c, n);
		end = omp_get_wtime();
		printf("Elapsed (outer-dyn)=%lf\n", end - start);
		check_correctness(a, b, c, n);
	}	

 	if (!strcmp(mode, "outer-gui") || !strcmp(mode, "all")) {
		start = omp_get_wtime();
		matadd_outer_guided(a, b, c, n);
		end = omp_get_wtime();
		printf("Elapsed (outer-gui)=%lf\n", end - start);
		check_correctness(a, b, c, n);
	}	

	if (!strcmp(mode, "collapse-sta") || !strcmp(mode, "all")) {
		start = omp_get_wtime();
		matadd_collapse_static(a, b, c, n);
		end = omp_get_wtime();
		printf("Elapsed (collapse-sts)=%lf\n", end - start);
		check_correctness(a, b, c, n);
	}	

	if (!strcmp(mode, "collapse-dyn") || !strcmp(mode, "all")) {
		start = omp_get_wtime();
		matadd_collapse_dynamic(a, b, c, n);
		end = omp_get_wtime();
		printf("Elapsed (collapse-dyn)=%lf\n", end - start);
		check_correctness(a, b, c, n);
	}	

 	if (!strcmp(mode, "collapse-gui") || !strcmp(mode, "all")) {
		start = omp_get_wtime();
		matadd_collapse_guided(a, b, c, n);
		end = omp_get_wtime();
		printf("Elapsed (collapse-gui)=%lf\n", end - start);
		check_correctness(a, b, c, n);
	}	

 return 0;
}
 
