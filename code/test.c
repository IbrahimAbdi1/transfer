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

#include <sys/times.h>
#include <time.h>
#include <stdio.h> 
#include <stdlib.h> 
#include <pthread.h> 
#include <string.h>

#define COUNT 10000000

typedef struct {
	int foo;
	int bar;
} doh;

static doh d;
static doh d2;

/* Assume that these functions are being run concurrently, each 
 * by a separate thread. Assume that each thread is running 
 * on a different core than other threads. You may not 
 * make any assumptions about any compiler optimizations.
 */
void *sum_foo() {
	int s = 0;
	for(int i = 0; i < COUNT; i++) {
		s += d.foo;
	}
}

void *add_bar() {
	for(int i = 0; i < COUNT; i++) {
		d.bar++;
	}
}

void *add_bar2() {
	for(int i = 0; i < COUNT; i++) {
		d2.bar++;
	}
}

// Store the timing-related activities
struct timespec tpBegin1,tpEnd1,tpBegin2,tpEnd2,tpBegin3,tpEnd3;  

// Computes time in milliseconds given endTime and startTime timespec structures
double compute(struct timespec start,struct timespec end) {
	double t;
	t=(end.tv_sec-start.tv_sec)*1000;
	t+=(end.tv_nsec-start.tv_nsec)*0.000001;
	
	return t;
}


int main(int argc, char *argv[]) { 
	double time1;
	double time2;
	double time3;
	pthread_t  thread_1;
	pthread_t  thread_2;

	if (argc != 2) {
		printf("Usage: %s <serial|false-sharing|no-false-sharing|all>\n", argv[0]);
		exit(1);
	}
			
	//--------START--------Serial Computation------------------------------------
	if (!strcmp(argv[1],"serial") || !strcmp(argv[1],"all") ) {
		clock_gettime(CLOCK_MONOTONIC,&tpBegin1);
		sum_foo();
		add_bar();
		clock_gettime(CLOCK_MONOTONIC,&tpEnd1);
		time1 = compute(tpBegin1,tpEnd1);
		printf("Serial time:                %f ms\n", time1);
	}
	//--------END----------Serial Computation------------------------------------
	
	//--------START--------parallel computation with False Sharing---------------
	if (!strcmp(argv[1],"false-sharing") || !strcmp(argv[1],"all") ) {
		clock_gettime(CLOCK_MONOTONIC,&tpBegin2);
		pthread_create(&thread_1, NULL, sum_foo, NULL);
		pthread_create(&thread_2, NULL, add_bar, NULL);
		pthread_join(thread_1, NULL);
		pthread_join(thread_2, NULL);
		clock_gettime(CLOCK_MONOTONIC,&tpEnd2);
		time2 = compute(tpBegin2,tpEnd2);
		printf("Time with false sharing:    %f ms\n", time2);
	}
	//--------END----------parallel computation with False Sharing---------------
	
	//--------START--------parallel computation without False Sharing------------
	if (!strcmp(argv[1],"no-false-sharing") || !strcmp(argv[1],"all") ) {
		clock_gettime(CLOCK_MONOTONIC,&tpBegin3);   
		pthread_create(&thread_1, NULL, sum_foo, NULL);
		pthread_create(&thread_2, NULL, add_bar2, NULL);
		pthread_join(thread_1, NULL);
		pthread_join(thread_2, NULL);
		clock_gettime(CLOCK_MONOTONIC,&tpEnd3);
		time3 = compute(tpBegin3,tpEnd3);
		printf("Time without false sharing: %f ms\n", time3);
	}
	//--------END--------parallel computation without False Sharing--------------
	
	return 0; 
}
