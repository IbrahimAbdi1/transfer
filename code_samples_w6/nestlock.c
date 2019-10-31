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

typedef struct {
  int x, y; 
  omp_nest_lock_t lck; 
} point;

void move_x(point *p, int dist) {
  p->x += dist;
}

void move_y(point *p, int dist) {
  omp_set_nest_lock(&p->lck);
  p->y += dist;
  omp_unset_nest_lock(&p->lck);
}

void move_xy(point *p, int d_x, int d_y) {
  omp_set_nest_lock(&p->lck);
  move_x(p, d_x);   
  move_y(p, d_y);
  omp_unset_nest_lock(&p->lck);
}

void process_point(point *p, int offset_x, int offset_y, int other_offset_y){

	#pragma omp parallel sections 
	{
		#pragma omp section
		move_xy(p, offset_x, offset_y);	

		#pragma omp section
		move_y(p, other_offset_y);
	}
}

int main(int argc, char *argv[]) {

	point p = {5, 7};

	process_point(&p, 4, 2, 1);	

	printf("Point(%d,%d)\n", p.x, p.y);

	return 0;
}
