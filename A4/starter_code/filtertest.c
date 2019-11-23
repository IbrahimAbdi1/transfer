#include "filters.h"
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>


const int8_t FILTER[] = {
    0, 1, 1, 2, 2, 2,   1,   1,   0, 1, 2, 4, 5, 5,   5,   4,   2,
    1, 1, 4, 5, 3, 0,   3,   5,   4, 1, 2, 5, 3, -12, -24, -12, 3,
    5, 2, 2, 5, 0, -24, -40, -24, 0, 5, 2, 2, 5, 3,   -12, -24, -12,
    3, 5, 2, 1, 4, 5,   3,   0,   3, 5, 4, 1, 1, 2,   4,   5,   5,
    5, 4, 2, 1, 0, 1,   1,   2,   2, 2, 1, 1, 0,
};

void apply_filter2d(const int8_t *filter, int32_t dimension, const int32_t *input, 
int32_t *output, int32_t width,int32_t height)){

    int32_t sum = 0;
    int filter_centre = dimension/2;
    for(int i = 0;i<width*height;i++){
        int s_row = row - filter_centre;
        int s_column = column - filter_centre;
        for(int r = 0;r<dimension;r++){
            int n_row = s_row + r;
            for(int c = 0;c<dimension;c++){
                int n_column = s_column + c;
                if((n_row >= 0) && (n_column >= 0) && (n_column < width) && (n_row < height)){
                    sum += (filter[r*dimension + c]) * (input[n_row*width + n_column]);
                    
                }
            }
        }
        output[i] = sum;
    }
}

int maint(){
    int32_t x[] = {0,1,2,3};
    int32_t y[] = {0,1,2,3};
}