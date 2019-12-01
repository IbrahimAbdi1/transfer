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

#include "kernels.h"
#include <stdio.h>
#include <string>
#include <unistd.h>
#include <math.h>
#include <pthread.h>
typedef struct common_work_t
{
    const int8_t *f;
    int32_t dimension;
            const int32_t *original_image;
            int32_t *output_image;
            int32_t width;
            int32_t height;
            int32_t max_threads;
            pthread_barrier_t barrier;
            int32_t work_chunk;
            int32_t minp;
            int32_t maxp;
            pthread_mutex_t lock;
} common_work;
typedef struct work_t
{
    common_work *common;
    int32_t id;
} work;

int32_t apply2d(const int8_t*f,int32_t dimension, const int32_t *original, int32_t *target,
    int32_t width, int32_t height,
    int row, int column)
{
int32_t sum = 0;
int filter_centre = f->dimension/2;

int s_row = row - filter_centre;
int s_column = column - filter_centre;
for(int r = 0;r<f->dimension;r++){
    int n_row = s_row + r;
    for(int c = 0;c<f->dimension;c++){
        int n_column = s_column + c;
        if((n_row >= 0) && (n_column >= 0) && (n_column < width) && (n_row < height)){
            sum += (f->matrix[access(r,c,f->dimension)]) * (original[access(n_row,n_column,width)]);
            
        }
    }
}
return sum;
}

void run_best_cpu(const int8_t *filter, int32_t dimension, const int32_t *input,int32_t *output, int32_t width, int32_t height) {
    common_work *x = malloc(sizeof(common_work));
    x->f = filter;
    x->dimension = dimension;
    x->original_image = input;
    x->output_image = output;
    pthread_barrier_init(&(x->barrier),NULL,8);
    pthread_mutex_init(&(x->lock), NULL);
    x->max_threads = 8;
    x->width = width; x->height = height;
    x->minp = 0; x->maxp = 255;
    pthread_t *t = (pthread_t*)malloc(8 * sizeof(pthread_t));

    for(int i = 0; i < 8; i++) {
        
        work *y = malloc(sizeof(work));
        y->common = x;
        y->id = i;
        pthread_create(&t[i], NULL,sharding_row_work , (void *)y);
    }

    for(int i = 0; i < 8; i++) {
        pthread_join(t[i], NULL);
    }


}


void *sharding_row_work(void *args){
    work *w = (work *)args;
    common_work *x = w->common;
    int num_rows =w->common->height/w->common->max_threads;
    int start_row = w->id * num_rows;
    int end_row = start_row + num_rows;
    double pix_min = INFINTY;
    double pix_max = -INFINTY;
    if(w->id == (w->common->max_threads - 1)){
        for(int i=start_row;i<x->height;i++){
            for(int j =0;j<w->common->width;j++){
                int32_t new_pix = apply2d(x->f,x->dimension,x->original_image,x->output_image,x->width,x->height,i,j);
                x->output_image[access(i,j,x->width)] = new_pix;
                if(new_pix < pix_min){
                    pix_min = new_pix;
                }
                else if (new_pix > pix_max){
                    pix_max = new_pix;
                }
            }
        }

        pthread_mutex_lock(&(x->lock));
        if(pix_min < x->minp){
            x->minp = (int32_t)pix_min;
        }
        if (pix_max > x->maxp){
            x->maxp = (int32_t)pix_max;
        }
        pthread_mutex_unlock(&(x->lock));

        pthread_barrier_wait(&(x->barrier));
        for(int i=start_row;i<x->height;i++){
            for(int j =0;j<w->common->width;j++){
                normalize_pixel(x->output_image,access(i,j,x->width),x->minp,x->maxp);
            }
        }

    }
    else{
        for(int i=start_row;i<end_row;i++){
            for(int j =0;j<w->common->width;j++){
                int32_t new_pix = apply2d(x->f,x->original_image,x->output_image,x->width,x->height,i,j);
                x->output_image[access(i,j,x->width)] = new_pix;
                if(new_pix < pix_min){
                    pix_min = new_pix;
                }
                else if (new_pix > pix_max){
                    pix_max = new_pix;
                }
            }
            
        }

        pthread_mutex_lock(&(x->lock));
        if(pix_min < x->minp){
            x->minp = pix_min;
        }
        if (pix_max > x->maxp){
            x->maxp = pix_max;
        }
        pthread_mutex_unlock(&(x->lock));

        pthread_barrier_wait(&(x->barrier));
        for(int i=start_row;i<end_row;i++){
            for(int j =0;j<w->common->width;j++){
                normalize_pixel(x->output_image,access(i,j,x->width),x->minp,x->maxp);
            }
        }
    }
    
    
    
   
    return NULL;
}
