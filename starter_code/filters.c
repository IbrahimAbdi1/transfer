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

#include "filters.h"
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>



typedef struct common_work_t
{
    const filter *f;
            const int32_t *original_image;
            int32_t *output_image;
            int32_t width;
            int32_t height;
            int32_t max_threads;
            pthread_barrier_t barrier;
} common_work;
typedef struct work_t
{
    common_work *common;
    int32_t id;
} work;

int pix_min = 0;
int pix_max = 255;

/************** FILTER CONSTANTS*****************/
/* laplacian */
int8_t lp3_m[] =
    {
        0, 1, 0,
        1, -4, 1,
        0, 1, 0,
    };
filter lp3_f = {3, lp3_m};

int8_t lp5_m[] =
    {
        -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1,
        -1, -1, 24, -1, -1,
        -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1,
    };
filter lp5_f = {5, lp5_m};

/* Laplacian of gaussian */
int8_t log_m[] =
    {
        0, 1, 1, 2, 2, 2, 1, 1, 0,
        1, 2, 4, 5, 5, 5, 4, 2, 1,
        1, 4, 5, 3, 0, 3, 5, 4, 1,
        2, 5, 3, -12, -24, -12, 3, 5, 2,
        2, 5, 0, -24, -40, -24, 0, 5, 2,
        2, 5, 3, -12, -24, -12, 3, 5, 2,
        1, 4, 5, 3, 0, 3, 5, 4, 1,
        1, 2, 4, 5, 5, 5, 4, 2, 1,
        0, 1, 1, 2, 2, 2, 1, 1, 0,
    };
filter log_f = {9, log_m};

/* Identity */
int8_t identity_m[] = {1};
filter identity_f = {1, identity_m};

filter *builtin_filters[NUM_FILTERS] = {&lp3_f, &lp5_f, &log_f, &identity_f};

/* Normalizes a pixel given the smallest and largest integer values
 * in the image */
void normalize_pixel(int32_t *target, int32_t pixel_idx, int32_t smallest, 
        int32_t largest)
{
    if (smallest == largest)
    {
        return;
    }
    
    target[pixel_idx] = ((target[pixel_idx] - smallest) * 255) / (largest - smallest);
}


int access(int row,int column,int width){
    return row*width+column;
}

/*************** COMMON WORK ***********************/
/* Process a single pixel and returns the value of processed pixel
 * TODO: you don't have to implement/use this function, but this is a hint
 * on how to reuse your code.
 * */
int32_t apply2d(const filter *f, const int32_t *original, int32_t *target,
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



/*********SEQUENTIAL IMPLEMENTATIONS ***************/
/* TODO: your sequential implementation goes here.
 * IMPORTANT: you must test this thoroughly with lots of corner cases and 
 * check against your own manual calculations on paper, to make sure that your code
 * produces the correct image.
 * Correctness is CRUCIAL here, especially if you re-use this code for filtering 
 * pieces of the image in your parallel implementations! 
 */
void apply_filter2d(const filter *f, 
        const int32_t *original, int32_t *target,
        int32_t width, int32_t height)
{

    for(int i = 0; i <height;i++){
        for(int j = 0; j<width;j++){
            int32_t new_pix = apply2d(f,original,target,width,height,i,j);
            target[access(i,j,width)] = new_pix;

            if(new_pix < pix_min){
                pix_min = new_pix;
            }
            else if (new_pix > pix_max){
                pix_max = new_pix;
            }
            
        }
    }

    for(int i = 0; i<(height*width);i++){
        normalize_pixel(target,i,pix_min,pix_max);
    }


}


void* sharding_row_work(void *args){
    work *w = (work *)args;
    common_work *x = w->common;
    int num_rows = w->common->height/w->common->max_threads;
    int start_row = w->id * num_rows;
    int end_row = start_row + num_rows;
    if(w->id == (w->common->max_threads - 1)){
        for(int i=start_row;i<x->height;i++){
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
    }

    pthread_barrier_wait(&(x->barrier));

    for(int i = 0; i<(x->height*x->width);i++){
        normalize_pixel(x->output_image,i,pix_min,pix_max);
    }

}


/****************** ROW/COLUMN SHARDING ************/
/* TODO: you don't have to implement this. It is just a suggestion for the
 * organization of the code.
 */

/* Recall that, once the filter is applied, all threads need to wait for
 * each other to finish before computing the smallest/largets elements
 * in the resulting matrix. To accomplish that, we declare a barrier variable:
 *      pthread_barrier_t barrier;
 * And then initialize it specifying the number of threads that need to call
 * wait() on it:
 *      pthread_barrier_init(&barrier, NULL, num_threads);
 * Once a thread has finished applying the filter, it waits for the other
 * threads by calling:
 *      pthread_barrier_wait(&barrier);
 * This function only returns after *num_threads* threads have called it.
 */
void* sharding_work(void *work)
{
    /* Your algorithm is essentially:
     *  1- Apply the filter on the image
     *  2- Wait for all threads to do the same
     *  3- Calculate global smallest/largest elements on the resulting image
     *  4- Scale back the pixels of the image. For the non work queue
     *      implementations, each thread should scale the same pixels
     *      that it worked on step 1.
     */
    return NULL;
}

/***************** WORK QUEUE *******************/
/* TODO: you don't have to implement this. It is just a suggestion for the
 * organization of the code.
 */
void* queue_work(void *work)
{
    return NULL;
}

/***************** MULTITHREADED ENTRY POINT ******/
/* TODO: this is where you should implement the multithreaded version
 * of the code. Use this function to identify which method is being used
 * and then call some other function that implements it.
 */
void apply_filter2d_threaded(const filter *f,
        const int32_t *original, int32_t *target,
        int32_t width, int32_t height,
        int32_t num_threads, parallel_method method, int32_t work_chunk)
{
    /* You probably want to define a struct to be passed as work for the
     * threads.
     * Some values are used by all threads, while others (like thread id)
     * are exclusive to a given thread. For instance:
     *   typedef struct common_work_t
     *   {
     *       const filter *f;
     *       const int32_t *original_image;
     *       int32_t *output_image;
     *       int32_t width;
     *       int32_t height;
     *       int32_t max_threads;
     *       pthread_barrier_t barrier;
     *   } common_work;
     *   typedef struct work_t
     *   {
     *       common_work *common;
     *       int32_t id;
     *   } work;
     *
     * An uglier (but simpler) solution is to define the shared variables
     * as global variables.
     */
    pthread_barrier_t b;
    pthread_barrier_init(&b,NULL,num_threads);
    common_work *x = malloc(sizeof(common_work));
    work *y = malloc(sizeof(work));
    x->f = f;
    x->original_image = original;
    x->output_image = target;
    x->barrier = b;
    x->max_threads = num_threads;
    x->width = width; x->height = height;
    y->common = x;

    pthread_t *t = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    
    if(method == SHARDED_ROWS){
        
        for(int i = 0; i < num_threads; i++) {
		    y->id = i;
		    pthread_create(&t[i], NULL,sharding_row_work , (void *)y);
	    }

        for(int i = 0; i < num_threads; i++) {
		    pthread_join(t[i], NULL);
	    }
    }


}
