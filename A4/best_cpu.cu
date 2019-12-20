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
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <pthread.h>

int32_t max_g = INT32_MIN;
int32_t min_g = INT32_MAX;

void normalize_pixel(int32_t *target, int32_t pixel_idx, int32_t smallest,
                     int32_t largest)
{
    if (smallest == largest)
    {
        return;
    }

    target[pixel_idx] = ((target[pixel_idx] - smallest) * 255) / (largest - smallest);
}

int access(int row, int column, int width)
{
    return row * width + column;
}

int32_t apply2d(const int8_t *f, int32_t dimension, const int32_t *original, int32_t *target,
                int32_t width, int32_t height,
                int row, int column)
{
    int32_t sum = 0;
    int filter_centre = dimension / 2;
    int s_row = row - filter_centre;
    int s_column = column - filter_centre;
    for (int r = 0; r < dimension; r++)
    {
        int n_row = s_row + r;
        for (int c = 0; c < dimension; c++)
        {
            int n_column = s_column + c;
            if ((n_row >= 0) && (n_column >= 0) && (n_column < width) && (n_row < height))
            {
                sum += (f[access(r, c, dimension)]) * (original[access(n_row, n_column, width)]);
            }
        }
    }
    return sum;
}

void *sharding_row_work(void *args)
{
    work *w = (work *)args;
    common_work *x = w->common;
    int num_rows = w->common->height / w->common->max_threads;
    int start_row = w->id * num_rows;
    int end_row = start_row + num_rows;
    int32_t pix_min = INT32_MAX;
    int32_t pix_max = INT32_MIN;
    if (w->id == (w->common->max_threads - 1))
    {
        for (int i = start_row; i < x->height; i++)
        {
            for (int j = 0; j < w->common->width; j++)
            {
                int32_t new_pix = apply2d(x->f, x->dimension, x->original_image, x->output_image, x->width, x->height, i, j);

                x->output_image[access(i, j, x->width)] = new_pix;

                if (new_pix < pix_min)
                {
                    pix_min = new_pix;
                }
                if (new_pix > pix_max)
                {
                    pix_max = new_pix;
                }
            }
        }

        pthread_mutex_lock(&(x->lock));
        if (pix_min < x->minp)
        {
            x->minp = pix_min;
        }
        if (pix_max > x->maxp)
        {
            x->maxp = pix_max;
        }
        pthread_mutex_unlock(&(x->lock));

        pthread_barrier_wait(&(x->barrier));

        for (int i = start_row; i < x->height; i++)
        {
            for (int j = 0; j < w->common->width; j++)
            {
                normalize_pixel(x->output_image, access(i, j, x->width), x->minp, x->maxp);
            }
        }
    }
    else
    {
        for (int i = start_row; i < end_row; i++)
        {
            for (int j = 0; j < w->common->width; j++)
            {
                int32_t new_pix = apply2d(x->f, x->dimension, x->original_image, x->output_image, x->width, x->height, i, j);
                x->output_image[access(i, j, x->width)] = new_pix;

                if (new_pix < pix_min)
                {
                    pix_min = new_pix;
                }
                else if (new_pix > pix_max)
                {
                    pix_max = new_pix;
                }
            }
        }

        pthread_mutex_lock(&(x->lock));
        if (pix_min < x->minp)
        {
            x->minp = pix_min;
        }

        if (pix_max > x->maxp)
        {
            x->maxp = pix_max;
        }
        pthread_mutex_unlock(&(x->lock));

        pthread_barrier_wait(&(x->barrier));
        for (int i = start_row; i < end_row; i++)
        {
            for (int j = 0; j < w->common->width; j++)
            {
                normalize_pixel(x->output_image, access(i, j, x->width), x->minp, x->maxp);
            }
        }
    }

    return NULL;
}

void run_best_cpu(const int8_t *filter, int32_t dimension, const int32_t *input, int32_t *output, int32_t width, int32_t height)
{
    common_work *x = (common_work *)malloc(sizeof(common_work));
    x->f = filter;
    x->dimension = dimension;
    x->original_image = input;
    x->output_image = output;
    pthread_barrier_init(&(x->barrier), NULL, 8);
    pthread_mutex_init(&(x->lock), NULL);
    x->max_threads = 8;
    x->width = width;
    x->height = height;
    x->minp = INT32_MAX;
    x->maxp = INT32_MIN;
    pthread_t *t = (pthread_t *)malloc(8 * sizeof(pthread_t));

    for (int i = 0; i < 8; i++)
    {

        work *y = (work *)malloc(sizeof(work));
        y->common = x;
        y->id = i;
        pthread_create(&t[i], NULL, sharding_row_work, (void *)y);
    }

    for (int i = 0; i < 8; i++)
    {
        pthread_join(t[i], NULL);
    }
}
