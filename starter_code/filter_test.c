#include "filters.h"
#include <stdio.h>
#include <pthread.h>

int8_t lp3_m[] =
    {
        0, 1, 0,
        1, -4, 1,
        0, 1, 0,
    };
filter lp3_f = {3, lp3_m};

int32_t image2_m[] =
    {
        2,2,
        2,2,
    };
int32_t target2_m[] =
    {
        2,2,
        2,2,
    };

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

int32_t apply2d(const filter *f, const int32_t *original, int32_t *target,
        int32_t width, int32_t height,
        int row, int column)
{
    int32_t sum = 0;
    int filter_centre = f->dimension/2;
    //printf("center = %d\n",filter_centre);
    int s_row = row - filter_centre;
    int s_column = column - filter_centre;
    for(int r = 0;r<f->dimension;r++){
        int n_row = s_row + r;
        for(int c = 0;c<f->dimension;c++){
            int n_column = s_column + c;
            if((n_row >= 0) && (n_column >= 0) && (n_column < width) && (n_row < height)){
                sum += (f->matrix[access(r,c,f->dimension)]) * (original[access(n_row,n_column,width)]);
                //printf("nr = %d; nc = %d; r = %d; c = %d; f = %d; o = %d\n",n_row,n_column,r,c, f->matrix[access(r,c,f->dimension)],original[access(n_row,n_column,width)]);
            }
        }
    }
    return sum;
}

void apply_filter2d(const filter *f, 
        const int32_t *original, int32_t *target,
        int32_t width, int32_t height)
{
    for(int i = 0; i <height;i++){
        for(int j = 0; j<width;j++){
            int32_t new_pix = apply2d(f,original,target,width,height,i,j);
            target2_m[access(i,j,width)] = new_pix;
        }
    }
    
    
}

int main(int argc, char **argv){
    int w,h;
    w = 2; h= 2;


    for(int i = (1*(3/2)); i < 3; i++){
            for(int j = 0; j<3;j++){
                printf("(%d,%d)\n",i,j);
            }
        }
    

}