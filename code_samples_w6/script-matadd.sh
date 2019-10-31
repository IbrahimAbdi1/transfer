#!/bin/bash

# change to -O3 to allow compiler optimizations 
gcc -fopenmp matadd.c -o matadd -Wall -std=c99 -O0

outputfile=results.txt


for mode in outer outer-sw inner inner-sw nested collapse \
            outer-sta outer-dyn outer-gui collapse-sta collapse-dyn collapse-gui
do
	echo "===== ${mode} =====" &>> $outputfile

	perf stat --repeat 5  \
          -e L1-dcache-loads -e L1-dcache-load-misses \
          -e L1-dcache-stores -e L1-dcache-store-misses \
          -- sh -c "./matadd 5000 ${mode}" &>> $outputfile
	#       -e cache-references -e cache-misses \
done
