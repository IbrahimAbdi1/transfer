#!/bin/bash

make clean ; make

# pgm creator or whatevers
gcc -O2 -Wall -Werror main2.c pgm.c filters.c very_big_sample.o very_tall_sample.o -o main2.out -lpthread
rm -f results.txt

for m in 2 3 4 5
do
    echo method ${m} >> results.txt
    for n in 1 2 4 8 16
    do
    ./main2.out -t 1 -b 2 -f 3 -m ${m} -n ${n} -c ${n} >> results.txt
    done
done


