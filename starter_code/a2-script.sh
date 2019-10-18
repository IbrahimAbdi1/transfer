#!/bin/bash

make clean ; make

# 
gcc -O2 -Wall -Werror main2.c pgm.c filters.c very_big_sample.o very_tall_sample.o -o main2.out -lpthread
rm -f results.txt
rm -f results2.txt

#threads experiment
for m in 1 2 3 4 5
do
    echo method ${m} >> results.txt
    for n in 1 2 4 8 16
    do
        for x in 1 2 3 4 5 6 7 8 9 10
        do
        ./main2.out -t 1 -b 1 -f 3 -m ${m} -n ${n} -c ${n} -u 1 >> results.txt
        done 
    done
done

#chunks experiment
for n in 1 2 4 8 16
do
    echo thread ${n} >> results2.txt
    for c 1 2 4 8 32
        for x in 1 2 3 4 5 6 7 8 9 10
        do
        ./main2.out -t 1 -b 1 -f 3 -m 5 -n ${n} -c ${c} -u 2 >> results2.txt
        done 
    done
done