#!/bin/bash

make clean ; make

# pgm creator or whatevers

rm -f results.txt

for m in 2 3 4 5
do
    echo method ${m} >> results.txt
    for n in 1 2 4 8 16
    do
    ./main.out -t 1 -b 2 -f 3 -m ${m} -n ${n} -c ${n} >> results.txt
    done
done


