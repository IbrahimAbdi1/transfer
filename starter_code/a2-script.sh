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
    for c in 1 2 4 8 16 32
    do
        for x in 1 2 3 4 5 6 7 8 9 10
        do
        ./main2.out -t 1 -b 1 -f 3 -m 5 -n ${n} -c ${c} -u 2 >> results2.txt
        done 
    done
done


Average time over 10 runs given n threads [4M pixels square image, filter = 9x9, chunk_size = n]

#gnuplot -e "set terminal pdf; set output 'datat.pdf';set xlabel '# Threads';set ylabel 'Average time over 10 runs (secounds)';set title '[4M pixel square image, filter = 9x9, chunk size = # of Threads]';plot 'result-average1.txt' with linespoints title 'sequential', 'result-average2.txt' with linespoints title 'sharded_rows', 'result-average3.txt' with linespoints title 'sharded_columns column major', 'result-average4.txt' with linespoints title 'sharded_columns row major', 'result-average5.txt' with linespoints title 'work queue'"

#gnuplot -e "set terminal pdf; set output 'datat2.pdf';set xlabel '# chunks';set ylabel 'Average time over 10 runs (secounds)';set title 'test';plot 'result2-average1.txt' with linespoints title 'Thread 1', 'result2-average2.txt' with linespoints title 'thread 2', 'result2-average3.txt' with linespoints title 'thread 4', 'result2-average4.txt' with linespoints title 'thread 8', 'result2-average6.txt' with linespoints title 'thread 16'"