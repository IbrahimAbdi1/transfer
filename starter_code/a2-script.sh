#!/bin/bash


# 
gcc -O2 -Wall -Werror main2.c pgm.c filters.c very_big_sample.o very_tall_sample.o -o main2.out -lpthread
rm -f results.txt results2.txt results3.txt
rm -f result-average1.txt result-average2.txt result-average3.txt result-average4.txt result-average5.txt result2-average1.txt result2-average2.txt result2-average3.txt result2-average4.txt result2-average6.txt result3-average1.txt result3-average2.txt result3-average3.txt result3-average4.txt result3-average5.txt
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

#filter experiment
for z in 1 2 3 4 5
do
    echo method ${z} >> results3.txt
    for f in 4 1 2 3
    do
        for x in 1 2 3 4 5 6 7 8 9 10
        do
        ./main2.out -t 1 -b 1 -f ${f} -m ${z} -n 8 -c 8 -u 3 >> results3.txt
        done
    done 
done

python a2-average.py

gnuplot -e "set terminal pdf; set output 'datat.pdf';set xlabel '# Threads';set ylabel 'Average time over 10 runs (secounds)';set title '[4M pixel square image, filter = 9x9, chunk size = # of Threads]';plot 'result-average1.txt' with linespoints title 'sequential', 'result-average2.txt' with linespoints title 'sharded_rows', 'result-average3.txt' with linespoints title 'sharded_columns column major', 'result-average4.txt' with linespoints title 'sharded_columns row major', 'result-average5.txt' with linespoints title 'work queue'"

gnuplot -e "set terminal pdf; set output 'datat2.pdf';set xlabel '# chunks';set ylabel 'Average time over 10 runs (secounds)';set title 'Vary chunks';plot 'result2-average1.txt' with linespoints title 'Thread 1', 'result2-average2.txt' with linespoints title 'thread 2', 'result2-average3.txt' with linespoints title 'thread 4', 'result2-average4.txt' with linespoints title 'thread 8', 'result2-average6.txt' with linespoints title 'thread 16'"

gnuplot -e "set terminal pdf; set output 'datat3.pdf';set xlabel 'filter';set ylabel 'Average time over 10 runs (secounds)';set title 'Varying filters';plot 'result3-average1.txt' with linespoints title 'sequential', 'result3-average2.txt' with linespoints title 'sharded_rows', 'result3-average3.txt' with linespoints title 'sharded_columns column major', 'result3-average4.txt' with linespoints title 'sharded_columns row major', 'result3-average5.txt' with linespoints title 'work queue'"


rm -f results.txt result-average1.txt result-average2.txt result-average3.txt result-average4.txt result-average5.txt

#Thread experiment for b 2 
for m in 1 2 3 4 5
do
    echo method ${m} >> results.txt
    for n in 1 2 4 8 16
    do
        for x in 1 2 3 4 5 6 7 8 9 10
        do
        ./main2.out -t 1 -b 2 -f 3 -m ${m} -n ${n} -c ${n} -u 1 >> results.txt
        done 
    done
done


python a2-average.py

gnuplot -e "set terminal pdf; set output 'datat4.pdf';set xlabel '# Threads';set ylabel 'Average time over 10 runs (secounds)';set title '[hardcoded big image, filter = 9x9, chunk size = # of Threads]';plot 'result-average1.txt' with linespoints title 'sequential', 'result-average2.txt' with linespoints title 'sharded_rows', 'result-average3.txt' with linespoints title 'sharded_columns column major', 'result-average4.txt' with linespoints title 'sharded_columns row major', 'result-average5.txt' with linespoints title 'work queue'"

python3 perfs_student.py