#!/bin/bash

make clean ; make

for prog in test test2
do
   for category in false-sharing no-false-sharing
   do
      # clear caches for each run; comment this out if you don't have sudo access
      sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"

      # run perf
      perf stat --repeat 5  \
          -e L1-dcache-loads -e L1-dcache-load-misses \
          -- sh -c "./${prog} ${category} > /dev/null" &>> outf_${prog}
   done
done
