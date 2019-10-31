CC = gcc
CFLAGS = -fopenmp -Wall -std=gnu99
EXEC = hello hello1 hello2 hello3 sum_simple exercise exercise-omp exercise-omp2 dotprod dotprod2 loops loops2 matadd matmul prefixsum-ordered dotprod-sync-wrong critical dotprod-sync-critical dotprod-sync-atomic dotprod-sync-atomic-localsums threadprivate copyinprivate nestlock
 
all: $(EXEC)
		clear
	
%.o : %.c
		gcc -c $(CFLAGS) $< -o $@

hello: hello.o
		gcc -o $@ $^ $(CFLAGS)

hello1: hello1.o
		gcc -o $@ $^ $(CFLAGS)

hello2: hello2.o
		gcc -o $@ $^ $(CFLAGS)

hello3: hello3.o
		gcc -o $@ $^ $(CFLAGS)

sum_simple: sum_simple.o
		gcc -o $@ $^ $(CFLAGS)

exercise: exercise.o
		gcc -o $@ $^ $(CFLAGS)

exercise-omp: exercise-omp.o
		gcc -o $@ $^ $(CFLAGS)

exercise-omp2: exercise-omp2.o
		gcc -o $@ $^ $(CFLAGS)

dotprod: dotprod.o
		gcc -o $@ $^ $(CFLAGS)

dotprod2: dotprod2.o
		gcc -o $@ $^ $(CFLAGS)

loops: loops.o
		gcc -o $@ $^ $(CFLAGS)

loops2: loops2.o
		gcc -o $@ $^ $(CFLAGS)

matadd: matadd.o
		gcc -o $@ $^ $(CFLAGS)

matmul: matmul.o
		gcc -o $@ $^ $(CFLAGS)

prefixsum-ordered: prefixsum-ordered.o
		gcc -o $@ $^ $(CFLAGS)

dotprod-sync-wrong: dotprod-sync-wrong.o
		gcc -o $@ $^ $(CFLAGS)

critical: critical.o
		gcc -o $@ $^ $(CFLAGS)

dotprod-sync-critical: dotprod-sync-critical.o
		gcc -o $@ $^ $(CFLAGS)

dotprod-sync-atomic: dotprod-sync-atomic.o
		gcc -o $@ $^ $(CFLAGS)

dotprod-sync-atomic-localsums: dotprod-sync-atomic-localsums.o
		gcc -o $@ $^ $(CFLAGS)

threadprivate: threadprivate.o
		gcc -o $@ $^ $(CFLAGS)

copyinprivate: copyinprivate.o
		gcc -o $@ $^ $(CFLAGS)

nestlock: nestlock.o	
		gcc -o $@ $^ $(CFLAGS)

clean:
		rm -f *.o $(EXEC)
