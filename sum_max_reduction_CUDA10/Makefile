CC=nvcc

CAPABILITY=-arch=sm_61
GCC_OPT = -O2 -std=c++11 $(CAPABILITY) -Xcompiler -Wall
OBJ = main.o kernel1.o  kernel2.o kernel3.o kernel4.o kernel5.o kernel8-10.o
DEPS = kernel6.h kernel7.h

%.o: %.cu $(DEPS)
	$(CC) -c -o $@ $< $(GCC_OPT)

all: $(OBJ)
	$(CC) $(GCC_OPT) $^ -o main.out
	
clean:
	rm -f *.o *.out
