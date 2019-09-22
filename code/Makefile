all: test test2

test: test.c
		gcc -O0 -std=gnu99 -o test test.c -pthread

test2: test2.c
		gcc -O0 -std=gnu99 -o test2 test2.c -pthread

clean:
		rm -f test test2 
