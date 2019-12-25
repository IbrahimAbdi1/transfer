all: ext2_cp ext2_mkdir ext2_ln ext2_rm ext2_restore ext2_checker

ext2_cp: ext2_cp.o new_funcs.o
	gcc -Wall -o $@ ext2_cp.o new_funcs.o

ext2_mkdir: ext2_mkdir.o new_funcs.o
	gcc -Wall -o $@ ext2_mkdir.o new_funcs.o

ext2_ln: ext2_ln.o new_funcs.o
	gcc -Wall -o $@ ext2_ln.o new_funcs.o

ext2_rm: ext2_rm.o new_funcs.o
	gcc -Wall -o $@ ext2_rm.o new_funcs.o

ext2_restore: ext2_restore.o new_funcs.o
	gcc -Wall -o $@ ext2_restore.o new_funcs.o

ext2_checker: ext2_checker.o new_funcs.o
	gcc -Wall -o $@ ext2_checker.o new_funcs.o

%.o: %.c new_funcs.h
	gcc -Wall -c $<

clean:
	rm -rf *.o
	rm -rf ext2_cp ext2_mkdir ext2_ln ext2_rm ext2_restore ext2_checker