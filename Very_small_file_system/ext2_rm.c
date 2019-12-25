#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <fcntl.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <libgen.h>
#include <sys/mman.h>
#include "new_funcs.h"

unsigned char *disk = NULL;


int main(int argc, char *argv[])
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <image file name> <file on native OS> <path on ext2 image>\n", argv[0]);
        exit(1);
    }
    disk = read_disk(argv[1]);
    char *target_path = argv[2];
    char *base_str = strdup(target_path);
    if(strcmp(argv[2],"/") == 0){
       // printf("Error: You cannot make the root directory, as the root directory already exists.\n");
        return 1;
    }
    if(argv[2][0] != '/'){
        //printf("Error: You must specify an absolute path.\n");
        return 1;
    }
    struct ext2_inode *check = check_target_path(dirname(target_path));
    char *base_name = basename(base_str);
    int result = attempt_remove(check,base_name,EXT2_FT_REG_FILE);
    int result2 = attempt_remove(check,base_name,EXT2_FT_SYMLINK);
    int result3 = attempt_remove(check,base_name,EXT2_FT_DIR);
    if(result != -1){
        remove_contents(result);
        return 0;
    }
    if(result2 != -1){
        remove_contents(result2);
        return 0;
    }
    if(result3 != -1){
        return EISDIR;
    }
    

    return ENOENT;
}