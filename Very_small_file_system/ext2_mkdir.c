#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <fcntl.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include "new_funcs.h"

unsigned char *disk = NULL;


int main(int argc, char *argv[])
{
    if(argc != 3){
        
        exit(ENOENT);
    }
    if(strcmp(argv[2],"/") == 0){
        
        return ENOENT;
    }
    if(argv[2][0] != '/'){
       
        return ENOENT;
    }
   
    disk = read_disk(argv[1]);
    struct ext2_inode *inode_table = get_inode_table();
    struct ext2_inode *current_inode = &inode_table[1];
    unsigned int curr_inode_num = 2;
    char *dir_path = argv[2];
    char *token = strtok(dir_path, "/");
    struct ext2_dir_entry *curr_dir = find_dir_entry(current_inode,token,EXT2_FT_DIR);
    while(token != NULL){
        char *next_token = strtok(NULL, "/");
        if(curr_dir == NULL){
            if(next_token == NULL){
                struct ext2_dir_entry *entry = create_dir_entry(current_inode,token,curr_inode_num,EXT2_FT_DIR);
                int c = init_inode_block(EXT2_FT_DIR);
                entry->inode = c+1;
                create_dir_entry(&inode_table[c],".",c+1,EXT2_FT_DIR);
                create_dir_entry(&inode_table[c],"..",curr_inode_num,EXT2_FT_DIR);
                get_groupdesc()->bg_used_dirs_count++;
                return 0;
            }
            else{
                return ENOENT;
            }
        }
        token = next_token;
        if(token != NULL){
        curr_inode_num = curr_dir->inode;
        current_inode = &inode_table[curr_inode_num - 1];
        curr_dir = find_dir_entry(current_inode,token,EXT2_FT_DIR);
        }
    }

    return EEXIST;
}