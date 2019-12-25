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

int check_file(const char *p) {
    struct stat stater;
    stat(p, &stater);
    return S_ISREG(stater.st_mode);
}

int main(int argc, char *argv[]) {
    
    if (argc != 4) {
        
        exit(1);
    }
    if(argv[3][0] != '/'){
        return ENOENT;
    }
    
    disk = read_disk(argv[1]);
    char * file_name = argv[2];
    char * path = argv[3];
    char *base_str = strdup(path);
    
    if(!check_file(file_name)){
        return ENOENT;
    }
    struct stat st;
    stat(file_name, &st);
    int size = st.st_size;
    FILE *fp = fopen(file_name,"rb");
    if(fp == NULL){
        return ENOENT;
    }
    struct ext2_inode *dir_inode = check_target_path(dirname(path));
    char *base_name = basename(base_str);
    if(find_dir_entry2(dir_inode,base_name) != NULL){
        return EEXIST;
    }
    struct ext2_inode *inode_table = get_inode_table();
    int inode_num = init_inode_block(EXT2_FT_REG_FILE);
    struct ext2_inode *inode = &inode_table[inode_num];
    inode->i_ctime = time(NULL);
   
    if(size > (get_groupdesc()->bg_free_blocks_count*1024)){
        return ENOENT;
    }
    copy_file_to_disk(inode,fp);
    create_dir_entry(dir_inode,base_name,inode_num+1,EXT2_FT_REG_FILE);
    inode->i_links_count++;
    fclose(fp);
    return 0;
}

