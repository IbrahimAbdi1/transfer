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
    
    if(argc == 4){
        disk = read_disk(argv[1]);
        char *link_to = argv[3];
        char *link_from = argv[2];
        char *base_from = strdup(link_from);
        char *base_to = strdup(link_to);

        if((strcmp(link_to,"/") == 0) || (strcmp(link_from,"/") == 0)){
            return ENOENT;
        }
        if(link_to[0] != '/' || link_from[0] != '/'){
            return ENOENT;
        }
        struct ext2_inode *check = check_target_path(dirname(link_from));
        char *f = basename(base_from);
       
        struct ext2_dir_entry *getfile = find_dir_entry(check,f,EXT2_FT_REG_FILE);
        if(getfile == NULL){
            if(find_dir_entry(check,basename(base_from),EXT2_FT_DIR) != NULL){
                return EISDIR;
            }
            return ENOENT;
        }
        struct ext2_inode *check2 = check_target_path(dirname(link_to));
        char * c = basename(base_to);
        struct ext2_dir_entry *checkfile = find_dir_entry(check,c,EXT2_FT_REG_FILE);
        if(checkfile != NULL){
            return EEXIST;
        }
        create_dir_entry(check2,c,getfile->inode,EXT2_FT_REG_FILE);
        get_inode_table()[getfile->inode-1].i_links_count++;
        return 0;
        
    }
    if(argc == 5){
        if(strcmp(argv[2],"-s") != 0){
            return ENOENT;
        }
        disk = read_disk(argv[1]);
        char *link_to = argv[4];
        char *link_from = argv[3];
        char *base_from = strdup(link_from);
        char *base_to = strdup(link_to);
        char *path = strdup(link_from);

        if((strcmp(link_to,"/") == 0) || (strcmp(link_from,"/") == 0)){
            return ENOENT;
        }
        if(link_to[0] != '/' || link_from[0] != '/'){
            return ENOENT;
        }
        struct ext2_inode *check = check_target_path(dirname(link_from));
        char *f = basename(base_from);
      
        struct ext2_dir_entry *getfile = find_dir_entry(check,f,EXT2_FT_REG_FILE);
        if(getfile == NULL){
            if(find_dir_entry(check,basename(base_from),EXT2_FT_DIR) != NULL){
                return EISDIR;
            }
            return ENOENT;
        }
        struct ext2_inode *check2 = check_target_path(dirname(link_to));
        char * c = basename(base_to);
        struct ext2_dir_entry *checkfile = find_dir_entry(check,c,EXT2_FT_REG_FILE);
        if(checkfile != NULL){
            return EEXIST;
        }
        struct ext2_inode *inode_table = get_inode_table(); 
        int inode_num = init_inode_block(EXT2_FT_SYMLINK);
        struct ext2_inode *inode = &inode_table[inode_num];
        inode->i_size = strlen(path);
        int next_free = get_free_block();
        unsigned char *new_block = (disk + 1024*(next_free));
        memcpy(new_block, path, strlen(path));
        inode->i_block[0] = next_free;
        inode->i_blocks += 2;
        create_dir_entry(check2,c,inode_num+1,EXT2_FT_SYMLINK);
        inode->i_links_count++;
    }
    return 0;
}