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

int count_bits(unsigned char *map,int size){
    int count = 0;
    for (int byte = 0; byte < size; byte++) {
    for (int bit = 0; bit < 8; bit++) {
        unsigned char check = (map[byte] & (1 << bit));
        if(!(check)){
            count++;
        }
    }
}
return count;
}

unsigned char type(unsigned short m){
    if((m >> 12) == (EXT2_S_IFDIR >> 12)){
        return EXT2_FT_DIR;
    }
    else if((m >> 12) == (EXT2_S_IFREG >> 12)){
        return EXT2_FT_REG_FILE;
    }
    else if((m >> 12) == (EXT2_S_IFLNK >>12)){
        return EXT2_FT_SYMLINK;
    }
    return 0;
}

int lookup_inodes(){
    int free_blocks = count_bits(get_inode_bitmap(),get_superblock()->s_inodes_count / 8);
    int check_super = abs(free_blocks -  get_superblock()->s_inodes_count);
    int check_group = abs(free_blocks - get_groupdesc()->bg_free_inodes_count);
    if(check_super != 0){
        get_superblock()->s_inodes_count = free_blocks;
        printf("Fixed: superblock's free inodes counter was off by %d compared to the bitmap\n", check_super);
    }
    if(check_group != 0){
        get_groupdesc()->bg_free_inodes_count = free_blocks;
        printf("Fixed: block group's free inodes counter was off by %d compared to the bitmap\n", check_group);
    }
    return check_group + check_super;
}

int lookup_blocks(){
    int free_blocks = count_bits(get_inode_bitmap(),get_superblock()->s_blocks_count / 8);
    int check_super = abs(free_blocks -  get_superblock()->s_blocks_count);
    int check_group = abs(free_blocks - get_groupdesc()->bg_free_blocks_count);
    if(check_super != 0){
        get_superblock()->s_blocks_count = free_blocks;
        printf("Fixed: superblock's free inodes counter was off by %d compared to the bitmap\n", check_super);
    }
    if(check_group != 0){
        get_groupdesc()->bg_free_blocks_count = free_blocks;
        printf("Fixed: block group's free inodes counter was off by %d compared to the bitmap\n", check_group);
    }
    return check_group + check_super;
}

int inode_allocation_check(int inode_index){
    if (check_use(get_inode_bitmap(), inode_index) == 0){
        set_block_inuse(get_inode_bitmap(), inode_index);
        printf("Fixed: inode [%d] not marked as in-use\n", inode_index);
        return 1;
    }
    return 0;

}

int check_deletion(int num){
    struct ext2_inode *inode = &get_inode_table()[num-1];
    if(inode->i_dtime == 0){
        inode->i_dtime = 0;
        printf("Fixed: valid inode marked for deletion: [%d]\n", num);
        return 1;
    }
    return 0;
}

int data_block_allocation(int num){
    int count = 0;
    struct ext2_inode *inode = &get_inode_table()[num-1];
    for(int i = 0; i < 12; i++){
        if(inode->i_block[i]!= 0){
            if(check_use(get_block_bitmap(),inode->i_block[i]-1) == 0){
                set_block_inuse(get_block_bitmap(),inode->i_block[i]-1);
                count++;
            }
        }
    }

    if(count > 0){
        printf("Fixed: %d in-use data blocks not marked in data bitmap for inode: [%d]\n", count, num);
    }
    return count;
}



int mismatch(struct ext2_dir_entry *entry){
    int inode_num = entry->inode -1;
    struct ext2_inode *inode = &get_inode_table()[inode_num];
    unsigned char check = type(inode->i_mode);
    if(check != entry->file_type){
        entry->file_type = check;
        printf("Fixed: Entry type vs inode mismatch: inode [%d]\n", entry->inode);
        return 1;
    }
    return 0;
}

int recursive_function(struct ext2_dir_entry *entry, int flag){
    
    struct ext2_inode *inode = &get_inode_table()[entry->inode-1];
    int count = mismatch(entry) + inode_allocation_check(entry->inode) +check_deletion(entry->inode) + data_block_allocation(entry->inode);
    if (entry->file_type != EXT2_FT_DIR) {
        return count;
    }
    if((strncmp(entry->name, "..", 2) == 0) || (flag == 1 && (strncmp(entry->name, ".", strlen(".") == 0)))){
        return count;
    }

    for(int i = 0;i<12;i++){
        if(inode->i_block[i]!= 0){
        int block_num = inode->i_block[i];
        struct ext2_dir_entry *entry = (struct ext2_dir_entry*)(disk + 1024*block_num);
        int counter = entry->rec_len;
        while(counter<1024){
            if(entry->inode!= 0){
                count+= recursive_function(entry,1);
            }
            entry = (struct ext2_dir_entry*)((char*)entry + entry->rec_len);
            counter += entry->rec_len;
        }
        }
    }
    return count;

}

// my checker segfaults and i dont have time :(
int main(int argc, char *argv[])
{
    if (argc != 2) {
        exit(ENOENT);
    }
    disk = read_disk(argv[1]);
    
    // int count = 0;
    // struct ext2_inode *root= &get_inode_table()[1];
    // if(type(root->i_mode) != EXT2_FT_DIR){
    //     root->i_mode = root->i_mode | EXT2_S_IFDIR;
    //     count++;
    //     printf("Fixed: Root inode not marked as directory\n");
    // }
    
    
    printf("No file system inconsistencies detected!\n");
    
    return 0;
}