#include "ext2.h"

int empty_iblock_index(struct ext2_inode *inode);
unsigned char *read_disk(char *disk_name);
unsigned short convert_type(unsigned char f);
int round_to_four(int x);
struct ext2_super_block *get_superblock();
struct ext2_group_desc *get_groupdesc();
struct ext2_inode *get_inode_table();
unsigned char *get_inode_bitmap();
unsigned char *get_block_bitmap();
void set_block_inuse(unsigned char *map, int index);
void set_block_unuse(unsigned char *map, int index);
int check_use(unsigned char *map, int index);
int next_free_block(unsigned char *map,int size);
int get_free_block();
struct ext2_dir_entry *find_dir_entry(struct ext2_inode *inode,char* dir_name, unsigned char file_tp);
struct ext2_dir_entry *attach_dir_entry(int block_num,char* name,int inode_num,unsigned char file_tp);
struct ext2_dir_entry *create_dir_entry(struct ext2_inode *inode,char* name,int inode_num,unsigned char file_tp);
int init_inode_block(unsigned char fd_type);
int check_path(char * path);
struct ext2_inode *get_dir_inode_from_path(char *path);
int copy_file_to_disk( struct ext2_inode *inode,FILE *fp);
struct ext2_inode *check_target_path(char *path);
int attempt_remove(struct ext2_inode *inode,char* dir_name, unsigned char file_tp);
struct ext2_dir_entry *find_dir_entry2(struct ext2_inode *inode,char* dir_name);
void remove_contents(int index);
int check_padding(struct ext2_dir_entry *dir_entry,int padd,char* dir_name,unsigned char file_tp);
int attempt_restore(struct ext2_inode *inode,char* dir_name, unsigned char file_tp);
int check_inode(struct ext2_inode *inode,int inode_num);
void set_back(struct ext2_inode *inode,int inode_num);
int try_restore(struct ext2_dir_entry *dir_entry,int padd,char* dir_name);
int restore(struct ext2_inode *inode,char* dir_name);