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


extern unsigned char *disk;

int empty_iblock_index(struct ext2_inode *inode){
    for(int i = 0; i<12; i++){
        if(inode->i_block[i] == 0){
            return i;
        }
    }

    exit(1);
}

unsigned short convert_type(unsigned char f){
    if(f == EXT2_FT_DIR){
        return EXT2_S_IFDIR;
    }
    else if(f == EXT2_FT_REG_FILE){
        return EXT2_S_IFREG;
    }
    else if(f == EXT2_FT_SYMLINK){
        return EXT2_S_IFLNK;
    }
    else{
        return 0;
    }
}

unsigned char *read_disk(char *disk_name){
    int fd = open(disk_name, O_RDWR);
    unsigned char *disk1 = mmap(NULL, 128 * EXT2_BLOCK_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if(disk1 == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }
    return disk1;
}

int round_to_four(int x){
    int rest = x%4;
    if(rest == 0){
        return x;
    }
    return x - rest + 4;
}

struct ext2_super_block *get_superblock(){
    struct ext2_super_block *sb = (struct ext2_super_block *)(disk + 1024);
    return sb;
}

struct ext2_group_desc *get_groupdesc(){
    struct ext2_group_desc *gd = (struct ext2_group_desc *)(disk + (2*1024));
    return gd;
}

struct ext2_inode *get_inode_table(){
    struct ext2_group_desc *gd = get_groupdesc();
    struct ext2_inode *inodes = (struct ext2_inode *)(disk + (1024 * (gd->bg_inode_table)));
    return inodes;
}
unsigned char *get_inode_bitmap(){
    struct ext2_group_desc *gd = get_groupdesc();
    unsigned char *map = (unsigned char *)(disk + 1024 * gd->bg_inode_bitmap);
    return map;
}

unsigned char *get_block_bitmap(){
    struct ext2_group_desc *gd = get_groupdesc();
    unsigned char *map = (unsigned char *)(disk + 1024 * gd->bg_block_bitmap);
    return map;
}

void set_block_inuse(unsigned char *map, int index){
    int byte_num = index / 8;
    int bit_num = index % 8;
    map[byte_num] |= 1 << bit_num;
    return;

}

void set_block_unuse(unsigned char *map, int index){
    int byte_num = index / 8;
    int bit_num = index % 8;
    map[byte_num] &= (~(1 << bit_num));
    return;

}

int check_use(unsigned char *map, int index){
    int byte_num = index / 8;
    int bit_num = index % 8;
    return (map[byte_num] & (1 << bit_num));
}

int next_free_block(unsigned char *map,int size){

for (int byte = 0; byte < size; byte++) {
    for (int bit = 0; bit < 8; bit++) {
        unsigned char check = (map[byte] & (1 << bit));
        if(!(check)){
            return 8*byte+bit;
        }
    }
}
exit(ENOENT);

}

int get_free_block(){
    unsigned char *block_bitmap = get_block_bitmap();
    struct ext2_super_block *sb = get_superblock();
    int next_free = next_free_block(block_bitmap,(sb->s_blocks_count/8));
    set_block_inuse(block_bitmap,next_free);
    sb->s_free_blocks_count--;
    get_groupdesc()->bg_free_blocks_count--;
    return next_free +1;
}

struct ext2_dir_entry *find_dir_entry(struct ext2_inode *inode,char* dir_name, unsigned char file_tp){
    if (dir_name == NULL){
        return NULL;
    }
    for(int i = 0; i<12; i++){
        if(inode->i_block[i] != 0){
            int block_num = inode->i_block[i];
            struct ext2_dir_entry *entry = (struct ext2_dir_entry*)(disk + 1024*block_num);
            if((file_tp == entry->file_type) && (entry->name_len == strlen(dir_name)) && (strncmp(dir_name,entry->name,entry->name_len) == 0)){
                return entry;
            }
            int count = entry->rec_len;
            while(count<1024){
                entry = (struct ext2_dir_entry*)((char*)entry + entry->rec_len);
                //check if lengths equal before comparing 
                if((file_tp == entry->file_type) && (entry->name_len == strlen(dir_name)) && (strncmp(dir_name,entry->name,entry->name_len) == 0)){
                    return entry;
                }
                count += entry->rec_len;
            }

        }
    }
    return NULL;

}

struct ext2_dir_entry *attach_dir_entry(int block_num,char* name,int inode_num,unsigned char file_tp){
    struct ext2_dir_entry *entry = (struct ext2_dir_entry*)(disk + 1024*block_num);
    int count = entry->rec_len;
    // go to the end of the block 
    while(count<1024){
        entry = (struct ext2_dir_entry*)((char*)entry + entry->rec_len);
        count += entry->rec_len;
    }
    int actual_length = round_to_four(sizeof(struct ext2_dir_entry) + entry->name_len);
    int padding = entry->rec_len - actual_length;
    int check = sizeof(struct ext2_dir_entry) + strlen(name);
    if(check <= padding){
        // can add dir
        count = count - entry->rec_len + actual_length;
        entry->rec_len = actual_length;
        struct ext2_dir_entry *new_dir = (struct ext2_dir_entry*)(disk + 1024*block_num + count);
        strncpy(new_dir->name,name,strlen(name));
        new_dir->rec_len = 1024-count;
        new_dir->file_type = file_tp;
        new_dir->name_len = strlen(name);
        new_dir ->inode = inode_num;
        return new_dir;
    }
    return NULL;

}

struct ext2_dir_entry *create_dir_entry(struct ext2_inode *inode,char* name,int inode_num,unsigned char file_tp){
    if(file_tp == EXT2_FT_DIR){
        inode->i_links_count++;
    }
    struct ext2_dir_entry *entry = NULL;
    for(int i = 0; i<12; i++){
        if(inode->i_block[i] != 0){
            entry = attach_dir_entry(inode->i_block[i],name,inode_num,file_tp); // 9 foo disk 2
            if(entry != NULL){
                return entry;
            }
            }

        }
    
    //no space in blocks in inode so create a new block
    for(int i = 0; i<12; i++){
        if(inode->i_block[i] == 0){
            //found a block to add to iblock list
            int index = get_free_block();
            struct ext2_dir_entry *new_dir = (struct ext2_dir_entry*)(disk + 1024*(index));
            strncpy(new_dir->name,name,strlen(name));
            new_dir->rec_len = 1024;
            new_dir->file_type = file_tp;
            new_dir->name_len = strlen(name);
            new_dir ->inode = inode_num;
            inode->i_block[i] = index;
            inode->i_blocks = 2;
            return new_dir;
        }
    }

    exit(1);

}


int init_inode_block(unsigned char fd_type){
    unsigned char *inode_bitmap = get_inode_bitmap();
    struct ext2_super_block *sb = get_superblock();
    if(sb->s_free_inodes_count == 0){
        exit(1);
    }
    int next_free_inode = next_free_block(inode_bitmap,(sb->s_inodes_count/8));
    int index = next_free_inode;
    struct ext2_inode *inodes = get_inode_table();
    struct ext2_inode *inode = &inodes[index];
    memset(inode,0,sizeof(struct ext2_inode));
    set_block_inuse(inode_bitmap,next_free_inode);
    inode->i_mode |= convert_type(fd_type);
    inode->i_size = 1024;
    inode->i_blocks = 0;

    sb->s_free_inodes_count--;
    get_groupdesc()->bg_free_inodes_count--;

     
    return next_free_inode;
}



int check_path(char * path){
    if(path[0] != '/'){
        return ENOENT;
    }
    char *token = strtok(path,"/"); // doh /foo 
    struct ext2_inode *inode_table = get_inode_table(); 
    struct ext2_inode *current_inode = &inode_table[1];
    unsigned int curr_inode_num = 1;
    struct ext2_dir_entry *curr_dir = find_dir_entry(current_inode,token,EXT2_FT_DIR);
    while(token != NULL){
        char *next_token = strtok(NULL,"/");
        if(curr_dir == NULL){
            if(next_token == NULL){
                struct ext2_dir_entry *is_file = find_dir_entry(current_inode,token,EXT2_FT_REG_FILE);
                if(is_file != NULL){
                    return -1;
                }
                return ENOENT;
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
    return 0;

}

// /foo/bar bar dont exsit 
// t:foo exists next
// t:bar 
struct ext2_inode *get_dir_inode_from_path(char *path){
    char *token = strtok(path,"/");
    struct ext2_inode *inode_table = get_inode_table(); 
    struct ext2_inode *current_inode = &inode_table[1];
    unsigned int curr_inode_num = 1;
    struct ext2_dir_entry *curr_dir = find_dir_entry(current_inode,token,EXT2_FT_DIR);
    while(token != NULL){
        if(curr_dir == NULL){
            exit(ENOENT);
        }
        curr_inode_num = curr_dir->inode;
        current_inode = &inode_table[curr_inode_num - 1];
        curr_dir = find_dir_entry(current_inode,token,EXT2_FT_DIR);
        
        token = strtok(NULL,"/");
    }
    return current_inode;


}


//inode is a new Inode made
//read file memcopy to a block 
//make as much blocks as needed 
//need to do indirect block case
int copy_file_to_disk( struct ext2_inode *inode,FILE *fp){
    unsigned char buf[EXT2_BLOCK_SIZE];
    inode->i_size = 0;
    int bytes;
    int counter = 0;
    while((counter < 12) && ((bytes = fread(buf,1,1024,fp)) > 0)){
        int next_free = get_free_block();
        unsigned char *new_block = (disk + 1024*(next_free));
        memcpy(new_block, buf, bytes);
        inode->i_block[counter] = next_free;
        inode->i_size += bytes;
        inode->i_blocks += 2;
        counter++;
    }
    if(bytes > 0){ 
    int next_free = get_free_block();
    inode->i_block[counter] = next_free;
    inode->i_blocks += 2;
    unsigned int *indrect_block = (unsigned int *)(disk + 1024*(next_free));
    int index = 0;
    while(((bytes = fread(buf,1,1024,fp)) > 0)){
        next_free = get_free_block();
        indrect_block[index] = next_free;
        unsigned char *new_block = (disk + 1024*(next_free));
        memcpy(new_block, buf, bytes);
        inode->i_size += bytes;
        inode->i_blocks += 2;
        index++;
    }
    }

    return 0;

}


/// /foo/bar 
struct ext2_inode *check_target_path(char *path){
    char *token = strtok(path,"/");
    struct ext2_inode *inode_table = get_inode_table(); 
    struct ext2_inode *current_inode = &inode_table[1];
    if(token == NULL){
        return current_inode;
    }
    unsigned int curr_inode_num = 1;
    struct ext2_dir_entry *curr_dir = find_dir_entry(current_inode,token,EXT2_FT_DIR); // find foo
    while(token != NULL){
        if(curr_dir == NULL){
            exit(ENOENT);
        }
        curr_inode_num = curr_dir->inode;
        current_inode = &inode_table[curr_inode_num - 1]; 
        token = strtok(NULL,"/");
        if(token != NULL){
        curr_dir = find_dir_entry(current_inode,token,EXT2_FT_DIR);
        }
    }
    return current_inode;
}


int attempt_remove(struct ext2_inode *inode,char* dir_name, unsigned char file_tp){
    if (dir_name == NULL){
        return -1;
    }
    for(int i = 0; i<12; i++){
        if(inode->i_block[i] != 0){
            int block_num = inode->i_block[i];
            struct ext2_dir_entry *entry = (struct ext2_dir_entry*)(disk + 1024*block_num);
            if((file_tp == entry->file_type) && (entry->name_len == strlen(dir_name)) && (strncmp(dir_name,entry->name,entry->name_len) == 0)){
                int t = entry->inode;
                return t;
            }
            int prev = 0; 
            int count = entry->rec_len;
            while(count<1024){
                entry = (struct ext2_dir_entry*)((char*)entry + entry->rec_len);
                //check if lengths equal before comparing 
                if((file_tp == entry->file_type) && (entry->name_len == strlen(dir_name)) && (strncmp(dir_name,entry->name,entry->name_len) == 0)){
                    struct ext2_dir_entry *prev_entry = (struct ext2_dir_entry*)(disk + 1024*block_num +prev);
                    prev_entry->rec_len += entry->rec_len;
                    return entry->inode;
                }
                prev = count;
                count += entry->rec_len;
            }

        }
    }
    return -1;

}

struct ext2_dir_entry *find_dir_entry2(struct ext2_inode *inode,char* dir_name){
    if (dir_name == NULL){
        return NULL;
    }
    for(int i = 0; i<12; i++){
        if(inode->i_block[i] != 0){
            int block_num = inode->i_block[i];
            struct ext2_dir_entry *entry = (struct ext2_dir_entry*)(disk + 1024*block_num);
            if((entry->name_len == strlen(dir_name)) && (strncmp(dir_name,entry->name,entry->name_len) == 0)){
                return entry;
            }
            int count = entry->rec_len;
            while(count<1024){
                entry = (struct ext2_dir_entry*)((char*)entry + entry->rec_len);
                //check if lengths equal before comparing 
                if((entry->name_len == strlen(dir_name)) && (strncmp(dir_name,entry->name,entry->name_len) == 0)){
                    return entry;
                }
                count += entry->rec_len;
            }

        }
    }
    return NULL;

}

void remove_contents(int index){
    int inode_num = index -1;
    struct ext2_inode *inode_table = get_inode_table();
    struct ext2_inode *inode = &inode_table[inode_num];
    inode->i_links_count--;
    if(inode->i_links_count == 0){
        set_block_unuse(get_inode_bitmap(),inode_num);
        get_superblock()->s_free_inodes_count++;
        get_groupdesc()->bg_free_inodes_count++;
        for(int i = 0; i < 12; i ++){
            if(inode->i_block[i] != 0){
                set_block_unuse(get_block_bitmap(),inode->i_block[i]-1);
                get_superblock()->s_free_blocks_count++;
                get_groupdesc()->bg_free_blocks_count++;
            }
        }
        if(inode->i_block[12] != 0){
            unsigned int *indrect_block = (unsigned int *)(disk + 1024*(inode->i_block[12]));
            int count = 0;
            while(indrect_block[count] != 0){
                set_block_unuse(get_block_bitmap(),indrect_block[count]-1);
                get_superblock()->s_free_blocks_count++;
                get_groupdesc()->bg_free_blocks_count++;
                count++;
            }
            set_block_unuse(get_block_bitmap(),inode->i_block[12] -1);
            get_superblock()->s_free_blocks_count++;
            get_groupdesc()->bg_free_blocks_count++;
        }
    }
    return;
}

int check_padding(struct ext2_dir_entry *dir_entry,int padd,char* dir_name,unsigned char file_tp){
    int count = padd;
    struct ext2_dir_entry *entry = dir_entry;
    while(count > 0){
        int actual_length = round_to_four(sizeof(struct ext2_dir_entry) + entry->name_len);
        entry = (struct ext2_dir_entry*)((char*)entry + actual_length);
        if((file_tp == entry->file_type) && (entry->name_len == strlen(dir_name)) && (strncmp(dir_name,entry->name,entry->name_len) == 0)){
            return entry->inode;
        }
        count = count - entry->rec_len;
    }
    return -1;
}

int attempt_restore(struct ext2_inode *inode,char* dir_name, unsigned char file_tp){
    if (dir_name == NULL){
        return -1;
    }
    for(int i = 0; i<12; i++){
        if(inode->i_block[i] != 0){
            int block_num = inode->i_block[i];
            struct ext2_dir_entry *entry = (struct ext2_dir_entry*)(disk + 1024*block_num);
            int actual_length = round_to_four(sizeof(struct ext2_dir_entry) + entry->name_len);
            if(actual_length < entry->rec_len){
                int padding = entry->rec_len - actual_length;
                entry->rec_len = entry->rec_len -padding;
                struct ext2_dir_entry *res = (struct ext2_dir_entry*)(disk + 1024*block_num + entry->rec_len);
                if((file_tp == res->file_type) && (res->name_len == strlen(dir_name)) && (strncmp(dir_name,res->name,res->name_len) == 0)){
                    return res->inode;
                }
                 entry->rec_len += padding;
            } 
            int count = entry->rec_len;
            while(count<1024){
                entry = (struct ext2_dir_entry*)((char*)entry + entry->rec_len);
                int actual_length = round_to_four(sizeof(struct ext2_dir_entry) + entry->name_len);
                if(actual_length < entry->rec_len){
                    int res = check_padding(entry,entry->rec_len,dir_name,file_tp);
                    if(res != -1){
                        return res;
                    }
                }
                count += entry->rec_len;
            }

        }
    }
    return -1;

}

int check_inode(struct ext2_inode *inode,int inode_num){
    int index = inode_num-1;
    int check = check_use(get_inode_bitmap(),index);
    if(check != 0){
        return -1;
    }
    for(int i = 0; i<12; i++){
        if(inode->i_block[i] != 0){
          if(check_use(get_block_bitmap(),inode->i_block[i]-1) != 0){
              return -1;
          }
        }
    }
    if(inode->i_block[12] != 0){
        if(check_use(get_block_bitmap(),inode->i_block[12]-1) != 0){
                return -1;
        }
        unsigned int *indrect_block = (unsigned int *)(disk + 1024*(inode->i_block[12]));
        int count = 0;
        while(indrect_block[count] != 0){
            if(check_use(get_block_bitmap(),indrect_block[count]-1) != 0){
                return -1;
            }
            count++;

        }
    }
    return 0;
}

void set_back(struct ext2_inode *inode,int inode_num){
    int index = inode_num-1;
    set_block_inuse(get_inode_bitmap(),index);
    get_superblock()->s_free_inodes_count--;
    get_groupdesc()->bg_free_inodes_count--;
    inode->i_links_count++;
    inode->i_dtime = 0;
    for(int i = 0; i<12; i++){
        if(inode->i_block[i] != 0){
          set_block_inuse(get_block_bitmap(),inode->i_block[i]-1);
          get_superblock()->s_free_blocks_count--;
                get_groupdesc()->bg_free_blocks_count--;
        }
    }
    if(inode->i_block[12] != 0){
        set_block_inuse(get_block_bitmap(),inode->i_block[12]-1);
        get_superblock()->s_free_blocks_count--;
                get_groupdesc()->bg_free_blocks_count--; 
        unsigned int *indrect_block = (unsigned int *)(disk + 1024*(inode->i_block[12]));
        int count = 0;
        while(indrect_block[count] != 0){
            set_block_inuse(get_block_bitmap(),indrect_block[count]-1);
            get_superblock()->s_free_blocks_count--;
                get_groupdesc()->bg_free_blocks_count--;
            count++;
        }
    }
    return;
}


int try_restore(struct ext2_dir_entry *dir_entry,int padd,char* dir_name){
    int checker = padd;
    struct ext2_dir_entry *entry = dir_entry;
    int count = 0;
    while(checker > 0){
        int actual_length = round_to_four(sizeof(struct ext2_dir_entry) + entry->name_len);
        entry = (struct ext2_dir_entry*)((char*)entry + actual_length);
        if((entry->name_len == strlen(dir_name)) && (strncmp(dir_name,entry->name,entry->name_len) == 0)){
            dir_entry->rec_len = count + actual_length;
            entry->rec_len = padd -count -actual_length;
            return 0;

        }
        checker = checker - entry->rec_len;
        count += entry->rec_len;
    }
    return -1;
}


int restore(struct ext2_inode *inode,char* dir_name){
    for(int i = 0; i<12; i++){
        if(inode->i_block[i] != 0){
            
            int block_num = inode->i_block[i];
            struct ext2_dir_entry *entry = (struct ext2_dir_entry*)(disk + 1024*block_num);
            int actual_length = round_to_four(sizeof(struct ext2_dir_entry) + entry->name_len);
            if(actual_length < entry->rec_len){
                
                int res = try_restore(entry,entry->rec_len,dir_name);
                if(res != -1){
                    return 0;
                }
            }
            
            int count = entry->rec_len;
            while(count<1024){
                entry = (struct ext2_dir_entry*)((char*)entry + entry->rec_len);
                int actual_length = round_to_four(sizeof(struct ext2_dir_entry) + entry->name_len);
                if(actual_length < entry->rec_len){
                    int res = try_restore(entry,entry->rec_len,dir_name);
                    if(res != -1){
                        return res;
                    }
                }
                count += entry->rec_len;
            }

        

        }
    }
    return -1;
}