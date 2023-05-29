#include "hasp_trigger.h"
#include <stdio.h>
#include <regex>
#include <iostream>
#include <string>

std::string parseFuncName(std::string overall_str, std::string head_str, std::string tail_str) {
    int head_idx = overall_str.find(head_str);
    int tail_idx = overall_str.find(tail_str);
    int head_len = head_str.size();
    std::string substr = overall_str.substr(head_idx + head_len, tail_idx - head_idx - head_len);
    return substr;
}

int parseNum(std::string overall_str, std::string head_str, std::string tail_str) {
    std::string substr = parseFuncName(overall_str, head_str, tail_str);
    std::regex num_reg("[0-9]+");
    bool ret = std::regex_match(substr, num_reg);
    if (ret) return std::stoi(substr);
    else return -1;
}

int parseNum(std::string overall_str, std::string head_str) {
    int head_idx = overall_str.find(head_str);
    int head_len = head_str.size();
    std::string substr = overall_str.substr(head_idx + head_len);
    int num = 0;
    for (int i = 0; i < substr.size(); i++) {
        if (substr[i] >= '0' && substr[i] <= '9') num = num * 10 + (substr[i] - '0');
        else if (i == 0) return -1;
        else break;
    }
    return num;
}

void hasp_trigger::print_table() const
{
    printf("------------------------Func----------------------------------\n");
    printf("Addr         \tCfg\tNsh\tNmem\tTid\tName\n");
    printf("--------------------------------------------------------------\n");
    for (auto &item : hasp_func_table) {
        printf("%p\t%d\t%d\t%d\t%d\t%s\n",  item.func_addr, item.cfg_state, item.shader_num,
                                            item.nmemory_partition_num, item.stream_id, item.func_name.c_str());
    }
    printf("--------------------------------------------------------------\n");
}

void hasp_trigger::print_memory_table() const
{
    printf("----------------------Memory----------------------------------\n");
    printf("StartAddr    \tEndAddr      \tStream\tSize\n");
    printf("--------------------------------------------------------------\n");
    for (auto &item : memory_map_table) {
        long long addr_num = (long long)item.addr;
        printf("%p\t%016llx\t%d\t%d\n",  item.addr, addr_num + item.size, item.stream, item.size);
    }
    printf("--------------------------------------------------------------\n");
}

void hasp_trigger::print_stream_table() const 
{
    printf("Current Table: ");
    for (auto streamIdx : (*shader_table_ptr)) printf("%d ", streamIdx);
    printf("\n");
}

int hasp_trigger::add_trigger_item(const void* func_ptr, char* func_name) const
{
    std::string new_func_name = func_name;
    hasp_func_item new_item;
    new_item.func_addr = (void*) func_ptr;
    new_item.nmemory_partition_num = -1;
    new_item.shader_num = -1;
    new_item.stream_id = -1;
    new_item.cfg_state = 0;

    // Parse Function Name
    std::string prefix = "haspSet_th";
    std::string unprefix = "haspUnset_th";

    int unprefix_idx = new_func_name.find(unprefix);
    int prefix_idx = new_func_name.find(prefix);
    
    if (unprefix_idx >= 0) {
        new_item.func_name = new_func_name;
        new_item.cfg_state = 2;
        new_item.stream_id = parseNum(new_func_name, unprefix);
        hasp_func_table.push_back(new_item);
        print_table();
        return 0;
    } else if (prefix_idx >= 0) {
        // HASP-Set Function
        std::string shHead  = "_sh";
        std::string memHead = "_mem";
        int thread_num = parseNum(new_func_name, prefix, shHead);
        int shcore_num = parseNum(new_func_name, shHead, memHead);
        int memory_num = parseNum(new_func_name, memHead);
        if (thread_num >= 0 && shcore_num >= 0 && memory_num >= 0) {
            new_item.func_name = new_func_name;
            new_item.cfg_state = 2;
            new_item.stream_id = thread_num;
            new_item.shader_num = shcore_num;
            new_item.nmemory_partition_num = memory_num;
            hasp_func_table.push_back(new_item);
            print_table();
            return 0;
        } else {
            printf("[Parsing Error] %s\n", new_func_name.c_str());
        }
    } else {
        funxname2addr_map.insert(std::pair<const char*, const void*>((const char*)func_name, func_ptr));
    }
    print_table();
    return 0;
}

int hasp_trigger::add_memory_item(const void* addr, size_t size) const {
    memory_item new_memory_item;
    new_memory_item.addr = addr;
    new_memory_item.size = size;
    new_memory_item.stream = cudaMalloc_target_stream;
    memory_map_table.push_back(new_memory_item);
    print_memory_table();
    cudaMalloc_target_stream = -1;
}

void hasp_trigger::push_back_func_stream(const void* func_ptr, int stream_id) const {

    addr2stream_item new_item;
    new_item.addr = func_ptr;
    new_item.stream_id = stream_id;
    addr2stream_table.push_back(new_item);

    for (auto item : funxname2addr_map) {
        if (func_ptr == item.second) {
            std::string func_name = item.first;
            std::string mallocprefix = "haspMalloc_th";
            int mallocprefix_idx = func_name.find(mallocprefix);
            if (mallocprefix_idx >= 0) {
                // Malloc Function: Configure Next cudaMalloc
                int memory_num = parseNum(func_name, mallocprefix);
                if (memory_num >= 0) {
                    cudaMalloc_target_stream = memory_num;
                    printf("[Mount Next cudaMalloc] To stream %d\n", cudaMalloc_target_stream);
                } else {
                    printf("[Parsing Error] %s\n", func_name.c_str());
                }
            }
            break;
        }
    }    
}

bool hasp_trigger::register_shader_table(const char* c_func_name, int shader_id) const {
    
    std::string func_name = c_func_name;
    std::string unset_prefix  = "haspUnset_th";
    bool is_target_function;

    // Judge HASP Function First
    for (auto hasp_func_item : hasp_func_table) {
        if (func_name.compare(hasp_func_item.func_name) == 0){
            // HASP-specific function
            printf("[Register Success] HASP-specific function %s\n", c_func_name);
            return true;
        }
    }
    // 从函数名找到对应的 Stream
    const void* addr = NULL;
    for (auto funxname2addr : funxname2addr_map) {
        std::string str1 = funxname2addr.first;
        if (str1.compare(func_name) == 0) addr = funxname2addr.second;
    }
    if (!addr) {
        printf("[Address Not Found] %p\n", addr);
        exit(1);
    }
    
    int cur_stream_id = -1;
    for (auto item : addr2stream_table) {
        if (item.addr == addr) {
            cur_stream_id = item.stream_id;
        }
    }
    if (cur_stream_id < 0) {
        // Undefined Function
        printf("[Register Error] ERROR undefined function\n");
        exit(1);
    }
    
    // Check the function is the function to be handled
    for (auto &hasp_func_item : hasp_func_table) {
        std::string prefix = "haspSet_th";
        int have_prefix = hasp_func_item.func_name.find(prefix);
        if (cur_stream_id == hasp_func_item.stream_id && have_prefix != -1){
            int n_cur_allocate_shader = 0;
            if ((*shader_table_ptr)[shader_id] < 0) {
                // Collect shaders for cur_stream
                for (auto & shader_item : (*shader_table_ptr)) {
                    if (shader_item == cur_stream_id) n_cur_allocate_shader += 1;
                }
                if (n_cur_allocate_shader < hasp_func_item.shader_num) {
                    // Can allocate more shader
                    (*shader_table_ptr)[shader_id] = cur_stream_id;
                    printf("[Register Success] Allocate stream %d to shader %d\n", cur_stream_id, shader_id);
                    // print_stream_table();
                    return true;
                } else {
                    return false;
                }
            } else {
                if ((*shader_table_ptr)[shader_id] == cur_stream_id) {
                    printf("[Register Success] Allocate stream %d to shader %d\n", cur_stream_id, shader_id);
                    // print_stream_table();
                    return true;
                } else {
                    return false;
                }
            }
        }
    }
    // return true;
    return false;
}

void hasp_trigger::clear_shader_table(const char * c_func_name) const {
    // Stream Validation
    std::string func_name = c_func_name;
    for (auto &item : hasp_func_table) {
        if (item.func_name.compare(func_name) == 0) {
            std::string prefix  = "haspUnset_th";
            int prefix_idx = item.func_name.find(prefix);
            if (prefix_idx >= 0) {
                printf("[release] %s: Stream %d\n", func_name.c_str(), item.stream_id);
                for (int i = 0; i < shader_table_ptr->size(); i++) {
                    if ((*shader_table_ptr)[i] == item.stream_id) (*shader_table_ptr)[i] = -1;
                }
                // print_stream_table();
            }
        }
    }
}

int hasp_trigger::get_stream_from_mem_addr(long long addr) const {
    for (auto item : memory_map_table) {
        const long long starting_addr = (long long)item.addr;
        const long long ending_addr = starting_addr + item.size;
        if (addr >= starting_addr && addr <= ending_addr) return item.stream;   
    }
    return -1;
}

int hasp_trigger::get_partion_num_from_stream(int stream) const {
    for (auto item : hasp_func_table) {
        if (item.stream_id == stream && item.nmemory_partition_num > 0) 
            return item.nmemory_partition_num;
    }
    return -1;
}

hasp_trigger::~hasp_trigger(){}
