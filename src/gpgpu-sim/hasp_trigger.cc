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
    printf("--------------------------------------------------------------\n");
    printf("Addr         \tCfg\tNsh\tNmem\tTid\tName\n");
    printf("--------------------------------------------------------------\n");
    for (auto &item : hasp_func_table) {
        printf("%p\t%d\t%d\t%d\t%d\t%s\n",  item.func_addr, item.cfg_state, item.shader_num,
                                            item.nmemory_partition_num, item.thread_id, item.func_name.c_str());
    }
    printf("--------------------------------------------------------------\n");
}

int hasp_trigger::add_hasp_item(const void* func_ptr, char* func_name) const
{
    std::string new_func_name = func_name;
    // printf("[hasp_trigger::add_hasp_item] %p-%s\n", func_ptr, new_func_name.c_str());
    // E.g. 1: 0x402704-_Z37haspSet_vectorSubKernel_th1_sh10_mem8v
    // E.g. 2: 0x402131-_Z15vectorAddKernelPfS_S_i
    
    hasp_func_item new_item;
    new_item.func_addr = (void*) func_ptr;
    new_item.nmemory_partition_num = -1;
    new_item.shader_num = -1;
    new_item.thread_id = -1;
    new_item.cfg_state = 0;

    // Parse Function Name
    std::string prefix = "haspSet_";
    std::string unprefix = "haspUnset_th";
    int unprefix_idx = new_func_name.find(unprefix);
    if (unprefix_idx >= 0) {
        new_item.func_name = new_func_name;
        new_item.cfg_state = 2;
        new_item.thread_id = parseNum(new_func_name, unprefix);
        hasp_func_table.push_back(new_item);
        return 0;
    }

    int prefix_idx = new_func_name.find(prefix);
    if (prefix_idx < 0) {
        // Original Function
        new_item.cfg_state = 1;
    } else {
        // HASP-Set Function
        std::string thHead  = "_th";
        std::string shHead  = "_sh";
        std::string memHead = "_mem";
        std::string pureFuncName = parseFuncName(new_func_name, prefix, thHead);
        int thread_num = parseNum(new_func_name, thHead, shHead);
        int shcore_num = parseNum(new_func_name, shHead, memHead);
        int memory_num = parseNum(new_func_name, memHead);
        if (thread_num >= 0 && shcore_num >= 0 && memory_num >= 0) {
            new_item.func_name = pureFuncName;
            new_item.thread_id = thread_num;
            new_item.shader_num = shcore_num;
            new_item.nmemory_partition_num = memory_num;
            new_item.cfg_state = -1;
        } else {
            printf("[Parsing Error] %s\n", new_func_name.c_str());
        }
    }

    /// Update the Table 
    bool find_target_func = false;
    for (auto &item : hasp_func_table) {
        if (item.cfg_state == -1 && new_item.cfg_state == 1) {
            if (new_item.func_name.find(item.func_name) >= 0) {
                // Target Function Get: Enable
                item.cfg_state = 2;
                item.func_addr = new_item.func_addr;
                find_target_func = true;
                break;
            }
        } else if (item.cfg_state == 1 && new_item.cfg_state == -1) {
            if (item.func_name.find(new_item.func_name) >= 0) {
                // Configure Function Get: Cfg!
                item.cfg_state = 2;
                item.copyCfg(new_item);
                find_target_func = true;
                break;
            }
        }
    }
    if (!find_target_func) hasp_func_table.push_back(new_item);
    print_table();
    return 0;
}

std::vector<int> hasp_trigger::register_shader_table(int stream_id, int kernel_id) const {
    std::vector<int> registered_shader_id;
    // TODO: Many Things to Do.
    return registered_shader_id;
}

void hasp_trigger::clear_shader_table(const char * c_func_name) const {
    // Stream Validation
    std::string func_name = c_func_name;
    for (auto &item : hasp_func_table) {
        if (item.func_name.compare(func_name) == 0) {
            std::string prefix  = "haspUnset_th";
            int prefix_idx = item.func_name.find(prefix);
            if (prefix_idx >= 0) {
                printf("[release] %s: Stream %d\n", func_name.c_str(), item.thread_id);
                // TODO: Update Runtime Table.
            }
        }
    }
}

hasp_trigger::~hasp_trigger(){}
