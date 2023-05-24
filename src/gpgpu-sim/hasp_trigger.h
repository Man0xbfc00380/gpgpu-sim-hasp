#ifndef HASP_TRIGGER_H
#define HASP_TRIGGER_H

#include <vector>
#include <string>
#include <map>

struct hasp_func_item {
    std::string func_name;
    // configuaration states
    int cfg_state;
    // -1: func_name == haspSet_th[]_..., waiting for [func]
    //  0: IDLE
    //  1: func_name == [func], waiting for haspSet_th[]_...
    //  2: CFG Done (Enable)
    void* func_addr;
    int stream_id;
    int shader_num;
    int nmemory_partition_num;

    void copyCfg(hasp_func_item others) {
        this->func_name = others.func_name;
        this->nmemory_partition_num = others.nmemory_partition_num;
        this->shader_num = others.shader_num;
        this->stream_id = others.stream_id;
    }
};

struct addr2stream_item {
    const void* addr;
    int stream_id;
};

class hasp_trigger
{
private:
    // backward pointer
    class gpgpu_context *gpgpu_ctx;
    mutable std::vector<hasp_func_item>         hasp_func_table;
    mutable std::vector<addr2stream_item>       addr2stream_table;
    mutable std::map<const char*, const void*>   funxname2addr_map;

    mutable std::vector<int>                    *shader_table_ptr;
    mutable std::vector<int>                    *mem_part_table_ptr;
public:
    // C/D-Function
    hasp_trigger(gpgpu_context *ctx) {
        gpgpu_ctx = ctx;
    }
    ~hasp_trigger();
    // Methods
    void print_table() const;
    void init(int n_shader, int n_mem_partition) const {
        shader_table_ptr = new std::vector<int>(n_shader, -1);
        mem_part_table_ptr = new std::vector<int>(n_mem_partition, -1);
    }
    int add_hasp_item(const void* func_ptr, char* func_name) const;
    bool register_shader_table(const char * c_func_name, int shader_id) const;
    void clear_shader_table(const char * func_name) const;
    void print_stream_table() const;
    void push_back_hasp_func_stream(const void* func_ptr, int stream_id) const {
        addr2stream_item new_item;
        new_item.addr = func_ptr;
        new_item.stream_id = stream_id;
        addr2stream_table.push_back(new_item);
    }
};

#endif