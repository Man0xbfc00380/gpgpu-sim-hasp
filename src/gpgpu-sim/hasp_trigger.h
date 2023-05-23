#ifndef HASP_TRIGGER_H
#define HASP_TRIGGER_H

#include <vector>
#include <string>

struct hasp_func_item {
    std::string func_name;
    // configuaration states
    int cfg_state;
    // -1: func_name == haspSet_[func]_th[]_..., waiting for [func]
    //  0: IDLE
    //  1: func_name == [func], waiting for haspSet_[func]_th[]_...
    //  2: CFG Done (Enable)
    void* func_addr;
    int thread_id; // Stream ID
    int shader_num;
    int nmemory_partition_num;

    void copyCfg(hasp_func_item others) {
        this->func_name = others.func_name;
        this->nmemory_partition_num = others.nmemory_partition_num;
        this->shader_num = others.shader_num;
        this->thread_id = others.thread_id;
    }
};

struct shader_rt_item {
    int  stream_id;
    int  current_kernel_id;
    bool busy;
    shader_rt_item() {
        stream_id = -1;
        current_kernel_id = -1;
        busy = false;
    }
};

struct mem_rt_item {
    int  stream_id;
    bool busy;
    mem_rt_item() {
        stream_id = -1;
        busy = false;
    }
};

class hasp_trigger
{
private:
    // backward pointer
    class gpgpu_context *gpgpu_ctx;
    mutable std::vector<hasp_func_item> hasp_func_table;
    mutable std::vector<shader_rt_item> *shader_table_ptr;
    mutable std::vector<mem_rt_item>    *mem_part_table_ptr;
    mutable std::vector<int>            active_stream_id;
public:
    // C/D-Function
    hasp_trigger(gpgpu_context *ctx) {
        gpgpu_ctx = ctx;
    }
    ~hasp_trigger();
    // Methods
    void print_table() const;
    void init(int n_shader, int n_mem_partition) const {
        shader_table_ptr = new std::vector<shader_rt_item>(n_shader);
        mem_part_table_ptr = new std::vector<mem_rt_item>(n_mem_partition);
    }
    int add_hasp_item(const void* func_ptr, char* func_name) const;
    std::vector<int> register_shader_table(int stream_id, int kernel_id) const;
    void clear_shader_table(const char * func_name) const;
};

#endif