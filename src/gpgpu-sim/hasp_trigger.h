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
    int thread_id;
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
    // TODO
};

struct mem_rt_item {
    // TODO
};

class hasp_trigger
{
private:
    // backward pointer
    class gpgpu_context *gpgpu_ctx;
    mutable std::vector<hasp_func_item> hasp_func_table;
    mutable std::vector<shader_rt_item> shader_table;
    mutable std::vector<mem_rt_item>    mem_part_table;

public:
    // C/D-Function
    hasp_trigger(gpgpu_context *ctx) {
        gpgpu_ctx = ctx;
    }
    ~hasp_trigger();
    // Methods
    void print_table() const;
    int add_hasp_item(const void* func_ptr, char* func_name) const;
};

#endif