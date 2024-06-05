#define BUFFER_SIZE 4096
#define DATA_SIZE 4096

__kernel void comm_kernel(__global int4 *dst, int NUM_ACCESS) {
    int id = get_global_id(0);
    for(long i = 0; i < NUM_ACCESS; i ++) {
        dst[id] = (int)dst ^ dst[id];
    }
}

// __kernel void comm_kernel(__global int4 *dst, int NUM_ACCESS) {
//     int id = get_global_id(0);
//     int4 local_value = dst[id];
//     float4 sin_input;

//     // 进行sin操作
//     for(long i = 0; i < NUM_ACCESS; i ++) {
//         // 将int4转换为float4以便进行sin操作
//         sin_input = convert_float4(local_value);
        
//         // 计算正弦值
//         sin_input = sin(sin_input);
        
//         // 将计算结果转换回int4
//         local_value = convert_int4(sin_input);
        
//         // 更新内存中的值
//         dst[id] = local_value;
//     }
// }

// Tripcount identifiers
__constant int c_size = DATA_SIZE;

//Includes 
// Read Data from Global Memory and write into buffer_in
static void read_input(__global int *in, int * buffer_in,int size){
    __attribute__((xcl_pipeline_loop(1))) __attribute__((xcl_loop_tripcount(c_size, c_size)))
    read: for (int i = 0 ; i < size ; i++){
        buffer_in[i] =  in[i];
    }
}

// Read Input data from buffer_in and write the result into buffer_out
static void compute_add(int * buffer_in , int * buffer_out
        , int inc, int size)
{
    __attribute__((xcl_pipeline_loop(1)))
    __attribute__((xcl_loop_tripcount(c_size, c_size)))
    compute: for (int i = 0 ; i < size ; i++){
        buffer_out[i] = buffer_in[i] + inc;
    }
}

// Read result from buffer_out and write the result to Global Memory
static void write_result(__global int *out, int* buffer_out,
        int size)
{
    __attribute__((xcl_pipeline_loop(1)))
    __attribute__((xcl_loop_tripcount(c_size, c_size)))
    write: for (int i = 0 ; i < size ; i++){
        out[i] = buffer_out[i];
    }
}

/*
    Vector Addition Kernel Implementation using dataflow in sub functions 
    Arguments:
        in   (input)  --> Input Vector
        out  (output) --> Output Vector
        inc  (input)  --> Increment
        size (input)  --> Size of Vector in Integer
*/


__kernel __attribute__((reqd_work_group_size(1, 1, 1))) __attribute__((xcl_dataflow)) void adder(__global int* in,
                                                                                                 __global int* out,
                                                                                                 int inc,
                                                                                                 int size) {
    int buffer_in[BUFFER_SIZE];
    int buffer_out[BUFFER_SIZE];
    while(true){
    read_input(in, buffer_in, size);
    compute_add(buffer_in, buffer_out, inc, size);
    write_result(out, buffer_out, size);
    }
}