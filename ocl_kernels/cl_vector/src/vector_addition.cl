#define BUFFER_SIZE 256
#define DATA_SIZE 1024

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


//TRIPCOUNT indentifier
__constant uint c_len = DATA_SIZE/BUFFER_SIZE;
__constant uint c_size = BUFFER_SIZE;

kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void vector_add(global int* c,
                global const int* a,
                global const int* b,
                       const int n_elements)
{
    int arrayA[BUFFER_SIZE];
    int arrayB[BUFFER_SIZE];
    while(true){
        __attribute__((xcl_loop_tripcount(c_len, c_len))) for (int i = 0 ; i < n_elements ; i += BUFFER_SIZE) {
                int size = BUFFER_SIZE;
        
        if (i + size > n_elements) size = n_elements - i;

        __attribute__((xcl_loop_tripcount(c_size, c_size))) __attribute__((xcl_pipeline_loop(1)))
        readA: for (int j = 0 ; j < size ; j++) {
                arrayA[j] = a[i+j]; }

        __attribute__((xcl_loop_tripcount(c_size, c_size))) __attribute__((xcl_pipeline_loop(1)))
        readB: for (int j = 0 ; j < size ; j++) {
                arrayB[j] = b[i+j]; }

        __attribute__((xcl_loop_tripcount(c_size, c_size))) __attribute__((xcl_pipeline_loop(1)))
        vadd_writeC: for (int j = 0 ; j < size ; j++) {
                c[i+j] = arrayA[j] + arrayB[j]; }
    }
    }
    
}
