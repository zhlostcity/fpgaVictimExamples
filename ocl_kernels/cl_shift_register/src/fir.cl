#define N_COEFF 11

#define SIGNAL_SIZE (1024*1024)

// Tripcount identifiers
__constant int c_n = N_COEFF;
__constant int c_size = SIGNAL_SIZE;


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


// A naive implementation of the Finite Impulse Response filter.
__kernel __attribute__ ((reqd_work_group_size(1, 1, 1)))
void fir_naive(__global int* restrict output_r,
               __global int* restrict signal_r,
               __global int* restrict coeff,
               int signal_length) {

    int coeff_reg[N_COEFF];
    __attribute__((xcl_loop_tripcount(c_n, c_n)))
    read_coef: for (int i = 0 ; i < N_COEFF ; i++) coeff_reg[i] = coeff[i];

    __attribute__((xcl_loop_tripcount(c_size, c_size)))
    outer_loop:
    for (int j = 0; j < signal_length; j++) {
        int acc = 0;
        shift_loop:
        __attribute__((xcl_pipeline_loop(1)))
        for (int i = min(j,N_COEFF-1); i >= 0; i--) {
            acc += signal_r[j-i] * coeff_reg[i];
        }
        output_r[j] = acc;
    }
}

// FIR using shift register
__kernel __attribute__ ((reqd_work_group_size(1, 1, 1)))
void fir_shift_register(__global int* restrict output_r,
                        __global int* restrict signal_r,
                        __global int* restrict coeff,
                        int signal_length) {
    int coeff_reg[N_COEFF];

    // Partitioning of this array is required because the shift register
    // operation will need access to each of the values of the array in
    // the same clock. Without partitioning the operation will need to
    // be performed over multiple cycles because of the limited memory
    // ports available to the array.
    int shift_reg[N_COEFF] __attribute__((xcl_array_partition(complete, 0)));
    while(true){
    __attribute__((xcl_loop_tripcount(c_n, c_n)))
    init_loop:
    for (int i = 0; i < N_COEFF; i++) {
        shift_reg[i] = 0;
        coeff_reg[i] = coeff[i];
    }

    outer_loop:
    __attribute__((xcl_pipeline_loop(1)))
    __attribute__((xcl_loop_tripcount(c_size, c_size)))
    for(int j = 0; j < signal_length; j++) {
        int acc = 0;
        int x = signal_r[j];

        // This is the shift register operation. The N_COEFF variable is defined
        // at compile time so the compiler knows the number of operations
        // performed by the loop. This loop does not require the unroll
        // attribute because the outer loop will be pipelined so
        // the compiler will unroll this loop in the process.
        __attribute__((xcl_loop_tripcount(c_n, c_n)))
        shift_loop:
        for (int i = N_COEFF-1; i >= 0; i--) {
            if (i == 0) {
                acc += x * coeff_reg[0];
                shift_reg[0] = x;
            } else {
                shift_reg[i] = shift_reg[i-1];
                acc += shift_reg[i] * coeff_reg[i];
            }
        }
        output_r[j] = acc;
    }
    }
}
