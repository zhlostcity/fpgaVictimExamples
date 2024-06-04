#include "xcl2.hpp"
#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

using std::default_random_engine;
using std::generate;
using std::uniform_int_distribution;
using std::vector;

// row major
void matmul(int *C, int *A, int *B, int M) {
    for (int k = 0; k < M; k++) {
        for (int j = 0; j < M; j++) {
            for (int i = 0; i < M; i++) {
                C[k * M + j] += A[k * M + i] * B[i * M + j];
            }
        }
    }
}

int gen_random() {
    static default_random_engine e;
    static uniform_int_distribution<int> dist(0, 10);

    return dist(e);
}

void print(int *data, int columns, int rows) {
    vector<int> out(columns * rows);
    for (int r = 0; r < 10; r++) {
        for (int c = 0; c < 10; c++) {
            printf("%4d ", data[r * columns + c]);
        }
        printf("…\n");
    }
    for (int r = 0; r < 10; r++) {
        printf("   %s ", "…");
    }
    printf("⋱\n\n");
}

void verify(vector<int, aligned_allocator<int>> &gold,
            vector<int, aligned_allocator<int>> &output) {
    for (int i = 0; i < (int)output.size(); i++) {
        if (output[i] != gold[i]) {
            printf("Mismatch %d: gold: %d device: %d\n", i, gold[i], output[i]);
            print(output.data(), 16, 16);
            exit(EXIT_FAILURE);
        }
    }
}

#define STRING_BUFFER_LEN 1024
#define CHUNK_SIZE 4 // Todo: 1, 2, 4, 8, 16; defualt: 4
#define REPEATS 10 //repeatedly executing benchmark, Todo: 1, 5, 10, 20, 50; defualt: 10
#define NUM_BUFFER 200  //number of buffers, Todo: 50, 100, 200, 400, 800; default: 200
#define NUM_ACCESS 1000 //number of access, to be transfered to the kernel, Todo: 250, 500, 1000, 2000, 4000; default: 1000

// Runtime constants
// Used to define the work set over which this kernel will execute.
static const size_t work_group_size = 1;  // 8 threads in the demo workgroup
// Defines kernel argument value, which is the workitem ID that will
// execute a printf call
static const int thread_id_to_output = 1;
// This example illustrates how to use array partitioning attributes in OpenCL
// kernels for FPGA devices using matmul.
int main(int argc, char **argv) {

    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    cl::NDRange wgSize(work_group_size, 1, 1); 
    cl::NDRange gSize(work_group_size, 1, 1); 
    unsigned long mSize = CHUNK_SIZE * (int)pow(2, 20);
    //allocating buffers
    std::vector<cl::Buffer> buffers2;
    std::vector<double> trace;

    std::string binaryFile = argv[1];
    static const int columns = 16;
    static const int rows = 16;
    cl_int err;
    cl::Program program;
    cl::CommandQueue q,q_bench;
    cl::Context context;

    vector<int, aligned_allocator<int>> A(columns * rows);
    vector<int, aligned_allocator<int>> B(columns * rows);
    vector<int, aligned_allocator<int>> gold(columns * rows, 0);
    vector<int, aligned_allocator<int>> C(columns * rows, 0);
    generate(begin(A), end(A), gen_random);
    generate(begin(B), end(B), gen_random);

    printf("A:\n");
    print(A.data(), columns, rows);
    printf("B:\n");
    print(B.data(), columns, rows);
    matmul(gold.data(), A.data(), B.data(), columns);

    printf("Gold:\n");
    print(gold.data(), columns, rows);
    auto devices = xcl::get_xil_devices();

    // read_binary_file() is a utility API which will load the binaryFile
    // and will return the pointer to file buffer.
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    int valid_device = 0;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
        OCL_CHECK(err,q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err,q_bench = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        std::cout << "Trying to program device[" << i
                  << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        program = cl::Program(context, {device}, bins, NULL, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i
                      << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            valid_device++;
            break; // we break because we found a valid device
        }
    }
    if (valid_device == 0) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    // compute the size of array in bytes
    size_t array_size_bytes = columns * rows * sizeof(int);
    OCL_CHECK(err,
              cl::Buffer buffer_a(context,
                                  CL_MEM_USE_HOST_PTR,
                                  array_size_bytes,
                                  A.data(),
                                  &err));
    OCL_CHECK(err,
              cl::Buffer buffer_b(context,
                                  CL_MEM_USE_HOST_PTR,
                                  array_size_bytes,
                                  B.data(),
                                  &err));
    OCL_CHECK(err,
              cl::Buffer buffer_c(context,
                                  CL_MEM_USE_HOST_PTR,
                                  array_size_bytes,
                                  C.data(),
                                  &err));

    //allocating benchmark buffer
    for(unsigned int i = 0; i < NUM_BUFFER; i ++) {
        OCL_CHECK(err, cl::Buffer buffer(context, CL_MEM_ALLOC_HOST_PTR, mSize, NULL, &err));
        buffers2.push_back(buffer);
    }


    OCL_CHECK(err, cl::Kernel matmul_kernel(program, "matmul", &err));
    OCL_CHECK(err, cl::Kernel kernel_bench(program, "comm_kernel", &err));
    OCL_CHECK(err, err = matmul_kernel.setArg(0, buffer_a));
    OCL_CHECK(err, err = matmul_kernel.setArg(1, buffer_b));
    OCL_CHECK(err, err = matmul_kernel.setArg(2, buffer_c));
    OCL_CHECK(err, err = matmul_kernel.setArg(3, columns));
    OCL_CHECK(err, err = q.enqueueNDRangeKernel(matmul_kernel, NULL, gSize, wgSize, NULL, NULL));
    printf("Assigned buffer number for benchmark = %ld\n", buffers2.size());

    int nr = 0;
    int memStart = 0;
    for(int i = 0; i < NUM_BUFFER; i++) {
        OCL_CHECK(err, err = kernel_bench.setArg(0, buffers2[i]));
        int num_access = NUM_ACCESS;
        OCL_CHECK(err, err = kernel_bench.setArg(1, num_access));
        cl_int liczba ;
        memStart += CHUNK_SIZE;
        
        //lanch benchmark kernel
        OCL_CHECK(err, err = q_bench.enqueueNDRangeKernel(kernel_bench, NULL, gSize, wgSize, NULL, NULL));
        
        double tick2 = 0;
        for(int i = 0; i < REPEATS; i++){
            cl::Event event;
            //run kernel
             OCL_CHECK(err, err = q_bench.enqueueNDRangeKernel(kernel_bench, NULL, gSize, wgSize, NULL, &event));
             OCL_CHECK(err, err = event.wait());
             size_t time_start, time_end;
             time_start = 0;
             time_end = 0;
             OCL_CHECK(err, err = event.getProfilingInfo<size_t>(CL_PROFILING_COMMAND_START, &time_start));
             OCL_CHECK(err, err = event.getProfilingInfo<size_t>(CL_PROFILING_COMMAND_END, &time_end));
             tick2 += time_end - time_start;
        }

        q_bench.enqueueReadBuffer(buffers2[i], CL_BLOCKING, 0, sizeof(liczba), &liczba, NULL, NULL);
        tick2 = tick2 / pow(10,6) / REPEATS;
        trace.push_back(mSize / tick2 / pow(2,10));   
        nr ++;        
    }

    // Wati for command queue to comlete pending events
    OCL_CHECK(err, err = q_bench.finish());


    for (auto t : trace) {
        printf("%f ", t);
    }
    printf("\n");

    return 0;

// 释放内核对象
if (matmul_kernel() != NULL) {
    clReleaseKernel(matmul_kernel.get());
}

if (kernel_bench() != NULL) {
    clReleaseKernel(kernel_bench.get());
}

// 释放命令队列对象
if (q() != NULL) {
    clReleaseCommandQueue(q.get());
}

if (q_bench() != NULL) {
    clReleaseCommandQueue(q_bench.get());
}

// 释放上下文对象
if (context() != NULL) {
    clReleaseContext(context.get());
}
    // OCL_CHECK(err,
    //           err = q.enqueueMigrateMemObjects({buffer_a, buffer_b},
    //                                            0 /* 0 means from host*/));

    // cl::Event event;
    // uint64_t nstimestart, nstimeend;

    // OCL_CHECK(err, err = q.enqueueTask(matmul_kernel, NULL, &event));
    // OCL_CHECK(err,
    //           err = q.enqueueMigrateMemObjects({buffer_c},
    //                                            CL_MIGRATE_MEM_OBJECT_HOST));
    // q.finish();

    // OCL_CHECK(err,
    //           err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START,
    //                                                  &nstimestart));
    // OCL_CHECK(err,
    //           err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END,
    //                                                  &nstimeend));
    // auto matmul_time = nstimeend - nstimestart;

    // verify(gold, C);
    // printf("| %-23s | %23lu |\n", "matmul: ", matmul_time);

    // OCL_CHECK(
    //     err,
    //     cl::Kernel matmul_partition_kernel(program, "matmul_partition", &err));

    // OCL_CHECK(err, err = matmul_partition_kernel.setArg(0, buffer_a));
    // OCL_CHECK(err, err = matmul_partition_kernel.setArg(1, buffer_b));
    // OCL_CHECK(err, err = matmul_partition_kernel.setArg(2, buffer_c));
    // OCL_CHECK(err, err = matmul_partition_kernel.setArg(3, columns));

    // OCL_CHECK(err, err = q.enqueueTask(matmul_partition_kernel, NULL, &event));
    // OCL_CHECK(err,
    //           err = q.enqueueMigrateMemObjects({buffer_c},
    //                                            CL_MIGRATE_MEM_OBJECT_HOST));
    // q.finish();

    // OCL_CHECK(err,
    //           err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START,
    //                                                  &nstimestart));
    // OCL_CHECK(err,
    //           err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END,
    //                                                  &nstimeend));
    // auto matmul_partition_time = nstimeend - nstimestart;

    // verify(gold, C);

    // printf("| %-23s | %23lu |\n", "matmul: partition", matmul_partition_time);

    // printf("|-------------------------+-------------------------|\n");
    // printf("Note: Wall Clock Time is meaningful for real hardware execution "
    //        "only, not for emulation.\n");
    // printf("Please refer to profile summary for kernel execution time for "
    //        "hardware emulation.\n");
    // printf("TEST PASSED\n\n");

    // return EXIT_SUCCESS;
}
