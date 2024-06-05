#include "xcl2.hpp"
#include <vector>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include <CL/cl.h>

#include "CL/opencl.h"

using std::vector;

static const int DATA_SIZE = 1024;
static const std::string error_message =
    "Error: Result mismatch:\n"
    "i = %d CPU result = %d Device result = %d\n";

#define STRING_BUFFER_LEN 1024
#define CHUNK_SIZE 4 // Todo: 1, 2, 4, 8, 16; defualt: 4
#define REPEATS 10 //repeatedly executing benchmark, Todo: 1, 5, 10, 20, 50; defualt: 10
#define NUM_BUFFER 200  //number of buffers, Todo: 50, 100, 200, 400, 800; default: 200
#define NUM_ACCESS 1000 //number of access, to be transfered to the kernel, Todo: 250, 500, 1000, 2000, 4000; default: 1000

static const size_t work_group_size = 1;


// This example illustrates the very simple OpenCL example that performs
// an addition on two vectors
int main(int argc, char **argv) {

    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];
    // compute the size of array in bytes
    cl::NDRange wgSize(work_group_size, 1, 1); 
    cl::NDRange gSize(work_group_size, 1, 1); 
 

    cl_int err;
    cl::CommandQueue q=NULL,q_bench=NULL;
    cl::Kernel krnl_vector_add;
    cl::Kernel krnl_benchmark;
    cl::Context context=NULL;

    // Creates a vector of DATA_SIZE elements with an initial value of 10 and 32
    vector<int, aligned_allocator<int>> source_a(DATA_SIZE, 10);
    vector<int, aligned_allocator<int>> source_b(DATA_SIZE, 32);
    vector<int, aligned_allocator<int>> source_results(DATA_SIZE);

    // The get_xil_devices will return vector of Xilinx Devices
    auto devices = xcl::get_xil_devices();
    // read_binary_file() is a utility API which will load the binaryFile
    // and will return the pointer to file buffer.

    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    int valid_device = 0;
    for(unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        //create context
        OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
        //create command queue
        OCL_CHECK(err,q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err, q_bench = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, NULL, &err);
        if(err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "]with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "] program successful!\n";
            OCL_CHECK(err, krnl_vector_add = cl::Kernel(program, "vector_add", &err));
            OCL_CHECK(err, krnl_benchmark = cl::Kernel(program,"comm_kernel", &err));
            valid_device++;
            break;  
        }
    }
    if(valid_device == 0) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    unsigned long mSize = CHUNK_SIZE * (int)pow(2, 20);
    //allocating buffers
    std::vector<cl::Buffer> buffers2;
    std::vector<double> trace;
    
     //allocating victim buffer
    size_t vector_size_bytes_a = DATA_SIZE * 10 * sizeof(int);
    size_t vector_size_bytes_b = DATA_SIZE * 32 * sizeof(int);
    size_t vector_size_bytes_result = DATA_SIZE * sizeof(int);
    OCL_CHECK(err,cl::Buffer buffer_a(context, CL_MEM_USE_HOST_PTR, vector_size_bytes_a, source_a.data(), &err));
    OCL_CHECK(err,cl::Buffer buffer_b(context, CL_MEM_USE_HOST_PTR, vector_size_bytes_b, source_b.data(), &err));
    OCL_CHECK(err,cl::Buffer buffer_result(context, CL_MEM_USE_HOST_PTR, vector_size_bytes_result, source_results.data(), &err));
    
    //allocating benchmark buffer
    for(unsigned int i = 0; i < NUM_BUFFER; i ++) {
        OCL_CHECK(err, cl::Buffer buffer(context, CL_MEM_ALLOC_HOST_PTR, mSize, NULL, &err));
        buffers2.push_back(buffer);
    }
    
    //set the kernel Arguments
    int narg = 0;
    OCL_CHECK(err, err = krnl_vector_add.setArg(narg++, buffer_result));
    OCL_CHECK(err, err = krnl_vector_add.setArg(narg++, buffer_a));
    OCL_CHECK(err, err = krnl_vector_add.setArg(narg++, buffer_b));
    OCL_CHECK(err, err = krnl_vector_add.setArg(narg++, DATA_SIZE));

    // These commands will load the source_a and source_b vectors from the host
    // application and into the buffer_a and buffer_b cl::Buffer objects. The data
    // will be be transferred from system memory over PCIe to the FPGA on-board
    // DDR memory.
    // OCL_CHECK(err,err = q.enqueueMigrateMemObjects({buffer_a, buffer_b},0 /* 0 means from host*/));

    // //Launch the Kernel
    // OCL_CHECK(err, err = q.enqueueTask(krnl_vector_add));
    // clEnqueueNDRangeKernel(q, krnl_vector_add, 1, NULL, gSize, wgSize, 0, NULL, NULL);
    // cl::Event vectorAddEvent;
    // cl_ulong startTime, endTime;
    // cl_ulong elapsedNanoSeconds = 0; // 初始化为0

    // // 定义5分钟的等待时间
    // const cl_ulong waitTimeNanoSeconds = 500 * 1000000ULL; // 5 minutes in nanoseconds

    // // 循环执行内核直到达到5分钟的执行时间
    // while (elapsedNanoSeconds < waitTimeNanoSeconds) {
    //     // 启动内核
    OCL_CHECK(err, err = q.enqueueNDRangeKernel(krnl_vector_add, NULL, gSize, wgSize, NULL, NULL));

    //     // // 等待内核完成
    //     // OCL_CHECK(err, err = vectorAddEvent.wait());

    //     // 获取内核执行开始和结束的时间
    //     OCL_CHECK(err, err = vectorAddEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &startTime));
    //     OCL_CHECK(err, err = vectorAddEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END, &endTime));

    //     // 计算已执行的时间
    //     elapsedNanoSeconds += endTime - startTime;
        
    //     // 如果执行时间超过5分钟，则退出循环
    //     if (elapsedNanoSeconds >= waitTimeNanoSeconds) {
    //         break;
    //     }
    // }


    // The result of the previous kernel execution will need to be retrieved in
    // order to view the results. This call will write the data from the
    // buffer_result cl_mem object to the source_results vector
    // OCL_CHECK(err,err = q.enqueueMigrateMemObjects({buffer_result},CL_MIGRATE_MEM_OBJECT_HOST));
    // q.finish();

    int nr = 0;
    int memStart = 0;
    for(int i = 0; i < NUM_BUFFER; i++) {
        OCL_CHECK(err, err = krnl_benchmark.setArg(0, buffers2[i]));
        int num_access = NUM_ACCESS;
        OCL_CHECK(err, err = krnl_benchmark.setArg(1, num_access));
        cl_int liczba ;
        printf("Chunk: %d\n", nr);
        printf("(%d - %d) MB\n", memStart, memStart+CHUNK_SIZE);
        memStart += CHUNK_SIZE;
        
        //lanch benchmark kernel
        OCL_CHECK(err, err = q_bench.enqueueNDRangeKernel(krnl_benchmark, NULL, gSize, wgSize, NULL, NULL));
        
        double tick2 = 0;
        for(int i = 0; i < REPEATS; i++){
            cl::Event event;
            //run kernel
             OCL_CHECK(err, err = q_bench.enqueueNDRangeKernel(krnl_benchmark, NULL, gSize, wgSize, NULL, &event));
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
        printf("Benchmark Speed: %6.6f \n", mSize / tick2 / pow(2,10));
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
if (krnl_vector_add() != NULL) {
    clReleaseKernel(krnl_vector_add.get());
}

if (krnl_benchmark() != NULL) {
    clReleaseKernel(krnl_benchmark.get());
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

for(int i = 0; i < buffers2.size(); i++) {
    if (buffers2[i]()) {
        clReleaseMemObject(buffers2[i].get());
    }
}

if (buffer_a()) {
    clReleaseMemObject(buffer_a.get());
}

if (buffer_b()) {
    clReleaseMemObject(buffer_b.get());
}

if (buffer_result()) {
    clReleaseMemObject(buffer_result.get());
}

}
