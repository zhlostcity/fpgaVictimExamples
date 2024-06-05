#include "xcl2.hpp"
#include <vector>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include "CL/opencl.h"
#include <algorithm>
#include <cstdio>
#include <random>

using std::vector;

#define STRING_BUFFER_LEN 1024
#define CHUNK_SIZE 4 // Todo: 1, 2, 4, 8, 16; defualt: 4
#define REPEATS 10 //repeatedly executing benchmark, Todo: 1, 5, 10, 20, 50; defualt: 10
#define NUM_BUFFER 200  //number of buffers, Todo: 50, 100, 200, 400, 800; default: 200
#define NUM_ACCESS 1000 //number of access, to be transfered to the kernel, Todo: 250, 500, 1000, 2000, 4000; default: 1000
#define DATA_SIZE 4096
#define INCR_VALUE 10

static const size_t work_group_size = 1;

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    auto binaryFile = argv[1];
    cl::NDRange wgSize(work_group_size, 1, 1); 
    cl::NDRange gSize(work_group_size, 1, 1); 

    //Allocate Memory in Host Memory
    cl_int err;
    cl::CommandQueue q=NULL,q_bench=NULL;
    cl::Context context=NULL;
    cl::Kernel krnl_adder;
    cl::Kernel krnl_bench;
    size_t vector_size_bytes = sizeof(int) * DATA_SIZE;
    std::vector<int, aligned_allocator<int>> source_input(DATA_SIZE);
    std::vector<int, aligned_allocator<int>> source_hw_results(DATA_SIZE);
    std::vector<int, aligned_allocator<int>> source_sw_results(DATA_SIZE);

    // Create the test data and Software Result
    for (int i = 0; i < DATA_SIZE; i++) {
        source_input[i] = i;
        source_sw_results[i] = i + INCR_VALUE;
        source_hw_results[i] = 0;
    }

    unsigned long mSize = CHUNK_SIZE * (int)pow(2, 20);
    //allocating buffers
    std::vector<cl::Buffer> buffers2;
    std::vector<double> trace;

    //OPENCL HOST CODE AREA START
    auto devices = xcl::get_xil_devices();

    // read_binary_file() is a utility API which will load the binaryFile
    // and will return the pointer to file buffer.
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    int valid_device = 0;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context({device}, NULL, NULL, NULL, &err));
        OCL_CHECK(err,q = cl::CommandQueue(context, {device}, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err,q_bench = cl::CommandQueue(context, {device}, CL_QUEUE_PROFILING_ENABLE, &err));
        std::cout << "Trying to program device[" << i
                  << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, NULL, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i
                      << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, krnl_adder = cl::Kernel(program, "adder", &err));
            OCL_CHECK(err, krnl_bench = cl::Kernel(program, "comm_kernel", &err));
            valid_device++;
            break; // we break because we found a valid device
        }
    }
    if (valid_device == 0) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    //Allocate Buffer in Global Memory
    OCL_CHECK(err,cl::Buffer buffer_input(context,CL_MEM_USE_HOST_PTR,vector_size_bytes,source_input.data(),&err));
    OCL_CHECK(err,cl::Buffer buffer_output(context,CL_MEM_USE_HOST_PTR,vector_size_bytes,source_hw_results.data(),&err));

    //allocating benchmark buffer
    for(unsigned int i = 0; i < NUM_BUFFER; i ++) {
        OCL_CHECK(err, cl::Buffer buffer(context, CL_MEM_ALLOC_HOST_PTR, mSize, NULL, &err));
        buffers2.push_back(buffer);
    }

    int inc = INCR_VALUE;
    int size = DATA_SIZE;

    //Set the Kernel Arguments
    int narg = 0;
    OCL_CHECK(err, err = krnl_adder.setArg(narg++, buffer_input));
    OCL_CHECK(err, err = krnl_adder.setArg(narg++, buffer_output));
    OCL_CHECK(err, err = krnl_adder.setArg(narg++, inc));
    OCL_CHECK(err, err = krnl_adder.setArg(narg++, size));
    OCL_CHECK(err, err = q.enqueueNDRangeKernel(krnl_adder, NULL, gSize, wgSize, NULL, NULL));
    printf("Assigned buffer number for benchmark = %ld\n", buffers2.size());

    // //Copy input data to device global memory
    // OCL_CHECK(err,err = q.enqueueMigrateMemObjects({buffer_input},0 /* 0 means from host*/));

    // //Launch the Kernel
    // OCL_CHECK(err, err = q.enqueueTask(krnl_adder));

    // //Copy Result from Device Global Memory to Host Local Memory
    // OCL_CHECK(err,err = q.enqueueMigrateMemObjects({buffer_output},CL_MIGRATE_MEM_OBJECT_HOST));
    // q.finish();

    //OPENCL HOST CODE AREA END

    int nr = 0;
    int memStart = 0;
    for(int i = 0; i < NUM_BUFFER; i++) {
        OCL_CHECK(err, err = krnl_bench.setArg(0, buffers2[i]));
        int num_access = NUM_ACCESS;
        OCL_CHECK(err, err = krnl_bench.setArg(1, num_access));
        cl_int liczba ;
        // printf("Chunk: %d\n", nr);
        // printf("(%d - %d) MB\n", memStart, memStart+CHUNK_SIZE);
        memStart += CHUNK_SIZE;
        
        //lanch benchmark kernel
        OCL_CHECK(err, err = q_bench.enqueueNDRangeKernel(krnl_bench, NULL, gSize, wgSize, NULL, NULL));
        
        double tick2 = 0;
        for(int i = 0; i < REPEATS; i++){
            cl::Event event;
            //run kernel
             OCL_CHECK(err, err = q_bench.enqueueNDRangeKernel(krnl_bench, NULL, gSize, wgSize, NULL, &event));
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
        // printf("Benchmark Speed: %6.6f \n", mSize / tick2 / pow(2,10));
        // printf("liczba = %d\n", liczba);
        trace.push_back(mSize / tick2 / pow(2,10));
    
        nr ++;        
    }

    // Wati for command queue to comlete pending events
    OCL_CHECK(err, err = q_bench.finish());
    printf("\nKernel execution is complete.\n");

    // Compare the results of the Device to the simulation
    bool match = true;
    for (int i = 0; i < DATA_SIZE; i++) {
        if (source_hw_results[i] != source_sw_results[i]) {
            std::cout << "Error: Result mismatch" << std::endl;
            std::cout << "i = " << i << " CPU result = " << source_sw_results[i]
                      << " Device result = " << source_hw_results[i]
                      << std::endl;
            match = false;
            break;
        }
    }

    printf("trace = [");
    for (auto t : trace) {
        printf("%f ", t);
    }
    printf("]\n");

    return 0;

// 释放内核对象
if (krnl_adder() != NULL) {
    clReleaseKernel(krnl_adder.get());
}

if (krnl_bench() != NULL) {
    clReleaseKernel(krnl_bench.get());
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

if (buffer_input()) {
    clReleaseMemObject(buffer_input.get());
}

if (buffer_output()) {
    clReleaseMemObject(buffer_output.get());
}


}
