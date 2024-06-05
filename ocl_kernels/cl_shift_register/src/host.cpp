#include "xcl2.hpp"
#include <algorithm>
#include <random>
#include <string>
#include <vector>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include "CL/opencl.h"


#define SIGNAL_SIZE (1024 * 1024)
#define SIGNAL_SIZE_IN_EMU 1024

using std::default_random_engine;
using std::inner_product;
using std::string;
using std::uniform_int_distribution;
using std::vector;

//helping functions
void fir_sw(vector<int, aligned_allocator<int>> &output,
            const vector<int, aligned_allocator<int>> &signal,
            const vector<int, aligned_allocator<int>> &coeff);

void verify(const vector<int, aligned_allocator<int>> &gold,
            const vector<int, aligned_allocator<int>> &out);
uint64_t get_duration_ns(const cl::Event &event);
void print_summary(std::string k1, std::string k2, uint64_t t1, uint64_t t2, int iterations);
int gen_random();

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


int main(int argc, char **argv) {

    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];

    cl::NDRange wgSize(work_group_size, 1, 1); 
    cl::NDRange gSize(work_group_size, 1, 1); 

    unsigned long mSize = CHUNK_SIZE * (int)pow(2, 20);
    //allocating buffers
    std::vector<cl::Buffer> buffers2;
    std::vector<double> trace;
 
    int signal_size = xcl::is_emulation() ? SIGNAL_SIZE_IN_EMU : SIGNAL_SIZE;
    vector<int, aligned_allocator<int>> signal(signal_size);
    vector<int, aligned_allocator<int>> out(signal_size);
    vector<int, aligned_allocator<int>> coeff = {
        {53, 0, -91, 0, 313, 500, 313, 0, -91, 0, 53}};
    vector<int, aligned_allocator<int>> gold(signal_size, 0);
    generate(begin(signal), end(signal), gen_random);

    fir_sw(gold, signal, coeff);

    size_t size_in_bytes = signal_size * sizeof(int);
    size_t coeff_size_in_bytes = coeff.size() * sizeof(int);
    cl_int err;
    cl::CommandQueue q,q_bench;
    cl::Context context;
    cl::Program program;

    // Initialize OpenCL context and load xclbin binary
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

    //Allocate Buffer in Global Memory
    OCL_CHECK(err,cl::Buffer buffer_signal(context,CL_MEM_USE_HOST_PTR,size_in_bytes,signal.data(),&err));
    OCL_CHECK(err,cl::Buffer buffer_coeff(context,CL_MEM_USE_HOST_PTR,coeff_size_in_bytes,coeff.data(),&err));
    OCL_CHECK(err,cl::Buffer buffer_output(context,CL_MEM_USE_HOST_PTR,size_in_bytes,out.data(),&err));

    //allocating benchmark buffer
    for(unsigned int i = 0; i < NUM_BUFFER; i ++) {
        OCL_CHECK(err, cl::Buffer buffer(context, CL_MEM_ALLOC_HOST_PTR, mSize, NULL, &err));
        buffers2.push_back(buffer);
    }

    //Creating FIR Shift Register Kernel object and setting args
    OCL_CHECK(err,cl::Kernel fir_sr_kernel(program, "fir_shift_register", &err));
    OCL_CHECK(err,cl::Kernel kernel_bench(program, "comm_kernel", &err));
    OCL_CHECK(err, err = fir_sr_kernel.setArg(0, buffer_output));
    OCL_CHECK(err, err = fir_sr_kernel.setArg(1, buffer_signal));
    OCL_CHECK(err, err = fir_sr_kernel.setArg(2, buffer_coeff));
    OCL_CHECK(err, err = fir_sr_kernel.setArg(3, signal_size));
    OCL_CHECK(err, err = q.enqueueNDRangeKernel(fir_sr_kernel, NULL, gSize, wgSize, NULL, NULL));
    printf("Assigned buffer number for benchmark = %ld\n", buffers2.size());

    int nr = 0;
    int memStart = 0;
    for(int i = 0; i < NUM_BUFFER; i++) {
        OCL_CHECK(err, err = kernel_bench.setArg(0, buffers2[i]));
        int num_access = NUM_ACCESS;
        OCL_CHECK(err, err = kernel_bench.setArg(1, num_access));
        cl_int liczba ;
        printf("Chunk: %d\n", nr);
        printf("(%d - %d) MB\n", memStart, memStart+CHUNK_SIZE);
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
if (fir_sr_kernel() != NULL) {
    clReleaseKernel(fir_sr_kernel.get());
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

}



// Finite Impulse Response Filter
void fir_sw(vector<int, aligned_allocator<int>> &output,
            const vector<int, aligned_allocator<int>> &signal,
            const vector<int, aligned_allocator<int>> &coeff) {
    auto out_iter = begin(output);
    auto rsignal_iter = signal.rend() - 1;

    int i = 0;
    while (rsignal_iter != signal.rbegin() - 1) {
        int elements = std::min((int)coeff.size(), i++);
        *(out_iter++) = inner_product(
            begin(coeff), begin(coeff) + elements, rsignal_iter--, 0);
    }
}

int gen_random() {
    static default_random_engine e;
    static uniform_int_distribution<int> dist(0, 100);

    return dist(e);
}

// Verifies the gold and the out data are equal
void verify(const vector<int, aligned_allocator<int>> &gold,
            const vector<int, aligned_allocator<int>> &out) {
    bool match = equal(begin(gold), end(gold), begin(out));
    if (!match) {
        printf("TEST FAILED\n");
        exit(EXIT_FAILURE);
    }
}

uint64_t get_duration_ns(const cl::Event &event) {
    uint64_t nstimestart, nstimeend;
    cl_int err;
    OCL_CHECK(err,
              err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START,
                                                     &nstimestart));
    OCL_CHECK(err,
              err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END,
                                                     &nstimeend));
    return (nstimeend - nstimestart);
}
void print_summary(
    std::string k1, std::string k2, uint64_t t1, uint64_t t2, int iterations) {
    double speedup = (double)t1 / (double)t2;
    printf("|-------------------------+-------------------------|\n"
           "| Kernel(%3d iterations)  |    Wall-Clock Time (ns) |\n"
           "|-------------------------+-------------------------|\n",
           iterations);
    printf("| %-23s | %23lu |\n", k1.c_str(), t1);
    printf("| %-23s | %23lu |\n", k2.c_str(), t2);
    printf("|-------------------------+-------------------------|\n");
    printf("| Speedup: | %23lf |\n", speedup);
    printf("|-------------------------+-------------------------|\n");
    printf("Note: Wall Clock Time is meaningful for real hardware execution "
           "only, not for emulation.\n");
    printf("Please refer to profile summary for kernel execution time for "
           "hardware emulation.\n");

    //Performance check for real hardware. t2 must be less than t1.
    if (!xcl::is_emulation() && (t1 < t2)) {
        printf("ERROR: Unexpected Performance is observed\n");
        exit(EXIT_FAILURE);
    }
}
