#include "bitmap.h"
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
#include <CL/cl.h>
#include "cmdlineparser.h"

using std::vector;
// define internal buffer max size
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

int main(int argc, char *argv[]) {
    sda::utils::CmdLineParser parser;
    parser.addSwitch("--xclbin_file", "-x", "input binary file string", "");
    parser.addSwitch("--input_file", "-i", "input test data flie", "");
    parser.addSwitch("--compare_file", "-c", "Compare File to compare result",
                   "");
    parser.parse(argc, argv);
      // Read settings
    auto binaryFile = parser.value("xclbin_file");
    std::string bitmapFilename = parser.value("input_file");
    std::string goldenFilename = parser.value("compare_file");
    if (argc != 7) {
        parser.printHelp();
        return EXIT_FAILURE;
    }

    cl_int err;
    cl::CommandQueue q,q_bench;
    cl::Context context;
    cl::Kernel krnl_applyWatermark;
    cl::Kernel krnl_bench;

    cl::NDRange wgSize(work_group_size, 1, 1); 
    cl::NDRange gSize(work_group_size, 1, 1); 

    unsigned long mSize = CHUNK_SIZE * (int)pow(2, 20);

    std::vector<cl::Buffer> buffers2;
    std::vector<double> trace;

    //Read the input bit map file into memory
    BitmapInterface image(bitmapFilename.data());
    bool result = image.readBitmapFile();
    if (!result) {
        std::cout << "ERROR:Unable to Read Input Bitmap File "
                  << bitmapFilename.data() << std::endl;
        return EXIT_FAILURE;
    }
    auto width = image.getWidth();
    auto height = image.getHeight();

    //Allocate Memory in Host Memory
    auto image_size = image.numPixels();
    size_t image_size_bytes = image_size * sizeof(int);
    std::vector<int, aligned_allocator<int>> inputImage(image_size);
    std::vector<int, aligned_allocator<int>> outImage(image_size);

    // Copy image host buffer
    memcpy(inputImage.data(), image.bitmap(), image_size_bytes);

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
        OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
        OCL_CHECK(err,q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err,q_bench = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE,&err));
        std::cout << "Trying to program device[" << i
                  << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, NULL, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i
                      << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err,krnl_applyWatermark =cl::Kernel(program, "apply_watermark", &err));
            OCL_CHECK(err,krnl_bench =cl::Kernel(program, "comm_kernel", &err));
            valid_device++;
            break; // we break because we found a valid device
        }
    }
    if (valid_device == 0) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    OCL_CHECK(err,cl::Buffer buffer_inImage(context, CL_MEM_USE_HOST_PTR,image_size_bytes,inputImage.data(),&err));
    OCL_CHECK(err,cl::Buffer buffer_outImage(context,CL_MEM_USE_HOST_PTR,image_size_bytes,outImage.data(),&err));

    //allocating benchmark buffer
    for(unsigned int i = 0; i < NUM_BUFFER; i ++) {
        OCL_CHECK(err, cl::Buffer buffer(context, CL_MEM_ALLOC_HOST_PTR, mSize, NULL, &err));
        buffers2.push_back(buffer);
    }

    krnl_applyWatermark.setArg(0, buffer_inImage);
    krnl_applyWatermark.setArg(1, buffer_outImage);
    krnl_applyWatermark.setArg(2, width);
    krnl_applyWatermark.setArg(3, height);
    OCL_CHECK(err, err = q.enqueueNDRangeKernel(krnl_applyWatermark, NULL, gSize, wgSize, NULL, NULL));
    printf("Assigned buffer number for benchmark = %ld\n", buffers2.size());

    int nr = 0;
    int memStart = 0;
    for(int i = 0; i < NUM_BUFFER; i++) {
        OCL_CHECK(err, err = krnl_bench.setArg(0, buffers2[i]));
        int num_access = NUM_ACCESS;
        OCL_CHECK(err, err = krnl_bench.setArg(1, num_access));
        cl_int liczba ;
        printf("Chunk: %d\n", nr);
        printf("(%d - %d) MB\n", memStart, memStart+CHUNK_SIZE);
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
        printf("Benchmark Speed: %6.6f \n", mSize / tick2 / pow(2,10));
        printf("liczba = %d\n", liczba);
        trace.push_back(mSize / tick2 / pow(2,10));
    
        nr ++;        
    }

    // Wati for command queue to comlete pending events
    OCL_CHECK(err, err = q_bench.finish());
    printf("\nKernel execution is complete.\n");

//     // Compare Golden Image with Output image
//     bool match = 1;
//     // Read the golden bit map file into memory
//     BitmapInterface goldenImage(goldenFilename.data());
//     result = goldenImage.readBitmapFile();
//     if (!result) {
//         std::cerr << "ERROR:Unable to Read Golden Bitmap File "
//                 << goldenFilename.data() << std::endl;
//         return EXIT_FAILURE;
//     }
//     if (image.getHeight() != goldenImage.getHeight() ||
//         image.getWidth() != goldenImage.getWidth()) {
//         match = 0;
//     } else {
//         int *goldImgPtr = goldenImage.bitmap();
//         for (unsigned int i = 0; i < image.numPixels(); i++) {
//         if (outImage[i] != goldImgPtr[i]) {
//             match = 0;
//             printf("Pixel %d Mismatch Output %x and Expected %x \n", i, outImage[i],
//                 goldImgPtr[i]);
//             break;
//         }
//         }
//     }
// //   Write the final image to disk
//   image.writeBitmapFile(outImage.data());



    for (auto t : trace) {
        printf("%f ", t);
    }
    printf("\n");
    return 0;

// 释放内核对象
if (krnl_applyWatermark() != NULL) {
    clReleaseKernel(krnl_applyWatermark.get());
}

if (krnl_bench() != NULL){
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

if(buffer_inImage() != NULL) {
    clReleaseMemObject(buffer_inImage.get());
}
}