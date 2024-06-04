#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include <vector>
#include "CL/opencl.h"
#include "xcl2.hpp"
#include <algorithm>
#include <cstdio>
#include <random>


#define BLOCK_SIZE 16                                                                                                                                         
#define STR_SIZE 256
# define EXPAND_RATE 2// add one iteration will extend the pyramid base by 2 per each borderline

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5

/* chip parameters	*/
const static float t_chip = 0.0005;
const static float chip_height = 0.016;
const static float chip_width = 0.016;
/* ambient temperature, assuming no package at all	*/
const static float amb_temp = 80.0;


using std::default_random_engine;
using std::generate;
using std::uniform_int_distribution;
using std::vector;

static const std::string error_message =
    "Error: Result mismatch:\n"
    "i = %d CPU result = %d Device result = %d\n";

#define STRING_BUFFER_LEN 1024
#define CHUNK_SIZE 4 // Todo: 1, 2, 4, 8, 16; defualt: 4
#define REPEATS 10 //repeatedly executing benchmark, Todo: 1, 5, 10, 20, 50; defualt: 10
#define NUM_BUFFER 200  //number of buffers, Todo: 50, 100, 200, 400, 800; default: 200
#define NUM_ACCESS 1000 //number of access, to be transfered to the kernel, Todo: 250, 500, 1000, 2000, 4000; default: 1000


// Runtime constants
// Used to define the work set over which this kernel will execute.
static const size_t work_group_size = 1;  // 8 threads in the demo workgroup



void readinput(float *vect, int grid_rows, int grid_cols, char *file) {

  int i,j;
	FILE *fp;
	char str[STR_SIZE];
	float val;

	if( (fp  = fopen(file, "r" )) ==0 )
            printf( "The file was not opened\n" );


	for (i=0; i <= grid_rows-1; i++) 
	 for (j=0; j <= grid_cols-1; j++)
	 {
		if (fgets(str, STR_SIZE, fp) == NULL) printf("Error reading file\n");
		if (feof(fp))
			printf("not enough lines in file\n");
		//if ((sscanf(str, "%d%f", &index, &val) != 2) || (index != ((i-1)*(grid_cols-2)+j-1)))
		if ((sscanf(str, "%f", &val) != 1))
			printf("invalid file format\n");
		vect[i*grid_cols+j] = val;
	}

	fclose(fp);	

}


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
    cl::Kernel krnl_hotspot;
    cl::Kernel krnl_benchmark;
    cl::Context context;

    int grid_rows = 1024;
    int grid_cols = 1024;
    int size = grid_rows * grid_cols;
    float *FilesavingTemp,*FilesavingPower; //,*MatrixOut; 
    char *tfile = "./src/temp_1024";
    char *pfile = "./src/power_1024";
    int total_iterations = 60;
    int pyramid_height = 4;
    // --------------- pyramid parameters --------------- 
    int borderCols = (pyramid_height)*EXPAND_RATE/2;
    int borderRows = (pyramid_height)*EXPAND_RATE/2;
    int smallBlockCol = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    int smallBlockRow = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    int blockCols = grid_cols/smallBlockCol+((grid_cols%smallBlockCol==0)?0:1);
    int blockRows = grid_rows/smallBlockRow+((grid_rows%smallBlockRow==0)?0:1);
    float grid_height = chip_height / grid_rows;
	float grid_width = chip_width / grid_cols;

	float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
	float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
	float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
	float Rz = t_chip / (K_SI * grid_height * grid_width);

	float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
	float step = PRECISION / max_slope;
	int t, itr;

    FilesavingTemp = (float *) malloc(size*sizeof(float));
    FilesavingPower = (float *) malloc(size*sizeof(float));


    OCL_CHECK(err,cl::Buffer buffer_a(context, CL_MEM_USE_HOST_PTR, sizeof(float) * size, FilesavingTemp, &err));
    OCL_CHECK(err,cl::Buffer buffer_b(context, CL_MEM_USE_HOST_PTR, sizeof(float) * size, NULL, &err));
    OCL_CHECK(err,cl::Buffer MatrixPower(context, CL_MEM_USE_HOST_PTR, sizeof(float) * size, FilesavingPower, &err));

	// Read input data from disk
    readinput(FilesavingTemp, grid_rows, grid_cols, tfile);
    readinput(FilesavingPower, grid_rows, grid_cols, pfile);   

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
            OCL_CHECK(err, krnl_hotspot = cl::Kernel(program, "hotspot", &err));
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
    
    
    //allocating benchmark buffer
    for(unsigned int i = 0; i < NUM_BUFFER; i ++) {
        OCL_CHECK(err, cl::Buffer buffer(context, CL_MEM_ALLOC_HOST_PTR, mSize, NULL, &err));
        buffers2.push_back(buffer);
    }
    
    //set the kernel Arguments
    int narg = 0;
    OCL_CHECK(err, err = krnl_hotspot.setArg(narg++, total_iterations));
    OCL_CHECK(err, err = krnl_hotspot.setArg(narg++, MatrixPower));
    OCL_CHECK(err, err = krnl_hotspot.setArg(narg++, buffer_a));
    OCL_CHECK(err, err = krnl_hotspot.setArg(narg++, buffer_b));
    OCL_CHECK(err, err = krnl_hotspot.setArg(narg++, grid_cols));
    OCL_CHECK(err, err = krnl_hotspot.setArg(narg++, grid_rows));
    OCL_CHECK(err, err = krnl_hotspot.setArg(narg++, borderCols));
    OCL_CHECK(err, err = krnl_hotspot.setArg(narg++, borderRows));
    OCL_CHECK(err, err = krnl_hotspot.setArg(narg++, Cap));
    OCL_CHECK(err, err = krnl_hotspot.setArg(narg++, Rx));
    OCL_CHECK(err, err = krnl_hotspot.setArg(narg++, Ry));
    OCL_CHECK(err, err = krnl_hotspot.setArg(narg++, Rz));
    OCL_CHECK(err, err = krnl_hotspot.setArg(narg++, step));


    OCL_CHECK(err, err = q.enqueueNDRangeKernel(krnl_hotspot, NULL, gSize, wgSize, NULL, NULL));
    printf("Assigned buffer number for benchmark = %ld\n", buffers2.size());



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
        printf("liczba = %d\n", liczba);
        trace.push_back(mSize / tick2 / pow(2,10));
    
        nr ++;        
    }

    // Wati for command queue to comlete pending events
    OCL_CHECK(err, err = q_bench.finish());
    printf("\nKernel execution is complete.\n");



    printf("trace = [");
    for (auto t : trace) {
        printf("%f ", t);
    }
    printf("]\n");

    return 0;

// 释放内核对象
if (krnl_hotspot() != NULL) {
    clReleaseKernel(krnl_hotspot.get());
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

}
