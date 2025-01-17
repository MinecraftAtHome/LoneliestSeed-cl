#define KERNEL_BUFFER_SIZE (0x493E0)
#define MAX_SEED_BUFFER_SIZE (0x100000)
#define __STDC_FORMAT_MACROS

#include <inttypes.h>
#include <stdbool.h>

#define CL_TARGET_OPENCL_VERSION 200

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <OpenCL/cl_platform.h>
#else

#include <CL/cl.h>
#include <CL/cl_platform.h>

#endif

#include "clutil.h"

#include <string.h>
#include <stdio.h>
#include <inttypes.h>
#include <math.h>


#define __STDC_FORMAT_MACROS 1

#include <stdlib.h>
#include <stddef.h>
#include <inttypes.h>

#ifdef BOINC
  #include "boinc_api.h"
  #include "boinc_opencl.h"
#if defined _WIN32 || defined _WIN64
  #include "boinc_win.h"
#endif
#endif

#include <time.h>
#include <chrono>
using namespace std::chrono;

#ifdef __GNUC__

#include <unistd.h>
#include <sys/time.h>

#endif

struct checkpoint_vars {
    unsigned long long offset;
    uint64_t elapsed_chkpoint;
};

uint64_t elapsed_chkpoint = 0;

int main(int argc, char **argv) {
    int old_gpu = 0;
    uint64_t block_min = 0;
    uint64_t block_max = 0;
    uint64_t checked = 0;
    int retval = 0;
    cl_platform_id platform_id = NULL;
    cl_device_id device_ids;
    cl_int err;
    cl_uint num_devices_standalone;
    num_devices_standalone = 1;
    cl_uint num_entries;
    num_entries = 1;
    size_t seedbuffer_size;
    const char* kernel_name = "loneliest.cl";
    //int radius = 6;
    int device = 0;
    //int villages = 0;
    for (int i = 1; i < argc; i += 2) {
		const char *param = argv[i];
		if (strcmp(param, "-d") == 0 || strcmp(param, "--device") == 0) {
			device = atoi(argv[i + 1]);
		} else if (strcmp(param, "-s") == 0 || strcmp(param, "--start") == 0) {
			sscanf(argv[i + 1], "%llu", &block_min);
		} else if (strcmp(param, "-e") == 0 || strcmp(param, "--end") == 0) {
			sscanf(argv[i + 1], "%llu", &block_max);
		}
        else {
			fprintf(stderr,"Unknown parameter: %s\n", param);
        }
    }
    uint64_t offsetStart = 0;
    uint64_t *out;
    //GPU Params
	int blocks = 32768;
	int threads = 32;
    //BOINC
  	#ifdef BOINC
        BOINC_OPTIONS options;
        boinc_options_defaults(options);
	    options.normal_thread_priority = true;
        boinc_init_options(&options);
        APP_INIT_DATA aid;
	    boinc_get_init_data(aid);
        retval = boinc_get_opencl_ids(argc, argv, 1, &device_ids, &platform_id);
        if (retval) {
            //Probably standalone mode
            fprintf(stderr, "Error: boinc_get_opencl_ids() failed with error %d\n", retval);
            retval = clGetPlatformIDs(num_entries, &platform_id, &num_devices_standalone);
            if (retval) {
                fprintf(stderr, "Error: clGetPlatformIDs() failed with error %d\n", retval);
                return 1;
            }
            retval = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, num_entries, &device_ids, &num_devices_standalone);
            if (retval) {
                fprintf(stderr, "Error: clGetDeviceIDs() failed with error %d\n", retval);
                return 1;
            }
        }


        FILE *checkpoint_data = boinc_fopen("checkpoint.txt", "rb");
        if(!checkpoint_data){
            fprintf(stderr, "No checkpoint to load\n");

        }
        else{
            boinc_begin_critical_section();
            struct checkpoint_vars data_store;
            fread(&data_store, sizeof(data_store), 1, checkpoint_data);
            offsetStart = data_store.offset;
            elapsed_chkpoint = data_store.elapsed_chkpoint;
            fprintf(stderr, "Checkpoint loaded, task time %d s, seed pos: %llu\n", elapsed_chkpoint, offsetStart);
            fclose(checkpoint_data);
            boinc_end_critical_section();
        }
    #endif

    int arguments[2] = {
            0,
            0
    };

    FILE *kernel_file = boinc_fopen(kernel_name, "r");
    FILE* seedsout = fopen("seeds.txt", "w+");

    if (!kernel_file) {
        fprintf(stderr,"Failed to open kernel\n");
        exit(1);
    }
    fprintf(stderr, "Beginning OpenCL init.\n");
    char *kernel_src = (char *)malloc(KERNEL_BUFFER_SIZE);
    size_t kernel_length = fread(kernel_src, 1, KERNEL_BUFFER_SIZE, kernel_file);
    fclose(kernel_file);
    //Get cl context props
    cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties) platform_id, 0};
    //Create cl context using props
    cl_context context = clCreateContext(cps, 1, &device_ids, NULL, NULL, &err);
    check(err, "clCreateContext ");
    fprintf(stderr, "Created cl context.\n");
    //Create cl command queue
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_ids, 0, &err);
    check(err, "clCreateCommandQueueWithProperties ");
    fprintf(stderr, "Created command queue.\n");

    seedbuffer_size = blocks * threads * sizeof(cl_ulong);
    // 4 MB of memory for seeds
    cl_mem seeds = clCreateBuffer(context, CL_MEM_READ_WRITE, seedbuffer_size, NULL, &err);
    check(err, "clCreateBuffer (seeds) ");
    cl_mem data = clCreateBuffer(context, CL_MEM_READ_ONLY, 10 * sizeof(int), NULL, &err);
    check(err, "clCreateBuffer (data) ");
    fprintf(stderr, "Initialized buffers for seeds and data.\n");

    //Create program from kernel
    cl_program program = clCreateProgramWithSource(
            context,
            1,
            (const char **) &kernel_src,
            &kernel_length,
            &err);

    check(err, "clCreateProgramWithSource ");
    char *opt = (char *) malloc(20 * sizeof(char));
    err = clBuildProgram(program, 1, &device_ids, NULL, NULL, NULL);
    fprintf(stderr, "Built kernel.\n");
    if (err != CL_SUCCESS) {
        size_t len;
        clGetProgramBuildInfo(program, device_ids, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);

        char *info = (char *) malloc(len);
        clGetProgramBuildInfo(program, device_ids, CL_PROGRAM_BUILD_LOG, len, info, NULL);
        printf("%s\n", info);
        free(info);
    }
    cl_kernel kernel = clCreateKernel(program, "find", &err);
    check(err, "clCreateKernel ");
    fprintf(stderr, "Created kernel from build.\n");
    check(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &data), "clSetKernelArg (0) ");
    check(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &seeds), "clSetKernelArg (1) ");
    fprintf(stderr, "Set kernel arguments\n");
    size_t work_unit_size = 1048576;
    size_t block_size = 256;
    arguments[1] = work_unit_size;
    cl_ulong offset = block_min;
    int block = 0;
    int total_seed_count = 0;
    int chkpoint_ready = 0;
    double seedrange = (block_max - block_min);
    int checkpointTemp = 0;
    cl_ulong found_seeds[MAX_SEED_BUFFER_SIZE];
    fprintf(stderr, "Beginning work...\n");
    auto start = high_resolution_clock::now();
    //Kernel loop
    for (uint64_t s = (uint64_t)block_min + offsetStart; s < (uint64_t)block_max; s++) {
        arguments[0] = s;
        //kernel<<<blocks, threads>>>(blocks * threads * s, out);

        //GPU_ASSERT(cudaPeekAtLastError());
        //GPU_ASSERT(cudaDeviceSynchronize());  

        check(clEnqueueWriteBuffer(command_queue, data, CL_TRUE, 0, 2 * sizeof(int), arguments, 0, NULL, NULL),"clEnqueueWriteBuffer ");
        check(clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &work_unit_size, &block_size, 0, NULL, NULL),"clEnqueueNDRangeKernel ");
        
        int *data_out = (int *) malloc(sizeof(int) * 2);
        check(clEnqueueReadBuffer(command_queue, data, CL_TRUE, 0, sizeof(int) * 2, data_out, 0, NULL, NULL),"clEnqueueReadBuffer (data) ");
        seedbuffer_size = sizeof(cl_ulong)* blocks * threads;
        cl_ulong *result = (cl_ulong *) malloc(seedbuffer_size);
        check(clEnqueueReadBuffer(command_queue, seeds, CL_TRUE, 0, seedbuffer_size, result, 0, NULL, NULL), "clEnqueueReadBuffer (seeds) ");

        checkpointTemp += 1;
        #ifdef BOINC
        if(checkpointTemp >= 15 || boinc_time_to_checkpoint()){
            //time_t elapsed = time(NULL) - start;
            auto checkpoint_end = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(checkpoint_end - start);
            boinc_begin_critical_section(); // Boinc should not interrupt this
            
            // Checkpointing section below
            boinc_delete_file("checkpoint.txt"); // Don't touch, same func as normal fdel
            FILE *checkpoint_data = boinc_fopen("checkpoint.txt", "wb");
            struct checkpoint_vars data_store;
            data_store.offset = s - block_min;
            data_store.elapsed_chkpoint = elapsed_chkpoint + duration.count()/1000;
            fwrite(&data_store, sizeof(data_store), 1, checkpoint_data);
            fclose(checkpoint_data);
            checkpointTemp = 0;
            boinc_end_critical_section();
            boinc_checkpoint_completed(); // Checkpointing completed
        }
        #endif
        fprintf(stderr, "Pulling seed data out to print to file...\n");
        for (unsigned long long i = 0; i < blocks * threads; i++){
            if(result[i] > 0){
			    fprintf(seedsout,"%llu\n", result[i]);
                result[i] = 0;
                //out_villages[i] = 0;
            }

		}
		fflush(seedsout);
        double frac = (double)(s+1 - block_min) / (double)(block_max - block_min);
        boinc_fraction_done(frac);
    }


    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    checked = blocks*threads*(block_max - block_min);
    fprintf(stderr, "checked = %" PRIu64 "\n", checked);
    fprintf(stderr, "time taken = %f\n", ((double)duration.count()/1000.0)+(double)elapsed_chkpoint);

	double seeds_per_second = checked / ((double)duration.count()/1000.0)+(double)elapsed_chkpoint;
	fprintf(stderr, "seeds per second: %f\n", seeds_per_second);
    boinc_finish(0);

}