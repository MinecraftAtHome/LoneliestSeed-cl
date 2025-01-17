
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
		} else if (strcmp(param, "-r") == 0 || strcmp(param, "--radius") == 0){
            radius = atoi(argv[i+1]);
        } //else if (strcmp(param, "-v") == 0 || strcmp(param, "--villages") == 0){
            //villages = atoi(argv[i+1]);
        //}
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
        if (aid.gpu_device_num >= 0) {
		    device = aid.gpu_device_num;
		    fprintf(stderr,"boinc gpu %i gpuindex: %i \n", aid.gpu_device_num, device);
		} else {
            device = -5;
            for (int i = 1; i < argc; i += 2) {
              	if(strcmp(argv[i], "--device") == 0){
                    sscanf(argv[i + 1], "%i", &device);
                }
  
            }
            if(device == -5){
                fprintf(stderr, "Error: No --device parameter provided! Defaulting to device 0...\n");
                device = 0;
            }
		    fprintf(stderr,"stndalone gpuindex %i (aid value: %i)\n", device, aid.gpu_device_num);
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
    //Kernel loop
    int retval = 0;
    cl_platform_id platform_id = NULL;
    cl_device_id device_ids;
    cl_int err;
    cl_uint num_devices_standalone;
    num_devices_standalone = 1;
    cl_uint num_entries;
    num_entries = 1;
    const char* kernel_name = "loneliest.cl";
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

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    checked = blocks*threads*(block_max - block_min);
    fprintf(stderr, "checked = %" PRIu64 "\n", checked);
    fprintf(stderr, "time taken = %f\n", ((double)duration.count()/1000.0)+(double)elapsed_chkpoint);

	double seeds_per_second = checked / ((double)duration.count()/1000.0)+(double)elapsed_chkpoint;
	fprintf(stderr, "seeds per second: %f\n", seeds_per_second);
    boinc_finish(0);

}