
#include <CL/cl.h>
#include <wb.h> //@@ wb include opencl.h for you

#define wbCheck(code) do { GPUAssert(code, __LINE__); } while (0)
inline void GPUAssert (cl_int code, int line) {
  if (code != CL_SUCCESS) {
    wbLog(ERROR, "==Assert== OpenCL error ", code, " at line ", line);
  }
}

constexpr size_t GROUP_SIZE = 256;

//@@ OpenCL Kernel
const char* vecAdd = "                                          \n\
__kernel void vecAdd (__global float* in1, __global float* in2, \n\
    __global float* out, int len) {                             \n\
  int i = get_global_id(0);                                     \n\
  if (i < len) out[i] = in1[i] + in2[i];                        \n\
}                                                               \n\
";

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  cl_mem deviceInput1;
  cl_mem deviceInput2;
  cl_mem deviceOutput;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 = ( float * )wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 = ( float * )wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = ( float * )malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);

  cl_int err = CL_SUCCESS;

  // query the number of platforms
  cl_uint numPlatforms;
  wbCheck(clGetPlatformIDs(0, NULL, &numPlatforms));
  // get all the platform IDs
  cl_platform_id* platforms = (cl_platform_id*) malloc(numPlatforms * sizeof(cl_platform_id));
  wbCheck(clGetPlatformIDs(numPlatforms, platforms, &numPlatforms));
  // set platform property - just pick the first one
  cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platforms[0], 0 };
  // now create the context
  cl_context ctx = clCreateContextFromType(properties, CL_DEVICE_TYPE_ALL, NULL, NULL, &err); wbCheck(err);
  // query the number of devices
  size_t numDevices;
  wbCheck(clGetContextInfo(ctx, CL_CONTEXT_DEVICES, 0, NULL, &numDevices));
  // get all the devices IDs
  cl_device_id* devices = (cl_device_id*) malloc(numDevices * sizeof(cl_device_id));
  wbCheck(clGetContextInfo(ctx, CL_CONTEXT_DEVICES, numDevices, devices, NULL));
  // create command queue - just using the first device
  cl_command_queue queue = clCreateCommandQueue(ctx, devices[0], 0, &err); wbCheck(err);
  // compile kernel
  cl_program prog = clCreateProgramWithSource(ctx, 1, &vecAdd, NULL, &err); wbCheck(err);
  wbCheck(clBuildProgram(prog, 0, NULL, "-cl-mad-enable", NULL, NULL));
  cl_kernel kernel = clCreateKernel(prog, "vecAdd", &err); wbCheck(err);
  free(devices); free(platforms);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  deviceInput1 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inputLength * sizeof(float), hostInput1, &err); wbCheck(err);
  deviceInput2 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inputLength * sizeof(float), hostInput2, &err); wbCheck(err);
  deviceOutput = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, inputLength * sizeof(float), NULL, &err); wbCheck(err);

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here (already did)

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  const size_t itemCount = ((inputLength - 1) / GROUP_SIZE + 1) * GROUP_SIZE;

  wbTime_start(Compute, "Performing GPU computation");
  //@@ Launch the GPU Kernel here
  wbCheck(clSetKernelArg(kernel, 0, sizeof(cl_mem), &deviceInput1));
  wbCheck(clSetKernelArg(kernel, 1, sizeof(cl_mem), &deviceInput2));
  wbCheck(clSetKernelArg(kernel, 2, sizeof(cl_mem), &deviceOutput));
  wbCheck(clSetKernelArg(kernel, 3, sizeof(int), &inputLength));
  cl_event event = NULL;
  wbCheck(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &itemCount, &GROUP_SIZE, 0, NULL, &event));
  wbCheck(clWaitForEvents(1, &event));

  wbTime_stop(Compute, "Performing GPU computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  clEnqueueReadBuffer(queue, deviceOutput, CL_TRUE, 0, inputLength * sizeof(float), hostOutput, 0, 0, NULL);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  clReleaseMemObject(deviceInput1);
  clReleaseMemObject(deviceInput2);
  clReleaseMemObject(deviceOutput);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
