// MP 5 Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this
#define SECTION_SIZE (BLOCK_SIZE << 1)

#define wbCheck(code) do { GPUAssert(code, __LINE__); } while (0)
inline void GPUAssert(cudaError_t code, int line) {
  if (code != cudaSuccess) {
    wbLog(ERROR, "==Assert== ", cudaGetErrorString(code), " at line ", line);
  }
}

__device__ void do_scan(float *section, int tid) {
  for (int stride = 1; stride < SECTION_SIZE; stride <<= 1) {
    int second = (tid + 2) * stride - 1;
    if (second < SECTION_SIZE) {
      section[second] += section[second - stride];
    }
    __syncthreads();
  }
  for (int stride = (BLOCK_SIZE >> 1); stride > 0; stride >>= 1) {
    int second = (tid + 3) * stride - 1;
    if (second < SECTION_SIZE) {
      section[second] += section[second - stride];
    }
    __syncthreads();
  }
}

__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from here
  __shared__ float section[SECTION_SIZE];
  __shared__ float sums_acc[SECTION_SIZE];
  int blocks = gridDim.x;
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int tid2 = BLOCK_SIZE + tid;
  int i = bid * SECTION_SIZE + tid;
  int i2 = BLOCK_SIZE + i;
  section[tid] = i < len ? input[i] : 0.0f;
  section[tid2] = i2 < len ? input[i2] : 0.0f;
  __syncthreads();
  do_scan(section, tid << 1);
  output[bid] = section[SECTION_SIZE - 1];
  __syncthreads();
  // assumes blocks less than (SECTION_SIZE - 1)
  sums_acc[tid] = sums_acc[tid2] = 0.0f; // exclusive scan
  if (tid < blocks) sums_acc[tid + 1] = output[tid];
  if (tid2 < blocks) sums_acc[tid2 + 1] = output[tid2];
  __syncthreads();
  do_scan(sums_acc, tid << 1);
  float acc = sums_acc[bid];
  if (i < len) output[i] = section[tid] + acc;
  if (i2 < len) output[i2] = section[tid2] + acc;
}

int main(int argc, char ** argv) {
  wbArg_t args;
  float *hostInput; // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float*) wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float*) malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ", numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
  wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid((numElements - 1) / SECTION_SIZE + 1, 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan <<<dimGrid, dimBlock>>> (deviceInput, deviceOutput, numElements);
  wbCheck(cudaPeekAtLastError());

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
