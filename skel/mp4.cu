// MP 4 Reduction
// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];

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

__global__ void total(float * input, float * output, int len) {
  //@@ Load a segment of the input vector into shared memory
  //@@ Traverse the reduction tree
  //@@ Write the computed sum of the block to the output vector at the
  //@@ correct index
  __shared__ float section[SECTION_SIZE];
  int bid = blockIdx.x;
  int tid = threadIdx.x << 1;
  int i = bid * SECTION_SIZE + tid;
  if (i < len) {
    section[tid] = input[i];
    section[tid + 1] = input[i + 1];
  } else {
    section[tid] = 0.0f;
    section[tid + 1] = 0.0f;
  }
  __syncthreads();
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
  if (tid == 0) output[bid] = section[SECTION_SIZE - 1];
}

int main(int argc, char ** argv) {
  int ii;
  wbArg_t args;
  float *hostInput; // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numInputElements; // number of elements in the input list
  int numOutputElements; // number of elements in the output list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numInputElements);

  numOutputElements = (numInputElements - 1) / SECTION_SIZE + 1;
  hostOutput = (float*) malloc(numOutputElements * sizeof(float));

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ", numInputElements);
  wbLog(TRACE, "The number of output elements in the input is ", numOutputElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  wbCheck(cudaMalloc((void**)&deviceInput, numInputElements * sizeof(float)));
  wbCheck(cudaMalloc((void**)&deviceOutput, numOutputElements * sizeof(float)));

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  wbCheck(cudaMemcpy(deviceInput, hostInput, numInputElements * sizeof(float), cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(deviceOutput, hostOutput, numOutputElements * sizeof(float), cudaMemcpyHostToDevice));

  wbTime_stop(GPU, "Copying input memory to the GPU.");
  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(numOutputElements, 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  total <<<dimGrid, dimBlock>>> (deviceInput, deviceOutput, numInputElements);
  wbCheck(cudaPeekAtLastError());

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numOutputElements * sizeof(float), cudaMemcpyDeviceToHost));

  wbTime_stop(Copy, "Copying output memory to the CPU");

  /********************************************************************
    * Reduce output vector on the host
    * NOTE: One could also perform the reduction of the output vector
    * recursively and support any size input. For simplicity, we do not
    * require that for this lab.
    ********************************************************************/
  for (ii = 1; ii < numOutputElements; ii++) {
    hostOutput[0] += hostOutput[ii];
  }

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  wbCheck(cudaFree(deviceInput));
  wbCheck(cudaFree(deviceOutput));

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, 1);

  free(hostInput);
  free(hostOutput);

  return 0;
}
