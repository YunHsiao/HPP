
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <wb.h>

#define wbCheck(code) do { GPUAssert(code, __LINE__); } while (0)
inline void GPUAssert (cudaError_t code, int line) {
  if (code != cudaSuccess) {
    wbLog(ERROR, "==Assert== ", cudaGetErrorString(code), " at line ", line);
  }
}

constexpr auto BLOCK_SIZE = 256;
constexpr auto BLOCK_COUNT = 8;
constexpr auto SEGMENT_SIZE = BLOCK_SIZE * BLOCK_COUNT;
constexpr auto STREAM_COUNT = 4;
constexpr auto USE_STREAMING = 0;

__global__ void vecAdd (float * in1, float * in2, float * out, int len) {
  //@@ Insert code to implement vector addition here
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) out[i] = in1[i] + in2[i];
}

int main (int argc, char ** argv) {
  wbArg_t args;
  int inputLength;
  float* hostInput1;
  float* hostInput2;
  float* hostOutput;
  float* deviceInput1[STREAM_COUNT];
  float* deviceInput2[STREAM_COUNT];
  float* deviceOutput[STREAM_COUNT];

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *) malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  if (USE_STREAMING) {
    wbTime_start(GPU, "Allocating GPU memory.");
    cudaStream_t streams[STREAM_COUNT];
    for (int s = 0; s < STREAM_COUNT; s++) {
      cudaStreamCreate(&streams[s]);
      cudaMalloc((void**) &deviceInput1[s], SEGMENT_SIZE * sizeof(float));
      cudaMalloc((void**) &deviceInput2[s], SEGMENT_SIZE * sizeof(float));
      cudaMalloc((void**) &deviceOutput[s], SEGMENT_SIZE * sizeof(float));
    }
    wbTime_stop(GPU, "Allocating GPU memory.");
    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");
    int offsets[STREAM_COUNT], sizes[STREAM_COUNT];
    for (int i = 0; i < inputLength; i += (SEGMENT_SIZE * STREAM_COUNT)) {
      int count = min((inputLength - i - 1) / SEGMENT_SIZE + 1, STREAM_COUNT);
      for (int s = 0; s < count; s++) {
        offsets[s] = i + SEGMENT_SIZE * s;
        sizes[s] = min(inputLength - offsets[s], SEGMENT_SIZE);
      }
      for (int s = 0; s < count; s++) {
        int size = sizes[s] * sizeof(float);
        cudaMemcpyAsync(deviceInput1[s], hostInput1 + offsets[s], size, cudaMemcpyHostToDevice, streams[s]);
        cudaMemcpyAsync(deviceInput2[s], hostInput2 + offsets[s], size, cudaMemcpyHostToDevice, streams[s]);
      }
      for (int s = 0; s < count; s++) {
        int size = sizes[s];
        vecAdd <<<(size - 1) / BLOCK_SIZE + 1, BLOCK_SIZE, 0, streams[s]>>>
          (deviceInput1[s], deviceInput2[s], deviceOutput[s], size);
      }
      for (int s = 0; s < count; s++) {
        cudaMemcpyAsync(hostOutput + offsets[s], deviceOutput[s], sizes[s] * sizeof(float), cudaMemcpyDeviceToHost, streams[s]);
      }
    }
    cudaDeviceSynchronize();
    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");
    for (int s = 0; s < STREAM_COUNT; s++) {
      cudaFree(deviceOutput[s]);
      cudaFree(deviceInput2[s]);
      cudaFree(deviceInput1[s]);
      cudaStreamDestroy(streams[s]);
    }
  } else {
    wbTime_start(GPU, "Allocating GPU memory.");
    cudaMalloc((void**) &deviceInput1[0], inputLength * sizeof(float));
    cudaMalloc((void**) &deviceInput2[0], inputLength * sizeof(float));
    cudaMalloc((void**) &deviceOutput[0], inputLength * sizeof(float));
    wbTime_stop(GPU, "Allocating GPU memory.");
    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");
    cudaMemcpy(deviceInput1[0], hostInput1, inputLength * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2[0], hostInput2, inputLength * sizeof(float), cudaMemcpyHostToDevice);
    vecAdd <<<(inputLength - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>> (deviceInput1[0], deviceInput2[0], deviceOutput[0], inputLength);
    cudaDeviceSynchronize();
    cudaMemcpy(hostOutput, deviceOutput[0], inputLength * sizeof(float), cudaMemcpyDeviceToHost);
    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");
    cudaFree(deviceInput1[0]);
    cudaFree(deviceInput2[0]);
    cudaFree(deviceOutput[0]);
  }

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
