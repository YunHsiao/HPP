// Histogram Equalization

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <wb.h>

#define BLOCK_SIZE 32
#define HISTOGRAM_LENGTH 256
#define THREAD_COUNT (HISTOGRAM_LENGTH >> 1)

#define wbCheck(code) do { GPUAssert(code, __LINE__); } while (0)
inline void GPUAssert (cudaError_t code, int line) {
  if (code != cudaSuccess) {
    wbLog(ERROR, "==Assert== ", cudaGetErrorString(code), " at line ", line);
  }
}

//@@ insert code here
__device__ void do_scan (float* section, int tid, bool sum) {
  for (int stride = 1; stride < HISTOGRAM_LENGTH; stride <<= 1) {
    int second = (tid + 2) * stride - 1;
    if (second < HISTOGRAM_LENGTH) {
      if (sum) section[second] += section[second - stride];
      else section[second] = min(section[second], section[second - stride]);
    }
    __syncthreads();
  }
  for (int stride = (THREAD_COUNT >> 1); stride > 0; stride >>= 1) {
    int second = (tid + 3) * stride - 1;
    if (second < HISTOGRAM_LENGTH) {
      if (sum) section[second] += section[second - stride];
      else section[second] = min(section[second], section[second - stride]);
    }
    __syncthreads();
  }
}

__device__ float correct_color (const float* __restrict__ CDF, float minCDF, unsigned char val) {
  return min(max((CDF[val] - minCDF) / (1.0f - minCDF), 0.0f), 1.0f);
}

__global__ void histogram (float* input, int* histogram, int width, int height) {
  int tx = threadIdx.x, ty = threadIdx.y;
  int bx = blockIdx.x, by = blockIdx.y;
  int row = bx * BLOCK_SIZE + tx, col = by * BLOCK_SIZE + ty;
  if (row < height && col < width) {
    float r = input[(row * width + col) * 3];
    float g = input[(row * width + col) * 3 + 1];
    float b = input[(row * width + col) * 3 + 2];
    unsigned char li = (0.21f * r + 0.71f * g + 0.07f * b) * 255.0f;
    atomicAdd(&histogram[li], 1);
  }
}

__global__ void scan (int* histogram, float* CDF, float pixels) {
  __shared__ float section[HISTOGRAM_LENGTH];
  int tid = threadIdx.x;
  int tid2 = THREAD_COUNT + tid;
  section[tid] = histogram[tid] * pixels;
  section[tid2] = histogram[tid2] * pixels;
  __syncthreads();
  do_scan(section, tid << 1, true);
  CDF[tid] = section[tid];
  CDF[tid2] = section[tid2];
}

__global__ void equalize (float* input, const float* __restrict__ CDF, float* output, int width, int height) {
  int tx = threadIdx.x, ty = threadIdx.y;
  int bx = blockIdx.x, by = blockIdx.y;
  int row = bx * BLOCK_SIZE + tx, col = by * BLOCK_SIZE + ty;
  if (row < height && col < width) {
    float minCDF = CDF[0];
    int idx = (row * width + col) * 3;
    output[idx] = correct_color(CDF, minCDF, input[idx] * 255.0f);
    output[idx + 1] = correct_color(CDF, minCDF, input[idx + 1] * 255.0f);
    output[idx + 2] = correct_color(CDF, minCDF, input[idx + 2] * 255.0f);
  }
}

int main(int argc, char** argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float* hostInputImageData;
  float* hostOutputImageData;
  const char* inputImageFile;

  //@@ Insert more code here
  int* hostHistogramData;
  float* hostCDFData;
  float* deviceInputImageData;
  float* deviceCDFData;
  float* deviceOutputImageData;
  int* deviceHistogramData;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  hostHistogramData = (int*) malloc(HISTOGRAM_LENGTH * sizeof(int));
  hostCDFData = (float*) malloc(HISTOGRAM_LENGTH * sizeof(float));

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **) &deviceHistogramData, HISTOGRAM_LENGTH * sizeof(int));
  cudaMalloc((void **) &deviceCDFData, HISTOGRAM_LENGTH * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputImageData,
              hostInputImageData,
              imageWidth * imageHeight * imageChannels * sizeof(float),
              cudaMemcpyHostToDevice);
  cudaMemset(deviceHistogramData, 0, HISTOGRAM_LENGTH * sizeof(int));
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");

  dim3 dimGrid((imageHeight - 1) / BLOCK_SIZE + 1, (imageWidth - 1) / BLOCK_SIZE + 1, 1);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

  histogram <<<dimGrid, dimBlock>>> (deviceInputImageData, deviceHistogramData, imageWidth, imageHeight);
  wbCheck(cudaPeekAtLastError());
  cudaDeviceSynchronize();

  scan <<<1, THREAD_COUNT>>> (deviceHistogramData, deviceCDFData, 1.0f / (imageWidth * imageHeight));
  wbCheck(cudaPeekAtLastError());
  cudaDeviceSynchronize();

  equalize <<<dimGrid, dimBlock>>> (deviceInputImageData, deviceCDFData, deviceOutputImageData, imageWidth, imageHeight);
  wbCheck(cudaPeekAtLastError());
  cudaDeviceSynchronize();

  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostCDFData,
            deviceCDFData,
            HISTOGRAM_LENGTH * sizeof(float),
            cudaMemcpyDeviceToHost);
  cudaMemcpy(hostHistogramData,
            deviceHistogramData,
            HISTOGRAM_LENGTH * sizeof(int),
            cudaMemcpyDeviceToHost);
  cudaMemcpy(hostOutputImageData,
            deviceOutputImageData,
            imageWidth * imageHeight * imageChannels * sizeof(float),
            cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // wbSolution(args, hostHistogramData, HISTOGRAM_LENGTH);
  // wbSolution(args, hostCDFData, HISTOGRAM_LENGTH);
  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  cudaFree(deviceHistogramData);
  free(hostHistogramData);
  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
