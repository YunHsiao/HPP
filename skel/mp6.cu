
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <wb.h>

#define wbCheck(code) do { GPUAssert(code, __LINE__); } while (0)
inline void GPUAssert (cudaError_t code, int line) {
  if (code != cudaSuccess) {
    wbLog(ERROR, "==Assert== ", cudaGetErrorString(code), " at line ", line);
  }
}

#define Mask_width 5
#define Mask_radius (Mask_width / 2)

//@@ INSERT CODE HERE
#define BLOCK_WIDTH 32
#define O_TILE_WIDTH (BLOCK_WIDTH - (Mask_width - 1))

__global__ void convolution (float* input, float* output, const float* __restrict__ mask, int width, int height, int channels) {
  __shared__ float section[BLOCK_WIDTH][BLOCK_WIDTH];
  int tx = threadIdx.x, ty = threadIdx.y;
  int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
  int oRow = bx * O_TILE_WIDTH + tx, oCol = by * O_TILE_WIDTH + ty;
  int iRow = oRow - Mask_radius, iCol = oCol - Mask_radius;
  if (iRow >= 0 && iRow < height && iCol >= 0 && iCol < width) {
    section[tx][ty] = input[(iRow * width + iCol) * channels + bz];
  } else {
    section[tx][ty] = 0.0f;
  }
  __syncthreads();
  if (tx < O_TILE_WIDTH && ty < O_TILE_WIDTH) {
    float t = 0.0f;
    for (int i = 0; i < Mask_width; i++) {
      for (int j = 0; j < Mask_width; j++) {
        t += mask[i * Mask_width + j] * section[tx + i][ty + j];
      }
    }
    if (oRow < height && oCol < width) output[(oRow * width + oCol) * channels + bz] = min(max(t, 0.0f), 1.0f);
  }
}

int main (int argc, char* argv[]) {
  wbArg_t arg;
  int maskRows;
  int maskColumns;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  char * inputImageFile;
  char * inputMaskFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float * hostInputImageData;
  float * hostOutputImageData;
  float * hostMaskData;
  float * deviceInputImageData;
  float * deviceOutputImageData;
  float * deviceMaskData;

  arg = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(arg, 0);
  inputMaskFile = wbArg_getInputFile(arg, 1);

  inputImage = wbImport(inputImageFile);
  hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

  assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
  assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);

  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputImageData,
              hostInputImageData,
              imageWidth * imageHeight * imageChannels * sizeof(float),
              cudaMemcpyHostToDevice);
  cudaMemcpy(deviceMaskData,
              hostMaskData,
              maskRows * maskColumns * sizeof(float),
              cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ INSERT CODE HERE
  dim3 dimGrid((imageHeight - 1) / O_TILE_WIDTH + 1, (imageWidth - 1) / O_TILE_WIDTH + 1, imageChannels);
  dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);

  convolution <<<dimGrid, dimBlock>>> (deviceInputImageData, deviceOutputImageData, deviceMaskData, imageWidth, imageHeight, imageChannels);
  wbCheck(cudaPeekAtLastError());
  cudaDeviceSynchronize();

  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutputImageData,
              deviceOutputImageData,
              imageWidth * imageHeight * imageChannels * sizeof(float),
              cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  wbSolution(arg, outputImage);

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  cudaFree(deviceMaskData);

  free(hostMaskData);
  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
