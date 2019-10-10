
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <wb.h>

constexpr int TILE_WIDTH = 32;

#define wbCheck(code) do { GPUAssert(code, __LINE__); } while (0)
inline void GPUAssert(cudaError_t code, int line) {
  if (code != cudaSuccess) {
    wbLog(ERROR, "==Assert== ", cudaGetErrorString(code), " at line ", line);
  }
}

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C, int numARows,
                                     int numAColumns, int numBRows,
                                     int numBColumns, int numCRows,
                                     int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float sB[TILE_WIDTH][TILE_WIDTH];
  float t = 0.0f;
  int tx = threadIdx.x, ty = threadIdx.y;
  int bx = blockIdx.x, by = blockIdx.y;
  int tiles = (numAColumns - 1) / TILE_WIDTH + 1;
  for (int i = 0; i < tiles; i++) {
    int aRow = TILE_WIDTH * by + ty, aCol = TILE_WIDTH *  i + tx;
    int bRow = TILE_WIDTH *  i + ty, bCol = TILE_WIDTH * bx + tx;
    if (aRow < numARows && aCol < numAColumns) sA[ty][tx] = A[numAColumns * aRow + aCol];
    else sA[ty][tx] = 0.0f;
    if (bRow < numBRows && bCol < numBColumns) sB[ty][tx] = B[numBColumns * bRow + bCol];
    else sB[ty][tx] = 0.0f;
    __syncthreads();
    for (int j = 0; j < TILE_WIDTH; j++) {
      t += sA[ty][j] * sB[j][tx];
    }
    __syncthreads();
  }
  int x = bx * blockDim.x + tx;
  int y = by * blockDim.y + ty;
  if (y < numCRows && x < numCColumns) {
    C[y * numCColumns + x] = t;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = ( float * )wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
  hostB = ( float * )wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  wbCheck(cudaMalloc((void**)&deviceA, numARows * numAColumns * sizeof(float)));
  wbCheck(cudaMalloc((void**)&deviceB, numBRows * numBColumns * sizeof(float)));
  wbCheck(cudaMalloc((void**)&deviceC, numCRows * numCColumns * sizeof(float)));

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  wbCheck(cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice));

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid((numCColumns - 1) / TILE_WIDTH + 1, (numCRows - 1) / TILE_WIDTH + 1, 1);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared <<<dimGrid, dimBlock>>> (deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  wbCheck(cudaPeekAtLastError());

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  wbCheck(cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost));

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  wbCheck(cudaFree(deviceA));
  wbCheck(cudaFree(deviceB));
  wbCheck(cudaFree(deviceC));

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
