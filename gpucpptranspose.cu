#include <iostream>
#include <math.h>
#include <cstdlib>
#include <ctime>

const int blockSizex = 32;
const int blockSizey = 8;
const int TILE_DIM = blockSizex;
#define imin(a, b) (a<b?a:b)

// Naive matrix transpose
__global__
void gpu_matrix_trans_naive(double *mat_in, double *mat_out) {
  int idx = blockIdx.x * TILE_DIM + threadIdx.x;
  int idy = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += blockSizey) {
    mat_out[idy + j + idx*width] = mat_in[(idy+j)*width + idx];
  }
}

// Transpose via shared memory
__global__
void gpu_matrix_trans_sharedmem(double *mat_in, double *mat_out) {
  // shared memory (48KB/N per block), N is the number of blocks on the same multiprocessor
  __shared__ double tile[TILE_DIM*TILE_DIM];

  int idx = blockIdx.x * TILE_DIM + threadIdx.x;
  int idy = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += blockSizey) {
    tile[threadIdx.x + (threadIdx.y+j)*TILE_DIM] = mat_in[(idy+j)*width + idx];
  }

  __syncthreads();


  for (int j = 0; j < TILE_DIM; j += blockSizey) {
    mat_out[(idy+j)*width + idx] = tile[threadIdx.x*TILE_DIM + threadIdx.y + j];
  }
}

// Coalesced Transpose via shared memory
__global__
void gpu_matrix_trans_coales_sharedmem(double *mat_in, double *mat_out) {
  // shared memory (48KB/N per block), N is the number of blocks on the same multiprocessor
  __shared__ double tile[TILE_DIM*TILE_DIM];

  int idx = blockIdx.x * TILE_DIM + threadIdx.x;
  int idy = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += blockSizey) {
    tile[threadIdx.x*TILE_DIM + threadIdx.y + j] = mat_in[(idy+j)*width + idx];
  }

  __syncthreads();

  idx = blockIdx.y * TILE_DIM + threadIdx.x;
  idy = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += blockSizey) {
    mat_out[(idy+j)*width + idx] = tile[threadIdx.x + (threadIdx.y+j)*TILE_DIM];
  }
}

// Coalesced Transpose via shared memory without bank conflict
__global__
void gpu_matrix_trans_coales_sharedmem_NoBankConfl(double *mat_in, double *mat_out) {
  // shared memory (48KB/N per block), N is the number of blocks on the same multiprocessor
  __shared__ double tile[TILE_DIM][TILE_DIM+1];

  int idx = blockIdx.x * TILE_DIM + threadIdx.x;
  int idy = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += blockSizey) {
    tile[threadIdx.y+j][threadIdx.x] = mat_in[(idy+j)*width + idx];
  }

  __syncthreads();

  idx = blockIdx.y * TILE_DIM + threadIdx.x;
  idy = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += blockSizey) {
    mat_out[(idy+j)*width + idx] = tile[threadIdx.x][threadIdx.y+j];
  }
}

int main(void) {
  int cols = 1<<10, rows = 1<<10;

  int grid_cols = imin(512, (cols + TILE_DIM - 1)/TILE_DIM);
  int grid_rows = imin(512, (rows + TILE_DIM - 1)/TILE_DIM);
  dim3 dimBlock(blockSizex, blockSizey, 1);
  dim3 dimGrid(grid_cols, grid_rows, 1);

  // Allocate memory in host RAM
  double *h_mat_in = new double[cols*rows];
  double *h_mat_out = new double[cols*rows];

  // Initialize h_mat_in
  std::srand(1103);
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
//      h_mat_in[i*cols+j] = double (((generator() % (1000 - 0 + 1)) + 0)/1000);
      h_mat_in[i*cols+j] = double(std::rand())/double(RAND_MAX);

  // capture the GPU start time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // Allocate memory on device
  double *d_mat_in, *d_mat_out;
  cudaMalloc(&d_mat_in, cols*rows*sizeof(double));
  cudaMalloc(&d_mat_out, cols*rows*sizeof(double));

  // Copy matrix in from host to device
  cudaMemcpy(d_mat_in, h_mat_in, cols*rows*sizeof(double), cudaMemcpyHostToDevice);

  // Run kernel
  gpu_matrix_trans_naive<<<dimGrid, dimBlock>>>(d_mat_in, d_mat_out);
  gpu_matrix_trans_sharedmem<<<dimGrid, dimBlock>>>(d_mat_in, d_mat_out);
  gpu_matrix_trans_coales_sharedmem<<<dimGrid, dimBlock>>>(d_mat_in, d_mat_out);
  gpu_matrix_trans_coales_sharedmem_NoBankConfl<<<dimGrid, dimBlock>>>(d_mat_in, d_mat_out);

  // Copy result from device to host
  cudaMemcpy(h_mat_out, d_mat_out, cols*rows*sizeof(double), cudaMemcpyDeviceToHost);
  cudaThreadSynchronize();

  // get GPU stop time
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);

  // Check results
  int check_flag = 1;
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
      if (h_mat_out[j*rows + i] != h_mat_in[i*cols + j])
        check_flag = 0;

  if (!check_flag)
    std::cout << "GPU matrix transpose not success!!!" << std::endl;
  else {
    std::cout << "GPU matrix transpose success!!!" << std::endl;
    std::cout << "GPU matrix multiplication time: " << elapsedTime << " ms." << std::endl;
  }
  // Free memory
  cudaFree(d_mat_in);
  cudaFree(d_mat_out);
  delete [] h_mat_in;
  delete [] h_mat_out;

  return 0;
}
