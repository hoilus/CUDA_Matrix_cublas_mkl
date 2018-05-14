#include <iostream>
#include <math.h>
#include <cstdlib>
#include <ctime>

const int blockSize = 16;

// Naive matrix transpose
__global__
void gpu_matrix_trans_naive(double *mat_in, double *mat_out, int cols, int rows) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx < cols && idy < rows) {
    mat_out[idy + idx*rows] = mat_in[idy*cols + idx];
  }
}

// Transpose via shared memory
__global__
void gpu_matrix_trans_sharedmem(double *mat_in, double *mat_out, int cols, int rows) {
  // shared memory (48KB/N per block), N is the number of blocks on the same multiprocessor
  __shared__ double tile[blockSize*blockSize];

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx < cols && idy < rows)
    tile[threadIdx.x + blockSize*threadIdx.y] = mat_in[idy*cols + idx];

  __syncthreads();

  if (idx < cols && idy < rows)
    mat_out[idy + idx*rows] = tile[threadIdx.x + blockSize*threadIdx.y];
}

int main(void) {
  int cols = 1<<10, rows = 1<<10;

  int grid_cols = (cols + blockSize - 1)/blockSize;
  int grid_rows = (rows + blockSize - 1)/blockSize;
  dim3 dimBlock(blockSize, blockSize, 1);
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
//  gpu_matrix_trans_naive<<<dimGrid, dimBlock>>>(d_mat_in, d_mat_out, cols, rows);
  gpu_matrix_trans_sharedmem<<<dimGrid, dimBlock>>>(d_mat_in, d_mat_out, cols, rows);

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
