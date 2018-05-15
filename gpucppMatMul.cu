#include <iostream>
#include <math.h>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cublas_v2.h>

#include <cstdio>
#include <stdlib.h>

#define imin(a, b) (a<b?a:b)

const int blockSize = 16;

// Naive matrix multiplication
// m x k matrix A, k x n matrix B, m x n matrix C = A x B
__global__
void gpu_matrix_multi(double *matA, double *matB, double *matC, int m, int k, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  double sum = 0.0;

  if (idx < n && idy < m) {
    for (int i = 0; i < k; i++) {
      sum += matA[idy*k + i] * matB[i*n + idx];
    }
    matC[idy*n + idx] = sum;
    idx += blockDim.x * gridDim.x;
    idy += blockDim.y * gridDim.y;
  }
}

// Matrix multiplication by cuBLAS
// m x k matrix A, k x n matrix B, m x n matrix C = A x B
// tran(C) = tran(B) x tran(A)
void gpu_blas_multi(double *matB, double *matA, double *matC, int m, int k, int n) {
  int lda = n, ldb = k, ldc = n;
  double alf = 1.0;
  double bet = 0.0;
  double *alpha = &alf;
  double *beta = &bet;

  // Create a handle for cuBLAS
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Do the actual multiplication
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, alpha, matB, lda, matA, ldb, beta, matC, ldc);

  // Destroy the handle
  cublasDestroy(handle);
}

// Naive cpu matrix multiplication
// m x k matrix A, k x n matrix B, m x n matrix C = A x B
void cpu_matrix_multi(double *matA, double *matB, double *matC, int m, int k, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      double tmp = 0.0;
      for (int l = 0; l < k; l++) {
        tmp += matA[i*k + l] * matB[l*n + j];
      }
      matC[i*n + j] = tmp;
    }
  }

}

// Matrix multiplication by BLAS
// m x k matrix A, k x n matrix B, m x n matrix C = A x B
// tran(C) = tran(B) x tran(A)
extern "C"{
  // product C= alphaA.B + betaC
 void dgemm_(char* TRANSA, char* TRANSB, const int* M,
             const int* N, const int* K, double* alpha, double* A,
             const int* LDA, double* B, const int* LDB, double* beta,
             double* C, const int* LDC);
}

void initvec(double* v, int N){
  for(int i= 0; i<N; ++i){
    v[i]= 0.0;
  }
}

void cpu_blas_multi(double *matB, double *matA, double *matC, int m, int k, int n) {
  double alpha= 1.0, beta= 0.0;
  char no= 'N', tr= 'T';
  int m1 = m, k1 = k, n1 = n, lda= n, incx= k, incy= n;
  double* tmp= new double[m*n];
  initvec(tmp, m*n);
  dgemm_(&no, &no, &n1, &m1, &k1, &alpha, matB, &lda, matA, &incx, &beta, tmp, &incy);
  for(int i= 0; i<m*n; ++i){
    matC[i]= tmp[i];
  }
  delete [] tmp;
}

int main(void) {
  int m = 1<<15, k = 1<<10, n = 1<<15;
//  int m = 100, k = 1<< 20, n = 100;

  int grid_cols = imin(1024, (n + blockSize - 1)/blockSize);
  int grid_rows = imin(1024, (m + blockSize - 1)/blockSize);
  dim3 dimBlock(blockSize, blockSize, 1);
  dim3 dimGrid(grid_cols, grid_rows, 1);

  // Allocate memory in host RAM
  double *h_matA = new double[m*k];
  double *h_matB = new double[k*n];
  double *h_matC = new double[m*n];
  double *h_matC_cpu = new double[m*n];

  // Initialize h_mat_in
  std::srand(1103);
  for (int i = 0; i < m; i++)
    for (int j = 0; j < k; j++)
      h_matA[i*k+j] = double(std::rand())/double(RAND_MAX);

  for (int i = 0; i < k; i++)
    for (int j = 0; j < n; j++)
      h_matB[i*n+j] = double(std::rand())/double(RAND_MAX);

  // capture the GPU start time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // Allocate memory on device
  double *d_matA, *d_matB, *d_matC;
  cudaMalloc(&d_matA, m*k*sizeof(double));
  cudaMalloc(&d_matB, k*n*sizeof(double));
  cudaMalloc(&d_matC, m*n*sizeof(double));

  // Copy matrix in from host to device
//  auto wallGPU0 = std::chrono::system_clock::now();
  cudaMemcpy(d_matA, h_matA, m*k*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_matB, h_matB, k*n*sizeof(double), cudaMemcpyHostToDevice);

  // Run kernel
//  gpu_matrix_multi<<<dimGrid, dimBlock>>>(d_matA, d_matB, d_matC, m, k, n);
  gpu_blas_multi(d_matB, d_matA, d_matC, m, k, n);

  // Copy result from device to host
  cudaMemcpy(h_matC, d_matC, m*n*sizeof(double), cudaMemcpyDeviceToHost);
  cudaThreadSynchronize();
//  std::chrono::duration<double> wallGPUduration = (std::chrono::system_clock::now() - wallGPU0);

  // get GPU stop time
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
//  cout << "Time to generate: " << elapsedTime << " ms." << endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Check results

  auto wallCPU0 = std::chrono::system_clock::now();
//  cpu_matrix_multi(h_matA, h_matB, h_matC_cpu, m, k, n);
  cpu_blas_multi(h_matB, h_matA, h_matC_cpu, m, k, n);
  std::chrono::duration<double> wallCPUduration = (std::chrono::system_clock::now() - wallCPU0);

  int check_flag = 1;
  double resol = 1e-5;
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      if (fabs(h_matC[i*n + j] - h_matC_cpu[i*n + j]) > resol)
        check_flag = 0;

  if (!check_flag)
    std::cout << "GPU matrix multiplication not success!!!" << std::endl;
  else {
    std::cout << "GPU matrix multiplication success!!!" << std::endl;
    std::cout << "GPU matrix multiplication by cublas costs: " << elapsedTime << " ms." << std::endl;
    std::cout << "CPU matrix multiplication by blas costs: " << wallCPUduration.count() << " s." << std::endl;
  }

  // Free memory
  cudaFree(d_matA);
  cudaFree(d_matB);
  cudaFree(d_matC);
  delete [] h_matA;
  delete [] h_matB;
  delete [] h_matC;
  delete [] h_matC_cpu;

  return 0;
}
