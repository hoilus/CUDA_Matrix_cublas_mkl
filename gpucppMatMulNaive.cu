#include <iostream>
#include <math.h>
#include <cstdio>

using namespace std;

// Thread block size
const int blockSize = 16;

// Matrices are stored in row-major order:
// M(row, clo) = *(M.elements + row*M.width + col);
typedef struct {
  int width;
  int height;
  float* elements;
} Matrix;

// CPU matrix multiplication for evaluating results
void cpu_matrix_multi(float *matA, float *matB, float *matC, int m, int k, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float tmp = 0.0;
      for (int l = 0; l < k; l++) {
        tmp += matA[i*k + l] * matB[l*n + j];
      }
      matC[i*n + j] = tmp;
    }
  }
}

// Matrix multiplication kernel called by MatMul()
__global__
void MatMulKernel_naive(const Matrix A, const Matrix B, Matrix C) {
  // Each thread computes one element of C
  // by accumulating results into Cvalue
  float Cvalue = 0;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  for (int e = 0; e < A.width; e++)
    Cvalue += A.elements[row*A.width+e] * B.elements[e*B.width+col];
  C.elements[row*C.width+col] = Cvalue;
}

// Matrix multiplication - host code
// Matrix dimensions are assumed to be multiples of blockSize
void MatMul(const Matrix A, const Matrix B, Matrix C) {
  // Load A and B to device memory
  Matrix d_A;
  d_A.width = A.width; d_A.height = A.height;
  size_t size = A.width * A.height * sizeof(float);
  cudaMalloc(&d_A.elements, size);
  cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
  Matrix d_B;
  d_B.width = B.width; d_B.height = B.width;
  size = B.width * B.height * sizeof(float);
  cudaMalloc(&d_B.elements, size);
  cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

  // Allocate C in device memory
  Matrix d_C;
  d_C.width = C.width; d_C.height = C.height;
  size = C.width * C.height * sizeof(float);
  cudaMalloc(&d_C.elements, size);
  
  // Invoke kernel
  dim3 dimBlock(blockSize, blockSize, 1);
  dim3 dimGrid(B.width/dimBlock.x, A.height/dimBlock.y, 1);
  MatMulKernel_naive<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

  // Read C from device
  cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
  cudaThreadSynchronize();

  // Free device memory
  cudaFree(d_A.elements);
  cudaFree(d_B.elements);
  cudaFree(d_C.elements);
}

int main() {
  // Initiate A and B elements on host memory
  Matrix h_A;
  h_A.height = 1024; h_A.width = 1024;
  float* h_matA = new float[h_A.height * h_A.width];
  std::srand(1103);
  for (int i = 0; i < h_A.height; i++)
    for (int j = 0; j < h_A.width; j++)
      h_matA[i*h_A.width+j] = float(std::rand())/float(RAND_MAX);
  h_A.elements = h_matA;

  Matrix h_B;
  h_B.height = 1024; h_B.width = 1024;
  float* h_matB = new float[h_B.height * h_B.width];
  for (int i = 0; i < h_B.height; i++)
    for (int j = 0; j < h_B.width; j++)
      h_matB[i*h_B.width+j] = float(std::rand())/float(RAND_MAX);
  h_B.elements = h_matB;

  // Matrix C size
  Matrix h_C;
  h_C.height = h_A.height; h_C.width = h_B.width;
  float* h_matC = new float[h_A.height * h_B.width];
  h_C.elements = h_matC;

  // Call MatMul()
  MatMul(h_A, h_B, h_C);

  // Evaluate results
  float* h_matC_cpu = new float[h_A.height * h_B.width];
//  cpu_matrix_multi(h_matA, h_matB, h_matC_cpu, h_A.height, h_A.width, h_B.width);
  cpu_matrix_multi(h_A.elements, h_B.elements, h_matC_cpu, h_A.height, h_A.width, h_B.width);
  bool res_flag = false;
  float resol = 0.000001;
  for (int i = 0; i < h_C.height; i++) {
    for (int j = 0; j < h_C.width; j++) {
      if (fabs(*(h_C.elements+i*h_C.width+j) - h_matC[i*h_C.width+j]) > resol)
	res_flag = true;
    }
  }
	
  if (res_flag == false)
    cout << "Matrix multiplication by GPU is right! " << endl;
  else
    cout << "Results are not right! " << endl;

  // Free memory on host 
  delete [] h_matA;
  delete [] h_matB;
  delete [] h_matC;
  delete [] h_matC_cpu;

  return 0;
}

