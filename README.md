## Level 3 large matrix multiplication by multi-cpus and multi-gpu cores
### Comparing computation efficiency of level 3 matrix multiplication using cublas and blas

### 1. gpucppadd.cu:
#### The hardware limits the number of blocks in a single lauch to 65,535 per dimension, and each block has a maximum of 1024 threads per block. In order to break this limit, we can use grid-stride loop to calculate the arrays with arbitrary length.
```
__global__
void add(int n, float *x, float *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}
```

### 2. gpucpptranspose.cu:
#### shared memory on-chip is much faster than local and global memory. It is thus a good idea to do the necessar arithemetic calculations on device, and minimize the data transit between gpu and cpu.
```
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
```
