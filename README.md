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
#### shared memory on-chip is much faster than local and global memory. It is thus a good idea to do the necessar arithemetic calculations on device, and minimize the data transit between gpu and cpu. Moreover, we can use share memory to achieve coalescing in both reads and writes. 
```
__global__
void gpu_matrix_trans_coales_sharedmem(double *mat_in, double *mat_out) {
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

```
