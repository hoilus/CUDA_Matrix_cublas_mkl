## Level 3 large matrix multiplication by multi-cpus and multi-gpu cores
### Comparing computation efficiency of level 3 matrix multiplication using cublas and blas

### - gpucppadd.cu:
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
