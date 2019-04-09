# operator

## BLAS

### gemv

- `y := alpha*A*x + beta*y`  
- `y := alpha*A'*x + beta*y`  
- `y := alpha*conjg(A')*x + beta*y` (not supported)  

Reference: [BLAS and Sparse BLAS Routines | Intel® Math Kernel Library for C](https://software.intel.com/en-us/mkl-developer-reference-c-blas-and-sparse-blas-routines)
