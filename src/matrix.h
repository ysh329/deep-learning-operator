#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>

#define DTYPE float
#define A(i, j) a[lda*(i)+(j)]
#define B(i, j) b[ldb*(i)+(j)]
#define C(i, j) c[ldc*(i)+(j)]
DTYPE* gemm(const int M, const int N, const int K,
            const DTYPE *a, const int lda,
            const DTYPE *b, const int ldb,
                  DTYPE *c, const int ldc);
DTYPE* rand_matrix(DTYPE *a, const size_t len, DTYPE min_value, DTYPE max_value);
DTYPE* calloc_matrix(const size_t len);
int free_matrix(DTYPE *mat);
void diff_matrix(const DTYPE *a, const DTYPE *b, const size_t len);
#define MAT(i, j) mat[cols*(i)+(j)]
void print_matrix(DTYPE *mat, const int rows, const int cols);
void copy_matrix(const DTYPE *src, DTYPE *dst, const size_t len);
