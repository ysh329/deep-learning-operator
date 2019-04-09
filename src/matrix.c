#include "matrix.h"

#define DTYPE float
#define A(i, j) a[lda*(i)+(j)]
#define B(i, j) b[ldb*(i)+(j)]
#define C(i, j) c[ldc*(i)+(j)]
DTYPE* gemm(const int M, const int N, const int K,
            const DTYPE *a, const int lda,
            const DTYPE *b, const int ldb,
                  DTYPE *c, const int ldc) {
#ifdef SAFE_MODE
    assert(a && b &&
           M>0 && N>0 && K>0 &&
           lda>0 && ldb>0 && ldc>0);
#endif
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int p = 0; p < K; ++p) {
                C(i, j) += A(i, p) * B(p, j);
            }
        }
    }
    return c;
}

DTYPE* calloc_matrix(const size_t len) {
    assert(len > 0);
    DTYPE *mat = calloc(len, sizeof(DTYPE));
    if (!mat) {
        printf("[ERRO] fail to calloc matrix: %.4lfMB cost\n", len*sizeof(DTYPE)/10e6);
        mat = NULL;
    }
    return mat;
}

int free_matrix(DTYPE *mat) {
    assert(mat);
    if (mat) {
        free(mat);
    }
    mat = NULL;
    return 0;
}

DTYPE* rand_matrix(DTYPE *a, const size_t len, DTYPE min_value, DTYPE max_value) {
#ifdef SAFE_MODE
    assert(a && len>0 && min_value<=max_value);
    srand(time( NULL ));
#endif
    int range_value = (int)(max_value - min_value);
    for (int i = 0; i < len; ++i) {
        a[i] = (rand() % range_value) + min_value;
    }
    return a;
}

void diff_matrix(const DTYPE *a, const DTYPE *b, const size_t len) {
#ifdef SAFE_MODE
    assert(a && b && len>0);
#endif
    double eps   = 1e-5;
    int diff_num = 0;
    double diff  = 0;

    for(int i = 0; i < len; ++i) {
        diff = a[i] - b[i];
        diff = diff > 0 ? diff : -diff;
        if(diff > eps) ++diff_num;
    }
    printf("[INFO] ---- diff_matrix ----\n");
    printf("[INFO] diff_num:%d\n", diff_num);
    printf("[INFO] diff_ratio:%f\n", diff_num/(float)len);
    return;
}

#define MAT(i, j) mat[cols*(i)+(j)]
void print_matrix(DTYPE *mat, const int rows, const int cols) {
#ifdef SAFE_MODE
    assert(mat && rows > 0 && cols > 0);
#endif
    printf("[INFO] ---- print_matrix ----\n");
    for(int r = 0; r < rows; ++r) {
        for(int c = 0; c < cols; ++c) {
            printf("%.2f ", MAT(r, c));
        }
        printf("\n");
    }
    printf("\n\n");
    return;
}

void copy_matrix(const DTYPE *src, DTYPE *dst, const size_t len) {
#ifdef SAFE_MODE
    assert(src && dst && len > 0);
#endif
    for(int i = 0; i < len; ++i) {
        dst[i] = src[i];
    }
    return;
}
