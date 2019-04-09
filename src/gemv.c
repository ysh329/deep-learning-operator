#include <stdlib.h>
#include <stdio.h>

/* header */
void gemv_cpu_naive(const int trans,
                    const int M, const int N, const float alpha,
                    const float *A, const int lda, const float *B,
                    const float beta, float *C);
#if 0
void cblas_sgemv(const bool trans,
                 const int M, const int N, const float alpha,
                 const float *A, const int lda, const float *B,
                 const float beta, float *C);
void sgemv_mx1(const bool trans,
               const int M, const int N, const float alpha,
               const float *A, const int lda, const float *B, const float beta,
               float *C);
void sgemv_trans_mx1_new(const int M, const int N, const float alpha,
                         const float *A, const int lda, const float *B,
                         const float beta, float *C);
void sgemv_trans_mx1_old(const int M, const int N, const float alpha,
                         const float *A, const int lda, const float *B,
                         const float beta, float *C);
void sgemv_notrans_mx1(const int M, const int N, const float alpha,
                       const float *A, const int lda, const float *B,
                       const float beta, float *C);
#endif

/* source */
#define a(i, j) A[(j)+lda*(i)]
#define b(i, j) B[(j)+ldb*(i)]
#define c(i, j) C[(j)+ldc*(i)]
void gemm_cpu_naive(const int M, const int N, const int K, 
                    const float alpha, const float beta,
                    const float *A, const int lda,
                    const float *B, const int ldb,
                    float *C, const int ldc) {
  for (register int m = 0; m < M; ++m) {
    for (register int n = 0; n < N; ++n) {
      for (register int k = 0; k < K; ++k) {
        c(m, n) += a(m, k) * c(k, n);
      }
    }
  }
}

#if 0
void sgemv_mx1(const bool trans,
               const int M, const int N, const float alpha,
               const float *A, const int lda, const float *B, const float beta,
               float *C) {
  if (trans) {
    sgemv_trans_mx1_new(M, N, alpha, A, lda, B, beta, C);
  } else {
    sgemv_notrans_mx1(M, N, alpha, A, lda, B, beta, C);
  }
}

void sgemv_notrans_mx1(const int M, const int N, const float alpha,
                       const float *A, const int lda, const float *B,
                       const float beta, float *C) {
  uint32_t mask[4] = {0, 1, 2, 3};
  int remain_n = N & 0x3;
  uint32x4_t vmask = vcltq_u32(vld1q_u32(mask), vdupq_n_u32(remain_n));
  float32x4_t _valpha = vdupq_n_f32(alpha);

  #pragma omp parallel for
  for (int m = 0; m < M - 3; m += 4) {
    const float *in0 = A + m * lda;
    const float *in1 = in0 + lda;
    const float *in2 = in1 + lda;
    const float *in3 = in2 + lda;
    float *output = C + m;

    float32x4_t _sum0, _sum1, _sum2, _sum3;
    _sum0 = vdupq_n_f32(0.f);
    _sum1 = vdupq_n_f32(0.f);
    _sum2 = vdupq_n_f32(0.f);
    _sum3 = vdupq_n_f32(0.f);
    int n = 0;
    for (; n < N - 3; n += 4) {
      float32x4_t _r0 = vld1q_f32(in0 + n);
      float32x4_t _r1 = vld1q_f32(in1 + n);
      float32x4_t _r2 = vld1q_f32(in2 + n);
      float32x4_t _r3 = vld1q_f32(in3 + n);
      float32x4_t _b = vld1q_f32(B + n);
      _sum0 = vmlaq_f32(_sum0, _r0, _b);
      _sum1 = vmlaq_f32(_sum1, _r1, _b);
      _sum2 = vmlaq_f32(_sum2, _r2, _b);
      _sum3 = vmlaq_f32(_sum3, _r3, _b);
    }
    if (n < N) {
      float32x4_t _r0 = vld1q_f32(in0 + n);
      float32x4_t _r1 = vld1q_f32(in1 + n);
      float32x4_t _r2 = vld1q_f32(in2 + n);
      float32x4_t _r3 = vld1q_f32(in3 + n);
      float32x4_t _b = vld1q_f32(B + n);
      _r0 = vandq_f32_u32(_r0, vmask);
      _r1 = vandq_f32_u32(_r1, vmask);
      _r2 = vandq_f32_u32(_r2, vmask);
      _r3 = vandq_f32_u32(_r3, vmask);
      _b = vandq_f32_u32(_b, vmask);
      _sum0 = vmlaq_f32(_sum0, _r0, _b);
      _sum1 = vmlaq_f32(_sum1, _r1, _b);
      _sum2 = vmlaq_f32(_sum2, _r2, _b);
      _sum3 = vmlaq_f32(_sum3, _r3, _b);
    }
    _sum0 = vpaddq_f32(_sum0, _sum1);
    _sum2 = vpaddq_f32(_sum2, _sum3);
    _sum0 = vpaddq_f32(_sum0, _sum2);
    _sum0 = vmulq_f32(_sum0, _valpha);
    if (beta != 0.f) {
      _sum2 = vmulq_n_f32(vld1q_f32(output), beta);
      _sum0 = vaddq_f32(_sum0, _sum2);
    }
    // restore
    vst1q_f32(output, _sum0);
  }
  // remain m
  for (int m = (M & 0xfffffffc); m < M; ++m) {
    const float *in0 = A + m * lda;
    float *output = C + m;
    float32x4_t _sum0 = vdupq_n_f32(0.f);

    int n = 0;
    for (; n < N - 3; n += 4) {
      float32x4_t _r0 = vld1q_f32(in0 + n);
      float32x4_t _b = vld1q_f32(B + n);
      _sum0 = vmlaq_f32(_sum0, _r0, _b);
    }
    if (n < N) {
      float32x4_t _r0 = vld1q_f32(in0 + n);
      float32x4_t _b = vld1q_f32(B + n);
      _r0 = vandq_f32_u32(_r0, vmask);
      _b = vandq_f32_u32(_b, vmask);
      _sum0 = vmlaq_f32(_sum0, _r0, _b);
    }
    _sum0 = vpaddq_f32(_sum0, _sum0);
    _sum0 = vmulq_f32(_sum0, _valpha);
    if (beta != 0.f) {
      float32x4_t _sum2 = vmulq_n_f32(vld1q_f32(output), beta);
      _sum0 = vpaddq_f32(_sum0, _sum2);
    }
    // restore
    *output = vgetq_lane_f32(_sum0, 0) + vgetq_lane_f32(_sum0, 1);
  }
}

void sgemv_trans_mx1_new(const int M, const int N, const float alpha,
                         const float *A, const int lda, const float *B,
                         const float beta, float *C) {
  // assign C with beta*C
  float32x4_t _valpha = vdupq_n_f32(alpha);
  if (beta == 0.f) {
    float32x4_t vzero = vdupq_n_f32(0.f);
    for (int m = 0; m < M - 3; m += 4) {
      vst1q_f32(C + m, vzero);
    }
    for (int m = (M & 0xfffffffc); m < M; ++m) {
      C[m] = 0.f;
    }
  } else {
    float32x4_t vbeta = vdupq_n_f32(beta);
    for (int m = 0; m < M - 3; m += 4) {
      float32x4_t _vc = vld1q_f32(C + m);
      _vc = vmulq_f32(_vc, vbeta);
      vst1q_f32(C + m, _vc);
    }
    for (int m = (M & 0xfffffffc); m < M; ++m) {
      C[m] *= beta;
    }
  }

  #pragma omp parallel for
  for (int m = 0; m < M - 3; m += 4) {
    // load A pointer
    register const float *ap = A + m;
    float32x4_t _sum = vdupq_n_f32(0.0f);
    float32x4_t _c00_10_20_30_vreg = vld1q_f32(C + m);
    int n = 0;
    for (; n < N - 3; n += 4) {
      // load a, b, c
      float32x4_t b_vreg = vld1q_f32(B + n);
      float32x4_t a00_10_20_30_vreg = vld1q_f32(ap + M * n);
      float32x4_t a01_11_21_31_vreg = vld1q_f32(ap + M * (n + 1));
      float32x4_t a02_12_22_32_vreg = vld1q_f32(ap + M * (n + 2));
      float32x4_t a03_13_23_33_vreg = vld1q_f32(ap + M * (n + 3));

      _sum = vmlaq_lane_f32(_sum, a00_10_20_30_vreg, vget_low_f32(b_vreg), 0);
      _sum = vmlaq_lane_f32(_sum, a01_11_21_31_vreg, vget_low_f32(b_vreg), 1);
      _sum = vmlaq_lane_f32(_sum, a02_12_22_32_vreg, vget_high_f32(b_vreg), 0);
      _sum = vmlaq_lane_f32(_sum, a03_13_23_33_vreg, vget_high_f32(b_vreg), 1);
    }

    // remain n, add to _sum
    for (; n < N; ++n) {
      float32x4_t a0n_1n_2n_3n_vreg = vld1q_f32(ap + M * n);
      float32x4_t bn_vreg = vdupq_n_f32(*(B + n));
      _sum = vmlaq_f32(_sum, a0n_1n_2n_3n_vreg, bn_vreg);
    }
    // _sum := _sum * valpha + _c
    _sum = vmlaq_f32(_c00_10_20_30_vreg, _sum, _valpha);
    // store
    vst1q_f32(C + m, _sum);
  }

  // remain m
  int remain_m = M & 3; // remain_m := M & (4-1)
  if (remain_m > 0) {
    const int remain_m_idx = M - remain_m;
    float32x4_t _c00_10_20_30_vreg = vld1q_f32(C + remain_m_idx);
    float32x4_t _sum = vdupq_n_f32(0.0f);
    register const float *ap = A + remain_m_idx;
    int n = 0;
    for (; n < N - 3; n += 4) {
      // load a, b
      float32x4_t a00_10_20_30_vreg = vld1q_f32(ap + M * (n));
      float32x4_t a01_11_21_31_vreg = vld1q_f32(ap + M * (n+1));
      float32x4_t a02_12_22_32_vreg = vld1q_f32(ap + M * (n+2));
      float32x4_t a03_13_23_33_vreg = vld1q_f32(ap + M * (n+3));
      float32x4_t b_vreg = vld1q_f32(B + n);

      _sum = vmlaq_lane_f32(_sum, a00_10_20_30_vreg, vget_low_f32(b_vreg), 0);
      _sum = vmlaq_lane_f32(_sum, a01_11_21_31_vreg, vget_low_f32(b_vreg), 1);
      _sum = vmlaq_lane_f32(_sum, a02_12_22_32_vreg, vget_high_f32(b_vreg), 0);
      _sum = vmlaq_lane_f32(_sum, a03_13_23_33_vreg, vget_high_f32(b_vreg), 1);
    }
    // remain n
    for (; n < N; ++n) {
      // load remain a, b
      float32x4_t a0n_1n_2n_3n_vreg = vld1q_f32(ap + M * n); 
      float32x4_t bn_vreg = vdupq_n_f32(*(B + n));
      _sum = vmlaq_f32(_sum, a0n_1n_2n_3n_vreg, bn_vreg);      
    }
    _sum = vmlaq_f32(_c00_10_20_30_vreg, _sum, _valpha);
    switch ( remain_m ) {
      case 3:
        vst1q_lane_f32(C + remain_m_idx + 2, _sum, 2);
      case 2:
        vst1_f32(C + remain_m_idx, vget_low_f32(_sum));
        break;
      case 1:
        vst1q_lane_f32(C + remain_m_idx, _sum, 0);
        break;
    }
  }
}

void sgemv_trans_mx1_old(const int M, const int N, const float alpha,
                         const float *A, const int lda, const float *B,
                         const float beta, float *C) {
  float32x4_t _valpha = vdupq_n_f32(alpha);
  if (beta == 0.f) {
    float32x4_t vzero = vdupq_n_f32(0.f);
    for (int m = 0; m < M - 3; m += 4) {
      vst1q_f32(C + m, vzero);
    }
    for (int m = (M & 0xfffffffc); m < M; ++m) {
      C[m] = 0.f;
    }
  } else {
    float32x4_t vbeta = vdupq_n_f32(beta);
    for (int m = 0; m < M - 3; m += 4) {
      float32x4_t _vc = vld1q_f32(C + m);
      _vc = vmulq_f32(_vc, vbeta);
      vst1q_f32(C + m, _vc);
    }
    for (int m = (M & 0xfffffffc); m < M; ++m) {
      C[m] *= beta;
    }
  }

  #pragma omp parallel for
  for (int n = 0; n < N - 3; n += 4) {
    const float *in0 = A + n * lda;
    const float *in1 = in0 + lda;
    const float *in2 = in1 + lda;
    const float *in3 = in2 + lda;
    float32x4_t _b = vld1q_f32(B + n);
    float32x4_t _sum0;
    int m = 0;
    for (; m < M - 3; m += 4) {
      float32x4_t _r0 = vld1q_f32(in0 + m);
      float32x4_t _r1 = vld1q_f32(in1 + m);
      float32x4_t _r2 = vld1q_f32(in2 + m);
      float32x4_t _r3 = vld1q_f32(in3 + m);
      float32x4_t _vc = vld1q_f32(C + m);

      _sum0 = vmulq_lane_f32(_r0, vget_low_f32(_b), 0);
      _sum0 = vmlaq_lane_f32(_sum0, _r1, vget_low_f32(_b), 1);
      _sum0 = vmlaq_lane_f32(_sum0, _r2, vget_high_f32(_b), 0);
      _sum0 = vmlaq_lane_f32(_sum0, _r3, vget_high_f32(_b), 1);
      _sum0 = vmulq_f32(_sum0, _valpha);
      _sum0 = vaddq_f32(_sum0, _vc);
      vst1q_f32(C + m, _sum0);
    }
    if (m < M) {
      float32x4_t _r0 = vld1q_f32(in0 + m);
      float32x4_t _r1 = vld1q_f32(in1 + m);
      float32x4_t _r2 = vld1q_f32(in2 + m);
      float32x4_t _r3 = vld1q_f32(in3 + m);
      float32x4_t _vc = vld1q_f32(C + m);

      _sum0 = vmulq_lane_f32(_r0, vget_low_f32(_b), 0);
      _sum0 = vmlaq_lane_f32(_sum0, _r1, vget_low_f32(_b), 1);
      _sum0 = vmlaq_lane_f32(_sum0, _r2, vget_high_f32(_b), 0);
      _sum0 = vmlaq_lane_f32(_sum0, _r3, vget_high_f32(_b), 1);
      _sum0 = vmulq_f32(_sum0, _valpha);
      _sum0 = vaddq_f32(_sum0, _vc);
      switch (M - m) {
        case 3:
          vst1q_lane_f32(C + m + 2, _sum0, 2);
        case 2:
          vst1_f32(C + m, vget_low_f32(_sum0));
          break;
        case 1:
          vst1q_lane_f32(C + m, _sum0, 0);
          break;
      }
    }
  }
  // remain n
  for (int n = (N & 0xfffffffc); n < N; ++n) {
    const float *in0 = A + n * lda;
    float32x4_t _b = vld1q_dup_f32(B + n);
    float32x4_t _sum0;
    int m = 0;
    for (; m < M - 3; m += 4) {
      float32x4_t _r0 = vld1q_f32(in0 + m);
      _sum0 = vld1q_f32(C + m);
      _r0 = vmulq_f32(_r0, _b);
      _r0 = vmulq_f32(_valpha, _r0);
      _sum0 = vaddq_f32(_sum0, _r0);
      vst1q_f32(C + m, _sum0);
    }
    for (; m < M; ++m) {
      C[m] += alpha * (in0[m] * B[n]);
    }
  }
}
#endif

int main(int argc, char *args[]) {
  #define COMMAND_LINE_INPUT_NUM (4)
  if (argc < COMMAND_LINE_INPUT_NUM) {
    printf("Usage: ./gemv m n k\n");
    exit(1);
  }

  printf("argc:%d\n", argc);
  for (int idx = 0; idx < argc; ++idx) {
    printf("%d %s\n", idx, args[idx]);
  }

  // init matrix a,b,c
  const int M = 1024;
  const int N = 1024;
  const int K = 1024;
  float *a = calloc(M * K, sizeof(float));
  float *b = calloc(K * N, sizeof(float));
  float *c = calloc(M * N, sizeof(float));
  if(!a || !b || !c) {
    printf("[ERRO] failed to calloc memory for a || b || c\n");
    if (a) free(a); a = NULL;
    if (b) free(b); b = NULL;
    if (c) free(c); c = NULL;
    exit(1);
  }

  // rand init matrix a,b
  void srand (unsigned int seed);
  for (int idx = )
  rand() % 10

  // free matrix a,b,c
  if (a) free(a); a = NULL;
  if (b) free(b); b = NULL;
  if (c) free(c); c = NULL;

  return 0;
}
