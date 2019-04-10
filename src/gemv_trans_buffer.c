void sgemv_trans_mx1_old_buffer(const int M, const int N, const float alpha,
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

  // todo:use paddle'alloc
  const int a_buf_cols = N/4 + (N%4==0 ? 0 : 1);
  float *a_buf = (float*)calloc(M * a_buf_cols, sizeof(float));
  if (!a_buf) {
    std::cout << "fail to calloc a_buff" << std::endl;
    exit(1);
  }

  #pragma omp parallel for
  for (int n = 0; n < N - 3; n += 4) {
    const float *in0 = A + n * lda;
    const float *in1 = in0 + lda;
    const float *in2 = in1 + lda;
    const float *in3 = in2 + lda;
    float *a_buf_start = a_buf + n / 4 * lda;
    float32x4_t _b = vld1q_f32(B + n);
    float32x4_t _sum0;
    int m = 0;
    for (; m < M - 3; m += 4) {
      float32x4_t _r0 = vld1q_f32(in0 + m);
      float32x4_t _r1 = vld1q_f32(in1 + m);
      float32x4_t _r2 = vld1q_f32(in2 + m);
      float32x4_t _r3 = vld1q_f32(in3 + m);
      float32x4_t _vc = vld1q_f32(C + m);
      //float32x4_t _a_buf = vld1q_f32(a_buf_start + m);

      _sum0 = vmulq_lane_f32(_r0, vget_low_f32(_b), 0);
      _sum0 = vmlaq_lane_f32(_sum0, _r1, vget_low_f32(_b), 1);
      _sum0 = vmlaq_lane_f32(_sum0, _r2, vget_high_f32(_b), 0);
      _sum0 = vmlaq_lane_f32(_sum0, _r3, vget_high_f32(_b), 1);
      _sum0 = vmulq_f32(_sum0, _valpha);
      _sum0 = vaddq_f32(_sum0, _vc);
      //vst1q_f32(C + m, _sum0);
      vst1q_f32(a_buf_start + m, _sum0);
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
          //vst1q_lane_f32(C + m + 2, _sum0, 2);
          vst1q_lane_f32(a_buf_start + m + 2, _sum0, 2);
        case 2:
          //vst1_f32(C + m, vget_low_f32(_sum0));
          vst1_f32(a_buf_start + m, vget_low_f32(_sum0));
          break;
        case 1:
          //vst1q_lane_f32(C + m, _sum0, 0);
          vst1q_lane_f32(a_buf_start + m, _sum0, 0);
          break;
      }
    }
  }
  // remain n
  for (int n = (N & 0xfffffffc); n < N; ++n) {
    const float *in0 = A + n * lda;
    float *a_buf_start = a_buf + n / 4 * lda;
    float32x4_t _b = vld1q_dup_f32(B + n);
    float32x4_t _sum0;
    int m = 0;
    for (; m < M - 3; m += 4) {
      float32x4_t _r0 = vld1q_f32(in0 + m);
      _sum0 = vld1q_f32(C + m);
      _r0 = vmulq_f32(_r0, _b);
      _r0 = vmulq_f32(_valpha, _r0);
      _sum0 = vaddq_f32(_sum0, _r0);
      //vst1q_f32(C + m, _sum0);
      vst1q_f32(a_buf_start + m, _sum0);
    }
    for (; m < M; ++m) {
      //C[m] += alpha * (in0[m] * B[n]);
      *(a_buf_start + m) += alpha * (in0[m] * B[n]);
    }
  }
  // TODO: reduce a_buf
  int m = 0;
  for (; m < M; m += 4) {
    float *a_buf_start = a_buf + m;
    float32x4_t _sum0;
    int n = 0;
    for (; n < a_buf_cols; n += 4) {
      float *a_buf0 = a_buf_start + lda * n;
      float *a_buf1 = a_buf0 + lda;
      float *a_buf2 = a_buf1 + lda;
      float *a_buf3 = a_buf2 + lda;
      float32x4_t a_buf0_v = vld1q_f32(a_buf0);
      float32x4_t a_buf1_v = vld1q_f32(a_buf1);
      float32x4_t a_buf2_v = vld1q_f32(a_buf2);
      float32x4_t a_buf3_v = vld1q_f32(a_buf3);
      _sum0 += a_buf0_v + a_buf1_v + a_buf2_v + a_buf3_v;
    }

    // TODO inner remain
    if (n < a_buf_cols) {
      float *a_buf0 = a_buf_start + lda * n;
      float *a_buf1 = a_buf0 + lda;
      float *a_buf2 = a_buf1 + lda;
      float32x4_t a_buf0_v = vld1q_f32(a_buf0);
      float32x4_t a_buf1_v = vld1q_f32(a_buf1);
      float32x4_t a_buf2_v = vld1q_f32(a_buf2);
      switch ( a_buf_cols - n ) {
        case 3:
          _sum0 += a_buf0_v + a_buf1_v + a_buf2_v;
          break;
        case 2:
          _sum0 += a_buf0_v + a_buf1_v;
          break;
        case 1:
          _sum0 += a_buf0_v;
          break;
      }
    }
    vst1q_f32(C + m, _sum0);
  }
  // TODO: remain of reduce a_buf
  if (m < M) {
    float *a_buf_start = a_buf + m;
    float32x4_t _sum0;
    int n = 0;
    for (; n < a_buf_cols; n += 4) {
      float *a_buf0 = a_buf_start + lda * n;
      float *a_buf1 = a_buf0 + lda;
      float *a_buf2 = a_buf1 + lda;
      float *a_buf3 = a_buf2 + lda;
      float32x4_t a_buf0_v = vld1q_f32(a_buf0);
      float32x4_t a_buf1_v = vld1q_f32(a_buf1);
      float32x4_t a_buf2_v = vld1q_f32(a_buf2);
      float32x4_t a_buf3_v = vld1q_f32(a_buf3);
      _sum0 += a_buf0_v + a_buf1_v + a_buf2_v + a_buf3_v;
    }
    // inner remain
    if (n < a_buf_cols) {
      float *a_buf0 = a_buf_start + lda * n;
      float *a_buf1 = a_buf0 + lda;
      float *a_buf2 = a_buf1 + lda;
      float32x4_t a_buf0_v = vld1q_f32(a_buf0);
      float32x4_t a_buf1_v = vld1q_f32(a_buf1);
      float32x4_t a_buf2_v = vld1q_f32(a_buf2);
      switch ( a_buf_cols - n ) {
        case 3:
          _sum0 += a_buf0_v + a_buf1_v + a_buf2_v;
          break;
        case 2:
          _sum0 += a_buf0_v + a_buf1_v;
          break;
        case 1:
          _sum0 += a_buf0_v;
          break;
      }
    }
    vst1q_f32(C + m, _sum0);
  }
  // TODO: use paddle free
  if (a_buf) free(a_buf); a_buf = nullptr;
}
