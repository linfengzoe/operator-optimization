#include <torch/extension.h>
#include <mma.h>
#include "conv2d_fp16.h"

using namespace nvcuda;

// ─── WMMA tile sizes ───────────────────────────────────────────────────────
#define WMMA_M    16
#define WMMA_N    16
#define WMMA_K    16
#define BLOCK_K   32    // 2 warp-rows  per block  (K direction)
#define BLOCK_OW  32    // 2 warp-cols  per block  (OhOw direction)
#define CRS_STEP  16    // inner-loop step over C*R*S

// ─── Forward kernel: Implicit GEMM using Tensor Core (WMMA) ───────────────
// Output [N, K, Oh*Ow] = Weight [K, C*R*S] × im2col-input [C*R*S, Oh*Ow]
// Each block: 128 threads (4 warps, 2×2), computes 32×32 output tile.
__global__ void implgemm_wmma(param_t param)
{
    const int tx      = threadIdx.x;
    const int warp_id = tx / 32;
    const int warp_row = warp_id / 2;   // K direction:    0 or 1
    const int warp_col = warp_id % 2;   // OhOw direction: 0 or 1

    const int block_k    = blockIdx.y * BLOCK_K;
    const int block_ohow = blockIdx.x * BLOCK_OW;
    const int n          = blockIdx.z;

    // smem_w[K_block][CRS_step]:  weight tile  (half)
    // smem_i[CRS_step][OW_block]: im2col tile  (half)
    // smem_o[K_block][OW_block]:  float output (accumulate from WMMA)
    __shared__ __half  smem_w[BLOCK_K ][CRS_STEP + 8];
    __shared__ __half  smem_i[CRS_STEP][BLOCK_OW + 8];
    __shared__ float   smem_o[BLOCK_K ][BLOCK_OW + 8];

    // WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   __half, wmma::row_major>  a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   __half, wmma::row_major>  b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    const int RS       = (int)(param.r * param.s);
    const int CRS      = (int)(param.c * param.r * param.s);
    const int weiKOff  = CRS;
    const int inCOff   = (int)(param.h * param.w);
    const int inNOff   = (int)(param.c * param.h * param.w);

    // ── Main loop: iterate over C*R*S in steps of CRS_STEP ─────────────────
    for (int crs0 = 0; crs0 < CRS; crs0 += CRS_STEP)
    {
        // ── Load smem_w[BLOCK_K][CRS_STEP]: 32×16 = 512 halves ─────────────
        // 128 threads × 4 values each = 512
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int flat    = tx * 4 + i;
            int kl      = flat / CRS_STEP;
            int crs_loc = flat % CRS_STEP;
            int k       = block_k + kl;
            int crs     = crs0   + crs_loc;
            __half val  = __float2half(0.0f);
            if (k < (int)param.k && crs < CRS) {
                int c = crs / RS;
                int r = (crs % RS) / (int)param.s;
                int s = crs % (int)param.s;
                val = param.weight[k * weiKOff + c * RS + r * (int)param.s + s];
            }
            smem_w[kl][crs_loc] = val;
        }

        // ── Load smem_i[CRS_STEP][BLOCK_OW]: 16×32 = 512 halves ───────────
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int flat     = tx * 4 + i;
            int crs_loc  = flat / BLOCK_OW;
            int ohow_loc = flat % BLOCK_OW;
            int crs      = crs0        + crs_loc;
            int ohow     = block_ohow  + ohow_loc;
            __half val   = __float2half(0.0f);
            if (ohow < (int)(param.Oh * param.Ow) && crs < CRS) {
                int c  = crs / RS;
                int r  = (crs % RS) / (int)param.s;
                int s  = crs % (int)param.s;
                int oh = ohow / (int)param.Ow;
                int ow = ohow % (int)param.Ow;
                int h  = oh * (int)param.u - (int)param.p + r;
                int w  = ow * (int)param.v - (int)param.q + s;
                if (h >= 0 && w >= 0 && h < (int)param.h && w < (int)param.w)
                    val = param.input[n * inNOff + c * inCOff + h * (int)param.w + w];
            }
            smem_i[crs_loc][ohow_loc] = val;
        }

        __syncthreads();

        // ── WMMA compute: c_frag += A[warp_row*16:+16, :] × B[:, warp_col*16:+16]
        wmma::load_matrix_sync(a_frag,
            (const __half*)&smem_w[warp_row * WMMA_M][0], CRS_STEP + 8);
        wmma::load_matrix_sync(b_frag,
            (const __half*)&smem_i[0][warp_col * WMMA_N], BLOCK_OW + 8);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    // ── Store c_frag to smem_o ─────────────────────────────────────────────
    wmma::store_matrix_sync(
        &smem_o[warp_row * WMMA_M][warp_col * WMMA_N],
        c_frag, BLOCK_OW + 8, wmma::mem_row_major);
    __syncthreads();

    // ── Write smem_o → global output (float → half) ────────────────────────
    // smem_o is BLOCK_K × BLOCK_OW = 32×32 = 1024 values
    // 128 threads × 8 = 1024 → each thread writes 8 values
    const int outNOff = (int)param.k * (int)(param.Oh * param.Ow);
    const int outKOff = (int)(param.Oh * param.Ow);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int flat     = tx * 8 + i;
        int kl       = flat / BLOCK_OW;
        int ohow_loc = flat % BLOCK_OW;
        int k        = block_k    + kl;
        int ohow     = block_ohow + ohow_loc;
        if (k < (int)param.k && ohow < outKOff)
            param.output[n * outNOff + k * outKOff + ohow] =
                __float2half(smem_o[kl][ohow_loc]);
    }
}

void conv2d_cuda_forward(param_t param)
{
    dim3 block(128, 1, 1);
    dim3 grid(
        ((int)(param.Oh * param.Ow) + BLOCK_OW - 1) / BLOCK_OW,
        ((int)param.k               + BLOCK_K  - 1) / BLOCK_K,
        (int)param.n);
    implgemm_wmma<<<grid, block>>>(param);
}

// ─── Backward (reuse FP16 scalar impl; inference only, never called) ──────
__global__ void implgemmbwddata(param_t param)
{
    uint32_t tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    const uint32_t lane_id = tx % 32, warp_id = tx / 32;
    const uint32_t mma_tid_x = (lane_id / 2) % 8;
    const uint32_t mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);
    uint32_t wl = (warp_id / 2) * 32 + mma_tid_y * 4;
    uint32_t il = (warp_id % 2) * 64 + mma_tid_x * 4;
    int x = bx * 128 + il, y = by * 128 + wl, z = blockIdx.z;
    __shared__ DTYPE sw[8 * 132], si[8 * 128];
    uint32_t ws = (tx % 8) * 132 + (tx / 8) * 4;
    uint32_t is_ = (tx / 32) * 128 + (tx % 32);
    int posH[4], posW[4];
    for (int i = 0; i < 4; ++i) {
        int hw = bx * 128 + tx % 32 + i * 32;
        posH[i] = hw / (int)param.w; posW[i] = hw % (int)param.w;
    }
    int outK = by * 128 + (tx / 8) * 4;
    int RS = param.r * param.s, CRS = param.c * RS;
    int kOhOw = param.k * param.Oh * param.Ow;
    DTYPE frag[8][8];
    for (int i = 0; i < 8; ++i) for (int j = 0; j < 8; ++j) frag[i][j] = __float2half(0.0f);
    for (int kk = 0; kk < (int)param.k; kk += 8) {
        int cur = kk + tx % 8;
        int cc = cur / RS, rr = (cur % RS) / (int)param.s, ss = cur % (int)param.s;
        DTYPE wv[4];
        for (int i = 0; i < 4; ++i) {
            int k = outK + i;
            wv[i] = (k < (int)param.k && cur < CRS) ?
                param.weight[k * CRS + cur] : __float2half(0.0f);
        }
        int cur2 = kk + (int)(tx / 32);
        DTYPE iv[4];
        for (int i = 0; i < 4; ++i) {
            int h = posH[i] + rr, w_ = posW[i] + ss;
            iv[i] = (h >= 0 && w_ >= 0 && h < (int)param.Oh && w_ < (int)param.Ow && cur2 < CRS) ?
                param.grad_output[z * kOhOw + (kk + (int)(tx/32)) * (int)(param.Oh * param.Ow) + posH[i] * (int)param.Ow + posW[i]]
                : __float2half(0.0f);
        }
        for (int i = 0; i < 4; ++i) { sw[ws + i] = wv[i]; si[is_ + i * 32] = iv[i]; }
        __syncthreads();
        DTYPE wf[8], inf[8];
        for (int sc = 0; sc < 8; ++sc) {
            for (int i = 0; i < 4; ++i) {
                wf[i]   = sw[wl + sc * 132 + i];
                wf[i+4] = sw[wl + sc * 132 + i + 16];
                inf[i]   = si[il + sc * 128 + i];
                inf[i+4] = si[il + sc * 128 + i + 32];
            }
            for (int i = 0; i < 8; ++i) for (int j = 0; j < 8; ++j)
                frag[i][j] = __hadd(frag[i][j], __hmul(wf[i], inf[j]));
        }
        __syncthreads();
    }
    int hOff = param.c * param.h * param.w, cOff = param.h * param.w;
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) {
        auto wr = [&](int yi, int xi) {
            if (xi < (int)(param.h * param.w) && yi < (int)param.c)
                param.grad_input[z * hOff + yi * cOff + xi] = frag[i][j];
        };
        wr(y+i, x+j); wr(y+i, x+j+32); wr(y+i+16, x+j); wr(y+i+16, x+j+32);
    }
}

__global__ void implgemmbwdweight(param_t param)
{
    // Minimal stub — not used during inference
}

void conv2d_cuda_backward(param_t param)
{
    // Inference-only operator; backward not needed
    (void)param;
}
