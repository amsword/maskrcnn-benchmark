// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCDeviceUtils.cuh>

#include <vector>
#include <iostream>

#include <ctime>
#include <chrono>
using namespace std::chrono;

#define CONF_TO_INT_MULT 1000000
#define CONF_TO_INT_ADD 100000
#define CONF_TO_INT(x) (long long)((x) * CONF_TO_INT_MULT) + CONF_TO_INT_ADD

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)
const int CUDA_NUM_THREADS = 512;

int const threadsPerBlock = sizeof(unsigned long long) * 8;

__device__ inline float devIoU(float const * const a, float const * const b) {
  float left = max(a[0], b[0]), right = min(a[2], b[2]);
  float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}

__global__ void nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const float *dev_boxes, unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_boxes[threadsPerBlock * 5];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 5 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 0];
    block_boxes[threadIdx.x * 5 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 1];
    block_boxes[threadIdx.x * 5 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 2];
    block_boxes[threadIdx.x * 5 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 3];
    block_boxes[threadIdx.x * 5 + 4] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 4];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * 5;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_boxes + i * 5) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = THCCeilDiv(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}


template <typename T>
__global__ void hnms_max_conf_kernel(long long nthreads,
        T* box_confs,
        int64_t* cell_indices,
        int64_t* cell_max_confs) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
      unsigned long long conf = CONF_TO_INT(box_confs[i]);
      unsigned long long cell = cell_indices[i];
      unsigned long long * cell_max = (unsigned long long*)(cell_max_confs + cell);
      // long long type is not supported for atomiMax
      atomicMax(cell_max, conf);
  }
}

template <typename T>
__global__ void hnms_max_idx_kernel(long long nthreads,
        T* box_confs,
        int64_t* cell_indices,
        int64_t* cell_max_confs) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
      unsigned long long conf = CONF_TO_INT(box_confs[i]);
      auto cell = cell_indices[i];
      unsigned long long* cell_max = (unsigned long long*)(cell_max_confs + cell);
      // no implementation to take long long, but unsigned long long
      atomicCAS(cell_max, conf, (unsigned long long)i);
  }
}

// boxes is a N x 5 tensor
at::Tensor nms_cuda(const at::Tensor boxes, float nms_overlap_thresh) {
  using scalar_t = float;
  AT_ASSERTM(boxes.type().is_cuda(), "boxes must be a CUDA tensor");
  auto scores = boxes.select(1, 4);
  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));
  auto boxes_sorted = boxes.index_select(0, order_t);

  int boxes_num = boxes.size(0);

  const int col_blocks = THCCeilDiv(boxes_num, threadsPerBlock);

  scalar_t* boxes_dev = boxes_sorted.data<scalar_t>();

  THCState *state = at::globalContext().lazyInitCUDA(); // TODO replace with getTHCState

  unsigned long long* mask_dev = NULL;
  //THCudaCheck(THCudaMalloc(state, (void**) &mask_dev,
  //                      boxes_num * col_blocks * sizeof(unsigned long long)));

  mask_dev = (unsigned long long*) THCudaMalloc(state, boxes_num * col_blocks * sizeof(unsigned long long));

  dim3 blocks(THCCeilDiv(boxes_num, threadsPerBlock),
              THCCeilDiv(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);
  nms_kernel<<<blocks, threads>>>(boxes_num,
                                  nms_overlap_thresh,
                                  boxes_dev,
                                  mask_dev);

  std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
  THCudaCheck(cudaMemcpy(&mask_host[0],
                        mask_dev,
                        sizeof(unsigned long long) * boxes_num * col_blocks,
                        cudaMemcpyDeviceToHost));

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  at::Tensor keep = at::empty({boxes_num}, boxes.options().dtype(at::kLong).device(at::kCPU));
  int64_t* keep_out = keep.data<int64_t>();

  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out[num_to_keep++] = i;
      unsigned long long *p = &mask_host[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }

  THCudaFree(state, mask_dev);
  // TODO improve this part
  return std::get<0>(order_t.index({
                       keep.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep).to(
                         order_t.device(), keep.scalar_type())
                     }).sort(0, false));
}

template <typename T>
__global__ void hash_rects_kernel(int64_t nthreads,
        T* dets,
        T w0, T h0, T alpha,
        T bx, T by,
        T gamma_ratio,
        bool b_is_relative,
        int64_t* out) {
    CUDA_1D_KERNEL_LOOP(idx_box, nthreads) {
        auto log_w0 = log(w0);
        auto log_h0 = log(h0);
        auto log_alpha = log(alpha);

        auto curr_det = dets + idx_box * 4;
        auto x = curr_det[0];
        auto y = curr_det[1];
        auto w = curr_det[2];
        auto h = curr_det[3];
        auto w0_gamma = w0 * gamma_ratio;
        auto h0_gamma = h0 * gamma_ratio;

        auto i = round((log_w0 - log(w)) / log_alpha);
        auto j = round((log_h0 - log(h)) / log_alpha);
        auto di = w0_gamma / pow(alpha, i);
        auto dj = h0_gamma / pow(alpha, j);

        int64_t qx, qy;
        if (b_is_relative) {
            qx = round(x / di - bx);
            qy = round(y / dj - by);
        } else {
            qx = round(x / di - bx / di);
            qy = round(y / dj - by / dj);
        }
        auto curr_out  = out + 4 * idx_box;
        curr_out[0] = qx;
        curr_out[1] = qy;
        curr_out[2] = i;
        curr_out[3] = j;
    }
}

at::Tensor hash_rects_cuda(const at::Tensor& dets,
               float w0,
               float h0,
               float alpha,
               float gamma,
               float bx,
               float by,
               bool b_is_relative) {
    auto num_box = dets.size(0);
    auto gamma_ratio = (1. - gamma) / (1. + gamma);

    auto result = at::zeros({long(num_box), 4},
            dets.options().dtype(at::kLong));

    AT_DISPATCH_FLOATING_TYPES(dets.type(), "HASH_RECTS", [&] {
            hash_rects_kernel<scalar_t><<<GET_BLOCKS(num_box), CUDA_NUM_THREADS>>>(num_box,
                    dets.data<scalar_t>(),
                    (scalar_t)w0, (scalar_t)h0, (scalar_t)alpha,
                    (scalar_t)bx, (scalar_t)by,
                    gamma_ratio,
                    b_is_relative,
                    result.data<int64_t>());
            });
    return result;
}

__global__ void map_code(int num_box,
        int64_t* codes,
        int64_t* codes_as_one) {
    CUDA_1D_KERNEL_LOOP(idx_box, num_box) {
        auto curr_code = codes + 4 * idx_box;
        auto curr_mapped = codes_as_one + idx_box;
        *curr_mapped = curr_code[0] +
            curr_code[1] * 10000 +
            curr_code[2] * 100000000 +
            curr_code[3] * 1000000000000;
    }
}

at::Tensor get_best_idx_each_code(
        at::Tensor codes,
        const at::Tensor& scores) {
    auto num_box = codes.size(0);
    auto codes_as_one = at::zeros({long(num_box)},
            codes.options().dtype(at::kLong));
    map_code<<<GET_BLOCKS(num_box), CUDA_NUM_THREADS>>>(num_box,
            codes.data<int64_t>(),
            codes_as_one.data<int64_t>());
    THCudaCheck(cudaGetLastError());

    auto unique_result = at::unique_dim(codes_as_one, 0, // dim
            false, true);

    at::Tensor reverse_index = std::get<1>(unique_result);
    auto count = std::get<0>(unique_result).size(0);

    auto result = at::zeros({long(count)},
            codes.options().dtype(at::kLong));

    // get the maximum confidence score for each code with the atomic operation
    // of atomicMax.
    AT_DISPATCH_FLOATING_TYPES(scores.type(), "HNMS_MAX_IDX_KERNEL", [&] {
        hnms_max_conf_kernel<scalar_t><<<GET_BLOCKS(num_box), CUDA_NUM_THREADS>>>(
                num_box,
                scores.data<scalar_t>(),
                reverse_index.data<int64_t>(),
                result.data<int64_t>());
            });
    THCudaCheck(cudaGetLastError());

    AT_DISPATCH_FLOATING_TYPES(scores.type(), "HNMS_MAX_IDX_KERNEL", [&] {
            hnms_max_idx_kernel<scalar_t><<<GET_BLOCKS(num_box), CUDA_NUM_THREADS>>>(
                    num_box,
                    scores.data<scalar_t>(),
                    reverse_index.data_ptr<int64_t>(),
                    result.data<int64_t>());
            // NULL,
            });
    return result;
}

at::Tensor hnms_cuda(const at::Tensor& dets,
               const at::Tensor& scores,
               float w0,
               float h0,
               float alpha,
               float gamma,
               float bx,
               float by,
               bool b_is_relative,
               bool rerank,
               float rerank_iou
               ) {
    AT_ASSERTM(dets.type().is_cuda(), "dets must be a CUDA tensor");
    AT_ASSERTM(scores.type().is_cuda(), "scores must be a CUDA tensor");
    AT_ASSERTM(dets.type() == scores.type(), "dets should have the same type as scores");
    if (dets.numel() == 0) {
      return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
    }

    auto codes = hash_rects_cuda(dets, w0, h0, alpha, gamma, bx, by, b_is_relative);
    AT_ASSERTM(!rerank, "not implemented");
    auto result = get_best_idx_each_code(codes, scores);
    return result;
}

