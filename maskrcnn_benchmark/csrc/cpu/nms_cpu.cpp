// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "cpu/vision.h"
#include <tgmath.h> // log
#include <torch/extension.h>
#include <torch/csrc/autograd/variable.h>
#include <map>
#include <vector>
#include <ctime>
#include <chrono>
using namespace std::chrono;

template <typename scalar_t>
at::Tensor nms_cpu_kernel_max(const at::Tensor& dets,
                          const at::Tensor& scores,
                          const float threshold,
                          int max_box) {
  AT_ASSERTM(!dets.type().is_cuda(), "dets must be a CPU tensor");
  AT_ASSERTM(!scores.type().is_cuda(), "scores must be a CPU tensor");
  AT_ASSERTM(dets.type() == scores.type(), "dets should have the same type as scores");
  AT_ASSERTM(max_box > 0, "max_box should be larger than 0");

  if (dets.numel() == 0) {
    return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
  }

  auto x1_t = dets.select(1, 0).contiguous();
  auto y1_t = dets.select(1, 1).contiguous();
  auto x2_t = dets.select(1, 2).contiguous();
  auto y2_t = dets.select(1, 3).contiguous();

  at::Tensor areas_t = (x2_t - x1_t + 1) * (y2_t - y1_t + 1);

  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));

  auto ndets = dets.size(0);
  at::Tensor suppressed_t = at::zeros({ndets}, dets.options().dtype(at::kByte).device(at::kCPU));

  auto suppressed = suppressed_t.data<uint8_t>();
  auto order = order_t.data<int64_t>();
  auto x1 = x1_t.data<scalar_t>();
  auto y1 = y1_t.data<scalar_t>();
  auto x2 = x2_t.data<scalar_t>();
  auto y2 = y2_t.data<scalar_t>();
  auto areas = areas_t.data<scalar_t>();

  for (int64_t _i = 0; _i < ndets; _i++) {
    auto i = order[_i];
    if (suppressed[i] == 1)
      continue;
    auto ix1 = x1[i];
    auto iy1 = y1[i];
    auto ix2 = x2[i];
    auto iy2 = y2[i];
    auto iarea = areas[i];

    max_box--;
    if (max_box <= 0) {
        for (int64_t _j = _i + 1; _j < ndets; _j++) {
            auto j = order[_j];
            suppressed[j] = 1;
        }
        break;
    }

    for (int64_t _j = _i + 1; _j < ndets; _j++) {
      auto j = order[_j];
      if (suppressed[j] == 1)
        continue;
      auto xx1 = std::max(ix1, x1[j]);
      auto yy1 = std::max(iy1, y1[j]);
      auto xx2 = std::min(ix2, x2[j]);
      auto yy2 = std::min(iy2, y2[j]);

      auto w = std::max(static_cast<scalar_t>(0), xx2 - xx1 + 1);
      auto h = std::max(static_cast<scalar_t>(0), yy2 - yy1 + 1);
      auto inter = w * h;
      auto ovr = inter / (iarea + areas[j] - inter);
      if (ovr >= threshold) {
        suppressed[j] = 1;
      }
   }
  }
  return at::nonzero(suppressed_t == 0).squeeze(1);
}


template <typename scalar_t>
at::Tensor nms_cpu_kernel(const at::Tensor& dets,
                          const at::Tensor& scores,
                          const float threshold) {
  AT_ASSERTM(!dets.type().is_cuda(), "dets must be a CPU tensor");
  AT_ASSERTM(!scores.type().is_cuda(), "scores must be a CPU tensor");
  AT_ASSERTM(dets.type() == scores.type(), "dets should have the same type as scores");

  if (dets.numel() == 0) {
    return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
  }

  auto x1_t = dets.select(1, 0).contiguous();
  auto y1_t = dets.select(1, 1).contiguous();
  auto x2_t = dets.select(1, 2).contiguous();
  auto y2_t = dets.select(1, 3).contiguous();

  at::Tensor areas_t = (x2_t - x1_t + 1) * (y2_t - y1_t + 1);

  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));

  auto ndets = dets.size(0);
  at::Tensor suppressed_t = at::zeros({ndets}, dets.options().dtype(at::kByte).device(at::kCPU));

  auto suppressed = suppressed_t.data<uint8_t>();
  auto order = order_t.data<int64_t>();
  auto x1 = x1_t.data<scalar_t>();
  auto y1 = y1_t.data<scalar_t>();
  auto x2 = x2_t.data<scalar_t>();
  auto y2 = y2_t.data<scalar_t>();
  auto areas = areas_t.data<scalar_t>();

  for (int64_t _i = 0; _i < ndets; _i++) {
    auto i = order[_i];
    if (suppressed[i] == 1)
      continue;
    auto ix1 = x1[i];
    auto iy1 = y1[i];
    auto ix2 = x2[i];
    auto iy2 = y2[i];
    auto iarea = areas[i];

    for (int64_t _j = _i + 1; _j < ndets; _j++) {
      auto j = order[_j];
      if (suppressed[j] == 1)
        continue;
      auto xx1 = std::max(ix1, x1[j]);
      auto yy1 = std::max(iy1, y1[j]);
      auto xx2 = std::min(ix2, x2[j]);
      auto yy2 = std::min(iy2, y2[j]);

      auto w = std::max(static_cast<scalar_t>(0), xx2 - xx1 + 1);
      auto h = std::max(static_cast<scalar_t>(0), yy2 - yy1 + 1);
      auto inter = w * h;
      auto ovr = inter / (iarea + areas[j] - inter);
      if (ovr >= threshold)
        suppressed[j] = 1;
   }
  }
  return at::nonzero(suppressed_t == 0).squeeze(1);
}

at::Tensor nms_cpu(const at::Tensor& dets,
               const at::Tensor& scores,
               const float threshold) {
  at::Tensor result;
  AT_DISPATCH_FLOATING_TYPES(dets.type(), "nms", [&] {
    result = nms_cpu_kernel<scalar_t>(dets, scores, threshold);
  });
  return result;
}

at::Tensor nms_cpu_max(const at::Tensor& dets,
               const at::Tensor& scores,
               const float threshold,
               const int max_box) {
  at::Tensor result;
  AT_DISPATCH_FLOATING_TYPES(dets.type(), "nms", [&] {
    result = nms_cpu_kernel_max<scalar_t>(dets, scores, threshold, max_box);
  });
  return result;
}

at::Tensor hash_rects2(const at::Tensor& dets,
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

    auto pdets = dets.data<float>();
    auto presult = result.data<int64_t>();
    auto log_w0 = log(w0);
    auto log_h0 = log(h0);
    auto log_alpha = log(alpha);

    auto w0_gamma = w0 * gamma_ratio;
    auto h0_gamma = h0 * gamma_ratio;

    for (auto idx_box = 0; idx_box < num_box; idx_box++) {
        auto curr_det = pdets + idx_box * 4;
        auto x = curr_det[0];
        auto y = curr_det[1];
        auto w = curr_det[2];
        auto h = curr_det[3];

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
        auto curr_out  = presult + 4 * idx_box;
        curr_out[0] = qx;
        curr_out[1] = qy;
        curr_out[2] = i;
        curr_out[3] = j;
    }

    return result;

}

at::Tensor hash_rects(const at::Tensor& dets,
               float w0,
               float h0,
               float alpha,
               float gamma,
               float bx,
               float by,
               bool b_is_relative) {
    auto log_w0 = log(w0);
    auto log_h0 = log(h0);
    auto log_alpha = log(alpha);

    // map the rects to the code
    auto x = dets.select(1, 0).contiguous();
    auto y = dets.select(1, 1).contiguous();
    auto w = dets.select(1, 2).contiguous();
    auto h = dets.select(1, 3).contiguous();
    auto gamma_ratio = (1. - gamma) / (1. + gamma);
    auto w0_gamma = w0 * gamma_ratio;
    auto h0_gamma = h0 * gamma_ratio;

    auto i = at::round((log_w0 - at::log(w)) / log_alpha);
    auto j = at::round((log_h0 - at::log(h)) / log_alpha);

    //auto start = high_resolution_clock::now();
    auto di = w0_gamma / at::pow(alpha, i);
    auto dj = h0_gamma / at::pow(alpha, j);
    //auto stop = high_resolution_clock::now();
    //auto duration = duration_cast<microseconds>(stop - start);
    //std::cout << "pow: " << duration.count() << std::endl;

    at::Tensor qx, qy;
    if (b_is_relative) {
        qx = at::round(x / di - bx);
        qy = at::round(y / dj - by);
    } else {
        qx = at::round(x / di - bx / di);
        qy = at::round(y / dj - by / dj);
    }
    auto result = at::stack({qx, qy, i, j}, 1);
    return at::_cast_Long(result).contiguous();
}

//typedef std::tuple<long, long, long, long> TCode;
//TCode get_code(const long* p_code) {
    //return TCode(p_code[0], p_code[1], p_code[2], p_code[3]);
//}

typedef long TCode;
TCode get_code(const long* p_code) {
    return p_code[0] + p_code[1] * 10000 +
        p_code[2] * 100000000 + p_code[3] * 1000000000000;
}

at::Tensor get_best_score_each_code(
        at::Tensor codes,
        const at::Tensor& scores) {
    std::map<TCode, long> code_to_idx;

    auto p_code = codes.data<long>();
    auto p_score = scores.data<float>();

    auto ndets = codes.size(0);
    for (auto i = 0; i < ndets; i++) {
        auto code = get_code(p_code);
        if (code_to_idx.count(code) == 0) {
            code_to_idx[code] = i;
        } else {
            auto &pre_idx = code_to_idx[code];
            if (p_score[pre_idx] < p_score[i]) {
                pre_idx = i;
            }
        }
        p_code += 4;
    }

    at::Tensor result = at::ones({long(code_to_idx.size())},
            scores.options().dtype(at::kLong).device(at::kCPU));
    auto p = result.data<long>();
    int idx = 0;
    for (auto i = code_to_idx.begin(); i != code_to_idx.end(); i++) {
        p[idx++] = i->second;
    }

    return result;
}

at::Tensor nms_idx(
        const at::Tensor &idxs_t,
        const at::Tensor &scores_t,
        const at::Tensor &dets_t,
        const at::Tensor &areas_t,
        float threshold
        ) {
    auto ndets = idxs_t.size(0);
    auto order_t = std::get<1>(scores_t.index(torch::autograd::make_variable(idxs_t)).sort(0, true));
    at::Tensor suppressed_t = at::zeros({ndets},
            scores_t.options().dtype(at::kByte).device(at::kCPU));
    auto suppressed = suppressed_t.data<uint8_t>();
    auto order = order_t.data<int64_t>();

    auto dets = dets_t.data<float>();
    auto areas = areas_t.data<float>();
    auto idxs = idxs_t.data<long>();
    for (auto _i = 0; _i < ndets; _i++) {
        auto i = order[_i];
        if (suppressed[i] == 1)
            continue;
        auto pi = dets + idxs[i] * 4;
        auto ix1 = pi[0] - pi[2] / 2.;
        auto iy1 = pi[1] - pi[3] / 2.;
        auto ix2 = pi[0] + pi[2] / 2.;
        auto iy2 = pi[1] + pi[3] / 2.;

        auto iarea = areas[idxs[i]];

        for (int64_t _j = _i + 1; _j < ndets; _j++) {
            auto j = order[_j];
            if (suppressed[j] == 1)
                continue;
            auto pj = dets + idxs[j] * 4;
            auto xx1 = std::max(ix1, pj[0] - pj[2] / 2.);
            auto yy1 = std::max(iy1, pj[1] - pj[3] / 2.);
            auto xx2 = std::min(ix2, pj[0] + pj[2] / 2.);
            auto yy2 = std::min(iy2, pj[1] + pj[3] / 2.);

            auto w = std::max(0., xx2 - xx1);
            auto h = std::max(0., yy2 - yy1);
            auto inter = w * h;
            auto ovr = inter / (iarea + areas[idxs[j]] - inter);
            if (ovr >= threshold)
                suppressed[j] = 1;
        }
    }
    return suppressed_t;
}

at::Tensor get_reranking_each_code(
        const at::Tensor &codes_t,
        const at::Tensor &scores_t,
        const at::Tensor &dets_t,
        float rerank_iou) {
    std::map<TCode, std::vector<long>> code_to_idxs;

    auto p_code = codes_t.data<long>();

    auto ndets = codes_t.size(0);
    for (auto i = 0; i < ndets; i++) {
        auto code = get_code(p_code);
        if (code_to_idxs.count(code) == 0) {
            code_to_idxs[code] = {i};
        } else {
            auto &pre_idx = code_to_idxs[code];
            pre_idx.push_back(i);
        }
        p_code += 4;
    }

    auto areas_t = dets_t.select(1, 2) * dets_t.select(1, 3);
    at::Tensor suppressed_t = at::zeros({codes_t.size(0)},
            codes_t.options().dtype(at::kByte).device(at::kCPU));
    auto suppressed = suppressed_t.data<uint8_t>();
    for (auto i = code_to_idxs.begin(); i != code_to_idxs.end(); i++) {
        auto &idxs = i->second;
        if (idxs.size() == 1) {
            continue;
        }
        auto idxs_t = at::from_blob(idxs.data(), {long(idxs.size())}, at::kLong);
        auto sub_suppressed_t = nms_idx(idxs_t, scores_t, dets_t, areas_t, rerank_iou);
        auto sub_suppressed = sub_suppressed_t.data<uint8_t>();

        for (auto j = 0; j < idxs.size(); j++) {
            suppressed[idxs[j]] = sub_suppressed[j];
        }
    }

    return at::nonzero(suppressed_t == 0).squeeze(1);
}

at::Tensor hnms_cpu(const at::Tensor& dets,
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
    AT_ASSERTM(!dets.type().is_cuda(), "dets must be a CPU tensor");
    AT_ASSERTM(!scores.type().is_cuda(), "scores must be a CPU tensor");
    AT_ASSERTM(dets.type() == scores.type(), "dets should have the same type as scores");
    if (dets.numel() == 0) {
      return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
    }

    auto codes = hash_rects(dets, w0, h0, alpha, gamma, bx, by, b_is_relative);

    // teh following implementation is slower
    //auto codes = hash_rects2(dets, w0, h0, alpha, gamma, bx, by, b_is_relative);

    if (!rerank) {
        auto result = get_best_score_each_code(codes, scores);
        return result;
    } else {
        auto result = get_reranking_each_code(codes, scores, dets,
                rerank_iou);
        return result;
    }
}
