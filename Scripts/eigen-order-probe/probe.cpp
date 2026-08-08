// Which floating-point reduction order does a2_fast's C=8 path actually use?
//
// a2_fast branches on channel count. The C=3 branch is hand-written scalar, so
// the slimmed-path lab could match it bit-for-bit just by writing the same
// chain. The C=8 branch is Eigen (a2_fast.cpp:558-606), and a hand-written NEON
// replacement can only be bit-identical to it if Eigen's reduction order is
// knowable and reproducible.
//
// This program answers that question before any kernel is written. It replicates
// a2_fast's exact expressions on its exact shapes, then compares the result
// bit-for-bit against several candidate hand-written orderings. Nothing here is
// a tolerance check: two float arrays either have identical bit patterns or they
// do not.
//
// Build and run:  ./Scripts/eigen-order-probe/run.sh

#include <Eigen/Dense>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace
{

// The A2 shape, read from the capture rather than assumed. See nam-files/README.md.
constexpr int C = 8; // channels == bottleneck
constexpr float kLeakySlope = 0.01f;
constexpr int kHeadKernelSize = 16;

// Buffers are column-major with channels contiguous: x[f * C + c].
using MatCC = Eigen::Matrix<float, C, C>;
using MatCDyn = Eigen::Matrix<float, C, Eigen::Dynamic>;
using VecC = Eigen::Matrix<float, C, 1>;
using RowDyn = Eigen::Matrix<float, 1, Eigen::Dynamic>;

bool bit_equal(const std::vector<float>& a, const std::vector<float>& b)
{
  if (a.size() != b.size())
    return false;
  return std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

/// How far apart two arrays are, in ULPs of the largest disagreement, plus the
/// count of differing elements. Reported for near-misses so a failing candidate
/// says *how* it failed rather than just "no".
struct Divergence
{
  size_t differing = 0;
  int max_ulps = 0;
  float max_abs = 0.0f;
};

Divergence divergence(const std::vector<float>& a, const std::vector<float>& b)
{
  Divergence d;
  for (size_t i = 0; i < a.size(); i++)
  {
    if (a[i] == b[i] && std::signbit(a[i]) == std::signbit(b[i]))
      continue;
    d.differing++;
    int32_t ia, ib;
    std::memcpy(&ia, &a[i], 4);
    std::memcpy(&ib, &b[i], 4);
    if (ia < 0)
      ia = INT32_MIN - ia;
    if (ib < 0)
      ib = INT32_MIN - ib;
    const int ulps = static_cast<int>(std::abs(static_cast<long>(ia) - static_cast<long>(ib)));
    if (ulps > d.max_ulps)
      d.max_ulps = ulps;
    const float diff = std::fabs(a[i] - b[i]);
    if (diff > d.max_abs)
      d.max_abs = diff;
  }
  return d;
}

// -----------------------------------------------------------------------------
// The reference: a2_fast's C=8 layer body, copied verbatim from
// vendor/upstream/NAM/wavenet/a2_fast.cpp:573-606.
// -----------------------------------------------------------------------------

struct LayerInputs
{
  int K = 0;
  std::vector<std::vector<float>> conv_w; // K taps, each C*C column-major
  std::vector<const float*> tap_src; // K pointers into history, each C x N col-major
  std::vector<float> conv_b;
  std::vector<float> mixin_w;
  std::vector<float> l1x1_w;
  std::vector<float> l1x1_b;
  const float* cond = nullptr;
};

/// Stage 1 only: the K conv GEMMs, before bias. This is the stage whose order is
/// in question; everything after it is elementwise or a second GEMM of the same
/// shape.
void ref_conv(const LayerInputs& in, std::vector<float>& z, int N)
{
  Eigen::Map<MatCDyn> ztile(z.data(), C, N);
  ztile.setZero();
  for (int k = 0; k < in.K; k++)
  {
    Eigen::Map<const MatCC> W(in.conv_w[k].data());
    Eigen::Map<const MatCDyn> input_block(in.tap_src[k], C, N);
    ztile.noalias() += W * input_block;
  }
}

/// The whole layer body, exactly as a2_fast runs it.
void ref_layer(const LayerInputs& in, std::vector<float>& z, std::vector<float>& head_sum,
               std::vector<float>& layer_in, int N)
{
  Eigen::Map<const VecC> conv_b_vec(in.conv_b.data());
  Eigen::Map<const VecC> mixin_vec(in.mixin_w.data());
  Eigen::Map<const MatCC> l1x1_mat(in.l1x1_w.data());
  Eigen::Map<const VecC> l1x1_b_vec(in.l1x1_b.data());
  Eigen::Map<const RowDyn> cond_row(in.cond, 1, N);

  Eigen::Map<MatCDyn> ztile(z.data(), C, N);
  Eigen::Map<MatCDyn> hsum_block(head_sum.data(), C, N);
  Eigen::Map<MatCDyn> lin_block(layer_in.data(), C, N);

  ztile.setZero();

  for (int k = 0; k < in.K; k++)
  {
    Eigen::Map<const MatCC> W(in.conv_w[k].data());
    Eigen::Map<const MatCDyn> input_block(in.tap_src[k], C, N);
    ztile.noalias() += W * input_block;
  }

  ztile.colwise() += conv_b_vec;
  ztile.noalias() += mixin_vec * cond_row;
  ztile = (ztile.array() < 0.0f).select(ztile.array() * kLeakySlope, ztile.array());
  hsum_block += ztile;
  lin_block.noalias() += l1x1_mat * ztile;
  lin_block.colwise() += l1x1_b_vec;
}

// -----------------------------------------------------------------------------
// Candidate hand-written orderings.
//
// Every candidate below turns contraction off and uses std::fmaf explicitly
// where a fused multiply-add is intended, so the ordering under test is the one
// written rather than whatever the optimiser chose.
// -----------------------------------------------------------------------------

// A: per tap, an inner sum over j from zero in increasing order with FMA; the
//    taps' partials then added into z in increasing k. The hypothesis.
#pragma clang fp contract(off)
void cand_per_tap_fma(const LayerInputs& in, std::vector<float>& z, int N)
{
  for (int f = 0; f < N; f++)
  {
    for (int i = 0; i < C; i++)
    {
      float acc = 0.0f;
      for (int k = 0; k < in.K; k++)
      {
        const float* W = in.conv_w[k].data();
        const float* x = in.tap_src[k] + static_cast<size_t>(f) * C;
        float t = 0.0f;
        for (int j = 0; j < C; j++)
          t = std::fmaf(W[j * C + i], x[j], t);
        acc = acc + t;
      }
      z[static_cast<size_t>(f) * C + i] = acc;
    }
  }
}

// B: as A, but the inner sum is a separate multiply and add (no contraction).
void cand_per_tap_nofma(const LayerInputs& in, std::vector<float>& z, int N)
{
  for (int f = 0; f < N; f++)
  {
    for (int i = 0; i < C; i++)
    {
      float acc = 0.0f;
      for (int k = 0; k < in.K; k++)
      {
        const float* W = in.conv_w[k].data();
        const float* x = in.tap_src[k] + static_cast<size_t>(f) * C;
        float t = 0.0f;
        for (int j = 0; j < C; j++)
          t = t + W[j * C + i] * x[j];
        acc = acc + t;
      }
      z[static_cast<size_t>(f) * C + i] = acc;
    }
  }
}

// C: one chain across every tap and channel, no per-tap partial. This is what
//    `fused` does (modulo seeding with bias), and is expected NOT to match.
void cand_single_chain(const LayerInputs& in, std::vector<float>& z, int N)
{
  for (int f = 0; f < N; f++)
  {
    for (int i = 0; i < C; i++)
    {
      float acc = 0.0f;
      for (int k = 0; k < in.K; k++)
      {
        const float* W = in.conv_w[k].data();
        const float* x = in.tap_src[k] + static_cast<size_t>(f) * C;
        for (int j = 0; j < C; j++)
          acc = std::fmaf(W[j * C + i], x[j], acc);
      }
      z[static_cast<size_t>(f) * C + i] = acc;
    }
  }
}

// D: per tap, the depth-8 inner sum split into two independent halves that are
//    added at the end. What a microkernel with two accumulators would produce.
void cand_per_tap_split2(const LayerInputs& in, std::vector<float>& z, int N)
{
  for (int f = 0; f < N; f++)
  {
    for (int i = 0; i < C; i++)
    {
      float acc = 0.0f;
      for (int k = 0; k < in.K; k++)
      {
        const float* W = in.conv_w[k].data();
        const float* x = in.tap_src[k] + static_cast<size_t>(f) * C;
        float t0 = 0.0f, t1 = 0.0f;
        for (int j = 0; j < C / 2; j++)
          t0 = std::fmaf(W[j * C + i], x[j], t0);
        for (int j = C / 2; j < C; j++)
          t1 = std::fmaf(W[j * C + i], x[j], t1);
        acc = acc + (t0 + t1);
      }
      z[static_cast<size_t>(f) * C + i] = acc;
    }
  }
}

// E: per tap, inner sum in decreasing j.
void cand_per_tap_reverse(const LayerInputs& in, std::vector<float>& z, int N)
{
  for (int f = 0; f < N; f++)
  {
    for (int i = 0; i < C; i++)
    {
      float acc = 0.0f;
      for (int k = 0; k < in.K; k++)
      {
        const float* W = in.conv_w[k].data();
        const float* x = in.tap_src[k] + static_cast<size_t>(f) * C;
        float t = 0.0f;
        for (int j = C - 1; j >= 0; j--)
          t = std::fmaf(W[j * C + i], x[j], t);
        acc = acc + t;
      }
      z[static_cast<size_t>(f) * C + i] = acc;
    }
  }
}

/// The whole layer body under candidate A's conv ordering, with every
/// post-conv step written the way a2_fast's Eigen expressions round: bias added
/// after the taps, mixin as a product then a separate add, layer1x1 accumulated
/// from zero and added to the existing residual before its own bias.
void cand_layer(const LayerInputs& in, std::vector<float>& z, std::vector<float>& head_sum,
                std::vector<float>& layer_in, int N)
{
  cand_per_tap_fma(in, z, N);

  for (int f = 0; f < N; f++)
  {
    float* zc = &z[static_cast<size_t>(f) * C];
    for (int i = 0; i < C; i++)
    {
      float v = zc[i] + in.conv_b[i];
      v = v + in.mixin_w[i] * in.cond[f]; // product then add: two roundings
      zc[i] = (v < 0.0f) ? v * kLeakySlope : v;
    }
  }

  for (int f = 0; f < N; f++)
  {
    const float* zc = &z[static_cast<size_t>(f) * C];
    float* hs = &head_sum[static_cast<size_t>(f) * C];
    float* lin = &layer_in[static_cast<size_t>(f) * C];
    for (int i = 0; i < C; i++)
      hs[i] = hs[i] + zc[i];
    for (int i = 0; i < C; i++)
    {
      float t = 0.0f;
      for (int j = 0; j < C; j++)
        t = std::fmaf(in.l1x1_w[j * C + i], zc[j], t);
      lin[i] = (lin[i] + t) + in.l1x1_b[i];
    }
  }
}

// -----------------------------------------------------------------------------
// The head. a2_fast's is plainly scalar (a2_fast.cpp:640-652): `y += wk[b] *
// src[b]`, one statement, so clang's default -ffp-contract=on turns it into an
// FMA. Two things need checking:
//
//   1. that the contraction really matters (it does, and a kernel written with
//      a separate multiply and add would not be bit-identical), and
//   2. that vectorising the chain across *frames* — four lanes each running the
//      identical scalar sequence — preserves the bits.
// -----------------------------------------------------------------------------

/// What a2_fast compiles to: sequential FMA from the bias, k outer, channel inner.
void ref_head_fma(const std::vector<std::vector<float>>& head_w, float head_b, float head_scale,
                  const std::vector<float>& history, int base, std::vector<float>& out, int N)
{
  for (int f = 0; f < N; f++)
  {
    float y = head_b;
    for (int k = 0; k < kHeadKernelSize; k++)
    {
      const float* src = &history[static_cast<size_t>(base + f + k) * C];
      const float* wk = head_w[k].data();
      for (int b = 0; b < C; b++)
        y = std::fmaf(wk[b], src[b], y);
    }
    out[f] = y * head_scale;
  }
}

/// The same order with the multiply and the add rounded separately, to show
/// that the contraction is load-bearing.
void ref_head_nofma(const std::vector<std::vector<float>>& head_w, float head_b, float head_scale,
                    const std::vector<float>& history, int base, std::vector<float>& out, int N)
{
  for (int f = 0; f < N; f++)
  {
    float y = head_b;
    for (int k = 0; k < kHeadKernelSize; k++)
    {
      const float* src = &history[static_cast<size_t>(base + f + k) * C];
      const float* wk = head_w[k].data();
      for (int b = 0; b < C; b++)
        y = y + wk[b] * src[b];
    }
    out[f] = y * head_scale;
  }
}

/// Four frames at a time, from planar history planes. Each lane executes exactly
/// ref_head_fma's sequence — this is the shape a NEON kernel would use, where
/// the vector spans frames and the weight is the scalar.
void cand_head_planar(const std::vector<std::vector<float>>& head_w, float head_b, float head_scale,
                      const std::vector<std::vector<float>>& planes, int base, std::vector<float>& out, int N)
{
  int f = 0;
  for (; f + 4 <= N; f += 4)
  {
    float y[4] = {head_b, head_b, head_b, head_b};
    for (int k = 0; k < kHeadKernelSize; k++)
    {
      const float* wk = head_w[k].data();
      for (int b = 0; b < C; b++)
      {
        const float* p = planes[b].data() + base + f + k;
        for (int lane = 0; lane < 4; lane++)
          y[lane] = std::fmaf(wk[b], p[lane], y[lane]);
      }
    }
    for (int lane = 0; lane < 4; lane++)
      out[f + lane] = y[lane] * head_scale;
  }
  for (; f < N; f++)
  {
    float y = head_b;
    for (int k = 0; k < kHeadKernelSize; k++)
    {
      const float* wk = head_w[k].data();
      for (int b = 0; b < C; b++)
        y = std::fmaf(wk[b], planes[b][static_cast<size_t>(base + f + k)], y);
    }
    out[f] = y * head_scale;
  }
}
#pragma clang fp contract(on)

// -----------------------------------------------------------------------------

struct Candidate
{
  const char* name;
  void (*run)(const LayerInputs&, std::vector<float>&, int);
};

const Candidate kCandidates[] = {
  {"A  per-tap partial, increasing j, FMA", &cand_per_tap_fma},
  {"B  per-tap partial, increasing j, mul+add", &cand_per_tap_nofma},
  {"C  single chain across taps, FMA (fused's shape)", &cand_single_chain},
  {"D  per-tap partial split into two halves", &cand_per_tap_split2},
  {"E  per-tap partial, decreasing j", &cand_per_tap_reverse},
};

} // namespace

int main()
{
  // The A2 kernel sizes, and the block sizes the benchmark actually uses. 32 is
  // both a sweep point and the tail block a 523,808-frame file produces at
  // --block-size 64 (523808 = 64 * 8184 + 32).
  const int kernel_sizes[] = {6, 15};
  const int block_sizes[] = {1, 2, 3, 4, 7, 8, 16, 31, 32, 33, 64, 128, 256, 512};
  constexpr int kTrials = 8;

  std::printf("Eigen reduction-order probe\n");
  std::printf("  Eigen %d.%d.%d, channels %d\n", EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION, C);
  std::printf("  replicating a2_fast.cpp:573-606 and :640-652\n\n");

  // matched[c][K-index] counts (block size, trial) pairs where candidate c was
  // bit-identical; total counts the pairs attempted.
  constexpr int kNumCandidates = static_cast<int>(sizeof(kCandidates) / sizeof(kCandidates[0]));
  int matched[kNumCandidates] = {};
  int total = 0;
  Divergence worst[kNumCandidates] = {};

  int layer_matched = 0, layer_total = 0;
  int head_matched = 0, head_total = 0, head_nofma_matched = 0;

  std::mt19937 rng(0x2A2FA57u);
  std::uniform_real_distribution<float> w_dist(-0.6f, 0.6f);
  std::uniform_real_distribution<float> x_dist(-2.5f, 2.5f);

  for (int trial = 0; trial < kTrials; trial++)
  {
    for (int K : kernel_sizes)
    {
      for (int N : block_sizes)
      {
        // History long enough for K taps at dilation 1 plus the block, laid out
        // exactly as a2_fast's ring is: column-major, channels contiguous.
        const int hist_cols = N + K + kHeadKernelSize;
        std::vector<float> history(static_cast<size_t>(hist_cols) * C);
        for (auto& v : history)
          v = x_dist(rng);

        LayerInputs in;
        in.K = K;
        in.conv_w.resize(K);
        in.tap_src.resize(K);
        for (int k = 0; k < K; k++)
        {
          in.conv_w[k].resize(static_cast<size_t>(C) * C);
          for (auto& v : in.conv_w[k])
            v = w_dist(rng);
          in.tap_src[k] = history.data() + static_cast<size_t>(k) * C;
        }
        in.conv_b.resize(C);
        in.mixin_w.resize(C);
        in.l1x1_w.resize(static_cast<size_t>(C) * C);
        in.l1x1_b.resize(C);
        for (auto& v : in.conv_b)
          v = w_dist(rng);
        for (auto& v : in.mixin_w)
          v = w_dist(rng);
        for (auto& v : in.l1x1_w)
          v = w_dist(rng);
        for (auto& v : in.l1x1_b)
          v = w_dist(rng);
        std::vector<float> cond(N);
        for (auto& v : cond)
          v = x_dist(rng);
        in.cond = cond.data();

        // --- the conv stage on its own ---
        std::vector<float> z_ref(static_cast<size_t>(C) * N, 0.0f);
        ref_conv(in, z_ref, N);

        for (int c = 0; c < kNumCandidates; c++)
        {
          std::vector<float> z_cand(static_cast<size_t>(C) * N, 0.0f);
          kCandidates[c].run(in, z_cand, N);
          if (bit_equal(z_ref, z_cand))
          {
            matched[c]++;
          }
          else
          {
            const Divergence d = divergence(z_ref, z_cand);
            if (d.max_ulps > worst[c].max_ulps)
              worst[c] = d;
          }
        }
        total++;

        // --- the whole layer body ---
        std::vector<float> hs_ref(static_cast<size_t>(C) * N), lin_ref(static_cast<size_t>(C) * N);
        for (size_t i = 0; i < hs_ref.size(); i++)
        {
          hs_ref[i] = x_dist(rng);
          lin_ref[i] = x_dist(rng);
        }
        std::vector<float> hs_cand = hs_ref, lin_cand = lin_ref;
        std::vector<float> z_a(static_cast<size_t>(C) * N, 0.0f), z_b(static_cast<size_t>(C) * N, 0.0f);
        ref_layer(in, z_a, hs_ref, lin_ref, N);
        cand_layer(in, z_b, hs_cand, lin_cand, N);
        if (bit_equal(z_a, z_b) && bit_equal(hs_ref, hs_cand) && bit_equal(lin_ref, lin_cand))
          layer_matched++;
        layer_total++;

        // --- the head ---
        std::vector<std::vector<float>> head_w(kHeadKernelSize, std::vector<float>(C));
        for (auto& t : head_w)
          for (auto& v : t)
            v = w_dist(rng);
        const float head_b = w_dist(rng);
        const float head_scale = 0.0114473553f;

        std::vector<std::vector<float>> planes(C, std::vector<float>(hist_cols));
        for (int b = 0; b < C; b++)
          for (int col = 0; col < hist_cols; col++)
            planes[b][col] = history[static_cast<size_t>(col) * C + b];

        std::vector<float> head_ref(N), head_cand(N), head_nofma(N);
        ref_head_fma(head_w, head_b, head_scale, history, 0, head_ref, N);
        ref_head_nofma(head_w, head_b, head_scale, history, 0, head_nofma, N);
        cand_head_planar(head_w, head_b, head_scale, planes, 0, head_cand, N);
        if (bit_equal(head_ref, head_cand))
          head_matched++;
        if (bit_equal(head_ref, head_nofma))
          head_nofma_matched++;
        head_total++;
      }
    }
  }

  std::printf("Conv stage — %d (block size, kernel size, trial) combinations\n\n", total);
  for (int c = 0; c < kNumCandidates; c++)
  {
    const bool all = matched[c] == total;
    std::printf("  %-48s  %4d/%-4d  %s", kCandidates[c].name, matched[c], total,
                all ? "BIT-IDENTICAL" : "differs");
    if (!all)
      std::printf(" (worst %d ULP, %.3e abs)", worst[c].max_ulps, static_cast<double>(worst[c].max_abs));
    std::printf("\n");
  }

  std::printf("\nWhole layer body (conv + bias + mixin + LeakyReLU + head_sum + layer1x1)\n");
  std::printf("  candidate A ordering throughout           %4d/%-4d  %s\n", layer_matched, layer_total,
              layer_matched == layer_total ? "BIT-IDENTICAL" : "differs");

  std::printf("\nHead (a2_fast's scalar chain, contracted to FMA, is the reference)\n");
  std::printf("  planar, one lane per frame, FMA           %4d/%-4d  %s\n", head_matched, head_total,
              head_matched == head_total ? "BIT-IDENTICAL" : "differs");
  std::printf("  same order but mul+add rounded apart      %4d/%-4d  %s\n", head_nofma_matched, head_total,
              head_nofma_matched == head_total ? "BIT-IDENTICAL" : "differs — the head must use FMA");

  const bool conv_ok = matched[0] == total;
  std::printf("\n%s\n", (conv_ok && layer_matched == layer_total && head_matched == head_total)
                          ? "VERDICT: a2_fast's C=8 order is reproducible. A bit-identical NEON kernel is possible."
                          : "VERDICT: see above — the hypothesised order is not the one Eigen uses.");
  return 0;
}
