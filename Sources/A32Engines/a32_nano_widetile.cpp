// Kernels 3-5: n_widetile{8,16,32} — find where ILP stops paying and spilling
// starts.
//
// `n_planar` uses a 4-frame tile: three live accumulators, three independent FMA
// chains through the tap loop. Wider tiles buy more chains and amortise the
// weight broadcast — each tap broadcasts C × C weights whatever the tile width,
// so at tile 4 that is nine vld1q_dup per nine FMAs and at tile 32 it is nine
// per seventy-two.
//
//   tile   live accumulators (C × T / 4)
//      4    3
//      8    6
//     16   12
//     32   24   more than the register file has; must spill
//
// The advance prediction, from SLIMMED-PATH.md's rule scaled to this part's
// sixteen Q registers rather than AArch64's thirty-two: the peak moves down from
// 32 to about 16.
//
// Measured, it moved further, and the object file says why. Per frame of tile,
// vldr/vstr in the K=6 layer body:
//
//   tile  4   50 spills   12.5 /frame   1.105x over a2_fast
//   tile  8  105 spills   13.1 /frame   1.218x   <- the peak
//   tile 12  321 spills   26.8 /frame   1.077x
//   tile 16  534 spills   33.4 /frame   1.010x
//   tile 32  848 spills   26.5 /frame   0.939x
//
// Phase 5 added the bottom rung, and the completed C=3 ladder is:
//
//   tile  2  0.888x     tile  8  1.242x  <- the peak
//   tile  4  1.116x     tile 12  1.086x
//                       tile 16  1.006x     tile 32  0.954x
//
// The C=8 ladder in a32_std_planar.cpp peaks at 8 as well. That is the finding
// the two rounds together produce and neither could produce alone: the best tile
// is the same at both widths, so it is not set by how many accumulators the
// channel count needs. It is a property of this core — where loop overhead stops
// dominating and weight-load amortisation stops paying — and the register model
// predicts the shape of the collapse either side of it, not the peak itself.
//
// Tile 12 was added after the first round, because the budget arithmetic
// (z 3T/4 + inputs T/4 + weight 1 = 13 registers at T=12) says it is the last
// width that still fits and the ladder had no rung between the last win and the
// collapse. It does not fit: it spills twice as much per frame as tile 8 and
// loses. So the budget under-counts, and it under-counts in a specific way — it
// prices the conv's working set and ignores the layer tail's, where the mixin,
// the layer1x1 weights and three destination pointers are all live at once.
// The usable rule on this part is nearer C * T / 4 <= 6 than <= 12. Writing 32 anyway is what makes that a prediction rather than
// an assertion — and Scripts/a32-codegen-check.sh's vldr/vstr count says
// whether a slow tile is slow because it spilled or for some other reason.

#if defined(NB_ENABLE_A32_LAB)

  #include "a32_planar_kernel.h"

namespace a32lab
{
namespace
{

/// Two frames in a D register. Below the width the ladder was written for, and
/// only expressible since the C=8 round needed half-width tiles anyway — the
/// sweep in the campaign plan names 2 as a value, so here it is rather than a
/// gap in the table.
struct Options2 : NanoPlanarOptions
{
  static constexpr int kTile = 2;
};

struct Options8 : NanoPlanarOptions
{
  static constexpr int kTile = 8;
};

struct Options12 : NanoPlanarOptions
{
  static constexpr int kTile = 12;
};

struct Options16 : NanoPlanarOptions
{
  static constexpr int kTile = 16;
};

struct Options32 : NanoPlanarOptions
{
  static constexpr int kTile = 32;
};

} // namespace

std::unique_ptr<nam::DSP> make_nano_widetile2(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options2>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_nano_widetile8(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options8>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_nano_widetile12(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options12>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_nano_widetile16(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options16>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_nano_widetile32(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options32>>(weights, sampleRate);
}

} // namespace a32lab

#endif // NB_ENABLE_A32_LAB
