// The a2* compositions: everything that paid, together.
//
// From the ablation against `a2p4`, the switches that paid were a wide frame
// tile, writing residuals straight into the next ring, a head tiled across
// frames, and skipping the two pieces of work nothing reads. `Pow2Lazy` (what
// fused uses), `ExactLazy` and `prefetch` all lost and are left out.
//
// Two ring strategies are carried forward rather than one, because the ring
// sweep was close: `Pow2Eager` is a2_fast's shipped strategy and `LinearRewind`
// is the one that won at C=3, and they were within a percent of each other here.
// Composed with ringdirect the balance can move, so both are measured.
//
// Several tile widths, because the best tile for the composed kernel need not be
// the best tile for `a2p` alone: ringdirect changes what is live in the tail,
// which is exactly where a wide tile runs out of registers.

#if defined(NB_ENABLE_FULL_LAB)

  #include "full_a2_planar_kernel.h"

namespace fulllab
{
namespace
{

/// Everything that paid, minus the ring choice.
struct Stacked : A2PlanarOptions
{
  static constexpr int kHeadVecs = 4;
  static constexpr bool kRingDirect = true;
  static constexpr bool kSkipLastL1x1 = true;
  static constexpr bool kFoldHeadInit = true;
};

struct S4 : Stacked
{
  static constexpr int kTile = 4;
};

struct S8 : Stacked
{
  static constexpr int kTile = 8;
};

struct S12 : Stacked
{
  static constexpr int kTile = 12;
};

struct S8L : S8
{
  static constexpr RingKind kRing = RingKind::LinearRewind;
};

struct S12L : S12
{
  static constexpr RingKind kRing = RingKind::LinearRewind;
};

// --- The head sweep ----------------------------------------------------------
//
// The head is 16 taps × 8 channels of sequential FMA from the bias — 128 deep,
// and a2_fast runs one such chain per frame. Planar layout makes each chain
// cover four frames; kHeadVecs says how many run at once. Cheap in registers
// (y + weights + one input), so the only question is where the latency stops
// being the limit.

struct Head2 : A2PlanarOptions
{
  static constexpr int kHeadVecs = 2;
};

struct Head8 : A2PlanarOptions
{
  static constexpr int kHeadVecs = 8;
};

// --- The register-pressure question ------------------------------------------
//
// a2_fast's association needs the running total `z` and the current tap's
// partial `t` live at the same time, so tile 8 wants 32 vector registers for
// accumulators alone and tile 16 collapsed. Splitting the conv into two passes
// over the output channels halves what the tap loop holds, at the cost of
// reading each input plane twice per tap.

struct Split8 : A2PlanarOptions
{
  static constexpr int kTile = 8;
  static constexpr int kOutPasses = 2;
};

struct Split16 : A2PlanarOptions
{
  static constexpr int kTile = 16;
  static constexpr int kOutPasses = 2;
};

/// The composition again, with the head sweep's answer folded in, and with and
/// without the output split.
struct S8H : Stacked
{
  static constexpr int kTile = 8;
  static constexpr int kHeadVecs = 8;
  static constexpr RingKind kRing = RingKind::LinearRewind;
};

struct Lane : A2PlanarOptions
{
  static constexpr bool kL1x1Lane = true;
};

struct S8HLane : S8H
{
  static constexpr bool kL1x1Lane = true;
};

struct S8HSplit : S8H
{
  static constexpr int kOutPasses = 2;
};

struct S16HSplit : S8H
{
  static constexpr int kTile = 16;
  static constexpr int kOutPasses = 2;
};

} // namespace

  #define NB_FULL_A2P(fn, opts)                                                                   \
    std::unique_ptr<nam::DSP> fn(const std::vector<float>& weights, double sampleRate)            \
    {                                                                                             \
      return std::make_unique<A2PlanarModel<opts>>(weights, sampleRate);                          \
    }

NB_FULL_A2P(make_a2s4, S4)
NB_FULL_A2P(make_a2s8, S8)
NB_FULL_A2P(make_a2s12, S12)
NB_FULL_A2P(make_a2s8_linear, S8L)
NB_FULL_A2P(make_a2s12_linear, S12L)
NB_FULL_A2P(make_a2p_head2, Head2)
NB_FULL_A2P(make_a2p_head8, Head8)
NB_FULL_A2P(make_a2p_split8, Split8)
NB_FULL_A2P(make_a2p_split16, Split16)
NB_FULL_A2P(make_a2s8_h8, S8H)
NB_FULL_A2P(make_a2p_l1x1lane, Lane)
NB_FULL_A2P(make_a2s8_h8_lane, S8HLane)
NB_FULL_A2P(make_a2s8_h8_split, S8HSplit)
NB_FULL_A2P(make_a2s16_split, S16HSplit)

  #undef NB_FULL_A2P

} // namespace fulllab

#endif // NB_ENABLE_FULL_LAB
