// The a2* family: planar kernels reproducing a2_fast's arithmetic exactly.
//
// Every candidate here is one switch away from `a2p4`, so the result table reads
// as an ablation rather than as a set of unrelated rewrites. `a2p4` itself is
// already three things at once relative to a2_baseline — planar layout,
// z-in-registers, and a head vectorised across frames — because none of the
// three is separable: the planar layout is what puts z in registers and what
// makes the head's per-frame chain vectorisable in the first place.

#if defined(NB_ENABLE_FULL_LAB)

  #include "full_a2_planar_kernel.h"

namespace fulllab
{
namespace
{

// --- The frame-tile sweep ----------------------------------------------------
//
// The lever that dominated everything else at C=3. It buys two things: more
// independent FMA chains, and weight-load amortisation (a tap's 16 weight
// vectors are loaded once however wide the tile is). The ceiling is lower here
// than at C=3 because a2_fast's association needs the running total `z` and the
// tap partial `t` live simultaneously — 16 registers at tile 4, 32 at tile 8,
// before inputs and weights. Where that actually stops paying is the question.

struct Opt4 : A2PlanarOptions
{
};

struct Opt8 : A2PlanarOptions
{
  static constexpr int kTile = 8;
};

struct Opt12 : A2PlanarOptions
{
  static constexpr int kTile = 12;
};

struct Opt16 : A2PlanarOptions
{
  static constexpr int kTile = 16;
};

// --- One switch each, against Opt4 -------------------------------------------

/// The head is a 128-long serial FMA chain. Four independent chains at once,
/// each still running the reference's per-frame order.
struct OptHeadTile : A2PlanarOptions
{
  static constexpr int kHeadVecs = 4;
};

/// Residual straight into the next layer's ring, head_sum straight into the head
/// ring. Removes 23 block-sized copies and the scratch pass that feeds them.
struct OptRingDirect : A2PlanarOptions
{
  static constexpr bool kRingDirect = true;
};

/// Two pieces of work nothing reads: the final layer1x1, and the per-block
/// memset of head_sum.
struct OptSkipLast : A2PlanarOptions
{
  static constexpr bool kSkipLastL1x1 = true;
  static constexpr bool kFoldHeadInit = true;
};

/// The ring sweep. Pow2Eager is a2_fast's shipped strategy and the default here,
/// so these three are the alternatives.
struct OptRingLazy : A2PlanarOptions
{
  static constexpr RingKind kRing = RingKind::Pow2Lazy;
};

struct OptRingExact : A2PlanarOptions
{
  static constexpr RingKind kRing = RingKind::ExactLazy;
};

struct OptRingLinear : A2PlanarOptions
{
  static constexpr RingKind kRing = RingKind::LinearRewind;
};

/// Expected to lose — it did at C=3 — but the premise is different here. The
/// far-dilation layers reach about 38 KB back at C=8 against 14 KB at C=3, and
/// planar layout means 8 streams per tap rather than 3.
struct OptPrefetch : A2PlanarOptions
{
  static constexpr bool kPrefetch = true;
};

} // namespace

  #define NB_FULL_A2P(fn, opts)                                                                   \
    std::unique_ptr<nam::DSP> fn(const std::vector<float>& weights, double sampleRate)            \
    {                                                                                             \
      return std::make_unique<A2PlanarModel<opts>>(weights, sampleRate);                          \
    }

NB_FULL_A2P(make_a2p4, Opt4)
NB_FULL_A2P(make_a2p8, Opt8)
NB_FULL_A2P(make_a2p12, Opt12)
NB_FULL_A2P(make_a2p16, Opt16)
NB_FULL_A2P(make_a2p_headtile, OptHeadTile)
NB_FULL_A2P(make_a2p_ringdirect, OptRingDirect)
NB_FULL_A2P(make_a2p_skiplast, OptSkipLast)
NB_FULL_A2P(make_a2p_ringlazy, OptRingLazy)
NB_FULL_A2P(make_a2p_ringexact, OptRingExact)
NB_FULL_A2P(make_a2p_ringlinear, OptRingLinear)
NB_FULL_A2P(make_a2p_prefetch, OptPrefetch)

  #undef NB_FULL_A2P

} // namespace fulllab

#endif // NB_ENABLE_FULL_LAB
