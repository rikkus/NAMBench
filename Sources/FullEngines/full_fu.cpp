// The fu* family: one switch at a time against fused's own settings.
//
// `fu_t4` instantiates the template with exactly fused's choices (conv tile 4,
// tail tile 2, one head chain per frame, three passes through _z, pow2 + lazy
// mirror). It should be bit-identical to `fused` *and* land on its time — a
// second control, this time for the template rather than for the lab, which is
// what makes every row below attributable to its one switch.

#if defined(NB_ENABLE_FULL_LAB)

  #include "full_fu_kernel.h"

namespace fulllab
{
namespace
{

/// fused's own settings.
struct T4 : FuOptions
{
};

// --- The conv tile -----------------------------------------------------------
//
// fused picks `(Q <= 4) ? 4 : 2`, which is right for C=16 and leaves C=8 with 8
// accumulator chains where four 4-cycle FMA pipes want about 16.

struct T8 : FuOptions
{
  static constexpr int kConvTile = 8;
};

struct T12 : FuOptions
{
  static constexpr int kConvTile = 12;
};

struct T16 : FuOptions
{
  static constexpr int kConvTile = 16;
};

struct T6 : FuOptions
{
  static constexpr int kConvTile = 6;
};

struct T10 : FuOptions
{
  static constexpr int kConvTile = 10;
};

// --- The tail tile -----------------------------------------------------------
//
// fused picks `(Q <= 4) ? 2 : 1`, leaving four chains and reloading all sixteen
// layer1x1 weight vectors every two frames.

struct Tail4 : FuOptions
{
  static constexpr int kTailTile = 4;
};

struct Tail8 : FuOptions
{
  static constexpr int kTailTile = 8;
};

// --- One structural switch each ----------------------------------------------

struct FuseZ : FuOptions
{
  static constexpr bool kFuseZ = true;
};

struct RingDirect : FuOptions
{
  static constexpr bool kRingDirect = true;
};

struct HeadTile : FuOptions
{
  static constexpr int kHeadFrames = 4;
};

struct StoreHead : FuOptions
{
  static constexpr bool kFoldHeadInit = true;
};

// --- The ring sweep ----------------------------------------------------------

struct RingEager : FuOptions
{
  static constexpr RingKind kRing = RingKind::Pow2Eager;
};

struct RingExact : FuOptions
{
  static constexpr RingKind kRing = RingKind::ExactLazy;
};

struct RingLinear : FuOptions
{
  static constexpr RingKind kRing = RingKind::LinearRewind;
};

// --- Compositions ------------------------------------------------------------

// kFuseZ is left out: it lost, and at conv tile 8 it would lose harder — the
// conv already needs 26 of the 32 vector registers there, and carrying the
// layer1x1 accumulators in the same tile needs another 16.
struct Stacked : FuOptions
{
  static constexpr int kConvTile = 8;
  static constexpr int kTailTile = 4;
  static constexpr bool kRingDirect = true;
  static constexpr RingKind kRing = RingKind::LinearRewind;
};

struct S8 : Stacked
{
};

struct S8Eager : Stacked
{
  static constexpr RingKind kRing = RingKind::Pow2Eager;
};

struct S8Lazy : Stacked
{
  static constexpr RingKind kRing = RingKind::Pow2Lazy;
};

struct S6 : Stacked
{
  static constexpr int kConvTile = 6;
};

struct S8Head : Stacked
{
  static constexpr int kHeadFrames = 4;
  static constexpr bool kFoldHeadInit = true;
};

/// The ring sweep was close enough inside the composition that both survivors
/// deserve the head switch as well.
struct S8HeadLazy : S8Head
{
  static constexpr RingKind kRing = RingKind::Pow2Lazy;
};

} // namespace

  #define NB_FULL_FU(fn, opts)                                                                    \
    std::unique_ptr<nam::DSP> fn(const std::vector<float>& weights, double sampleRate)            \
    {                                                                                             \
      return std::make_unique<FuModel<opts>>(weights, sampleRate);                                \
    }

NB_FULL_FU(make_fu_t4, T4)
NB_FULL_FU(make_fu_t8, T8)
NB_FULL_FU(make_fu_t12, T12)
NB_FULL_FU(make_fu_t16, T16)
NB_FULL_FU(make_fu_tail4, Tail4)
NB_FULL_FU(make_fu_tail8, Tail8)
NB_FULL_FU(make_fu_fusez, FuseZ)
NB_FULL_FU(make_fu_ringdirect, RingDirect)
NB_FULL_FU(make_fu_headtile, HeadTile)
NB_FULL_FU(make_fu_storehead, StoreHead)
NB_FULL_FU(make_fu_ringeager, RingEager)
NB_FULL_FU(make_fu_ringexact, RingExact)
NB_FULL_FU(make_fu_ringlinear, RingLinear)
NB_FULL_FU(make_fu_t6, T6)
NB_FULL_FU(make_fu_t10, T10)
NB_FULL_FU(make_fu_s6, S6)
NB_FULL_FU(make_fu_s8, S8)
NB_FULL_FU(make_fu_s8_eager, S8Eager)
NB_FULL_FU(make_fu_s8_lazy, S8Lazy)
NB_FULL_FU(make_fu_s8_head, S8Head)
NB_FULL_FU(make_fu_s8_head_lazy, S8HeadLazy)

  #undef NB_FULL_FU

} // namespace fulllab

#endif // NB_ENABLE_FULL_LAB
