// Planar history rings, in the four strategies worth comparing at C=8.
//
// The nano lab found that the ring strategy mattered more than most kernels did
// — compiling a2_fast at NAM_A2_RING_MODE=0 was worth 14% on the 3-channel
// submodel — and SLIMMED-PATH.md deferred the same question at C=8 explicitly.
// So here the strategy is a template parameter rather than a #define: the sweep
// needs no rebuild, and every candidate can be measured against the same
// arithmetic under all four.
//
// Layout is planar: kChannels planes side by side, `stride` floats apart, so a
// NEON register can hold four consecutive frames of one channel. That is what
// lets each lane run a2_fast's per-frame scalar reduction verbatim.

#pragma once

#if defined(NB_ENABLE_FULL_LAB)

  #include <algorithm>
  #include <cstring>
  #include <vector>

  #include "full_common.h"

namespace fulllab
{

/// How a ring is sized and kept readable.
enum class RingKind
{
  /// Power-of-two capacity, tail mirrored on every write. a2_fast's default
  /// (NAM_A2_RING_MODE=1): masked reads, constant work per block, and a mirror
  /// memcpy every block whether or not anything reads it.
  Pow2Eager,
  /// Power-of-two capacity, mirrored only when a read genuinely wraps. What the
  /// fused engine does: same footprint as Pow2Eager, but the common case is
  /// copy-free.
  Pow2Lazy,
  /// Exactly-sized capacity, mirrored only when a read wraps. Smallest
  /// footprint, and the one the nano lab expected to win and did not — an
  /// exactly-sized ring wraps almost every block, and then the lazy mirror
  /// fires once per *tap*.
  ExactLazy,
  /// One linear buffer per ring, written forward until it runs out and then
  /// memmoved back. a2_fast's NAM_A2_RING_MODE=0, and the winner at C=3. No
  /// mirror, no masking, every read contiguous, at the price of an occasional
  /// large memmove.
  LinearRewind,
};

inline const char* ring_kind_name(RingKind k)
{
  switch (k)
  {
    case RingKind::Pow2Eager: return "pow2+eager";
    case RingKind::Pow2Lazy: return "pow2+lazy";
    case RingKind::ExactLazy: return "exact+lazy";
    case RingKind::LinearRewind: return "linear+rewind";
  }
  return "?";
}

/// Planar ring buffer, kChannels channel planes side by side.
///
/// In the three wrapping modes a write is always one contiguous run at `wpos`:
/// the region past the ring proper gives at least `mbs` columns of slack, so the
/// run can overhang the end and be folded back afterwards. That keeps the frame
/// loop straight-line even when a block straddles the wrap, which is what makes
/// writing residuals straight into the ring (kRingDirect) practical at all. In
/// LinearRewind mode there is no wrap to straddle: the buffer is rewound
/// *before* the write if the write would not fit.
///
/// Whichever mode, the contract is the same three calls in the same order:
/// prepare(n), write into write_ptr(c), commit(n).
template <RingKind Kind>
struct PlanarRing
{
  std::vector<float> data;
  int cap = 0; ///< usable columns (LinearRewind: total columns)
  int mask = 0; ///< cap - 1, only meaningful for the pow2 kinds
  int stride = 0; ///< distance between channel planes
  int wpos = 0;
  int mbs = 0;
  int lookback = 0;

  void reset(int max_lookback, int max_buffer)
  {
    mbs = max_buffer;
    lookback = max_lookback;
    switch (Kind)
    {
      case RingKind::Pow2Eager:
      case RingKind::Pow2Lazy:
        cap = next_pow2(max_lookback + max_buffer);
        stride = cap + max_buffer;
        break;
      case RingKind::ExactLazy:
        cap = max_lookback + max_buffer;
        stride = cap + max_buffer;
        break;
      case RingKind::LinearRewind:
        // Matches a2_fast's ring mode 0 sizing: enough slack that the rewind
        // memmove is amortised over many blocks on the long-lookback layers.
        cap = 2 * max_lookback + max_buffer;
        stride = cap;
        break;
    }
    mask = cap - 1;
    data.assign(static_cast<size_t>(kChannels) * stride, 0.0f);
    wpos = max_lookback;
  }

  float* plane(int c) { return data.data() + static_cast<size_t>(c) * stride; }
  const float* plane(int c) const { return data.data() + static_cast<size_t>(c) * stride; }

  int wrap(int v) const
  {
    if constexpr (Kind == RingKind::Pow2Eager || Kind == RingKind::Pow2Lazy)
      return v & mask;
    else if constexpr (Kind == RingKind::ExactLazy)
    {
      if (v >= cap)
        v -= cap;
      if (v < 0)
        v += cap;
      return v;
    }
    else
      return v; // LinearRewind: positions are monotonic, never wrapped
  }

  /// Make room for an n-frame write. Only LinearRewind has anything to do.
  void prepare(int n)
  {
    if constexpr (Kind == RingKind::LinearRewind)
    {
      if (wpos + n > cap)
      {
        for (int c = 0; c < kChannels; c++)
          std::memmove(plane(c), plane(c) + (wpos - lookback), static_cast<size_t>(lookback) * sizeof(float));
        wpos = lookback;
      }
    }
  }

  /// Where an n-frame block is written. Contiguous; see the note above.
  float* write_ptr(int c) { return plane(c) + wpos; }

  /// Fold any overhang back into the ring, refresh the eager mirror, advance.
  void commit(int n)
  {
    if constexpr (Kind != RingKind::LinearRewind)
    {
      const int overflow = wpos + n - cap;
      if (overflow > 0)
      {
        for (int c = 0; c < kChannels; c++)
          std::memcpy(plane(c), plane(c) + cap, static_cast<size_t>(overflow) * sizeof(float));
      }
      if constexpr (Kind == RingKind::Pow2Eager)
      {
        for (int c = 0; c < kChannels; c++)
          std::memcpy(plane(c) + cap, plane(c), static_cast<size_t>(mbs) * sizeof(float));
      }
      wpos = wrap(wpos + n);
    }
    else
      wpos += n;
  }

  /// First column of an n-frame read looking `lookback_frames` further back than
  /// the block just written.
  int tap(int lookback_frames, int n)
  {
    const int base = wrap(wpos - n - lookback_frames);
    if constexpr (Kind == RingKind::Pow2Lazy || Kind == RingKind::ExactLazy)
    {
      // Lazy mirror: only the blocks whose read actually wraps pay for it.
      const int overflow = base + n - cap;
      if (overflow > 0)
      {
        for (int c = 0; c < kChannels; c++)
          std::memcpy(plane(c) + cap, plane(c), static_cast<size_t>(overflow) * sizeof(float));
      }
    }
    return base;
  }
};

/// The same four strategies over a channel-major buffer: one column per frame,
/// kChannels floats contiguous within it. This is a2_fast's and fused's storage,
/// and what the fu* family needs — its vectors span channels, so a frame's
/// channels have to be adjacent.
///
/// Same three-call contract as PlanarRing: prepare(n), write into write_ptr(),
/// commit(n).
template <RingKind Kind>
struct ChannelRing
{
  std::vector<float> data;
  int cap = 0;
  int mask = 0;
  int wpos = 0;
  int mbs = 0;
  int lookback = 0;

  void reset(int max_lookback, int max_buffer)
  {
    mbs = max_buffer;
    lookback = max_lookback;
    int cols = 0;
    switch (Kind)
    {
      case RingKind::Pow2Eager:
      case RingKind::Pow2Lazy:
        cap = next_pow2(max_lookback + max_buffer);
        cols = cap + max_buffer;
        break;
      case RingKind::ExactLazy:
        cap = max_lookback + max_buffer;
        cols = cap + max_buffer;
        break;
      case RingKind::LinearRewind:
        cap = 2 * max_lookback + max_buffer;
        cols = cap;
        break;
    }
    mask = cap - 1;
    data.assign(static_cast<size_t>(kChannels) * cols, 0.0f);
    wpos = max_lookback;
  }

  int wrap(int v) const
  {
    if constexpr (Kind == RingKind::Pow2Eager || Kind == RingKind::Pow2Lazy)
      return v & mask;
    else if constexpr (Kind == RingKind::ExactLazy)
    {
      if (v >= cap)
        v -= cap;
      if (v < 0)
        v += cap;
      return v;
    }
    else
      return v;
  }

  void prepare(int n)
  {
    if constexpr (Kind == RingKind::LinearRewind)
    {
      if (wpos + n > cap)
      {
        std::memmove(data.data(), data.data() + static_cast<size_t>(wpos - lookback) * kChannels,
                     static_cast<size_t>(lookback) * kChannels * sizeof(float));
        wpos = lookback;
      }
    }
  }

  float* write_ptr() { return data.data() + static_cast<size_t>(wpos) * kChannels; }

  void commit(int n)
  {
    if constexpr (Kind != RingKind::LinearRewind)
    {
      const int overflow = wpos + n - cap;
      if (overflow > 0)
      {
        std::memcpy(data.data(), data.data() + static_cast<size_t>(cap) * kChannels,
                    static_cast<size_t>(overflow) * kChannels * sizeof(float));
      }
      if constexpr (Kind == RingKind::Pow2Eager)
      {
        std::memcpy(data.data() + static_cast<size_t>(cap) * kChannels, data.data(),
                    static_cast<size_t>(mbs) * kChannels * sizeof(float));
      }
      wpos = wrap(wpos + n);
    }
    else
      wpos += n;
  }

  const float* tap(int lookback_frames, int n)
  {
    const int base = wrap(wpos - n - lookback_frames);
    if constexpr (Kind == RingKind::Pow2Lazy || Kind == RingKind::ExactLazy)
    {
      const int overflow = base + n - cap;
      if (overflow > 0)
      {
        std::memcpy(data.data() + static_cast<size_t>(cap) * kChannels, data.data(),
                    static_cast<size_t>(overflow) * kChannels * sizeof(float));
      }
    }
    return data.data() + static_cast<size_t>(base) * kChannels;
  }
};

} // namespace fulllab

#endif // NB_ENABLE_FULL_LAB
