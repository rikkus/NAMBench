// History rings for the 32-bit ARM lab, in the strategies worth comparing.
//
// Both baselines arrived with a private, identical copy of the same ring. A
// dozen candidates on top of that duplication is what the AArch64 labs avoided
// by extracting Sources/FullEngines/full_ring.h, and the reason is not tidiness:
// ring strategy is a *measured axis* — compiling a2_fast at NAM_A2_RING_MODE=0
// was worth 14% at C=3 on AArch64, the single biggest surprise in
// SLIMMED-PATH.md — so varying it must cost a template argument rather than a
// rewrite, and every candidate must be able to be measured under all of them
// against unchanged arithmetic.
//
// Three shapes live here, and they are deliberately not one:
//
//   PlanarRing    channel planes side by side, so a Q register holds four
//                 consecutive frames of one channel. What the planar kernels
//                 read.
//   ChannelRing   one column per frame with the channels adjacent inside it.
//                 a2_fast's own storage, and what a channel-major candidate
//                 (n_pad4, n_framemajor, s_chanmajor) needs.
//   BaselineRing  the ring the two controls already had, moved here verbatim.
//                 See the note on it below for why it is not just
//                 ChannelRing<C, Pow2Eager>.
//
// The A2 shapes are known and fixed, so C is a template parameter: one
// instantiation per width, no runtime stride arithmetic in the inner loops.

#pragma once

#if defined(NB_ENABLE_A32_LAB)

  #include <algorithm>
  #include <cstring>
  #include <vector>

  #include "a32_common.h"

namespace a32lab
{

/// How a ring is sized and kept readable.
///
/// Three, where full_ring.h has four. ExactLazy is missing on purpose: an
/// exactly-sized ring wraps on very nearly every block, so its lazy mirror
/// fires once per *tap* instead of once per block, and it lost at both widths
/// on AArch64. Carrying it here would add a third instantiation of every
/// candidate to answer a question that has been answered twice.
enum class RingKind
{
  /// Power-of-two capacity, tail mirrored on every write. a2_fast's default
  /// (NAM_A2_RING_MODE=1): masked reads, constant work per block, and a mirror
  /// memcpy every block whether or not anything goes on to read it.
  Pow2Eager,
  /// Power-of-two capacity, mirrored only when a read genuinely wraps. Same
  /// footprint as Pow2Eager, but the common case is copy-free.
  Pow2Lazy,
  /// One linear buffer per ring, written forward until it runs out and then
  /// memmoved back. a2_fast's NAM_A2_RING_MODE=0, and the winner at C=3 on
  /// AArch64. No mirror, no masking, every read contiguous, at the price of an
  /// occasional large memmove.
  LinearRewind,
};

inline const char* ring_kind_name(RingKind k)
{
  switch (k)
  {
    case RingKind::Pow2Eager: return "pow2+eager";
    case RingKind::Pow2Lazy: return "pow2+lazy";
    case RingKind::LinearRewind: return "linear+rewind";
  }
  return "?";
}

/// Planar ring buffer: `C` channel planes side by side, `stride` floats apart.
///
/// In the two wrapping modes a write is always one contiguous run at `wpos`:
/// the region past the ring proper gives at least `mbs` columns of slack, so the
/// run may overhang the end and be folded back afterwards. That keeps the frame
/// loop straight-line even when a block straddles the wrap, which is what makes
/// writing a layer's residual straight into the next layer's ring (kRingDirect)
/// practical at all. In LinearRewind mode there is no wrap to straddle: the
/// buffer is rewound *before* the write if the write would not fit.
///
/// Whichever mode, the contract is the same three calls in the same order:
/// prepare(n), write into write_ptr(c), commit(n).
///
/// `TailPad` is extra readable columns past the mirror, for kernels that build
/// their tap windows with vextq_f32 and therefore over-read the last window by
/// up to three frames.
template <int C, RingKind Kind, int TailPad = 0>
struct PlanarRing
{
  std::vector<float> data;
  int cap = 0;    ///< usable columns (LinearRewind: total columns)
  int mask = 0;   ///< cap - 1, only meaningful for the pow2 kinds
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
        stride = cap + max_buffer + TailPad;
        break;
      case RingKind::LinearRewind:
        // Matches a2_fast's ring mode 0 sizing: enough slack that the rewind
        // memmove is amortised over many blocks on the long-lookback layers.
        cap = 2 * max_lookback + max_buffer;
        stride = cap + TailPad;
        break;
    }
    mask = cap - 1;
    data.assign(static_cast<size_t>(C) * stride, 0.0f);
    wpos = max_lookback;
  }

  float* plane(int c) { return data.data() + static_cast<size_t>(c) * stride; }
  const float* plane(int c) const { return data.data() + static_cast<size_t>(c) * stride; }

  int wrap(int v) const
  {
    if constexpr (Kind == RingKind::LinearRewind)
      return v; // positions are monotonic, never wrapped
    else
      return v & mask;
  }

  /// Make room for an n-frame write. Only LinearRewind has anything to do.
  void prepare(int n)
  {
    if constexpr (Kind == RingKind::LinearRewind)
    {
      if (wpos + n > cap)
      {
        for (int c = 0; c < C; c++)
          std::memmove(plane(c), plane(c) + (wpos - lookback),
                       static_cast<size_t>(lookback) * sizeof(float));
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
        for (int c = 0; c < C; c++)
          std::memcpy(plane(c), plane(c) + cap, static_cast<size_t>(overflow) * sizeof(float));
      }
      if constexpr (Kind == RingKind::Pow2Eager)
      {
        for (int c = 0; c < C; c++)
          std::memcpy(plane(c) + cap, plane(c), static_cast<size_t>(mbs) * sizeof(float));
      }
      wpos = wrap(wpos + n);
    }
    else
      wpos += n;
  }

  /// First column of an n-frame read looking `lookback_frames` further back
  /// than the block just written.
  int tap(int lookback_frames, int n)
  {
    const int base = wrap(wpos - n - lookback_frames);
    if constexpr (Kind == RingKind::Pow2Lazy)
    {
      // Lazy mirror: only the blocks whose read actually wraps pay for it. The
      // pad is mirrored too, so a vext kernel's over-read stays in bounds.
      const int overflow = base + n + TailPad - cap;
      if (overflow > 0)
      {
        for (int c = 0; c < C; c++)
          std::memcpy(plane(c) + cap, plane(c), static_cast<size_t>(overflow) * sizeof(float));
      }
    }
    return base;
  }
};

/// The same strategies over channel-major storage: one column per frame, `C`
/// floats contiguous inside it.
///
/// This is a2_fast's own layout, and what a candidate that vectorises across
/// channels rather than frames needs — n_pad4 (C=3 padded to a full Q register),
/// n_framemajor, s_chanmajor. Same three-call contract as PlanarRing.
template <int C, RingKind Kind>
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
      case RingKind::LinearRewind:
        cap = 2 * max_lookback + max_buffer;
        cols = cap;
        break;
    }
    mask = cap - 1;
    data.assign(static_cast<size_t>(C) * cols, 0.0f);
    wpos = max_lookback;
  }

  int wrap(int v) const
  {
    if constexpr (Kind == RingKind::LinearRewind)
      return v;
    else
      return v & mask;
  }

  void prepare(int n)
  {
    if constexpr (Kind == RingKind::LinearRewind)
    {
      if (wpos + n > cap)
      {
        std::memmove(data.data(), data.data() + static_cast<size_t>(wpos - lookback) * C,
                     static_cast<size_t>(lookback) * C * sizeof(float));
        wpos = lookback;
      }
    }
  }

  float* write_ptr() { return data.data() + static_cast<size_t>(wpos) * C; }

  void commit(int n)
  {
    if constexpr (Kind != RingKind::LinearRewind)
    {
      const int overflow = wpos + n - cap;
      if (overflow > 0)
      {
        std::memcpy(data.data(), data.data() + static_cast<size_t>(cap) * C,
                    static_cast<size_t>(overflow) * C * sizeof(float));
      }
      if constexpr (Kind == RingKind::Pow2Eager)
      {
        std::memcpy(data.data() + static_cast<size_t>(cap) * C, data.data(),
                    static_cast<size_t>(mbs) * C * sizeof(float));
      }
      wpos = wrap(wpos + n);
    }
    else
      wpos += n;
  }

  const float* tap(int lookback_frames, int n)
  {
    const int base = wrap(wpos - n - lookback_frames);
    if constexpr (Kind == RingKind::Pow2Lazy)
    {
      const int overflow = base + n - cap;
      if (overflow > 0)
      {
        std::memcpy(data.data() + static_cast<size_t>(cap) * C, data.data(),
                    static_cast<size_t>(overflow) * C * sizeof(float));
      }
    }
    return data.data() + static_cast<size_t>(base) * C;
  }
};

/// The ring the two controls carry, moved out of them unchanged.
///
/// It is not ChannelRing<C, RingKind::Pow2Eager>, and the difference is the
/// point of the controls. This one writes a block as two runs split at the wrap
/// where ChannelRing writes one run with an overhang and folds it back
/// afterwards; both leave the same bytes in the same places, but they are not
/// the same instructions. A control's whole job is to reproduce `upstream`'s
/// number, so it keeps the copying pattern it was measured with — the shared
/// type is for candidates, which are supposed to differ.
///
/// Layout: cols [pow2, pow2 + mbs) always duplicate cols [0, mbs), so every read
/// of up to mbs frames starting inside [0, pow2) is contiguous.
template <int C>
struct BaselineRing
{
  std::vector<float> data;
  int pow2 = 0;
  int mask = 0;
  int write_pos = 0;
  int mbs = 0;

  void reset(int max_lookback, int max_buffer)
  {
    mbs = max_buffer;
    pow2 = next_pow2(max_lookback + max_buffer);
    mask = pow2 - 1;
    data.assign(static_cast<size_t>(C) * (pow2 + max_buffer), 0.0f);
    write_pos = max_lookback;
  }

  void write(const float* src, int num_frames)
  {
    float* const hist = data.data();
    const int wp = write_pos;
    const int first = std::min(num_frames, pow2 - wp);
    std::memcpy(hist + static_cast<size_t>(wp) * C, src,
                static_cast<size_t>(first) * C * sizeof(float));
    if (first < num_frames)
    {
      std::memcpy(hist, src + static_cast<size_t>(first) * C,
                  static_cast<size_t>(num_frames - first) * C * sizeof(float));
    }
    std::memcpy(hist + static_cast<size_t>(pow2) * C, hist,
                static_cast<size_t>(mbs) * C * sizeof(float));
    write_pos = (wp + num_frames) & mask;
  }
};

} // namespace a32lab

#endif // NB_ENABLE_A32_LAB
