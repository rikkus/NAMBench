// Heap in use after building a mono ImpulseResponse from a 500 ms, 48 kHz IR
// (24000 samples, what 1529 of the 1538 files in the library look like) and
// calling Reset(64). macOS zone statistics see every allocation, Eigen's too.
//
// macOS only (malloc zone statistics). Build one binary per tree and path:
//   c++ -std=c++17 -O2 -Ivendor/adt-partitioned -isystem vendor/eigen-adt \
//     -DHAS_IMPL -DIMPL=FFT ir-study/heap_footprint.cpp \
//     vendor/adt-partitioned/dsp/{ImpulseResponse,dsp,wav,PartitionedConvolution}.cpp -o heap_fft
// Upstream has no ConvolutionImplementation, so build it without HAS_IMPL.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <malloc/malloc.h>
#include "dsp/ImpulseResponse.h"
static size_t inUse(){ malloc_statistics_t s; malloc_zone_statistics(nullptr, &s); return s.size_in_use; }
int main(int argc, char** argv){
  const int samples = argc > 1 ? atoi(argv[1]) : 24000;
  dsp::ImpulseResponse::IRData d;
  d.mRawAudio.resize(samples);
  for (int i = 0; i < samples; i++) d.mRawAudio[i] = std::exp(-i / 2000.0f) * ((i * 7919) % 200 - 100) / 100.0f;
  d.mRawAudioSampleRate = 48000.0;
  const size_t before = inUse();
  {
#if defined(HAS_IMPL)
    dsp::ImpulseResponse ir(d, 48000.0, dsp::ConvolutionImplementation::IMPL);
#else
    dsp::ImpulseResponse ir(d, 48000.0);
#endif
    ir.Reset(64, 1);
    const size_t after = inUse();
    printf("%zu\n", (after - before));
  }
}
