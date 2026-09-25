// ADT side of linear_ab.cpp: dsp::ImpulseResponse with the FFT path forced.
#include <cstring>
#include "ImpulseResponse.h"
#include "ab_iface.h"

namespace
{
struct Adt : AbConv
{
  dsp::ImpulseResponse ir;
  Adt(const dsp::ImpulseResponse::IRData& d, double rate, int maxFrames)
  : ir(d, rate, dsp::ConvolutionImplementation::FFT)
  {
    ir.Reset(maxFrames, 1);
  }
  void Process(const double* in, double* out, int n) override
  {
    double* p = const_cast<double*>(in);
    double** o = ir.Process(&p, 1, n);
    std::memcpy(out, o[0], n * sizeof(double));
  }
};
} // namespace

AbConv* make_adt(const std::vector<float>& ir, double rate, int maxFrames)
{
  dsp::ImpulseResponse::IRData d;
  d.mRawAudio = ir;
  d.mRawAudioSampleRate = rate;
  return new Adt(d, rate, maxFrames);
}
