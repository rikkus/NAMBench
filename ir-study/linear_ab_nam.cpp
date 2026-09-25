// NAM core side of linear_ab.cpp: nam::Linear with the FFT path forced.
#include <cstring>
#include "NAM/linear.h"
#include "ab_iface.h"

namespace
{
struct Nam : AbConv
{
  nam::Linear lin;
  Nam(const std::vector<float>& ir, double rate, int maxFrames)
  : lin(1, 1, (int)ir.size(), false, ir, rate, nam::LinearImplementation::FFT)
  {
    lin.Reset(rate, maxFrames);
  }
  void Process(const double* in, double* out, int n) override
  {
    NAM_SAMPLE* i = const_cast<NAM_SAMPLE*>(in);
    lin.process(&i, &out, n);
  }
};
} // namespace

AbConv* make_nam(const std::vector<float>& ir, double rate, int maxFrames) { return new Nam(ir, rate, maxFrames); }
