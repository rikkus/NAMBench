// Does storing half of each spectrum matter, separately from multiplying half?
//
// V0 = ba646f4 (all 1024 bins multiplied, full storage)
// V1 = loop stops at N/2, storage still full size
// V2 = committed: loop stops at N/2, storage N/2+1
// Same input, same kernel, forced FFT, 64-frame callbacks. The three take turns
// in a rotating order each round, so drift lands on all of them alike; that is
// what lets a same-machine comparison get away with short runs. Generate the
// PC0/PC1/PC2 sources with make_storage_variants.py.
//
//   ./storage_ab [cpu to pin to, Linux] [rounds]
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#if defined(__linux__)
#include <sched.h>
#endif
#include "PC0.h"
#include "PC1.h"
#include "PC2.h"
using namespace dsp;
static uint64_t now(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
static void ftz(){
#if defined(__aarch64__)
  uint64_t v; __asm__ __volatile__("mrs %0, fpcr":"=r"(v)); v|=1ull<<24; __asm__ __volatile__("msr fpcr, %0"::"r"(v));
#elif defined(__arm__)
  uint32_t v; __asm__ __volatile__("vmrs %0, fpscr":"=r"(v)); v|=1u<<24; __asm__ __volatile__("vmsr fpscr, %0"::"r"(v));
#endif
}
struct Stats { std::vector<double> heavy, all; };
template<class C> void pass(C& c, const std::vector<float>& in, std::vector<float>& out, int period, Stats* st){
  c.Reset(64);
  const size_t n = in.size()/64;
  for(size_t b=0;b<n;b++){ uint64_t t0=now(); c.Process(in.data()+b*64, out.data()+b*64, 64); uint64_t t1=now();
    if(st){ double us=(t1-t0)/1e3; st->all.push_back(us); if(b%period==(size_t)period-1) st->heavy.push_back(us);} }
}
static double med(std::vector<double> v){ std::sort(v.begin(),v.end()); return v[v.size()/2]; }
static double mean(const std::vector<double>& v){ double s=0; for(double x:v) s+=x; return s/v.size(); }
int main(int argc, char** argv){
#if defined(__linux__)
  cpu_set_t set; CPU_ZERO(&set); CPU_SET(argc>1?atoi(argv[1]):3,&set); sched_setaffinity(0,sizeof(set),&set);
#endif
  ftz();
  const int rounds = argc>2 ? atoi(argv[2]) : 15;
  std::vector<float> in(48000*1); uint32_t s=12345;
  for(auto& v:in){ s=s*1664525u+1013904223u; v=((int32_t)(s>>16)-32768)/65536.f; }
  printf("taps  | transform callback median (us): V0 / V1 / V2 | mean callback (us): V0 / V1 / V2 | V2 vs V1 | outputs\n");
  for(int taps : {1024, 2048, 4096, 8192}){
    std::vector<float> k(taps); float env=1; uint32_t q=0x9E3779B9u;
    for(int i=0;i<taps;i++){ q=q*1664525u+1013904223u; k[i]=((int32_t)(q>>16)-32768)/32768.f*env; env*=0.9975f; if(env<1e-6f) env=1e-6f; } k[0]=1;
    PC0 a; PC1 b; PC2 c;
    a.SetKernel(k.data(),taps,ConvolutionImplementation::FFT); b.SetKernel(k.data(),taps,ConvolutionImplementation::FFT); c.SetKernel(k.data(),taps,ConvolutionImplementation::FFT);
    const int period = (int)a.GetFftBlockSize()/64;
    std::vector<float> oa(in.size()), ob(in.size()), oc(in.size());
    pass(a,in,oa,period,nullptr); pass(b,in,ob,period,nullptr); pass(c,in,oc,period,nullptr);
    const bool same = oa==ob && ob==oc;
    Stats sa, sb, sc;
    for(int r=0;r<rounds;r++){
      // rotate the order each round so no variant always runs first
      switch(r%3){
        case 0: pass(a,in,oa,period,&sa); pass(b,in,ob,period,&sb); pass(c,in,oc,period,&sc); break;
        case 1: pass(b,in,ob,period,&sb); pass(c,in,oc,period,&sc); pass(a,in,oa,period,&sa); break;
        case 2: pass(c,in,oc,period,&sc); pass(a,in,oa,period,&sa); pass(b,in,ob,period,&sb); break;
      }
    }
    const double h0=med(sa.heavy),h1=med(sb.heavy),h2=med(sc.heavy);
    const double m0=mean(sa.all),m1=mean(sb.all),m2=mean(sc.all);
    printf("%5d | %7.2f / %7.2f / %7.2f | %6.3f / %6.3f / %6.3f | heavy %+5.1f%% mean %+5.1f%% | %s\n",
      taps,h0,h1,h2,m0,m1,m2,100*(h2/h1-1),100*(m2/m1-1), same?"bit-identical":"DIFFER");
  }
}
