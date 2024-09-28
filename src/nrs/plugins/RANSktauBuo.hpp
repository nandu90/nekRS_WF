#if !defined(nekrs_RANSktauBuo_hpp_)
#define nekrs_RANSktauBuo_hpp_

#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"

namespace RANSbuo
{
  void buildKernel(occa::properties kernelInfo);
  void updateProperties();
  void updateSourceTerms();
  void setup(int startIndex, dfloat RiIn, dfloat *gIn);
  void updateForce(occa::memory o_FU);
  occa::memory implicitBuo(double time, int scalarIdx);
}
#endif
