#if !defined(nekrs_RANSktau_hpp_)
#define nekrs_RANSktau_hpp_

#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"

namespace RANSktau
{
void buildKernel(occa::properties kernelInfo);
void updateSourceTerms();
void setup(dfloat mue, dfloat rho, int startIndex);
void setup(dfloat mue, dfloat rho, int startIndex, std::string &model);
void setup(dfloat mue, dfloat rho, int startIndex, std::string &model, occa::memory &o_ywd);
void updateProperties();
const deviceMemory<dfloat> o_mue_t();
occa::memory implicitK(double time, int scalarIdx);
bool setup();
occa::properties RANSInfo();
}

#endif
