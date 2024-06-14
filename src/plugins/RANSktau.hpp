#if !defined(nekrs_RANSktau_hpp_)
#define nekrs_RANSktau_hpp_

#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"

namespace RANSktau
{
void buildKernel(occa::properties kernelInfo);
void updateSourceTerms();
void setup(nrs_t* nrsIn, dfloat mue, dfloat rho, int startIndex, std::string & model);
void setup(nrs_t* nrsIn, dfloat mue, dfloat rho, int startIndex, std::string & model, occa::memory o_ywdIn);
void setup(nrs_t* nrsIn, dfloat mue, dfloat rho, int startIndex, std::string & model, occa::memory o_ywdIn, occa::memory o_wfIn);
void updateProperties();
occa::memory o_mue_t();
}

#endif
