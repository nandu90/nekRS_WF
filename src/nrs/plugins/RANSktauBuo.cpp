#include "nrs.hpp"
#include "platform.hpp"
#include "nekInterfaceAdapter.hpp"
#include "RANSktau.hpp"
#include "linAlg.hpp"
#include "RANSktauBuo.hpp"

namespace{
static nrs_t* nrs;

int kFieldIndex;

static dfloat rho;
static dfloat mueLam;
static dfloat specificHeat;
static dfloat conductivity;
static dfloat Ri;

static occa::memory o_gvec;
static occa::memory o_T;
static occa::memory o_k;
static occa::memory o_tau;
static occa::memory o_implicitBuo;

static occa::kernel diffKernel;
static occa::kernel computeKernel;

static bool setupCalled = 0;
static bool buildKernelCalled = 0;

static dfloat coeff[] = {
  1./0.85,         // inverse turbulent Prandtl number
  0.3              // Cs_buo
};
} //namespace

occa::memory RANSbuo::implicitBuo(double time, int scalarIdx)
{
  mesh_t* mesh = nrs->mesh;
  auto o_implicitKtau = RANSktau::implicitK(time, scalarIdx);

  if (scalarIdx == kFieldIndex){
    auto o_term = o_implicitBuo.slice(0 * nrs->fieldOffset, nrs->fieldOffset);
    platform->linAlg->axpby(mesh->Nlocal, 1.0, o_implicitKtau, 1.0, o_term);
    return o_term;
  }
  if (scalarIdx == kFieldIndex + 1){
    auto o_term = o_implicitBuo.slice(1 * nrs->fieldOffset, nrs->fieldOffset);
    platform->linAlg->axpby(mesh->Nlocal, 1.0, o_implicitKtau, 1.0, o_term);
    return o_term;
  }
  return o_NULL;
}

void RANSbuo::buildKernel(occa::properties _kernelInfo)
{
  RANSktau::buildKernel(_kernelInfo);

  occa::properties kernelInfo;

  if (!kernelInfo.get<std::string>("defines/p_iPrt").size())
    kernelInfo["defines/p_iPrt"] = coeff[0];
  if (!kernelInfo.get<std::string>("defines/p_Cs").size())
    kernelInfo["defines/p_Cs"] = coeff[1];

  const int verbose = platform->options.compareArgs("VERBOSE", "TRUE") ? 1 : 0;

  if(platform->comm.mpiRank == 0 && verbose) {
    std::cout << "\nRANSktau Buoyancy settings\n";
    std::cout << kernelInfo << std::endl;
  }

  kernelInfo += _kernelInfo + RANSktau::RANSInfo();

  auto buildKernel = [&kernelInfo](const std::string &kernelName) {
    const auto path = getenv("NEKRS_KERNEL_DIR") + std::string("/nrs/plugins/");
    const auto fileName = path + "RANSktauBuo.okl";
    const auto reqName = "RANSBuo::";
    if (platform->options.compareArgs("REGISTER ONLY", "TRUE")) {
      platform->kernelRequests.add(reqName, fileName, kernelInfo);
      return occa::kernel();
    }
    else{
      buildKernelCalled = 1;
      return platform->kernelRequests.load(reqName, kernelName);
    }
  };

  diffKernel = buildKernel("ktauBuoDiff");
  computeKernel = buildKernel("ktauBuoCompute");
}

void RANSbuo::updateProperties()
{
  nekrsCheck(!setupCalled || !buildKernelCalled,
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s\n",
             "called prior to RANSBuo::setup()!");

  mesh_t *mesh = nrs->mesh;
  cds_t *cds = nrs->cds;
  
  auto o_temp_mue = cds->o_diff;

  //SGDH
  auto o_mut = RANSktau::o_mue_t();
  diffKernel(mesh->Nlocal, conductivity, specificHeat, o_mut, o_temp_mue);
}

void RANSbuo::updateSourceTerms()
{
  nekrsCheck(!setupCalled || !buildKernelCalled,
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s\n",
             "called prior to RANSBuo::setup()!");

  mesh_t *mesh = nrs->mesh;
  cds_t *cds = nrs->cds;

  occa::memory o_Tgrad = platform->o_memPool.reserve<dfloat>(nrs->NVfields * nrs->fieldOffset);

  nrs->gradientVolumeKernel(mesh->Nelements,
                            mesh->o_vgeo,
                            mesh->o_D,
                            nrs->fieldOffset,
                            o_T,
                            o_Tgrad);

  oogs::startFinish(o_Tgrad,
                    nrs->NVfields,
                    nrs->fieldOffset,
                    ogsDfloat,
                    ogsAdd,
                    nrs->gsh);

  platform->linAlg->axmyVector(mesh->Nlocal,
			       nrs->fieldOffset,
			       0,
			       1.0,
			       mesh->o_invLMM,
			       o_Tgrad);

  
  computeKernel(mesh->Nelements * mesh->Np,
                nrs->fieldOffset,
                rho,
								Ri,
								o_gvec,
								o_k,
								o_tau,
								o_Tgrad,
								o_implicitBuo);
}

void RANSbuo::setup(int ifld, dfloat RiIn, dfloat *gIn)
{
  nekrsCheck(!RANSktau::setup(),
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s\n",
             "called prior to RANSktau::setup()!");

  if(setupCalled)
    return;

  nrs = dynamic_cast<nrs_t *>(platform->solver);

  platform->options.getArgs("VISCOSITY", mueLam);
  platform->options.getArgs("DENSITY", rho);

  kFieldIndex = ifld;
  Ri = RiIn;
  o_gvec = platform->device.malloc<dfloat>(3, gIn);

  cds_t *cds = nrs->cds;
  mesh_t *mesh = nrs->mesh;

  o_T = cds->o_S + cds->fieldOffsetScan[0];
  o_k = cds->o_S + cds->fieldOffsetScan[kFieldIndex];
  o_tau = cds->o_S + cds->fieldOffsetScan[kFieldIndex + 1];

  o_implicitBuo = platform->device.malloc<dfloat>(2 * nrs->fieldOffset);
  cds->userImplicitLinearTerm = RANSbuo::implicitBuo;

  dfloat rhoCp;
  platform->options.getArgs("SCALAR00 DENSITY",rhoCp);
  specificHeat = rhoCp / rho;	
  platform->options.getArgs("SCALAR00 DIFFUSIVITY",conductivity);

  setupCalled = 1;
}
