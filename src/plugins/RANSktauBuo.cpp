#include "nrs.hpp"
#include "platform.hpp"
#include "nekInterfaceAdapter.hpp"
#include "RANSktau.hpp"
#include "postProcessing.hpp"
#include "linAlg.hpp"
#include "RANSktauBuo.hpp"

namespace{
static nrs_t* nrs;

int kFieldIndex;

static dfloat rho;
static dfloat mueLam;
static dfloat conductivity;
static dfloat Ri;

static occa::memory o_gvec;
static occa::memory o_T;
static occa::memory o_k;
static occa::memory o_tau;

static occa::kernel ktauBuoDiffKernel;
static occa::kernel ktauBuoComputeKernel;
static occa::kernel ktauBuoForceKernel;

static bool setupCalled = 0;

static dfloat coeff[] = {
  1.0,             // alpinf_str
  0.52,            // alp_inf
  1./0.9,          // inverse turbulent Prandtl number
  0.3              // Cs_buo
};
}

void RANSbuo::buildKernel(occa::properties _kernelInfo)
{
  occa::properties kernelInfo;
  if (!kernelInfo.get<std::string>("defines/p_alpinf_str").size())
    kernelInfo["defines/p_alpinf_str"] = coeff[0];
  if (!kernelInfo.get<std::string>("defines/p_alp_inf").size())
    kernelInfo["defines/p_alp_inf"] = coeff[1];
  if (!kernelInfo.get<std::string>("defines/p_iPrt").size())
    kernelInfo["defines/p_iPrt"] = coeff[2];
  if (!kernelInfo.get<std::string>("defines/p_Cs").size())
    kernelInfo["defines/p_Cs"] = coeff[3];

  const int verbose = platform->options.compareArgs("VERBOSE", "TRUE") ? 1 : 0;

  if(platform->comm.mpiRank == 0 && verbose) {
    std::cout << "\nRANSktau Buoyancy settings\n";
    std::cout << kernelInfo << std::endl;
  }

  kernelInfo += _kernelInfo;

  int rank = platform->comm.mpiRank;
  const std::string oklpath = getenv("NEKRS_KERNEL_DIR");
  const std::string path = oklpath + "/plugins/";
  std::string fileName, kernelName;
  const std::string extension = ".okl";
  {
    kernelName = "ktauBuoDiff";
    fileName = path + kernelName + extension;
    ktauBuoDiffKernel = platform->device.buildKernel(fileName, kernelInfo, true);

    kernelName = "ktauBuoCompute";
    fileName = path + kernelName + extension;
    ktauBuoComputeKernel = platform->device.buildKernel(fileName, kernelInfo, true);

    kernelName = "ktauBuoForce";
    fileName = path + kernelName + extension;
    ktauBuoForceKernel = platform->device.buildKernel(fileName, kernelInfo, true);
  }
}

void RANSbuo::updateForce(occa::memory o_FU)
{
  mesh_t *mesh = nrs->meshV;
  cds_t *cds = nrs->cds;

  ktauBuoForceKernel(mesh->Nlocal,
                     nrs->fieldOffset,
                     Ri,
                     o_gvec,
                     o_T,
                     o_FU);
}

void RANSbuo::updateProperties()
{
  mesh_t *mesh = nrs->meshV;
  cds_t *cds = nrs->cds;
  
  occa::memory o_temp_mue = cds->o_diff + 0 * cds->fieldOffset[0] * sizeof(dfloat);

  //SGDH
  occa::memory o_mut = RANSktau::o_mue_t();
  ktauBuoDiffKernel(mesh->Nlocal, conductivity, o_mut, o_temp_mue);
}

void RANSbuo::updateSourceTerms()
{
  mesh_t *mesh = nrs->meshV;
  cds_t *cds = nrs->cds;

  occa::memory o_FS = cds->o_FS + cds->fieldOffsetScan[kFieldIndex] * sizeof(dfloat);
  occa::memory o_BFdiag = cds->o_BFDiag + cds->fieldOffsetScan[kFieldIndex] * sizeof(dfloat);

  occa::memory o_Tgrad = platform->o_mempool.slice0;

  nrs->gradientVolumeKernel(mesh->Nelements,
                            mesh->o_vgeo,
                            mesh->o_D,
                            nrs->fieldOffset,
                            o_T,
                            o_Tgrad);

  oogs::startFinish(o_Tgrad,
                    3,
                    nrs->fieldOffset,
                    ogsDfloat,
                    ogsAdd,
                    nrs->gsh);

  platform->linAlg->axmyMany(mesh->Nlocal,
                             3,
                             nrs->fieldOffset,
                             0,
                             1.0,
                             mesh->o_invLMM,
                             o_Tgrad);

  
  ktauBuoComputeKernel(mesh->Nelements,
                       nrs->fieldOffset,
                       rho,
                       Ri,
                       o_gvec,
                       o_k,
                       o_tau,
                       o_Tgrad,
                       o_BFdiag,
                       o_FS);
}

void RANSbuo::setup(nrs_t *nrsIn, dfloat mueIn, dfloat rhoIn, int ifld, dfloat RiIn, dfloat *gIn)
{
  if(setupCalled)
    return;

  nrs = nrsIn;
  mueLam = mueIn;
  rho = rhoIn;
  kFieldIndex = ifld;
  Ri = RiIn;
  o_gvec = platform->device.malloc(3 * sizeof(dfloat), gIn);

  cds_t *cds = nrs->cds;
  mesh_t *mesh = nrs->meshV;

  o_T = cds->o_S + cds->fieldOffsetScan[0] * sizeof(dfloat);
  o_k = cds->o_S + cds->fieldOffsetScan[kFieldIndex] * sizeof(dfloat);
  o_tau = cds->o_S + cds->fieldOffsetScan[kFieldIndex + 1] * sizeof(dfloat);

  platform->options.getArgs("SCALAR00 DIFFUSIVITY",conductivity);

  setupCalled = 1;
}
