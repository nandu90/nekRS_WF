#include "nrs.hpp"
#include "platform.hpp"
#include "nekInterfaceAdapter.hpp"
#include "RANSktau.hpp"
#include "postProcessing.hpp"
#include "linAlg.hpp"

// private members
namespace {
static nrs_t *nrs;

int kFieldIndex;
int mid;

dfloat rho;
dfloat mueLam;

static occa::memory o_mut;

static occa::memory o_k;
static occa::memory o_tau;

static occa::memory o_xk;
static occa::memory o_xt;
static occa::memory o_xtq;
static occa::memory o_OiOjSk;
static occa::memory o_SijMag2;
static occa::memory o_ywd;

static occa::kernel computeKernel;
static occa::kernel computeGradKernel;
static occa::kernel mueKernel;
static occa::kernel limitKernel;
  
static occa::kernel SijMag2OiOjSkKernel;

static bool setupCalled = 0;

static dfloat coeff[] = {
    0.6,       // sigma_k
    0.5,       // sigma_tau
    1.0,       // alpinf_str
    0.0708,    // beta0
    0.41,      // kappa
    0.09,      // betainf_str
    0.0,       // sigd_min
    1.0 / 8.0, // sigd_max
    400.0,     // fb_c1st
    400.0,     // fb_c2st
    85.0,      // fb_c1
    100.0,     // fb_c2
    0.52,      // alp_inf
    1e-8,      // TINY
    0,         // Pope correction

    //Additional SST parameters
    0.85,      // sigma_k_SST
    0.075,     // beta0_SST
    5.0 / 9.0, // alp_inf_SST
    0.31,      // alp1
    0.0828,    // beta2
    1.0,       // sigk2
    0.856,     // sigom2
    0.44,      // gamma2
    1e-10      // tinySST
};
} // namespace

void RANSktau::buildKernel(occa::properties _kernelInfo)
{

  occa::properties kernelInfo;
  if (!kernelInfo.get<std::string>("defines/p_sigma_k").size())
    kernelInfo["defines/p_sigma_k"] = coeff[0];
  if (!kernelInfo.get<std::string>("defines/p_sigma_tau").size())
    kernelInfo["defines/p_sigma_tau"] = coeff[1];
  if (!kernelInfo.get<std::string>("defines/p_alpinf_str").size())
    kernelInfo["defines/p_alpinf_str"] = coeff[2];
  if (!kernelInfo.get<std::string>("defines/p_beta0").size())
    kernelInfo["defines/p_beta0"] = coeff[3];
  if (!kernelInfo.get<std::string>("defines/p_kappa").size())
    kernelInfo["defines/p_kappa"] = coeff[4];
  if (!kernelInfo.get<std::string>("defines/p_betainf_str").size())
    kernelInfo["defines/p_betainf_str"] = coeff[5];
  if (!kernelInfo.get<std::string>("defines/p_ibetainf_str3").size())
    kernelInfo["defines/p_ibetainf_str3"] = 1 / pow(coeff[5], 3);
  if (!kernelInfo.get<std::string>("defines/p_sigd_min").size())
    kernelInfo["defines/p_sigd_min"] = coeff[6];
  if (!kernelInfo.get<std::string>("defines/p_sigd_max").size())
    kernelInfo["defines/p_sigd_max"] = coeff[7];
  if (!kernelInfo.get<std::string>("defines/p_fb_c1st").size())
    kernelInfo["defines/p_fb_c1st"] = coeff[8];
  if (!kernelInfo.get<std::string>("defines/p_fb_c2st").size())
    kernelInfo["defines/p_fb_c2st"] = coeff[9];
  if (!kernelInfo.get<std::string>("defines/p_fb_c1").size())
    kernelInfo["defines/p_fb_c1"] = coeff[10];
  if (!kernelInfo.get<std::string>("defines/p_fb_c2").size())
    kernelInfo["defines/p_fb_c2"] = coeff[11];
  if (!kernelInfo.get<std::string>("defines/p_alp_inf").size())
    kernelInfo["defines/p_alp_inf"] = coeff[12];
  if (!kernelInfo.get<std::string>("defines/p_tiny").size())
    kernelInfo["defines/p_tiny"] = coeff[13];
  if (!kernelInfo.get<std::string>("defines/p_pope").size())
    kernelInfo["defines/p_pope"] = coeff[14];

  if (!kernelInfo.get<std::string>("defines/p_sigma_k_SST").size())
    kernelInfo["defines/p_sigma_k_SST"] = coeff[15];
  if (!kernelInfo.get<std::string>("defines/p_beta0_SST").size())
    kernelInfo["defines/p_beta0_SST"] = coeff[16];
  if (!kernelInfo.get<std::string>("defines/p_alp_inf_SST").size())
    kernelInfo["defines/p_alp_inf_SST"] = coeff[17];
  if (!kernelInfo.get<std::string>("defines/p_alp1").size())
    kernelInfo["defines/p_alp1"] = coeff[18];
  if (!kernelInfo.get<std::string>("defines/p_beta2").size())
    kernelInfo["defines/p_beta2"] = coeff[19];
  if (!kernelInfo.get<std::string>("defines/p_sigk2").size())
    kernelInfo["defines/p_sigk2"] = coeff[20];
  if (!kernelInfo.get<std::string>("defines/p_sigom2").size())
    kernelInfo["defines/p_sigom2"] = coeff[21];
  if (!kernelInfo.get<std::string>("defines/p_gamma2").size())
    kernelInfo["defines/p_gamma2"] = coeff[22];
  if (!kernelInfo.get<std::string>("defines/p_tinySST").size())
    kernelInfo["defines/p_tinySST"] = coeff[23];
  
  const int verbose = platform->options.compareArgs("VERBOSE", "TRUE") ? 1 : 0;

  if (platform->comm.mpiRank == 0 && verbose) {
    std::cout << "\nRANSktau settings\n";
    std::cout << kernelInfo << std::endl;
  }

  kernelInfo += _kernelInfo;

  int rank = platform->comm.mpiRank;
  const std::string oklpath = getenv("NEKRS_KERNEL_DIR");
  const std::string path = oklpath + "/plugins/";
  std::string fileName, kernelName;
  const std::string extension = ".okl";
  {
    kernelName = "RANSktauGradHex3D";
    fileName = path + kernelName + extension;
    computeGradKernel = platform->device.buildKernel(fileName, kernelInfo, true);

    kernelName = "limit";
    fileName = path + kernelName + extension;
    limitKernel = platform->device.buildKernel(fileName, kernelInfo, true);

    kernelName = "SijMag2OiOjSk";
    fileName = path + kernelName + extension;
    SijMag2OiOjSkKernel = platform->device.buildKernel(fileName, kernelInfo, true);

    const std::string headerFile = path + "RANSktauSSTBlendingFunc" + extension;
    kernelInfo["includes"] += headerFile.c_str();
    kernelName = "RANSktauComputeHex3D";
    fileName = path + kernelName + extension;
    computeKernel = platform->device.buildKernel(fileName, kernelInfo, true);

    kernelName = "mue";
    fileName = path + kernelName + extension;
    mueKernel = platform->device.buildKernel(fileName, kernelInfo, true);
  }

  int Nscalar;
  platform->options.getArgs("NUMBER OF SCALARS", Nscalar);

  nrsCheck(Nscalar < 2, platform->comm.mpiComm, EXIT_FAILURE,
           "%s\n", "Nscalar needs to be >= 2!");
  platform->options.setArgs("VELOCITY STRESSFORMULATION", "TRUE");
}

void RANSktau::updateProperties()
{
  mesh_t *mesh = nrs->meshV;
  cds_t *cds = nrs->cds;

  occa::memory o_mue = nrs->o_mue;
  occa::memory o_diff = cds->o_diff + cds->fieldOffsetScan[kFieldIndex];

  limitKernel(mesh->Nelements * mesh->Np, o_k, o_tau);

  occa::memory o_SijOij = platform->o_memPool.reserve<dfloat>(3 * nrs->NVfields * nrs->fieldOffset);
  postProcessing::strainRotationRate(nrs, true, true, o_SijOij);

  SijMag2OiOjSkKernel(mesh->Nelements * mesh->Np, nrs->fieldOffset, 1, o_SijOij, o_OiOjSk, o_SijMag2);

  computeGradKernel(mesh->Nelements,
                    nrs->cds->fieldOffset[kFieldIndex],
                    mesh->o_vgeo,
                    mesh->o_D,
                    o_k,
                    o_tau,
                    o_xk,
                    o_xt,
                    o_xtq);

  mueKernel(mesh->Nelements * mesh->Np, nrs->fieldOffset, mid, rho, mueLam, o_k, o_tau, o_SijMag2, o_xk, o_ywd, o_mut, o_mue, o_diff);
}

occa::memory RANSktau::o_mue_t() { return o_mut; }

void RANSktau::updateSourceTerms()
{
  mesh_t *mesh = nrs->meshV;
  cds_t *cds = nrs->cds;

  occa::memory o_FS = cds->o_FS + cds->fieldOffsetScan[kFieldIndex];
  occa::memory o_BFDiag = cds->o_BFDiag + cds->fieldOffsetScan[kFieldIndex];
    
  computeKernel(mesh->Nelements * mesh->Np,
                nrs->cds->fieldOffset[kFieldIndex],
                mid,
                rho,
                mueLam,
                o_k,
                o_tau,
                o_SijMag2,
                o_OiOjSk,
                o_xk,
                o_xt,
                o_xtq,
                o_ywd,
		o_BFDiag,
                o_FS);
}

void RANSktau::setup(nrs_t *nrsIn, dfloat mueIn, dfloat rhoIn, int ifld, std::string & model)
{
  if (setupCalled)
    return;

  upperCase(model);
  if(model.compare("DEFAULT") == 0 || model.compare("KTAU") == 0) mid = 0;
  if(model.compare("SST") == 0){
    mid = 1;
    if(o_ywd == o_NULL){
      printf("\nSST model requires wall distance\nCheck usage\n");
      exit(1);
    }
  }

  nrs = nrsIn;
  mueLam = mueIn;
  rho = rhoIn;
  kFieldIndex = ifld;

  cds_t *cds = nrs->cds;
  mesh_t *mesh = nrs->meshV;

  o_k = cds->o_S + cds->fieldOffsetScan[kFieldIndex];
  o_tau = cds->o_S + cds->fieldOffsetScan[kFieldIndex + 1];

  o_mut = platform->device.malloc<dfloat>(cds->fieldOffset[kFieldIndex]);

  if (!cds->o_BFDiag.ptr()) {
    cds->o_BFDiag = platform->device.malloc<dfloat>(cds->fieldOffsetSum);
    platform->linAlg->fill(cds->fieldOffsetSum, 0.0, cds->o_BFDiag);
  }

  o_OiOjSk = platform->device.malloc<dfloat>(nrs->fieldOffset);
  o_SijMag2 = platform->device.malloc<dfloat>(nrs->fieldOffset);
  o_xk = platform->device.malloc<dfloat>(cds->fieldOffset[kFieldIndex]);
  o_xt = platform->device.malloc<dfloat>(cds->fieldOffset[kFieldIndex]);
  o_xtq = platform->device.malloc<dfloat>(cds->fieldOffset[kFieldIndex]);

  setupCalled = 1;
}

void RANSktau::setup(nrs_t *nrsIn, dfloat mueIn, dfloat rhoIn, int ifld, std::string & model, double *ywd)
{
  o_ywd = platform->device.malloc<dfloat>(nrsIn->fieldOffset);
  o_ywd.copyFrom(ywd, nrsIn->fieldOffset);

  RANSktau::setup(nrsIn, mueIn, rhoIn, ifld, model);
}
