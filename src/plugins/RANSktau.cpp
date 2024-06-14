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
static std::string mid;

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
static occa::memory o_OijMag2;
static occa::memory o_ywd;
static occa::memory o_ywdg;
static occa::memory o_dgrd;
static occa::memory o_wf;

static occa::kernel computeKernel;
static occa::kernel computeSSTKernel;
static occa::kernel computeGradKernel;
static occa::kernel mueKernel;
static occa::kernel mueSSTKernel;
static occa::kernel limitKernel;
static occa::kernel desLenScaleKernel;
  
static occa::kernel OiOjSkKernel;
static occa::kernel wallFuncKernel;

static bool buildKernelCalled = false;
static bool setupCalled = false;
static bool desScaleCalled = false;
static bool ywdCalled = false;

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
    1e-10,     // tinySST

    //DES parameters
    0.78,      // cdes1
    0.61,      // cdes2
    20.0,      // c_d1
    3.0,       // c_d2
    0.41,      // vkappa

    //Free-stream limiter
    0.01,      // edd_free
    0.5,       // ywlim

    //Wall-function parameters
    30.0,      // yplus
    9.0        // Econ
};
} // namespace

void RANSktau::buildKernel(occa::properties _kernelInfo)
{
  static bool isInitialized = false;
  if (isInitialized) return;
  isInitialized = true;

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
  
  if (!kernelInfo.get<std::string>("defines/p_cdes1").size())
    kernelInfo["defines/p_cdes1"] = coeff[24];
  if (!kernelInfo.get<std::string>("defines/p_cdes2").size())
    kernelInfo["defines/p_cdes2"] = coeff[25];
  if (!kernelInfo.get<std::string>("defines/p_cd1").size())
    kernelInfo["defines/p_cd1"] = coeff[26];
  if (!kernelInfo.get<std::string>("defines/p_cd2").size())
    kernelInfo["defines/p_cd2"] = coeff[27];
  if (!kernelInfo.get<std::string>("defines/p_vkappa").size())
    kernelInfo["defines/p_vkappa"] = coeff[28];
	
  if (!kernelInfo.get<std::string>("defines/p_edd_free").size())
    kernelInfo["defines/p_edd_free"] = coeff[29];
  if (!kernelInfo.get<std::string>("defines/p_ywlim").size())
    kernelInfo["defines/p_ywlim"] = coeff[30];

  if (!kernelInfo.get<std::string>("defines/p_yplus").size())
    kernelInfo["defines/p_yplus"] = coeff[31];
  if (!kernelInfo.get<std::string>("defines/p_Econ").size())
    kernelInfo["defines/p_Econ"] = coeff[32];

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

  std::string installDir;
  installDir.assign(getenv("NEKRS_HOME"));
  occa::properties kernelInfoBC = kernelInfo;
  const std::string bcDataFile = installDir + "/include/bdry/bcData.h";
  kernelInfoBC["includes"] += bcDataFile.c_str();
  {
    kernelName = "RANSktauGradHex3D";
    fileName = path + kernelName + extension;
    computeGradKernel = platform->device.buildKernel(fileName, kernelInfo, true);

    kernelName = "limit";
    fileName = path + kernelName + extension;
    limitKernel = platform->device.buildKernel(fileName, kernelInfo, true);

    kernelName = "OiOjSk";
    fileName = path + kernelName + extension;
    OiOjSkKernel = platform->device.buildKernel(fileName, kernelInfo, true);

    kernelName = "desLenScale";
    fileName = path + kernelName + extension;
    desLenScaleKernel = platform->device.buildKernel(fileName, kernelInfo, true);

    kernelName = "RANSktauComputeHex3D";
    fileName = path + kernelName + extension;
    computeKernel = platform->device.buildKernel(fileName, kernelInfo, true);

    kernelName = "mue";
    fileName = path + kernelName + extension;
    mueKernel = platform->device.buildKernel(fileName, kernelInfo, true);

    const std::string headerFile = path + "RANSktauSSTBlendingFunc.h";
    kernelInfo["includes"] += headerFile.c_str();
    kernelName = "RANSktauComputeSSTHex3D";
    fileName = path + kernelName + extension;
    computeSSTKernel = platform->device.buildKernel(fileName, kernelInfo, true);
    
    kernelName = "mueSST";
    fileName = path + kernelName + extension;
    mueSSTKernel = platform->device.buildKernel(fileName, kernelInfo, true);

    kernelName = "wallFunc";
    fileName = path + kernelName + extension;
    wallFuncKernel = platform->device.buildKernel(fileName, kernelInfoBC, true);
  }


  int Nscalar;
  platform->options.getArgs("NUMBER OF SCALARS", Nscalar);

  nrsCheck(Nscalar < 2, platform->comm.mpiComm, EXIT_FAILURE,
           "%s\n", "Nscalar needs to be >= 2!");
  platform->options.setArgs("VELOCITY STRESSFORMULATION", "TRUE");

  buildKernelCalled = true;
}

void RANSktau::updateProperties()
{
  nrsCheck(!setupCalled || !buildKernelCalled, MPI_COMM_SELF, EXIT_FAILURE,
           "%s\n", "called prior to tavg::setup()!");

  mesh_t *mesh = nrs->meshV;
  cds_t *cds = nrs->cds;

  auto o_mue = nrs->o_mue;
  auto o_diff = cds->o_diff + cds->fieldOffsetScan[kFieldIndex];

  limitKernel(mesh->Nelements * mesh->Np, o_k, o_tau);

  auto o_SijOij = platform->o_memPool.reserve<dfloat>(3 * nrs->NVfields * nrs->fieldOffset);
  postProcessing::strainRotationRate(nrs, true, true, nrs->o_U, o_SijOij);

  platform->linAlg->magSqrSymTensor(mesh->Nelements * mesh->Np, nrs->fieldOffset, o_SijOij, o_SijMag2);

  if(mid == "KTAU" || mid == "KTAU+SWF") OiOjSkKernel(mesh->Nelements * mesh->Np, nrs->fieldOffset, o_SijOij, o_OiOjSk);

  if(mid == "SST+DES") {
    auto o_Oij = o_SijOij.slice(6 * nrs->fieldOffset);
    platform->linAlg->magSqrVector(mesh->Nelements * mesh->Np, nrs->fieldOffset, o_Oij, o_OijMag2);
  }

  //not sure whether i want this here
  o_SijOij.free();

  computeGradKernel(mesh->Nelements,
                    nrs->cds->fieldOffset[kFieldIndex],
                    mesh->o_vgeo,
                    mesh->o_D,
                    o_k,
                    o_tau,
                    o_xk,
                    o_xt,
                    o_xtq);

  if(mid == "KTAU" || mid == "KTAU+SWF"){
    mueKernel(mesh->Nelements * mesh->Np, nrs->fieldOffset, rho, mueLam, o_k, o_tau, o_mut, o_mue, o_diff);
  }
  else if(mid == "SST" || mid == "SST+DES"){
    mueSSTKernel(mesh->Nelements * mesh->Np, nrs->fieldOffset, rho, mueLam, o_k, o_tau, o_SijMag2, o_xk, o_ywd, o_mut, o_mue, o_diff);
  }
}

occa::memory RANSktau::o_mue_t() { return o_mut; }

void RANSktau::updateSourceTerms()
{
  nrsCheck(!setupCalled || !buildKernelCalled, MPI_COMM_SELF, EXIT_FAILURE,
           "%s\n", "called prior to tavg::setup()!");

  mesh_t *mesh = nrs->meshV;
  cds_t *cds = nrs->cds;

  auto o_FS = cds->o_FS + cds->fieldOffsetScan[kFieldIndex];
  auto o_BFDiag = cds->o_BFDiag + cds->fieldOffsetScan[kFieldIndex];

  if(mid == "KTAU" || mid == "KTAU+SWF") {
    computeKernel(mesh->Nelements * mesh->Np,
                  nrs->cds->fieldOffset[kFieldIndex],
                  rho,
                  mueLam,
                  o_k,
                  o_tau,
                  o_SijMag2,
                  o_OiOjSk,
                  o_xk,
                  o_xt,
                  o_xtq,
                  o_BFDiag,
                  o_FS);
    if(mid == "KTAU+SWF") {
      if(platform->options.compareArgs("MOVING MESH", "TRUE") || !ywdCalled) {
        nrs->gradientVolumeKernel(mesh->Nelements,
            mesh->o_vgeo,
            mesh->o_D,
            nrs->fieldOffset,
            o_ywd,
            o_ywdg);
        oogs::startFinish(o_ywdg, nrs->NVfields, nrs->fieldOffset, ogsDfloat, ogsAdd, nrs->gsh);

        platform->linAlg->axmyVector(mesh->Nlocal, nrs->fieldOffset, 0, 1.0, nrs->meshV->o_invLMM, o_ywdg);

        ywdCalled = true;
      }
    }
    wallFuncKernel(mesh->Nelements,
                   nrs->fieldOffset,
                   rho,
                   mueLam,
                   mesh->o_sgeo,
                   mesh->o_vmapM,
                   mesh->o_EToB,
                   nrs->o_EToB,
                   nrs->o_U,
                   o_k,
                   o_tau,
                   o_ywdg,
                   o_wf);
  }
  else {
    bool ifDES = false;
    if(mid == "SST+DES") {
      ifDES = true;
      if(platform->options.compareArgs("MOVING MESH", "TRUE") || !desScaleCalled) {
        desLenScaleKernel(mesh->Nelements,
                          nrs->fieldOffset,
                          mesh->o_x,
                          mesh->o_y,
                          mesh->o_z,
                          o_dgrd);
        desScaleCalled = true;
      }
    }
    computeSSTKernel(mesh->Nelements * mesh->Np,
                     nrs->cds->fieldOffset[kFieldIndex],
                     static_cast<int>(ifDES),
                     rho,
                     mueLam,
                     o_k,
                     o_tau,
                     o_SijMag2,
                     o_xk,
                     o_xt,
                     o_xtq,
                     o_dgrd,
                     o_ywd,
                     o_OijMag2,
                     o_BFDiag,
                     o_FS);
  }
}

void RANSktau::setup(nrs_t *nrsIn, dfloat mueIn, dfloat rhoIn, int ifld, std::string & model)
{
  static bool isInitialized = false;
  if (isInitialized) return; 
  isInitialized = true;

  upperCase(model);
  if(model == "DEFAULT" || model == "KTAU") mid = "KTAU";
  if(model == "SST" || model == "SST+DES" || model == "KTAU+SWF"){
    mid = model;
    if(o_ywd == o_NULL){
      printf("\n%s model requires wall distance\nCheck usage\n",model.c_str());
      exit(1);
    }
  }
  if(model == "KTAU+SWF") {
    if(o_wf == o_NULL) {
      printf("\n%s model requires WF arrays\nCheck usage\n",model.c_str());
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
  
  if(mid == "SST+DES"){
    o_dgrd = platform->device.malloc<dfloat>(mesh->Nelements);
    o_OijMag2 = platform->device.malloc<dfloat>(nrs->fieldOffset);
  }
  else if(mid == "KTAU+SWF"){
    o_ywdg = platform->device.malloc<dfloat>(3 * nrs->fieldOffset);
  }
  setupCalled = true;
}

void RANSktau::setup(nrs_t *nrsIn, dfloat mueIn, dfloat rhoIn, int ifld, std::string & model, occa::memory o_ywdIn)
{
  o_ywd = o_ywdIn;

  RANSktau::setup(nrsIn, mueIn, rhoIn, ifld, model);
}

void RANSktau::setup(nrs_t *nrsIn, dfloat mueIn, dfloat rhoIn, int ifld, std::string & model, occa::memory o_ywdIn, occa::memory o_wfIn)
{
  o_ywd = o_ywdIn;

  o_wf = o_wfIn;

  RANSktau::setup(nrsIn, mueIn, rhoIn, ifld, model);
}
