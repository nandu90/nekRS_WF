#include <math.h>

static int ciMode = 0;

#define EPS 3e-2

void ciSetup(MPI_Comm comm, setupAide &options)
{
  /* if(caseType == "DES") { */
    options.setArgs("POLYNOMIAL DEGREE", std::string("3"));
    options.setArgs("CUBATURE POLYNOMIAL DEGREE", std::string("4"));
    options.setArgs("CONSTANT FLOW RATE", "TRUE");
    options.setArgs("CONSTANT FLOW DIRECTION", "X");
    options.setArgs("RESTART FILE NAME", std::string("des.fld+time=0"));
    options.setArgs("SOLUTION OUTPUT INTERVAL", "-1");
    options.setArgs("VISCOSITY", std::to_string(1.0/10595.0));
    options.setArgs("DENSITY", std::string("1.0"));
    options.setArgs("END TIME", std::string("1"));
    options.setArgs("ADVECTION TYPE", "CUBATURE+CONVECTIVE");
    options.setArgs("DT", std::string("5e-3"));

    options.setArgs("PRESSURE SOLVER TOLERANCE", std::string("1e-4"));
    options.setArgs("PRESSURE PRECONDITIONER", "MULTIGRID");
    options.setArgs("PRESSURE MULTIGRID COARSE SOLVE", "TRUE");
    options.setArgs("PRESSURE MULTIGRID SMOOTHER", "FOURTHOPTCHEBYSHEV+ASM");
    options.setArgs("PRESSURE MULTIGRID CHEBYSHEV DEGREE", std::string("3"));
    options.setArgs("PRESSURE MULTIGRID CHEBYSHEV MAX EIGENVALUE BOUND FACTOR", std::string("1.1"));
    options.setArgs("PRESSURE INITIAL GUESS", "PROJECTION-ACONJ");

    options.setArgs("VELOCITY SOLVER TOLERANCE", std::string("1e-6"));
    options.setArgs("VELOCITY BLOCK SOLVER", "TRUE");
    options.setArgs("VELOCITY INITIAL GUESS", "EXTRAPOLATION");

    options.setArgs("SCALAR00 SOLVER TOLERANCE", std::string("1e-6"));
    options.setArgs("SCALAR00 INITIAL GUESS", "EXTRAPOLATION");

    options.setArgs("SCALAR01 SOLVER TOLERANCE", std::string("1e-8"));
    options.setArgs("SCALAR01 INITIAL GUESS", "EXTRAPOLATION");

    options.setArgs("SCALAR02 SOLVER TOLERANCE", std::string("1e-6"));
    options.setArgs("SCALAR02 INITIAL GUESS", "EXTRAPOLATION");
  /* } */
}

void ciTestErrors(nrs_t *nrs, double time, int tstep, occa::memory o_avgU)
{
  if(!nrs->lastStep) return;

  mesh_t *mesh = nrs->meshV;
  cds_t *cds = nrs->cds;

  auto o_Sij = nrs->strainRate();

  occa::memory o_tauw = platform->device.malloc<dfloat>(nrs->fieldOffset);
  platform->linAlg->fill(mesh->Nlocal, 0.0, o_tauw);

  tauWallKernel(mesh->Nelements,
    nrs->fieldOffset,
    mesh->o_sgeo,
    mesh->o_vmapM,
    nrs->o_EToB,
    nrs->o_mue,
    o_Sij,
    o_tauw);

	double norm = platform->linAlg->norm2(mesh->Nlocal, o_tauw, platform->comm.mpiComm);

	printf("\nNorm2: %.4e\n",norm);

  occa::memory o_xintercept = platform->device.malloc<dfloat>(mesh->Nelements);
  const int bID = 1; //lowerWall
  intercept(mesh->Nelements,
            bID,
            4.0,
            5.0,
            mesh->o_sgeo,
            mesh->o_vmapM,
            mesh->o_EToB,
            mesh->o_x,
            mesh->o_z,
            o_tauw,
            o_xintercept);

  dfloat *xintercept = (dfloat *) calloc(mesh->Nelements, sizeof(dfloat));
  o_xintercept.copyTo(xintercept);

  auto relen = -1e9;
  for (dlong i = 0; i < mesh->Nelements; i++) {
    relen = (relen > xintercept[i]) ? relen : xintercept[i];
  }

  MPI_Allreduce(MPI_IN_PLACE, &relen, 1, MPI_DFLOAT, MPI_MAX, platform->comm.mpiComm);

  //https://turbmodels.larc.nasa.gov/Other_LES_Data/2Dhill_periodic/hill_LES_cf_digitized.dat
  const auto lenLES = 4.69;
  const auto lenErr = std::abs(relen - lenLES) / lenLES; 

  if (platform->comm.mpiRank == 0) {
    printf("\n Reattachment length = %g : relative error = %g\n", relen, lenErr);
  }

	CiEvalTest("Re-attachment length", fabs(lenErr) < EPS, "with relative LinfErr:" + to_string_f(lenErr));

  occa::memory o_tau = cds->o_S;
  platform->linAlg->fill(mesh->Nlocal, 1.0, o_tau);
  platform->linAlg->axmy(mesh->Nlocal, 1.0, o_tauw, o_tau);

	norm = platform->linAlg->norm2(mesh->Nlocal, cds->o_S, platform->comm.mpiComm);

	printf("\nNorm2: %.4e\n",norm);
}
