#include <limits>
#include "nrs.hpp"
#include "linAlg.hpp"
#include "plugins/RANSktau.hpp"

occa::memory cdsSolve(const int is, cds_t* cds, dfloat time, int stage)
{
  platform->timer.tic("scalar rhs", 1);  
  mesh_t* mesh = cds->mesh[0];
  oogs_t* gsh = cds->gshT;
  if(is) {
    mesh = cds->meshV;
    gsh = cds->gsh;
  }

  occa::memory o_Si = cds->o_S.slice(cds->fieldOffsetScan[is] * sizeof(dfloat), cds->fieldOffset[is] * sizeof(dfloat));

  occa::memory o_k = RANSktau::o_k_t();
  occa::memory o_tau = RANSktau::o_tau_t();
  
  platform->o_mempool.slice1.copyFrom(cds->o_BF, cds->fieldOffset[is] * sizeof(dfloat), 0,  cds->fieldOffsetScan[is] * sizeof(dfloat));
  cds->helmholtzRhsBCKernel(mesh->Nelements,
                            mesh->o_sgeo,
                            mesh->o_vmapM,
                            mesh->o_EToB,
                            is,
                            time,
                            cds->fieldOffset[is],
                            mesh->o_x,
                            mesh->o_y,
                            mesh->o_z,
                            o_Si,
                            cds->o_EToB[is],
                            *(cds->o_usrwrk),
			    o_k,
			    o_tau,
			    cds->o_U,
                            platform->o_mempool.slice1);

  platform->timer.toc("scalar rhs");  
  platform->o_mempool.slice0.copyFrom(o_Si, mesh->Nlocal * sizeof(dfloat));
  ellipticSolve(cds->solver[is], platform->o_mempool.slice1, platform->o_mempool.slice0);

  return platform->o_mempool.slice0;
}


