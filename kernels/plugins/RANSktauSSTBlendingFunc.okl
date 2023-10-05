void SSTBlendingFunc(const dfloat rho, const dfloat mueLam, const dfloat k, const dfloat tau, const dfloat stmag2, const dfloat xk, const dfloat yw, dfloat &f1, dfloat &mut)
{
  dfloat iyd = 0.0;
  if(yw > p_tiny)
    iyd = 1 / yw;
  const dfloat iyd2 = iyd * iyd;

  const dfloat k2 = sqrt(k);
  const dfloat crit = yw * k2 / (500.0 * mueLam * p_betainf_str);
  dfloat arg1_1 = 500.0 * p_beta0_SST / 6.0;
  dfloat f2 = 1.0;
  if(crit > p_tiny){
    const dfloat arg2_1 = k2 * tau * iyd / p_betainf_str;
    const dfloat arg2_2 = 500.0 * mueLam * tau * iyd2;
    arg1_1 = fmax(arg2_1, arg2_2);
    const dfloat arg2 = fmax(2.0 * arg2_1, arg2_2);
    const dfloat arg2sq = arg2 * arg2;
    f2 = tanh(arg2sq);
  }

  f1 = 1.0;
  if(yw > p_tiny){
    const dfloat itau = 1 / (tau + p_tiny);
    const dfloat den1_1 = 2.0 * p_sigom2 * xk * itau;
    const dfloat den1_2 = p_tinySST * tau * tau;
    const dfloat den1 = fmax(den1_1, den1_2);
    const dfloat arg1_2 = 4.0 * p_sigom2 * k * iyd2 / den1;
    const dfloat arg1 = fmin(arg1_1, arg1_2);
    const dfloat arg1_4 = arg1 * arg1 * arg1 * arg1;
    f1 = tanh(arg1_4);
  }

  mut = rho * k * tau;
  const dfloat stmagn = sqrt(2.0 * stmag2);
  const dfloat argn = f2 * stmagn; //note: stmag2 is missing factor 2
  const dfloat argtau = argn * tau;
  if(p_alp1 < argtau){
    mut = 0.0;
    if(argn > 0.0)
      mut = rho * p_alp1 * k / argn;
  }
}

