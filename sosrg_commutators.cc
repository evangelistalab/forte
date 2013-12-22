
#include <cmath>

#include <libpsio/psio.hpp>
#include <libmints/wavefunction.h>
#include <libmints/molecule.h>

#include "sosrg.h"

using namespace std;
using namespace psi;

namespace psi{ namespace libadaptive{

void SOSRG::commutator_A1_B1_C0(TwoIndex restrict A,TwoIndex restrict B,double sign,double& C)
{
    double sum = 0.0;
    if(srgcomm == SRCommutators){
        loop_p loop_q{
            sum += A[p][q] * B[q][p] * (No_[p] - No_[q]);
        }
    }
    if(srgcomm == MRNOCommutators){
        loop_p loop_q loop_r{
            sum += A[p][q] * B[q][r] * G1_[r][p] - A[p][q] * G1_[q][r] * B[r][p];
        }
    }
    C += sign * sum;
}

void SOSRG::commutator_A1_B1_C1(TwoIndex restrict A,TwoIndex restrict B,double sign,TwoIndex C)
{
    loop_p loop_q{
        double sum = 0.0;
        loop_r{
            sum += A[p][r] * B[r][q] - B[p][r] * A[r][q];
        }
        C[p][q] += sign * sum;
    }
}

void SOSRG::commutator_A1_B2_C0(TwoIndex restrict A,FourIndex restrict B,double sign,double& C)
{
    double sum = 0.0;
    if(srgcomm == MRNOCommutators){  // NOT CHECKED
        loop_p loop_q loop_r loop_s{
            double partial_sum = 0.0;
            loop_t{
                partial_sum += A[r][t] * B[t][s][p][q] - A[t][p] * B[r][s][t][q];
            }
            sum += partial_sum * L2_[p][q][r][s];
        }
    }
    C += sign * sum;
}

void SOSRG::commutator_A1_B2_C1(TwoIndex restrict A,FourIndex restrict B,double sign,TwoIndex C)
{
    if(srgcomm == SRCommutators){
        loop_p loop_q{
            double sum = 0.0;
            loop_r loop_s {
                sum += A[r][s] * B[p][s][q][r] * (No_[r] - No_[s]);
            }
            C[p][q] += sign * sum;
        }
    }
    if(srgcomm == MRNOCommutators){  // NOT CHECKED
        loop_p loop_q{
            double sum = 0.0;
            loop_r loop_s loop_t{
                sum += A[r][s] * B[p][s][q][t] * G1_[t][r] - A[r][s] * B[p][t][q][r] * G1_[s][t];
            }
            C[p][q] += sign * sum;
        }
    }
}

void SOSRG::commutator_A1_B2_C2(TwoIndex restrict A,FourIndex restrict B,double sign,FourIndex C)
{
    loop_p loop_q loop_r loop_s{
        double sum = 0.0;
        loop_t{
            sum += A[p][t] * B[t][q][r][s] - A[q][t] * B[t][p][r][s] - A[t][r] * B[p][q][t][s] + A[t][s] * B[p][q][t][r];
        }
        C[p][q][r][s] += sign * sum;
    }
}

void SOSRG::commutator_A2_B2_C0(FourIndex restrict A,FourIndex restrict B,double sign,double& C)
{
    double sum = 0.0;
    loop_p loop_q loop_r loop_s{
        sum += 0.25 * (A[p][q][r][s] * B[r][s][p][q] - B[p][q][r][s] * A[r][s][p][q]) * (No_[p] * No_[q] * Nv_[r] * Nv_[s]);
    }
    C += sign * sum;
}

void SOSRG::commutator_A2_B2_C1(FourIndex restrict A,FourIndex restrict B,double sign,TwoIndex C)
{
    loop_p loop_q{
        double sum = 0.0;
        loop_r loop_s loop_t{
            sum += 0.5 * (A[t][p][r][s] * B[r][s][t][q] - B[t][p][r][s] * A[r][s][t][q]) * (No_[r] * No_[s] * Nv_[t] + Nv_[r] * Nv_[s] * No_[t]);
        }
        C[p][q] += sign * sum;
    }
}

void SOSRG::commutator_A2_B2_C2(FourIndex restrict A,FourIndex restrict B,double sign,FourIndex C)
{


    loop_p loop_q loop_r loop_s{
        double sum = 0.0;
        loop_t loop_u{
            sum += 0.5 * (A[p][q][t][u] * B[t][u][r][s] - A[t][u][r][s] * B[p][q][t][u]) * (1.0 - No_[t] - No_[u]);
            sum += (  A[p][t][r][u] * B[q][u][s][t]
                    - A[q][t][r][u] * B[p][u][s][t]
                    - A[p][t][s][u] * B[q][u][r][t]
                    + A[q][t][s][u] * B[p][u][r][t]) * (No_[t] - No_[u]);
        }
        C[p][q][r][s] += sign * sum;
    }
}

}} // EndNamespaces
