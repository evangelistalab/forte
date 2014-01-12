#include <cmath>

#include <boost/timer.hpp>

#include <libpsio/psio.hpp>
#include <libmints/wavefunction.h>
#include <libmints/molecule.h>

#include "mosrg.h"

using namespace std;
using namespace psi;

double t_commutator_A1_B1_C0 = 0;
double t_commutator_A1_B1_C1 = 0;
double t_commutator_A1_B2_C0 = 0;
double t_commutator_A1_B2_C1 = 0;
double t_commutator_A1_B2_C2 = 0;
double t_commutator_A2_B2_C0 = 0;
double t_commutator_A2_B2_C1 = 0;
double t_commutator_A2_B2_C2 = 0;
double t_tensor = 0;
double t_four = 0;

namespace psi{ namespace libadaptive{

void MOSRG::commutator_A_B_C(double factor,
                             MOTwoIndex restrict A1,
                             MOFourIndex restrict A2,
                             MOTwoIndex restrict B1,
                             MOFourIndex restrict B2,
                             double& C0,
                             MOTwoIndex restrict C1,
                             MOFourIndex restrict C2)
{
    commutator_A1_B1_C0(A1,B1,+factor,C0);
    commutator_A1_B2_C0(A1,B2,+factor,C0);
    commutator_A1_B2_C0(B1,A2,-factor,C0);
    commutator_A2_B2_C0(A2,B2,+factor,C0);

    commutator_A1_B1_C1(A1,B1,+factor,C1);
    commutator_A1_B2_C1(A1,B2,+factor,C1);
    commutator_A1_B2_C1(B1,A2,-factor,C1);
    commutator_A2_B2_C1(A2,B2,+factor,C1);

    commutator_A1_B2_C2(A1,B2,+factor,C2);
    commutator_A1_B2_C2(B1,A2,-factor,C2);
    commutator_A2_B2_C2(A2,B2,+factor,C2);
}

void MOSRG::commutator_A_B_C_fourth_order(double factor,
                                          MOTwoIndex restrict A1,
                                          MOFourIndex restrict A2,
                                          MOTwoIndex restrict B1,
                                          MOFourIndex restrict B2,
                                          double& C0,
                                          MOTwoIndex restrict C1,
                                          MOFourIndex restrict C2)
{
    commutator_A1_B1_C0(A1,B1,+factor,C0);
    commutator_A1_B2_C0(A1,B2,+factor,C0);
    commutator_A1_B2_C0(B1,A2,-factor,C0);
    commutator_A2_B2_C0(A2,B2,+factor,C0);

    commutator_A1_B1_C1(A1,B1,+factor,C1);
    commutator_A1_B2_C1(A1,B2,+factor,C1);
    commutator_A1_B2_C1(B1,A2,-factor,C1);
    commutator_A2_B2_C1(A2,B2,+2.0 * factor,C1);

    commutator_A1_B2_C2(A1,B2,+factor,C2);
    commutator_A1_B2_C2(B1,A2,-factor,C2);
    commutator_A2_B2_C2(A2,B2,+factor,C2);
}

void MOSRG::commutator_A_B_C_SRG2(double factor,
                                  MOTwoIndex restrict A1,
                                  MOFourIndex restrict A2,
                                  MOTwoIndex restrict B1,
                                  MOFourIndex restrict B2,
                                  double& C0,
                                  MOTwoIndex restrict C1,
                                  MOFourIndex restrict C2)
{
    //    commutator_A1_B1_C0(A1,B1,+factor,C0);
    //    commutator_A1_B2_C0(A1,B2,+factor,C0);
    //    commutator_A1_B2_C0(B1,A2,-factor,C0);
    commutator_A2_B2_C0(A2,B2,+factor,C0);

    commutator_A1_B1_C1(A1,B1,+factor,C1);
    //    commutator_A1_B2_C1(A1,B2,+factor,C1);
    //    commutator_A1_B2_C1(B1,A2,-factor,C1);
    commutator_A2_B2_C1(A2,B2,+factor,C1);

    //    commutator_A1_B2_C2(A1,B2,+factor,C2);
    commutator_A1_B2_C2(B1,A2,-factor,C2);
    commutator_A2_B2_C2(A2,B2,+factor,C2);
}

void MOSRG::commutator_A1_B1_C0(MOTwoIndex restrict A,MOTwoIndex restrict B,double sign,double& C)
{
    boost::timer t;
    double sum = 0.0;
    if(srgcomm == SRCommutators){
        loop_mo_p loop_mo_q{
            sum += A.aa[p][q] * B.aa[q][p] * (No_.a[p] - No_.a[q]);
        }
        loop_mo_p loop_mo_q{
            sum += A.bb[p][q] * B.bb[q][p] * (No_.b[p] - No_.b[q]);
        }
    }
    if(srgcomm == MRNOCommutators){
        loop_mo_p loop_mo_q loop_mo_r{
            sum += A.aa[p][q] * B.aa[q][r] * G1_.aa[r][p] - A.aa[p][q] * G1_.aa[q][r] * B.aa[r][p];
            sum += A.bb[p][q] * B.bb[q][r] * G1_.bb[r][p] - A.bb[p][q] * G1_.bb[q][r] * B.bb[r][p];
        }
    }
    C += sign * sum;
    if(print_ > 1){
        fprintf(outfile,"\n  Time for [A1,B1] -> C0 : %.4f",t.elapsed());
    }
    t_commutator_A1_B1_C0 += t.elapsed();
}

void MOSRG::commutator_A1_B1_C1(MOTwoIndex restrict A,MOTwoIndex restrict B,double sign,MOTwoIndex C)
{
    boost::timer t;
    loop_mo_p loop_mo_q{
        double sum = 0.0;
        loop_mo_r{
            sum += A.aa[p][r] * B.aa[r][q] - B.aa[p][r] * A.aa[r][q];
        }
        C.aa[p][q] += sign * sum;
    }
    loop_mo_p loop_mo_q{
        double sum = 0.0;
        loop_mo_r{
            sum += A.bb[p][r] * B.bb[r][q] - B.bb[p][r] * A.bb[r][q];
        }
        C.bb[p][q] += sign * sum;
    }

    if(print_ > 1){
        fprintf(outfile,"\n  Time for [A1,B1] -> C1 : %.4f",t.elapsed());
    }
    t_commutator_A1_B1_C1 += t.elapsed();
}

void MOSRG::commutator_A1_B2_C0(MOTwoIndex restrict A,MOFourIndex restrict B,double sign,double& C)
{
    boost::timer t;
    double sum = 0.0;
    if(srgcomm == MRNOCommutators){  // NOT CHECKED
    }
    C += sign * sum;

    if(print_ > 1){
        fprintf(outfile,"\n  Time for [A1,B2] -> C0 : %.4f",t.elapsed());
    }
    t_commutator_A1_B2_C0 += t.elapsed();
}

void MOSRG::commutator_A1_B2_C1(MOTwoIndex restrict A,MOFourIndex restrict B,double sign,MOTwoIndex C)
{
    boost::timer t;
    if(srgcomm == SRCommutators){
        loop_mo_p loop_mo_q{
            double sum = 0.0;
            loop_mo_r loop_mo_s {
                sum += A.aa[r][s] * B.aaaa[p][s][q][r] * (No_.a[r] - No_.a[s]);
                sum += A.bb[r][s] * B.abab[p][s][q][r] * (No_.b[r] - No_.b[s]);
            }
            C.aa[p][q] += sign * sum;
        }
        loop_mo_p loop_mo_q{
            double sum = 0.0;
            loop_mo_r loop_mo_s {
                sum += A.aa[r][s] * B.abab[s][p][r][q] * (No_.a[r] - No_.a[s]);
                sum += A.bb[r][s] * B.bbbb[p][s][q][r] * (No_.b[r] - No_.b[s]);
            }
            C.bb[p][q] += sign * sum;
        }
    }
    if(srgcomm == MRNOCommutators){  // NOT CHECKED
        loop_mo_p loop_mo_q{
            //            double sum = 0.0;
            //            loop_mo_r loop_mo_s loop_mo_t{
            //                sum += A[r][s] * B[p][s][q][t] * G1_[t][r] - A[r][s] * B[p][t][q][r] * G1_[s][t];
            //            }
            //            C[p][q] += sign * sum;
        }
    }

    if(print_ > 1){
        fprintf(outfile,"\n  Time for [A1,B2] -> C1 : %.4f",t.elapsed());
    }
    t_commutator_A1_B2_C1 += t.elapsed();
}

void MOSRG::commutator_A1_B2_C2(MOTwoIndex restrict A,MOFourIndex restrict B,double sign,MOFourIndex C)
{
    boost::timer t;
    loop_mo_p loop_mo_q loop_mo_r loop_mo_s{
        double sum = 0.0;
        loop_mo_t{
            sum += A.aa[p][t] * B.aaaa[t][q][r][s] - A.aa[q][t] * B.aaaa[t][p][r][s] - A.aa[t][r] * B.aaaa[p][q][t][s] + A.aa[t][s] * B.aaaa[p][q][t][r];
        }
        C.aaaa[p][q][r][s] += sign * sum;
    }
    loop_mo_p loop_mo_q loop_mo_r loop_mo_s{
        double sum = 0.0;
        loop_mo_t{
            sum += A.aa[p][t] * B.abab[t][q][r][s] + A.bb[q][t] * B.abab[p][t][r][s] - A.aa[t][r] * B.abab[p][q][t][s] - A.bb[t][s] * B.abab[p][q][r][t];
        }
        C.abab[p][q][r][s] += sign * sum;
    }
    loop_mo_p loop_mo_q loop_mo_r loop_mo_s{
        double sum = 0.0;
        loop_mo_t{
            sum += A.bb[p][t] * B.bbbb[t][q][r][s] - A.bb[q][t] * B.bbbb[t][p][r][s] - A.bb[t][r] * B.bbbb[p][q][t][s] + A.bb[t][s] * B.bbbb[p][q][t][r];
        }
        C.bbbb[p][q][r][s] += sign * sum;
    }

    if(print_ > 1){
        fprintf(outfile,"\n  Time for [A1,B2] -> C2 : %.4f",t.elapsed());
    }
    t_commutator_A1_B2_C2 += t.elapsed();
}

void MOSRG::commutator_A2_B2_C0(MOFourIndex restrict A,MOFourIndex restrict B,double sign,double& C)
{
    boost::timer t;
    double sum = 0.0;
    loop_mo_p loop_mo_q loop_mo_r loop_mo_s{
        sum += 0.25 * (A.aaaa[p][q][r][s] * B.aaaa[r][s][p][q] - B.aaaa[p][q][r][s] * A.aaaa[r][s][p][q]) * (No_.a[p] * No_.a[q] * Nv_.a[r] * Nv_.a[s]);
    }
    loop_mo_p loop_mo_q loop_mo_r loop_mo_s{
        sum += (A.abab[p][q][r][s] * B.abab[r][s][p][q] - B.abab[p][q][r][s] * A.abab[r][s][p][q]) * (No_.a[p] * No_.b[q] * Nv_.a[r] * Nv_.b[s]);
    }
    loop_mo_p loop_mo_q loop_mo_r loop_mo_s{
        sum += 0.25 * (A.bbbb[p][q][r][s] * B.bbbb[r][s][p][q] - B.bbbb[p][q][r][s] * A.bbbb[r][s][p][q]) * (No_.b[p] * No_.b[q] * Nv_.b[r] * Nv_.b[s]);
    }
    C += sign * sum;

    if(print_ > 1){
        fprintf(outfile,"\n  Time for [A2,B2] -> C0 : %.4f",t.elapsed());
    }
    t_commutator_A2_B2_C0 += t.elapsed();
}

void MOSRG::commutator_A2_B2_C1(MOFourIndex restrict A,MOFourIndex restrict B,double sign,MOTwoIndex C)
{
    boost::timer t;
    loop_mo_p loop_mo_q{
        double sum = 0.0;
        loop_mo_r loop_mo_s loop_mo_t{
            sum += 0.5 * (A.aaaa[t][p][r][s] * B.aaaa[r][s][t][q] - B.aaaa[t][p][r][s] * A.aaaa[r][s][t][q])
                    * (No_.a[r] * No_.a[s] * Nv_.a[t] + Nv_.a[r] * Nv_.a[s] * No_.a[t]);
            sum += (A.abab[p][t][r][s] * B.abab[r][s][q][t] - B.abab[p][t][r][s] * A.abab[r][s][q][t])
                    * (No_.a[r] * No_.b[s] * Nv_.b[t] + Nv_.a[r] * Nv_.b[s] * No_.b[t]);
        }
        C.aa[p][q] += sign * sum;
    }
    loop_mo_p loop_mo_q{
        double sum = 0.0;
        loop_mo_r loop_mo_s loop_mo_t{
            sum += 0.5 * (A.bbbb[t][p][r][s] * B.bbbb[r][s][t][q] - B.bbbb[t][p][r][s] * A.bbbb[r][s][t][q])
                    * (No_.b[r] * No_.b[s] * Nv_.b[t] + Nv_.b[r] * Nv_.b[s] * No_.b[t]);
            sum += (A.abab[t][p][r][s] * B.abab[r][s][t][q] - B.abab[t][p][r][s] * A.abab[r][s][t][q])
                    * (No_.a[r] * No_.b[s] * Nv_.a[t] + Nv_.a[r] * Nv_.b[s] * No_.a[t]);
        }
        C.bb[p][q] += sign * sum;
    }

    if(print_ > 1){
        fprintf(outfile,"\n  Time for [A2,B2] -> C1 : %.4f",t.elapsed());
    }
    t_commutator_A2_B2_C1 += t.elapsed();
}

void MOSRG::commutator_A2_B2_C2(MOFourIndex restrict A,MOFourIndex restrict B,double sign,MOFourIndex C)
{
    boost::timer t;
    bool use_tensor = false;
    if(use_tensor){
        boost::timer t1;
        loop_mo_p loop_mo_q{
            D_aa(p,q) = (p == q) ? No_.a[p] : 0.0;
            D_bb(p,q) = (p == q) ? No_.b[p] : 0.0;
            CD_aa(p,q) = (p == q) ? 1.0 - No_.a[p] : 0.0;
            CD_bb(p,q) = (p == q) ? 1.0 - No_.b[p] : 0.0;
        }

        loop_mo_p loop_mo_q loop_mo_r loop_mo_s{
            A_aaaa(p,q,r,s) = A.aaaa[p][q][r][s];
            B_aaaa(p,q,r,s) = B.aaaa[p][q][r][s];
            A_abab(p,q,r,s) = A.abab[p][q][r][s];
            B_abab(p,q,r,s) = B.abab[p][q][r][s];
            A_bbbb(p,q,r,s) = A.bbbb[p][q][r][s];
            B_bbbb(p,q,r,s) = B.bbbb[p][q][r][s];
        }

        C_aaaa.zero();
        C_abab.zero();
        C_bbbb.zero();

        // AAAA case
        // Term I
        I4("abcd") = CD_aa("ac") * CD_aa("bd");
        I4("abcd") += -1.0 * D_aa("ac") * D_aa("bd");
        Bm_aaaa("abrs") = I4("abcd") * B_aaaa("cdrs");
        C_aaaa("pqrs") +=  0.5 * sign * A_aaaa("pqab") * Bm_aaaa("abrs");
        Am_aaaa("abrs") = I4("abcd") * A_aaaa("cdrs");
        C_aaaa("pqrs") += -0.5 * sign * B_aaaa("pqab") * Am_aaaa("abrs");

        // Term II
        I4("abcd") = D_aa("ac") * CD_aa("bd");
        I4("abcd") += -1.0 * CD_aa("ac") * D_aa("bd");

        Bm_aaaa("qbsc") = I4("abcd") * B_aaaa("qdsa");
        I4("pqrs") = sign * A_aaaa("pcrb") * Bm_aaaa("qbsc");
        C_aaaa("pqrs") += +1.0 * I4("pqrs");
        C_aaaa("pqrs") += -1.0 * I4("qprs");
        C_aaaa("pqrs") += -1.0 * I4("pqsr");
        C_aaaa("pqrs") += +1.0 * I4("qpsr");

        I4("abcd") = D_bb("ac") * CD_bb("bd");
        I4("abcd") += -1.0 * CD_bb("ac") * D_bb("bd");

        Bm_aaaa("qbsc") = I4("abcd") * B_abab("qdsa");
        I4("pqrs") = sign * A_abab("pcrb") * Bm_aaaa("qbsc");
        C_aaaa("pqrs") += +1.0 * I4("pqrs");
        C_aaaa("pqrs") += -1.0 * I4("qprs");
        C_aaaa("pqrs") += -1.0 * I4("pqsr");
        C_aaaa("pqrs") += +1.0 * I4("qpsr");


        // ABAB case
        // Term I
        I4("abcd") = CD_aa("ac") * CD_bb("bd");
        I4("abcd") += -1.0 * D_aa("ac") * D_bb("bd");
        Bm_abab("abrs") = I4("abcd") * B_abab("cdrs");
        C_abab("pqrs") +=  sign * A_abab("pqab") * Bm_abab("abrs");
        Am_abab("abrs") = I4("abcd") * A_abab("cdrs");
        C_abab("pqrs") += -sign * B_abab("pqab") * Am_abab("abrs");

        // Term II
        I4("abcd") = D_aa("ac") * CD_aa("bd");
        I4("abcd") += -1.0 * CD_aa("ac") * D_aa("bd");
        Bm_abab("bqcs") = I4("abcd") * B_abab("dqas");
        C_abab("pqrs") += sign * A_aaaa("pcrb") * Bm_abab("bqcs");
        Bm_aaaa("pbrc") = I4("abcd") * B_aaaa("pdra");
        C_abab("pqrs") += sign * A_abab("cqbs") * Bm_aaaa("pbrc");


        I4("abcd") = D_bb("ac") * CD_bb("bd");
        I4("abcd") += -1.0 * CD_bb("ac") * D_bb("bd");
        Bm_bbbb("qbsc") = I4("abcd") * B_abab("qdsa");
        C_abab("pqrs") += sign * A_abab("pcrb") * Bm_bbbb("qbsc");
        Bm_abab("pbrc") = I4("abcd") * B_abab("pdra");
        C_abab("pqrs") += sign * A_bbbb("qcsb") * Bm_abab("pbrc");


        loop_mo_p loop_mo_q loop_mo_r loop_mo_s{
            double sum = 0.0;
            loop_mo_t loop_mo_u{
//                sum += (/* A.aaaa[p][t][r][u] * B.abab[u][q][t][s] */+ A.abab[t][q][u][s] * B.aaaa[p][u][r][t]) * (No_.a[t] - No_.a[u]);
//                sum += (A.abab[p][t][r][u] * B.bbbb[q][u][s][t] + A.bbbb[t][q][u][s] * B.abab[p][u][r][t]) * (No_.b[t] - No_.b[u]);
                sum += -(A.abab[t][q][r][u] * B.abab[p][u][t][s]) * (No_.a[t] - No_.b[u]);
                sum += -(A.abab[p][t][u][s] * B.abab[u][q][r][t]) * (No_.b[t] - No_.a[u]);
            }
            C.abab[p][q][r][s] += sign * sum;
        }


        // BBBB case
        // Term I
        I4("abcd") = CD_bb("ac") * CD_bb("bd");
        I4("abcd") += -1.0 * D_bb("ac") * D_bb("bd");
        Bm_bbbb("abrs") = I4("abcd") * B_bbbb("cdrs");
        C_bbbb("pqrs") +=  0.5 * sign * A_bbbb("pqab") * Bm_bbbb("abrs");
        Am_bbbb("abrs") = I4("abcd") * A_bbbb("cdrs");
        C_bbbb("pqrs") += -0.5 * sign * B_bbbb("pqab") * Am_bbbb("abrs");

        // Term II
        I4("abcd") = D_bb("ac") * CD_bb("bd");
        I4("abcd") += -1.0 * CD_bb("ac") * D_bb("bd");

        Bm_bbbb("qbsc") = I4("abcd") * B_bbbb("qdsa");
        I4("pqrs") = sign * A_bbbb("pcrb") * Bm_bbbb("qbsc");
        C_bbbb("pqrs") += +1.0 * I4("pqrs");
        C_bbbb("pqrs") += -1.0 * I4("qprs");
        C_bbbb("pqrs") += -1.0 * I4("pqsr");
        C_bbbb("pqrs") += +1.0 * I4("qpsr");

        I4("abcd") = D_aa("ac") * CD_aa("bd");
        I4("abcd") += -1.0 * CD_aa("ac") * D_aa("bd");

        Bm_bbbb("qbsc") = I4("abcd") * B_abab("dqas");
        I4("pqrs") = sign * A_abab("cpbr") * Bm_bbbb("qbsc");
        C_bbbb("pqrs") += +1.0 * I4("pqrs");
        C_bbbb("pqrs") += -1.0 * I4("qprs");
        C_bbbb("pqrs") += -1.0 * I4("pqsr");
        C_bbbb("pqrs") += +1.0 * I4("qpsr");

        loop_mo_p loop_mo_q loop_mo_r loop_mo_s{
            C.aaaa[p][q][r][s] += C_aaaa(p,q,r,s);
            C.abab[p][q][r][s] += C_abab(p,q,r,s);
            C.bbbb[p][q][r][s] += C_bbbb(p,q,r,s);
        }
        t_tensor += t1.elapsed();
    }else{
        loop_mo_p loop_mo_q loop_mo_r loop_mo_s{
            double sum = 0.0;
            loop_mo_t loop_mo_u{
                sum += 0.5 * (A.aaaa[p][q][t][u] * B.aaaa[t][u][r][s] - A.aaaa[t][u][r][s] * B.aaaa[p][q][t][u]) * (1.0 - No_.a[t] - No_.a[u]);
                sum += (  A.aaaa[p][t][r][u] * B.aaaa[q][u][s][t]
                          - A.aaaa[q][t][r][u] * B.aaaa[p][u][s][t]
                          - A.aaaa[p][t][s][u] * B.aaaa[q][u][r][t]
                          + A.aaaa[q][t][s][u] * B.aaaa[p][u][r][t]) * (No_.a[t] - No_.a[u]);
                sum += (  A.abab[p][t][r][u] * B.abab[q][u][s][t]
                          - A.abab[q][t][r][u] * B.abab[p][u][s][t]
                          - A.abab[p][t][s][u] * B.abab[q][u][r][t]
                          + A.abab[q][t][s][u] * B.abab[p][u][r][t]) * (No_.b[t] - No_.b[u]);
            }
            C.aaaa[p][q][r][s] += sign * sum;
        }
        loop_mo_p loop_mo_q loop_mo_r loop_mo_s{
            double sum = 0.0;
            loop_mo_t loop_mo_u{
                sum += (A.abab[p][q][t][u] * B.abab[t][u][r][s] - A.abab[t][u][r][s] * B.abab[p][q][t][u]) * (1.0 - No_.a[t] - No_.b[u]);
                sum += (A.aaaa[p][t][r][u] * B.abab[u][q][t][s] + A.abab[t][q][u][s] * B.aaaa[p][u][r][t]) * (No_.a[t] - No_.a[u]);
                sum += (A.abab[p][t][r][u] * B.bbbb[q][u][s][t] + A.bbbb[t][q][u][s] * B.abab[p][u][r][t]) * (No_.b[t] - No_.b[u]);
                sum += -(A.abab[t][q][r][u] * B.abab[p][u][t][s]) * (No_.a[t] - No_.b[u]);
                sum += -(A.abab[p][t][u][s] * B.abab[u][q][r][t]) * (No_.b[t] - No_.a[u]);
            }
            C.abab[p][q][r][s] += sign * sum;
        }

        loop_mo_p loop_mo_q loop_mo_r loop_mo_s{
            double sum = 0.0;
            loop_mo_t loop_mo_u{
                sum += 0.5 * (A.bbbb[p][q][t][u] * B.bbbb[t][u][r][s] - A.bbbb[t][u][r][s] * B.bbbb[p][q][t][u]) * (1.0 - No_.b[t] - No_.b[u]);
                sum += (  A.bbbb[p][t][r][u] * B.bbbb[q][u][s][t]
                          - A.bbbb[q][t][r][u] * B.bbbb[p][u][s][t]
                          - A.bbbb[p][t][s][u] * B.bbbb[q][u][r][t]
                          + A.bbbb[q][t][s][u] * B.bbbb[p][u][r][t]) * (No_.b[t] - No_.b[u]);
                sum += (  A.abab[t][p][u][r] * B.abab[u][q][t][s]
                          - A.abab[t][q][u][r] * B.abab[u][p][t][s]
                          - A.abab[t][p][u][s] * B.abab[u][q][t][r]
                          + A.abab[t][q][u][s] * B.abab[u][p][t][r]) * (No_.a[t] - No_.a[u]);
            }
            C.bbbb[p][q][r][s] += sign * sum;
        }
    }

    if(print_ > 1){
        fprintf(outfile,"\n  Time for [A2,B2] -> C2 : %.4f",t.elapsed());
    }
    t_commutator_A2_B2_C2 += t.elapsed();
}

void MOSRG::print_timings()
{
    fprintf(outfile,"\n\n  =========== TIMINGS =========");
    fprintf(outfile,"\n  Time for [A1,B1] -> C0 : %.4f",t_commutator_A1_B1_C0);
    fprintf(outfile,"\n  Time for [A1,B1] -> C1 : %.4f",t_commutator_A1_B1_C1);
    fprintf(outfile,"\n  Time for [A1,B2] -> C0 : %.4f",t_commutator_A1_B2_C0);
    fprintf(outfile,"\n  Time for [A1,B2] -> C1 : %.4f",t_commutator_A1_B2_C1);
    fprintf(outfile,"\n  Time for [A1,B2] -> C2 : %.4f",t_commutator_A1_B2_C2);
    fprintf(outfile,"\n  Time for [A2,B2] -> C0 : %.4f",t_commutator_A2_B2_C0);
    fprintf(outfile,"\n  Time for [A2,B2] -> C1 : %.4f",t_commutator_A2_B2_C1);
    fprintf(outfile,"\n  Time for [A2,B2] -> C2 : %.4f",t_commutator_A2_B2_C2);
    fprintf(outfile,"\n  =============================\n");
    fprintf(outfile,"\n  Time for tensor : %.4f",t_tensor);
    fprintf(outfile,"\n  Time for four   : %.4f",t_four);
}

}} // EndNamespaces

