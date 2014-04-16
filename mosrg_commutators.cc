//#define _MULTIPLE_CONTRACTIONS_

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
                             MOTwoIndex A1,
                             MOFourIndex A2,
                             MOTwoIndex B1,
                             MOFourIndex B2,
                             double& C0,
                             MOTwoIndex C1,
                             MOFourIndex C2)
{
    commutator_A1_B1_C0(A1,B1,+factor,C0);
//    commutator_A1_B2_C0(A1,B2,+factor,C0);
//    commutator_A1_B2_C0(B1,A2,-factor,C0);
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
                                          MOTwoIndex A1,
                                          MOFourIndex A2,
                                          MOTwoIndex B1,
                                          MOFourIndex B2,
                                          double& C0,
                                          MOTwoIndex C1,
                                          MOFourIndex C2)
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
                                  MOTwoIndex A1,
                                  MOFourIndex A2,
                                  MOTwoIndex B1,
                                  MOFourIndex B2,
                                  double& C0,
                                  MOTwoIndex C1,
                                  MOFourIndex C2)
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

void MOSRG::commutator_A1_B1_C0(MOTwoIndex A,MOTwoIndex B,double sign,double& C)
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

void MOSRG::commutator_A1_B1_C1(MOTwoIndex A,MOTwoIndex B,double sign,MOTwoIndex C)
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

void MOSRG::commutator_A1_B2_C0(MOTwoIndex A,MOFourIndex B,double sign,double& C)
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

void MOSRG::commutator_A1_B2_C1(MOTwoIndex A,MOFourIndex B,double sign,MOTwoIndex C)
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

void MOSRG::commutator_A1_B2_C2(MOTwoIndex A,MOFourIndex B,double sign,MOFourIndex C)
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

void MOSRG::commutator_A2_B2_C0(MOFourIndex A,MOFourIndex B,double sign,double& C)
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

void MOSRG::commutator_A2_B2_C1(MOFourIndex A,MOFourIndex B,double sign,MOTwoIndex C)
{
    boost::timer t;
    if(use_tensor_class_){
        loop_mo_p loop_mo_q{
            D_a(p,q) = (p == q) ? No_.a[p] : 0.0;
            D_b(p,q) = (p == q) ? No_.b[p] : 0.0;
            CD_a(p,q) = (p == q) ? 1.0 - No_.a[p] : 0.0;
            CD_b(p,q) = (p == q) ? 1.0 - No_.b[p] : 0.0;
        }

        loop_mo_p loop_mo_q loop_mo_r loop_mo_s{
            A4_aa(p,q,r,s) = A.aaaa[p][q][r][s];
            B4_aa(p,q,r,s) = B.aaaa[p][q][r][s];
            A4_ab(p,q,r,s) = A.abab[p][q][r][s];
            B4_ab(p,q,r,s) = B.abab[p][q][r][s];
            A4_bb(p,q,r,s) = A.bbbb[p][q][r][s];
            B4_bb(p,q,r,s) = B.bbbb[p][q][r][s];
        }

        C_a.zero();
        C_b.zero();

        A4m_aa("cpdb") = A4_aa("cpab") * D_a("ad");
        A4m_aa("cpde") = A4m_aa("cpdb") * D_a("be");
        A4m_aa("fpde") = A4m_aa("cpde") * CD_a("fc");
        C_a("pq") += 0.5 * sign * A4m_aa("fpde") * B4_aa("defq");

        B4m_aa("cpdb") = B4_aa("cpab") * D_a("ad");
        B4m_aa("cpde") = B4m_aa("cpdb") * D_a("be");
        B4m_aa("fpde") = B4m_aa("cpde") * CD_a("fc");
        C_a("pq") += -0.5 * sign * B4m_aa("fpde") * A4_aa("defq");

        A4m_aa("cpdb") = A4_aa("cpab") * CD_a("ad");
        A4m_aa("cpde") = A4m_aa("cpdb") * CD_a("be");
        A4m_aa("fpde") = A4m_aa("cpde") * D_a("fc");
        C_a("pq") += 0.5 * sign * A4m_aa("fpde") * B4_aa("defq");

        B4m_aa("cpdb") = B4_aa("cpab") * CD_a("ad");
        B4m_aa("cpde") = B4m_aa("cpdb") * CD_a("be");
        B4m_aa("fpde") = B4m_aa("cpde") * D_a("fc");
        C_a("pq") += -0.5 * sign * B4m_aa("fpde") * A4_aa("defq");


        A4m_ab("pCbD") = A4_ab("pCbA") * D_b("AD");
        A4m_ab("pCeD") = A4m_ab("pCbD") * D_a("be");
        A4m_ab("pFeD") = A4m_ab("pCeD") * CD_b("FC");
        C_a("pq") += sign * A4m_ab("pFeD") * B4_ab("eDqF");

        B4m_ab("pCbD") = B4_ab("pCbA") * D_b("AD");
        B4m_ab("pCeD") = B4m_ab("pCbD") * D_a("be");
        B4m_ab("pFeD") = B4m_ab("pCeD") * CD_b("FC");
        C_a("pq") += -sign * B4m_ab("pFeD") * A4_ab("eDqF");

        A4m_ab("pCbD") = A4_ab("pCbA") * CD_b("AD");
        A4m_ab("pCeD") = A4m_ab("pCbD") * CD_a("be");
        A4m_ab("pFeD") = A4m_ab("pCeD") * D_b("FC");
        C_a("pq") += sign * A4m_ab("pFeD") * B4_ab("eDqF");

        B4m_ab("pCbD") = B4_ab("pCbA") * CD_b("AD");
        B4m_ab("pCeD") = B4m_ab("pCbD") * CD_a("be");
        B4m_ab("pFeD") = B4m_ab("pCeD") * D_b("FC");
        C_a("pq") += -sign * B4m_ab("pFeD") * A4_ab("eDqF");


        A4m_bb("cpdb") = A4_bb("cpab") * D_b("ad");
        A4m_bb("cpde") = A4m_bb("cpdb") * D_b("be");
        A4m_bb("fpde") = A4m_bb("cpde") * CD_b("fc");
        C_b("pq") += 0.5 * sign * A4m_bb("fpde") * B4_bb("defq");

        B4m_bb("cpdb") = B4_bb("cpab") * D_b("ad");
        B4m_bb("cpde") = B4m_bb("cpdb") * D_b("be");
        B4m_bb("fpde") = B4m_bb("cpde") * CD_b("fc");
        C_b("pq") += -0.5 * sign * B4m_bb("fpde") * A4_bb("defq");

        A4m_bb("cpdb") = A4_bb("cpab") * CD_b("ad");
        A4m_bb("cpde") = A4m_bb("cpdb") * CD_b("be");
        A4m_bb("fpde") = A4m_bb("cpde") * D_b("fc");
        C_b("pq") += 0.5 * sign * A4m_bb("fpde") * B4_bb("defq");

        B4m_bb("cpdb") = B4_bb("cpab") * CD_b("ad");
        B4m_bb("cpde") = B4m_bb("cpdb") * CD_b("be");
        B4m_bb("fpde") = B4m_bb("cpde") * D_b("fc");
        C_b("pq") += -0.5 * sign * B4m_bb("fpde") * A4_bb("defq");

        A4m_ab("cPdB") = A4_ab("cPaB") * D_a("ad");
        A4m_ab("cPdE") = A4m_ab("cPdB") * D_b("BE");
        A4m_ab("fPdE") = A4m_ab("cPdE") * CD_a("fc");
        C_b("PQ") += sign * A4m_ab("fPdE") * B4_ab("dEfQ");

        B4m_ab("cPdB") = B4_ab("cPaB") * D_a("ad");
        B4m_ab("cPdE") = B4m_ab("cPdB") * D_b("BE");
        B4m_ab("fPdE") = B4m_ab("cPdE") * CD_a("fc");
        C_b("PQ") += -sign * B4m_ab("fPdE") * A4_ab("dEfQ");

        A4m_ab("cPdB") = A4_ab("cPaB") * CD_a("ad");
        A4m_ab("cPdE") = A4m_ab("cPdB") * CD_b("BE");
        A4m_ab("fPdE") = A4m_ab("cPdE") * D_a("fc");
        C_b("PQ") += sign * A4m_ab("fPdE") * B4_ab("dEfQ");

        B4m_ab("cPdB") = B4_ab("cPaB") * CD_a("ad");
        B4m_ab("cPdE") = B4m_ab("cPdB") * CD_b("BE");
        B4m_ab("fPdE") = B4m_ab("cPdE") * D_a("fc");
        C_b("PQ") += -sign * B4m_ab("fPdE") * A4_ab("dEfQ");


        loop_mo_p loop_mo_q{
            C.aa[p][q] += C_a(p,q);
            C.bb[p][q] += C_b(p,q);
        }
    }else{
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
    }
    if(print_ > 1){
        fprintf(outfile,"\n  Time for [A2,B2] -> C1 : %.4f",t.elapsed());
    }
    t_commutator_A2_B2_C1 += t.elapsed();
}

void MOSRG::commutator_A2_B2_C2(MOFourIndex A,MOFourIndex B,double sign,MOFourIndex C)
{
    boost::timer t;
    if(use_tensor_class_){
        boost::timer t1;
        loop_mo_p loop_mo_q{
            D_a(p,q) = (p == q) ? No_.a[p] : 0.0;
            D_b(p,q) = (p == q) ? No_.b[p] : 0.0;
            CD_a(p,q) = (p == q) ? 1.0 - No_.a[p] : 0.0;
            CD_b(p,q) = (p == q) ? 1.0 - No_.b[p] : 0.0;
        }

        loop_mo_p loop_mo_q loop_mo_r loop_mo_s{
            A4_aa(p,q,r,s) = A.aaaa[p][q][r][s];
            B4_aa(p,q,r,s) = B.aaaa[p][q][r][s];
            A4_ab(p,q,r,s) = A.abab[p][q][r][s];
            B4_ab(p,q,r,s) = B.abab[p][q][r][s];
            A4_bb(p,q,r,s) = A.bbbb[p][q][r][s];
            B4_bb(p,q,r,s) = B.bbbb[p][q][r][s];
        }

        C4_aa.zero();
        C4_ab.zero();
        C4_bb.zero();

#ifdef _MULTIPLE_CONTRACTIONS_
        // Term I
        C4_aa("pqrs") +=  0.5 * sign * A4_aa("pqab") * B4_aa("cdrs") * CD_a("ac") * CD_a("bd");
        C4_aa("pqrs") += -0.5 * sign * A4_aa("pqab") * B4_aa("cdrs") * D_a("ac") * D_a("bd");
        C4_aa("pqrs") += -0.5 * sign * B4_aa("pqab") * A4_aa("cdrs") * CD_a("ac") * CD_a("bd");
        C4_aa("pqrs") +=  0.5 * sign * B4_aa("pqab") * A4_aa("cdrs") * D_a("ac") * D_a("bd");

        // Term II
        I4("pqrs")  = +sign * A4_aa("pcrb") * B4_aa("qdsa") * D_a("ac") * CD_a("bd");
        I4("pqrs") += -sign * A4_aa("pcrb") * B4_aa("qdsa") * CD_a("ac") * D_a("bd");
        C4_aa("pqrs") += +1.0 * I4("pqrs");
        C4_aa("pqrs") += -1.0 * I4("qprs");
        C4_aa("pqrs") += -1.0 * I4("pqsr");
        C4_aa("pqrs") += +1.0 * I4("qpsr");

        I4("pqrs")  = +sign * A4_ab("pcrb") * B4_ab("qdsa") * D_b("ac") * CD_b("bd");
        I4("pqrs") += -sign * A4_ab("pcrb") * B4_ab("qdsa") * CD_b("ac") * D_b("bd");
        C4_aa("pqrs") += +1.0 * I4("pqrs");
        C4_aa("pqrs") += -1.0 * I4("qprs");
        C4_aa("pqrs") += -1.0 * I4("pqsr");
        C4_aa("pqrs") += +1.0 * I4("qpsr");
#else
        // AAAA case
        // Term I
        I4("abcd") = CD_a("ac") * CD_a("bd");
        I4("abcd") += -1.0 * D_a("ac") * D_a("bd");
        B4m_aa("abrs") = I4("abcd") * B4_aa("cdrs");
        C4_aa("pqrs") +=  0.5 * sign * A4_aa("pqab") * B4m_aa("abrs");
        A4m_aa("abrs") = I4("abcd") * A4_aa("cdrs");
        C4_aa("pqrs") += -0.5 * sign * B4_aa("pqab") * A4m_aa("abrs");

        // Term II
        I4("abcd") = D_a("ac") * CD_a("bd");
        I4("abcd") += -1.0 * CD_a("ac") * D_a("bd");

        B4m_aa("qbsc") = I4("abcd") * B4_aa("qdsa");
        I4("pqrs") = sign * A4_aa("pcrb") * B4m_aa("qbsc");

        C4_aa("pqrs") += +1.0 * I4("pqrs");
        C4_aa("pqrs") += -1.0 * I4("qprs");
        C4_aa("pqrs") += -1.0 * I4("pqsr");
        C4_aa("pqrs") += +1.0 * I4("qpsr");

        I4("abcd") = D_b("ac") * CD_b("bd");
        I4("abcd") += -1.0 * CD_b("ac") * D_b("bd");

        B4m_ab("qbsc") = I4("abcd") * B4_ab("qdsa");
        I4("pqrs") = sign * A4_ab("pcrb") * B4m_ab("qbsc");

        C4_aa("pqrs") += +1.0 * I4("pqrs");
        C4_aa("pqrs") += -1.0 * I4("qprs");
        C4_aa("pqrs") += -1.0 * I4("pqsr");
        C4_aa("pqrs") += +1.0 * I4("qpsr");
#endif

        // ABAB case
        // Term I
        I4("abcd") = CD_a("ac") * CD_b("bd");
        I4("abcd") += -1.0 * D_a("ac") * D_b("bd");
        B4m_ab("abrs") = I4("abcd") * B4_ab("cdrs");
        C4_ab("pqrs") +=  sign * A4_ab("pqab") * B4m_ab("abrs");
        A4m_ab("abrs") = I4("abcd") * A4_ab("cdrs");
        C4_ab("pqrs") += -sign * B4_ab("pqab") * A4m_ab("abrs");

        // Term II
        I4("abcd") = D_a("ac") * CD_a("bd");
        I4("abcd") += -1.0 * CD_a("ac") * D_a("bd");
        B4m_ab("bqcs") = I4("abcd") * B4_ab("dqas");
        C4_ab("pqrs") += sign * A4_aa("pcrb") * B4m_ab("bqcs");
        B4m_aa("pbrc") = I4("abcd") * B4_aa("pdra");
        C4_ab("pqrs") += sign * A4_ab("cqbs") * B4m_aa("pbrc");

        I4("abcd") = D_b("ac") * CD_b("bd");
        I4("abcd") += -1.0 * CD_b("ac") * D_b("bd");
        B4m_bb("qbsc") = I4("abcd") * B4_bb("qdsa");
        C4_ab("pqrs") += sign * A4_ab("pcrb") * B4m_bb("qbsc");
        B4m_ab("pbrc") = I4("abcd") * B4_ab("pdra");
        C4_ab("pqrs") += sign * A4_bb("qcsb") * B4m_ab("pbrc");

        I4("aBcD") = D_a("ac") * CD_b("BD");
        I4("aBcD") += -1.0 * CD_a("ac") * D_b("BD");
        B4m_ab("pBcS") = I4("aBcD") * B4_ab("pDaS");
        C4_ab("pQrS") += -sign * A4_ab("cQrB") * B4m_ab("pBcS");

        B4m_ab("bQrC") = I4("bAdC") * B4_ab("dQrA");
        C4_ab("pQrS") += +sign * A4_ab("pCbS") * B4m_ab("bQrC");

        // BBBB case
        // Term I
        I4("abcd") = CD_b("ac") * CD_b("bd");
        I4("abcd") += -1.0 * D_b("ac") * D_b("bd");
        B4m_bb("abrs") = I4("abcd") * B4_bb("cdrs");
        C4_bb("pqrs") +=  0.5 * sign * A4_bb("pqab") * B4m_bb("abrs");
        A4m_bb("abrs") = I4("abcd") * A4_bb("cdrs");
        C4_bb("pqrs") += -0.5 * sign * B4_bb("pqab") * A4m_bb("abrs");

        // Term II
        I4("abcd") = D_b("ac") * CD_b("bd");
        I4("abcd") += -1.0 * CD_b("ac") * D_b("bd");

        B4m_bb("qbsc") = I4("abcd") * B4_bb("qdsa");
        I4("pqrs") = sign * A4_bb("pcrb") * B4m_bb("qbsc");
        C4_bb("pqrs") += +1.0 * I4("pqrs");
        C4_bb("pqrs") += -1.0 * I4("qprs");
        C4_bb("pqrs") += -1.0 * I4("pqsr");
        C4_bb("pqrs") += +1.0 * I4("qpsr");

        I4("abcd") = D_a("ac") * CD_a("bd");
        I4("abcd") += -1.0 * CD_a("ac") * D_a("bd");

        B4m_bb("qbsc") = I4("abcd") * B4_ab("dqas");
        I4("pqrs") = sign * A4_ab("cpbr") * B4m_bb("qbsc");
        C4_bb("pqrs") += +1.0 * I4("pqrs");
        C4_bb("pqrs") += -1.0 * I4("qprs");
        C4_bb("pqrs") += -1.0 * I4("pqsr");
        C4_bb("pqrs") += +1.0 * I4("qpsr");

        loop_mo_p loop_mo_q loop_mo_r loop_mo_s{
            C.aaaa[p][q][r][s] += C4_aa(p,q,r,s);
            C.abab[p][q][r][s] += C4_ab(p,q,r,s);
            C.bbbb[p][q][r][s] += C4_bb(p,q,r,s);
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
