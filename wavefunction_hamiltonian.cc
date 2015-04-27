///*
// *  wavefunction_hamiltonian.cpp
// *  Capriccio
// *
// *  Created by Francesco Evangelista on 3/9/09.
// *  Copyright 2009 __MyCompanyName__. All rights reserved.
// *
// */

//#include <boost/timer.hpp>

//#include <libmoinfo/libmoinfo.h>
//#include <liboptions/liboptions.h>
//#include <libqt/qt.h>

//#include "wavefunction.h"

//using namespace std;
//using namespace psi;
//using namespace boost;

//#include <psi4-dec.h>

//double h1_aa_timer = 0.0;
//double h1_bb_timer = 0.0;
//double h2_aaaa_timer = 0.0;
//double h2_aabb_timer = 0.0;
//double h2_bbbb_timer = 0.0;

//double** C1;
//double** Y1;


///**
// * Apply the Hamiltonian to the wave function
// * @param result Wave function object which stores the resulting vector
// */
//void FCIWfn::Hamiltonian(FCIWfn& result)
//{
//  check_temp_space();
//  result.zero();
  
//  // H0
//  {
//    H0(result);
//  }
  
//  // H1_aa
//  { timer t;
//    H1(result,true);
//    h1_aa_timer += t.elapsed();
//  }
//  // H1_bb
//  { timer t;
//    H1(result,false);
//    h1_bb_timer += t.elapsed();
//  }
//  // H2_aabb
//  { timer t;
//    H2_aabb(result);
//    h2_aabb_timer += t.elapsed();
//  }
//  if(Process::environment.options.get_bool("FASTALGRM")){
//    // H2_aaaa
//    { timer t;
//      H2_aaaa2(result,true);
//      h2_aaaa_timer += t.elapsed();
//    }
//    // H2_bbbb
//    { timer t;
//      H2_aaaa2(result,false);
//      h2_bbbb_timer += t.elapsed();
//    }
//  }else{
//    // H2_aaaa
//    { timer t;
//      H2_aaaa(result,true);
//      h2_aaaa_timer += t.elapsed();
//    }
//    // H2_bbbb
//    { timer t;
//      H2_aaaa(result,false);
//      h2_bbbb_timer += t.elapsed();
//    }
//  }
//}


///**
// * Apply the scalar part of the Hamiltonian to the wave function
// */
//void FCIWfn::H0(FCIWfn& result)
//{
//  double core_energy = ints->frozen_core_energy();
//  for(int alfa_sym = 0; alfa_sym < nirreps; ++alfa_sym){
//    int beta_sym = alfa_sym ^ symmetry_;
//    if(detpi[alfa_sym] > 0){
//      size_t maxIa = alfa_graph_->strpi(alfa_sym);
//      size_t maxIb = beta_graph_->strpi(beta_sym);
//      for(size_t Ia = 0; Ia < maxIa; ++Ia)
//        for(size_t Ib = 0; Ib < maxIb; ++Ib)
//          result.coefficients[alfa_sym][Ia][Ib] = core_energy * coefficients[alfa_sym][Ia][Ib];
//    }
//  }
//}

///**
// * Apply the one-particle Hamiltonian to the wave function
// * @param alfa flag for alfa or beta component, true = alfa, false = beta
// */
//void FCIWfn::H1(FCIWfn& result, bool alfa)
//{
//  for(int alfa_sym = 0; alfa_sym < nirreps; ++alfa_sym){
//    int beta_sym = alfa_sym ^ symmetry_;
//    if(detpi[alfa_sym] > 0){
//      double** Y = alfa ? result.coefficients[alfa_sym] : Y1;
//      double** C = alfa ? coefficients[alfa_sym]        : C1;
      
//      if(!alfa){
//        memset(&(C[0][0]), 0, sizeC1);
//        memset(&(Y[0][0]), 0, sizeC1);
//        size_t maxIa = alfa_graph_->strpi(alfa_sym);
//        size_t maxIb = beta_graph_->strpi(beta_sym);
//        // Copy C transposed in C1
//        for(size_t Ia = 0; Ia < maxIa; ++Ia)
//          for(size_t Ib = 0; Ib < maxIb; ++Ib)
//            C[Ib][Ia] = coefficients[alfa_sym][Ia][Ib];
//      }

//      size_t maxL = alfa ? beta_graph_->strpi(beta_sym) : alfa_graph_->strpi(alfa_sym);

//      for(int p_sym = 0; p_sym < nirreps; ++p_sym){
//        int q_sym = p_sym;  // Select the totat symmetric irrep
//        for(int p_rel = 0; p_rel < cmos[p_sym]; ++p_rel){
//          for(int q_rel = 0; q_rel < cmos[q_sym]; ++q_rel){
//            int p_abs = p_rel + cmos_offset[p_sym];
//            int q_abs = q_rel + cmos_offset[q_sym];
//            double Hpq = 0.0;
//            if(Process::environment.options.get_bool("FASTALGRM")){
//              Hpq = alfa ? oei_aa(p_abs,q_abs) : oei_bb(p_abs,q_abs); // Grab the integral
//            }else{
//              Hpq = alfa ? h_aa(p_abs,q_abs) : h_bb(p_abs,q_abs); // Grab the integral
//            }
//            std::vector<StringSubstitution>& vo = alfa ? lists->get_alfa_vo_list(p_abs,q_abs,alfa_sym)
//                                                       : lists->get_beta_vo_list(p_abs,q_abs,beta_sym);
//            // TODO loop in a differen way
//            int maxss = vo.size();
//            for(int ss = 0; ss < maxss; ++ss){
//#if CAPRICCIO_USE_DAXPY
//              C_DAXPY(maxL,static_cast<double>(vo[ss].sign) * Hpq, &C[vo[ss].I][0], 1, &Y[vo[ss].J][0], 1);
//#else
//              double H = static_cast<double>(vo[ss].sign) * Hpq;
//              double* y = &Y[vo[ss].J][0];
//              double* c = &C[vo[ss].I][0];
//              for(size_t L = 0; L < maxL; ++L)
//                y[L] += c[L] * H;
//#endif
//            }
//          }
//        }
//      }
//      if(!alfa){
//        size_t maxIa = alfa_graph_->strpi(alfa_sym);
//        size_t maxIb = beta_graph_->strpi(beta_sym);
//        // Add Y1 transposed to Y
//        for(size_t Ia = 0; Ia < maxIa; ++Ia)
//          for(size_t Ib = 0; Ib < maxIb; ++Ib)
//            result.coefficients[alfa_sym][Ia][Ib] += Y[Ib][Ia];
//      }
//    }
//  } // End loop over h
//}



///**
// * Apply the same-spin two-particle Hamiltonian to the wave function
// * @param alfa flag for alfa or beta component, true = alfa, false = beta
// */
//void FCIWfn::H2_aaaa(FCIWfn& result, bool alfa)
//{
//  for(int alfa_sym = 0; alfa_sym < nirreps; ++alfa_sym){
//    int beta_sym = alfa_sym ^ symmetry_;
//    if(detpi[alfa_sym] > 0){
//      double** Y = alfa ? result.coefficients[alfa_sym] : Y1;
//      double** C = alfa ? coefficients[alfa_sym]        : C1;
//      if(!alfa){
//        memset(&(C[0][0]), 0, sizeC1);
//        memset(&(Y[0][0]), 0, sizeC1);
//        size_t maxIa = alfa_graph_->strpi(alfa_sym);
//        size_t maxIb = beta_graph_->strpi(beta_sym);
//        // Copy C transposed in C1
//        for(size_t Ia = 0; Ia < maxIa; ++Ia)
//          for(size_t Ib = 0; Ib < maxIb; ++Ib)
//            C[Ib][Ia] = coefficients[alfa_sym][Ia][Ib];
//      }

//      size_t maxL = alfa ? beta_graph_->strpi(beta_sym) : alfa_graph_->strpi(alfa_sym);

//      for(int p_sym = 0; p_sym < nirreps; ++p_sym){
//        for(int q_sym = 0; q_sym < nirreps; ++q_sym){
//          for(int r_sym = 0; r_sym < nirreps; ++r_sym){
//            int s_sym = p_sym ^ q_sym ^ r_sym;
//            for(int p_rel = 0; p_rel < cmos[p_sym]; ++p_rel){
//              int p_abs = p_rel + cmos_offset[p_sym];
//              for(int q_rel = 0; q_rel < cmos[q_sym]; ++q_rel){
//                int q_abs = q_rel + cmos_offset[q_sym];
//                for(int r_rel = 0; r_rel < cmos[r_sym]; ++r_rel){
//                  int r_abs = r_rel + cmos_offset[r_sym];
//                  for(int s_rel = 0; s_rel < cmos[s_sym]; ++s_rel){
//                    int s_abs = s_rel + cmos_offset[s_sym];
//                    double integral = alfa ? 0.5 * tei_aaaa(p_abs,q_abs,r_abs,s_abs)
//                                           : 0.5 * tei_bbbb(p_abs,q_abs,r_abs,s_abs); // Grab the integral
//                    std::vector<StringSubstitution>& VOVO = alfa ? lists->get_alfa_vovo_list(p_abs,q_abs,r_abs,s_abs,alfa_sym)
//                                                                 : lists->get_beta_vovo_list(p_abs,q_abs,r_abs,s_abs,beta_sym);
//                    // TODO loop in a differen way
//                    size_t maxss = VOVO.size();
//                    for(size_t ss = 0; ss < maxss; ++ss){
//#if CAPRICCIO_USE_DAXPY
//                      C_DAXPY(maxL,static_cast<double>(VOVO[ss].sign) * integral, &C[VOVO[ss].I][0], 1, &Y[VOVO[ss].J][0], 1);
//#elif CAPRICCIO_USE_UNROLL
//                      double V = static_cast<double>(VOVO[ss].sign) * integral;
//                      double* y = &Y[VOVO[ss].J][0];
//                      double* c = &C[VOVO[ss].I][0];
//                      size_t L1 = maxL % 8; //
//                      size_t L2 = maxL - L1;
//                      for(size_t L = 0; L < L1; ++L)
//                        y[L] += c[L] * V;
//                      for(size_t L = L1; L < L2; L += 8){
//                        y[L]   += c[L]   * V;
//                        y[L+1] += c[L+1] * V;
//                        y[L+2] += c[L+2] * V;
//                        y[L+3] += c[L+3] * V;
//                        y[L+4] += c[L+4] * V;
//                        y[L+5] += c[L+5] * V;
//                        y[L+6] += c[L+6] * V;
//                        y[L+7] += c[L+7] * V;
//                      }
//#else
//                      double V = static_cast<double>(VOVO[ss].sign) * integral;
//                      double* y = &Y[VOVO[ss].J][0];
//                      double* c = &C[VOVO[ss].I][0];
//                      for(size_t L = 0; L < maxL; ++L)  // TODO Use daxpy
//                        y[L] += c[L] * V;
//#endif
//                    }
//                  }
//                }
//              }
//            }
//          }
//        }
//      }
//      if(!alfa){
//        size_t maxIa = alfa_graph_->strpi(alfa_sym);
//        size_t maxIb = beta_graph_->strpi(beta_sym);
//        // Add Y1 transposed to Y
//        for(size_t Ia = 0; Ia < maxIa; ++Ia)
//          for(size_t Ib = 0; Ib < maxIb; ++Ib)
//            result.coefficients[alfa_sym][Ia][Ib] += Y[Ib][Ia];
//      }
//    }
//  } // End loop over h
//}
