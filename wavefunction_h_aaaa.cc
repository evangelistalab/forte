///*
// *  wavefunction_h_aaaa.cc
// *  Capriccio
// *
// *  Created by Francesco Evangelista on 3/17/09.
// *  Copyright 2009 __MyCompanyName__. All rights reserved.
// *
// */

//#include <boost/timer.hpp>

//#include <libmoinfo/libmoinfo.h>
//#include <libqt/qt.h>

//#include "wavefunction.h"

//using namespace std;
//using namespace psi;
//using namespace boost;

///**
// * Apply the same-spin two-particle Hamiltonian to the wave function
// * @param alfa flag for alfa or beta component, true = alfa, false = beta
// */
//void FCIWfn::H2_aaaa2(FCIWfn& result, bool alfa)
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
//      // Loop over (p>q) == (p>q)
//      for(int pq_sym = 0; pq_sym < nirreps; ++pq_sym){
//        size_t max_pq = lists->get_pairpi(pq_sym);
//        for(size_t pq = 0; pq < max_pq; ++pq){
//          const Pair& pq_pair = lists->get_nn_list_pair(pq_sym,pq);
//          int p_abs = pq_pair.first;
//          int q_abs = pq_pair.second;
//          double integral = alfa ? tei_aaaa(p_abs,p_abs,q_abs,q_abs)
//                                 : tei_bbbb(p_abs,p_abs,q_abs,q_abs); // Grab the integral
//          integral -= alfa ? tei_aaaa(p_abs,q_abs,q_abs,p_abs)
//                           : tei_bbbb(p_abs,q_abs,q_abs,p_abs); // Grab the integral

//          std::vector<StringSubstitution>& OO = alfa ? lists->get_alfa_oo_list(pq_sym,pq,alfa_sym)
//                                                     : lists->get_beta_oo_list(pq_sym,pq,beta_sym);

//          size_t maxss = OO.size();
//          for(size_t ss = 0; ss < maxss; ++ss)
//            C_DAXPY(maxL,static_cast<double>(OO[ss].sign) * integral, &C[OO[ss].I][0], 1, &Y[OO[ss].J][0], 1);
//        }
//      }
//      // Loop over (p>q) > (r>s)
//      for(int pq_sym = 0; pq_sym < nirreps; ++pq_sym){
//        size_t max_pq = lists->get_pairpi(pq_sym);
//        for(size_t pq = 0; pq < max_pq; ++pq){
//          const Pair& pq_pair = lists->get_nn_list_pair(pq_sym,pq);
//          int p_abs = pq_pair.first;
//          int q_abs = pq_pair.second;
//          for(size_t rs = 0; rs < pq; ++rs){
//              const Pair& rs_pair = lists->get_nn_list_pair(pq_sym,rs);
//              int r_abs = rs_pair.first;
//              int s_abs = rs_pair.second;
//              double integral = alfa ? tei_aaaa(p_abs,r_abs,q_abs,s_abs)
//                                     : tei_bbbb(p_abs,r_abs,q_abs,s_abs); // Grab the integral
//              integral -= alfa ? tei_aaaa(p_abs,s_abs,q_abs,r_abs)
//                               : tei_bbbb(p_abs,s_abs,q_abs,r_abs); // Grab the integral

//              {
//                std::vector<StringSubstitution>& VVOO = alfa ? lists->get_alfa_vvoo_list(p_abs,q_abs,r_abs,s_abs,alfa_sym)
//                                                             : lists->get_beta_vvoo_list(p_abs,q_abs,r_abs,s_abs,beta_sym);
//                // TODO loop in a differen way
//                size_t maxss = VVOO.size();
//                for(size_t ss = 0; ss < maxss; ++ss)
//                  C_DAXPY(maxL,static_cast<double>(VVOO[ss].sign) * integral, &C[VVOO[ss].I][0], 1, &Y[VVOO[ss].J][0], 1);
//              }
//              {
//                std::vector<StringSubstitution>& VVOO = alfa ? lists->get_alfa_vvoo_list(r_abs,s_abs,p_abs,q_abs,alfa_sym)
//                                                             : lists->get_beta_vvoo_list(r_abs,s_abs,p_abs,q_abs,beta_sym);
//                // TODO loop in a differen way
//                size_t maxss = VVOO.size();
//                for(size_t ss = 0; ss < maxss; ++ss)
//                  C_DAXPY(maxL,static_cast<double>(VVOO[ss].sign) * integral, &C[VVOO[ss].I][0], 1, &Y[VVOO[ss].J][0], 1);
//              }
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
// * Apply the different-spin component of two-particle Hamiltonian to the wave function
// */
//void FCIWfn::H2_aabb(FCIWfn& result)
//{
//    // Loop over blocks of matrix C
//    for(int Ia_sym = 0; Ia_sym < nirreps; ++Ia_sym){
//        size_t maxIa = alfa_graph_->strpi(Ia_sym);
//        int Ib_sym = Ia_sym ^ symmetry_;
//        double** C = coefficients[Ia_sym];

//        // Loop over all r,s
//        for(int rs_sym = 0; rs_sym < nirreps; ++rs_sym){
//            int Ja_sym = Ia_sym ^ rs_sym;
//            size_t maxJa = alfa_graph_->strpi(Ja_sym);
//            double** Y = result.coefficients[Ja_sym];
//            for(int r_sym = 0; r_sym < nirreps; ++r_sym){
//                int s_sym = rs_sym ^ r_sym;

//                for(int r_rel = 0; r_rel < cmos[r_sym]; ++r_rel){
//                    for(int s_rel = 0; s_rel < cmos[s_sym]; ++s_rel){
//                        int r_abs = r_rel + cmos_offset[r_sym];
//                        int s_abs = s_rel + cmos_offset[s_sym];

//                        // Grab list (r,s,Ib_sym)
//                        std::vector<StringSubstitution>& vo_beta = lists->get_beta_vo_list(r_abs,s_abs,Ib_sym);
//                        size_t maxSSb = vo_beta.size();

//                        //            // Gather cols of C into C1
//                        //            for(size_t SSb = 0; SSb < maxSSb; ++SSb){
//                        //              for(size_t Ia = 0; Ia < maxIa; ++Ia){
//                        //                C1[Ia][SSb] = C[Ia][vo_beta[SSb].I] * static_cast<double>(vo_beta[SSb].sign);
//                        //              }
//                        //            }
//                        memset(&(C1[0][0]), 0, sizeC1);
//                        memset(&(Y1[0][0]), 0, sizeC1);

//                        // Gather cols of C into C1
//                        for(size_t Ia = 0; Ia < maxIa; ++Ia){
//                            if(maxSSb > 0){
//                                double* c1 = &C1[Ia][0];
//                                double* c  = &C[Ia][0];
//                                for(size_t SSb = 0; SSb < maxSSb; ++SSb){
//                                    c1[SSb] = c[vo_beta[SSb].I] * static_cast<double>(vo_beta[SSb].sign);
//                                }
//                            }
//                        }


//                        // Loop over all p,q
//                        int pq_sym = rs_sym;
//                        for(int p_sym = 0; p_sym < nirreps; ++p_sym){
//                            int q_sym = pq_sym ^ p_sym;
//                            for(int p_rel = 0; p_rel < cmos[p_sym]; ++p_rel){
//                                int p_abs = p_rel + cmos_offset[p_sym];
//                                for(int q_rel = 0; q_rel < cmos[q_sym]; ++q_rel){
//                                    int q_abs = q_rel + cmos_offset[q_sym];
//                                    // Grab the integral
//                                    double integral = tei_aabb(p_abs,q_abs,r_abs,s_abs);

//                                    std::vector<StringSubstitution>& vo_alfa = lists->get_alfa_vo_list(p_abs,q_abs,Ia_sym);


//                                    // ORIGINAL CODE
//                                    size_t maxSSa = vo_alfa.size();
//                                    for(size_t SSa = 0; SSa < maxSSa; ++SSa){
//#if CAPRICCIO_USE_DAXPY
//                                        C_DAXPY(maxSSb,integral * static_cast<double>(vo_alfa[SSa].sign), &C1[vo_alfa[SSa].I][0], 1, &Y1[vo_alfa[SSa].J][0], 1);
//#else
//                                        double V = integral * static_cast<double>(vo_alfa[SSa].sign);
//                                        for(size_t SSb = 0; SSb < maxSSb; ++SSb){
//                                            Y1[vo_alfa[SSa].J][SSb] += C1[vo_alfa[SSa].I][SSb] * V;
//                                        }
//#endif
////
////                                    long maxSSa = static_cast<long>(vo_alfa.size());
////                                    long SSa;
////                                    for(SSa = 0; SSa < maxSSa; ++SSa){
////                                        double V = integral * static_cast<double>(vo_alfa[SSa].sign);
////                                        double* y1 = &Y1[vo_alfa[SSa].J][0];
////                                        double* c1 = &C1[vo_alfa[SSa].I][0];
////                                        for(size_t SSb = 0; SSb < maxSSb; ++SSb){
////                                            y1[SSb] += c1[SSb] * V;
////                                        }
//                                    }
//                                }
//                            }
//                        } // End loop over p,q
//                        // Scatter cols of Y1 into Y
//                        for(size_t Ja = 0; Ja < maxJa; ++Ja){
//                            if(maxSSb > 0){
//                                double* y = &Y[Ja][0];
//                                double* y1 = &Y1[Ja][0];
//                                for(size_t SSb = 0; SSb < maxSSb; ++SSb){
//                                    y[vo_beta[SSb].J] += y1[SSb];
//                                }
//                            }
//                        }
//                        //            // Scatter cols of Y1 into Y
//                        //            for(size_t SSb = 0; SSb < maxSSb; ++SSb){
//                        //              for(size_t Ja = 0; Ja < maxJa; ++Ja){
//                        //                Y[Ja][vo_beta[SSb].J] += Y1[Ja][SSb];
//                        //              }
//                        //            }
//                    }
//                } // End loop over r_rel,s_rel

//            }
//        }
//    }
//}


//void FCIWfn::form_H_diagonal(boost::shared_ptr<Integrals> ints)
//{
//  timer t;

//  int wfn_sym = symmetry_;
//  int n  = ncmos;
//  int ka = alfa_graph_->nones();
//  int kb = beta_graph_->nones();

//  bool* Ia = new bool[n];
//  bool* Ib = new bool[n];

//  // Generate the alfa string 1111000000
//  //                          {ka}{n-ka}
//  for(int i = 0; i < n - ka; ++i) Ia[i] = false; // 0
//  for(int i = n - ka; i < n; ++i) Ia[i] = true;  // 1
//  // Loop over all alfa strings
//  do{
//    // Compute irrep
//    int alfa_sym = alfa_graph_->sym(Ia);
//    int beta_sym = alfa_sym ^ wfn_sym;

//    // Generate the beta string 1111000000
//    //                          {kb}{n-kb}
//    for(int i = 0; i < n - kb; ++i) Ib[i] = false; // 0
//    for(int i = n - kb; i < n; ++i) Ib[i] = true;  // 1
//    // Loop over all beta strings
//    do{
//      // Check if the product of strings gives the right irrep
//      if(beta_graph_->sym(Ib) == beta_sym){
//        size_t addIa = alfa_graph_->rel_add(Ia);
//        size_t addIb = beta_graph_->rel_add(Ib);
//        coefficients[alfa_sym][addIa][addIb] = determinant_energy(Ia,Ib,n,ints);
////        outfile->Printf("\n |[%1d][%3d][%3d]> energy = %20.12f",alfa_sym,static_cast<int> (addIa),
////                                                                         static_cast<int> (addIb),coefficients[alfa_sym][addIa][addIb]);
//      }
//    } while (std::next_permutation(Ib,Ib+n));

//  } while (std::next_permutation(Ia,Ia+n));

//  hdiag_timer += t.elapsed();
//  outfile->Printf("\n  timing for Hdiag     = %10.3f s\n",hdiag_timer);
//  outfile->Flush();
//}

//double FCIWfn::determinant_energy(bool*& Ia,bool*& Ib,int n,boost::shared_ptr<Integrals>& ints)
//{
//  double energy(ints->frozen_core_energy());
//  for(int p = 0; p < n; ++p){
//    if(Ia[p]) energy += oei_aa(p,p);
//    if(Ib[p]) energy += oei_bb(p,p);
//    for(int q = 0; q < n; ++q){
//      if(Ia[p] && Ia[q])
//        energy += 0.5 * tei_aaaa(p,p,q,q) - 0.5 * tei_aaaa(p,q,p,q);
//      if(Ib[p] && Ib[q])
//        energy += 0.5 * tei_bbbb(p,p,q,q) - 0.5 * tei_bbbb(p,q,p,q);
//      if(Ia[p] && Ib[q])
//        energy += tei_aabb(p,p,q,q);
//    }
//  }
//  return(energy);
//}
