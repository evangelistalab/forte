///*
// *  wavefunction_h_abab.cc
// *  Capriccio
// *
// *  Created by Francesco Evangelista on 3/18/09.
// *  Copyright 2009 __MyCompanyName__. All rights reserved.
// *
// */

#include <libqt/qt.h>

#include "wavefunction.h"

namespace psi{ namespace libadaptive{

/**
 * Apply the different-spin component of two-particle Hamiltonian to the wave function
 */
void FCIWfn::H2_aabb(FCIWfn& result)
{
    // Loop over blocks of matrix C
    for(int Ia_sym = 0; Ia_sym < nirrep_; ++Ia_sym){
        size_t maxIa = alfa_graph_->strpi(Ia_sym);
        int Ib_sym = Ia_sym ^ symmetry_;
        double** C = C_[Ia_sym]->pointer();
        
        // Loop over all r,s
        for(int rs_sym = 0; rs_sym < nirrep_; ++rs_sym){
            int Ja_sym = Ia_sym ^ rs_sym;
            size_t maxJa = alfa_graph_->strpi(Ja_sym);
            double** Y = result.C_[Ja_sym]->pointer();
            for(int r_sym = 0; r_sym < nirrep_; ++r_sym){
                int s_sym = rs_sym ^ r_sym;
                
                for(int r_rel = 0; r_rel < cmopi_[r_sym]; ++r_rel){
                    for(int s_rel = 0; s_rel < cmopi_[s_sym]; ++s_rel){
                        int r_abs = r_rel + cmopi_offset_[r_sym];
                        int s_abs = s_rel + cmopi_offset_[s_sym];
                        
                        // Grab list (r,s,Ib_sym)
                        std::vector<StringSubstitution>& vo_beta = lists_->get_beta_vo_list(r_abs,s_abs,Ib_sym);
                        size_t maxSSb = vo_beta.size();
                        
                        //            // Gather cols of C into C1
                        //            for(size_t SSb = 0; SSb < maxSSb; ++SSb){
                        //              for(size_t Ia = 0; Ia < maxIa; ++Ia){
                        //                C1[Ia][SSb] = C[Ia][vo_beta[SSb].I] * static_cast<double>(vo_beta[SSb].sign);
                        //              }
                        //            }
                        C1->zero();
                        Y1->zero();
                        
                        // Gather cols of C into C1
                        for(size_t Ia = 0; Ia < maxIa; ++Ia){
                            if(maxSSb > 0){
                                double* c1 = &(C1->pointer()[Ia][0]);//&C1[Ia][0];
                                double* c  = &(C[Ia][0]);
                                for(size_t SSb = 0; SSb < maxSSb; ++SSb){
                                    c1[SSb] = c[vo_beta[SSb].I] * static_cast<double>(vo_beta[SSb].sign);
                                }
                            }
                        }
                        
                        
                        // Loop over all p,q
                        int pq_sym = rs_sym;
                        for(int p_sym = 0; p_sym < nirrep_; ++p_sym){
                            int q_sym = pq_sym ^ p_sym;
                            for(int p_rel = 0; p_rel < cmopi_[p_sym]; ++p_rel){
                                int p_abs = p_rel + cmopi_offset_[p_sym];
                                for(int q_rel = 0; q_rel < cmopi_[q_sym]; ++q_rel){
                                    int q_abs = q_rel + cmopi_offset_[q_sym];
                                    // Grab the integral
//                                    double integral = tei_aabb(p_abs,q_abs,r_abs,s_abs);  NB: original
                                    double integral = tei_aabb(p_abs,r_abs,q_abs,s_abs);
                                    
                                    std::vector<StringSubstitution>& vo_alfa = lists_->get_alfa_vo_list(p_abs,q_abs,Ia_sym);


                                    // ORIGINAL CODE
                                    size_t maxSSa = vo_alfa.size();
                                    for(size_t SSa = 0; SSa < maxSSa; ++SSa){
#if CAPRICCIO_USE_DAXPY
                                        C_DAXPY(maxSSb,integral * static_cast<double>(vo_alfa[SSa].sign),
                                                &(C1->pointer()[vo_alfa[SSa].I][0]), 1, &(Y1->pointer()[vo_alfa[SSa].J][0]), 1);
#else
                                        double V = integral * static_cast<double>(vo_alfa[SSa].sign);
                                        for(size_t SSb = 0; SSb < maxSSb; ++SSb){
                                            Y1[vo_alfa[SSa].J][SSb] += C1[vo_alfa[SSa].I][SSb] * V;
                                        }
#endif
                                        //
                                        //                                    long maxSSa = static_cast<long>(vo_alfa.size());
                                        //                                    long SSa;
                                        //                                    for(SSa = 0; SSa < maxSSa; ++SSa){
                                        //                                        double V = integral * static_cast<double>(vo_alfa[SSa].sign);
                                        //                                        double* y1 = &Y1[vo_alfa[SSa].J][0];
                                        //                                        double* c1 = &C1[vo_alfa[SSa].I][0];
                                        //                                        for(size_t SSb = 0; SSb < maxSSb; ++SSb){
                                        //                                            y1[SSb] += c1[SSb] * V;
                                        //                                        }
                                    }
                                }
                            }
                        } // End loop over p,q
                        // Scatter cols of Y1 into Y
                        for(size_t Ja = 0; Ja < maxJa; ++Ja){
                            if(maxSSb > 0){
                                double* y = &Y[Ja][0];
                                double* y1 = &(Y1->pointer()[Ja][0]);
                                for(size_t SSb = 0; SSb < maxSSb; ++SSb){
                                    y[vo_beta[SSb].J] += y1[SSb];
                                }
                            }
                        }
                        //            // Scatter cols of Y1 into Y
                        //            for(size_t SSb = 0; SSb < maxSSb; ++SSb){
                        //              for(size_t Ja = 0; Ja < maxJa; ++Ja){
                        //                Y[Ja][vo_beta[SSb].J] += Y1[Ja][SSb];
                        //              }
                        //            }
                    }
                } // End loop over r_rel,s_rel
                
            }
        }
    }
}

}}
