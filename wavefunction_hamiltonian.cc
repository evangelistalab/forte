///*
// *  wavefunction_hamiltonian.cpp
// *  Capriccio
// *
// *  Created by Francesco Evangelista on 3/9/09.
// *  Copyright 2009 __MyCompanyName__. All rights reserved.
// *
// */

#include <boost/timer.hpp>

#include <libqt/qt.h>

#include "wavefunction.h"

namespace psi{ namespace libadaptive{

/**
 * Apply the Hamiltonian to the wave function
 * @param result Wave function object which stores the resulting vector
 */
void FCIWfn::Hamiltonian(FCIWfn& result,RequiredLists required_lists)
{
//    check_temp_space();
    result.zero();

    // H0
    {
        H0(result);
    }

    // H1_aa
    { boost::timer t;
        H1(result,true);
        h1_aa_timer += t.elapsed();
    }
    // H1_bb
    { boost::timer t;
        H1(result,false);
        h1_bb_timer += t.elapsed();
    }
    // H2_aabb
    { boost::timer t;
        H2_aabb(result);
        h2_aabb_timer += t.elapsed();
    }
    // H2_aaaa
    { boost::timer t;
        H2_aaaa2(result,true);
        h2_aaaa_timer += t.elapsed();
    }
    // H2_bbbb
    { boost::timer t;
        H2_aaaa2(result,false);
        h2_bbbb_timer += t.elapsed();
    }
}


/**
 * Apply the scalar part of the Hamiltonian to the wave function
 */
void FCIWfn::H0(FCIWfn& result)
{
    double core_energy = ints_->frozen_core_energy();
    for(int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym){
        result.C_[alfa_sym]->copy(C_[alfa_sym]);
        result.C_[alfa_sym]->scale(core_energy);
    }
}

/**
 * Apply the one-particle Hamiltonian to the wave function
 * @param alfa flag for alfa or beta component, true = alfa, false = beta
 */
void FCIWfn::H1(FCIWfn& result, bool alfa)
{
    for(int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym){
        int beta_sym = alfa_sym ^ symmetry_;
        if(detpi_[alfa_sym] > 0){
            SharedMatrix C = alfa ? C_[alfa_sym] : C1;
            SharedMatrix Y = alfa ? result.C_[alfa_sym] : Y1;
            double** Ch = C->pointer();
            double** Yh = Y->pointer();

            C_[alfa_sym]->print();
            if(!alfa){
                C->zero();
                Y->zero();
                size_t maxIa = alfa_graph_->strpi(alfa_sym);
                size_t maxIb = beta_graph_->strpi(beta_sym);

                double** C0h = C_[alfa_sym]->pointer();

                // Copy C0 transposed in C1
                for(size_t Ia = 0; Ia < maxIa; ++Ia)
                    for(size_t Ib = 0; Ib < maxIb; ++Ib)
                        Ch[Ib][Ia] = C0h[Ia][Ib];
            }

            size_t maxL = alfa ? beta_graph_->strpi(beta_sym) : alfa_graph_->strpi(alfa_sym);

            for(int p_sym = 0; p_sym < nirrep_; ++p_sym){
                int q_sym = p_sym;  // Select the totat symmetric irrep
                for(int p_rel = 0; p_rel < cmopi_[p_sym]; ++p_rel){
                    for(int q_rel = 0; q_rel < cmopi_[q_sym]; ++q_rel){
                        int p_abs = p_rel + cmopi_offset_[p_sym];
                        int q_abs = q_rel + cmopi_offset_[q_sym];

                        double Hpq = alfa ? oei_aa(p_abs,q_abs) : oei_bb(p_abs,q_abs); // Grab the integral
                        std::vector<StringSubstitution>& vo = alfa ? lists_->get_alfa_vo_list(p_abs,q_abs,alfa_sym)
                                                                   : lists_->get_beta_vo_list(p_abs,q_abs,beta_sym);
                        outfile->Printf("\n  p,q = %zu,%zu <p|h|q> = %f",p_abs,q_abs,Hpq);
                        // TODO loop in a differen way
                        int maxss = vo.size();
                        for(int ss = 0; ss < maxss; ++ss){
#if CAPRICCIO_USE_DAXPY
                            C_DAXPY(maxL,static_cast<double>(vo[ss].sign) * Hpq, &(Ch[vo[ss].I][0]), 1, &(Yh[vo[ss].J][0]), 1);
#else
                            double H = static_cast<double>(vo[ss].sign) * Hpq;
                            double* y = &Y[vo[ss].J][0];
                            double* c = &C[vo[ss].I][0];
                            for(size_t L = 0; L < maxL; ++L)
                                y[L] += c[L] * H;
#endif
                        }
                    }
                }
            }
            Y->print();
            if(!alfa){
                size_t maxIa = alfa_graph_->strpi(alfa_sym);
                size_t maxIb = beta_graph_->strpi(beta_sym);

                double** HC = result.C_[alfa_sym]->pointer();
                // Add Y1 transposed to Y
                for(size_t Ia = 0; Ia < maxIa; ++Ia)
                    for(size_t Ib = 0; Ib < maxIb; ++Ib)
                        HC[Ia][Ib] += Yh[Ib][Ia];
            }
        }
    } // End loop over h
}

}}
