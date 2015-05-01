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
 * Compute the one-particle density matrix for a given wave function
 * @param alfa flag for alfa or beta component, true = alfa, false = beta
 */
void FCIWfn::compute_1rdm(std::vector<double>& rdm, bool alfa)
{
    rdm.assign(ncmo_ * ncmo_,0.0);

    for(int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym){
        int beta_sym = alfa_sym ^ symmetry_;
        if(detpi_[alfa_sym] > 0){
            SharedMatrix C = alfa ? C_[alfa_sym] : C1;
            double** Ch = C->pointer();

            if(!alfa){
                C->zero();
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
                        std::vector<StringSubstitution>& vo = alfa ? lists_->get_alfa_vo_list(p_abs,q_abs,alfa_sym)
                                                                   : lists_->get_beta_vo_list(p_abs,q_abs,beta_sym);
                        int maxss = vo.size();
                        for(int ss = 0; ss < maxss; ++ss){
                            double H = static_cast<double>(vo[ss].sign);
                            double* y = &(Ch[vo[ss].J][0]);
                            double* c = &(Ch[vo[ss].I][0]);
                            for(size_t L = 0; L < maxL; ++L){
                                rdm[oei_index(p_abs,q_abs)] += c[L] * y[L] * H;
                            }
                        }
                    }
                }
            }
        }
    } // End loop over h
    outfile->Printf("\n OPDM:");
    for (int p = 0; p < ncmo_; ++p) {
        outfile->Printf("\n");
        for (int q = 0; q < ncmo_; ++q) {
            outfile->Printf("%9.6f ",rdm[oei_index(p,q)]);
        }
    }
}

}}
