///*
// *  wavefunction_hamiltonian.cpp
// *  Capriccio
// *
// *  Created by Francesco Evangelista on 3/9/09.
// *  Copyright 2009 __MyCompanyName__. All rights reserved.
// *
// */

#include <cmath>

#include <boost/timer.hpp>

#include <libqt/qt.h>

#include "wavefunction.h"

namespace psi{ namespace libadaptive{

/**
 * Compute the one-particle density matrix for a given wave function
 * @param alfa flag for alfa or beta component, true = alfa, false = beta
 */
void FCIWfn::compute_rdms(int max_order)
{
    compute_1rdm(opdm_a_,true);
    compute_1rdm(opdm_b_,false);
    compute_2rdm_aa(tpdm_aa_,true);
}

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
                                rdm[p_abs * ncmo_ + q_abs] += c[L] * y[L] * H;
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
            outfile->Printf("%15.12f ",rdm[oei_index(p,q)]);
        }
    }
}


/**
 * Compute the aa/bb two-particle density matrix for a given wave function
 * @param alfa flag for alfa or beta component, true = aa, false = bb
 */
void FCIWfn::compute_2rdm_aa(std::vector<double>& rdm, bool alfa)
{
    rdm.assign(ncmo_ * ncmo_ * ncmo_ * ncmo_,0.0);
    // Notation
    // ha - symmetry of alpha strings
    // hb - symmetry of beta strings
    for(int ha = 0; ha < nirrep_; ++ha){
        int hb = ha ^ symmetry_;
        if(detpi_[ha] > 0){
            SharedMatrix C = alfa ? C_[ha] : C1;
            double** Ch = C->pointer();

            if(!alfa){
                C->zero();
                size_t maxIa = alfa_graph_->strpi(ha);
                size_t maxIb = beta_graph_->strpi(hb);

                double** C0h = C_[ha]->pointer();

                // Copy C0 transposed in C1
                for(size_t Ia = 0; Ia < maxIa; ++Ia)
                    for(size_t Ib = 0; Ib < maxIb; ++Ib)
                        Ch[Ib][Ia] = C0h[Ia][Ib];
            }

            size_t maxL = alfa ? beta_graph_->strpi(hb) : alfa_graph_->strpi(ha);
            // Loop over (p>q) == (p>q)
            for(int pq_sym = 0; pq_sym < nirrep_; ++pq_sym){
                size_t max_pq = lists_->pairpi(pq_sym);
                for(size_t pq = 0; pq < max_pq; ++pq){
                    const Pair& pq_pair = lists_->get_nn_list_pair(pq_sym,pq);
                    int p_abs = pq_pair.first;
                    int q_abs = pq_pair.second;

                    double integral = alfa ? tei_aaaa(p_abs,q_abs,p_abs,q_abs) : tei_bbbb(p_abs,q_abs,p_abs,q_abs);

                    std::vector<StringSubstitution>& OO = alfa ? lists_->get_alfa_oo_list(pq_sym,pq,ha)
                                                               : lists_->get_beta_oo_list(pq_sym,pq,hb);

                    size_t maxss = OO.size();
                    for(size_t ss = 0; ss < maxss; ++ss){
//                        C_DAXPY(maxL,static_cast<double>(OO[ss].sign) * integral, &(C->pointer()[OO[ss].I][0]), 1, &(Y->pointer()[OO[ss].J][0]), 1);
                        double H = static_cast<double>(OO[ss].sign);
                        double* y = &(Ch[OO[ss].J][0]);
                        double* c = &(Ch[OO[ss].I][0]);
                        for(size_t L = 0; L < maxL; ++L){
                            rdm[tei_index(p_abs,q_abs,p_abs,q_abs)] += c[L] * y[L] * H;
                        }
                    }
                }
            }
            // Loop over (p>q) > (r>s)
            for(int pq_sym = 0; pq_sym < nirrep_; ++pq_sym){
                size_t max_pq = lists_->pairpi(pq_sym);
                for(size_t pq = 0; pq < max_pq; ++pq){
                    const Pair& pq_pair = lists_->get_nn_list_pair(pq_sym,pq);
                    int p_abs = pq_pair.first;
                    int q_abs = pq_pair.second;
                    for(size_t rs = 0; rs < pq; ++rs){
                        const Pair& rs_pair = lists_->get_nn_list_pair(pq_sym,rs);
                        int r_abs = rs_pair.first;
                        int s_abs = rs_pair.second;
                        double integral = alfa ? tei_aaaa(p_abs,q_abs,r_abs,s_abs) : tei_bbbb(p_abs,q_abs,r_abs,s_abs);

                        {
                            std::vector<StringSubstitution>& VVOO = alfa ? lists_->get_alfa_vvoo_list(p_abs,q_abs,r_abs,s_abs,ha)
                                                                         : lists_->get_beta_vvoo_list(p_abs,q_abs,r_abs,s_abs,hb);
                            // TODO loop in a differen way
                            size_t maxss = VVOO.size();
                            for(size_t ss = 0; ss < maxss; ++ss){
                                double H = static_cast<double>(VVOO[ss].sign);
                                double* y = &(Ch[VVOO[ss].J][0]);
                                double* c = &(Ch[VVOO[ss].I][0]);
                                for(size_t L = 0; L < maxL; ++L){
                                    rdm[tei_index(p_abs,q_abs,r_abs,s_abs)] += c[L] * y[L] * H;
                                }
                            }
//                                C_DAXPY(maxL,static_cast<double>(VVOO[ss].sign) * integral, &(C->pointer()[VVOO[ss].I][0]), 1, &(Y->pointer()[VVOO[ss].J][0]), 1);
                        }
                        {
                            std::vector<StringSubstitution>& VVOO = alfa ? lists_->get_alfa_vvoo_list(r_abs,s_abs,p_abs,q_abs,ha)
                                                                         : lists_->get_beta_vvoo_list(r_abs,s_abs,p_abs,q_abs,hb);
                            // TODO loop in a differen way
                            size_t maxss = VVOO.size();
                            for(size_t ss = 0; ss < maxss; ++ss){
                                double H = static_cast<double>(VVOO[ss].sign);
                                double* y = &(Ch[VVOO[ss].J][0]);
                                double* c = &(Ch[VVOO[ss].I][0]);
                                for(size_t L = 0; L < maxL; ++L){
                                    rdm[tei_index(p_abs,q_abs,r_abs,s_abs)] += c[L] * y[L] * H;
                                }
                            }
//                                C_DAXPY(maxL,static_cast<double>(VVOO[ss].sign) * integral, &(C->pointer()[VVOO[ss].I][0]), 1, &(Y->pointer()[VVOO[ss].J][0]), 1);
                        }
                    }
                }
            }
        }
    } // End loop over h

    outfile->Printf("\n TPDM:");
    for (int p = 0; p < ncmo_; ++p) {
        for (int q = 0; q <= p; ++q) {
            for (int r = 0; r < ncmo_; ++r) {
                for (int s = 0; s <= r; ++s) {
                    if (std::fabs(rdm[tei_index(p,q,r,s)]) > 1.0e-12){
                        outfile->Printf("\n %+-20.12f : %3d %3d %3d %3d",rdm[tei_index(p,q,r,s)],p,q,r,s);
                    }
                }
            }
        }
    }
}


/**
 * Compute the ab two-particle density matrix for a given wave function
 * @param alfa flag for alfa or beta component, true = aa, false = bb
 */
void FCIWfn::compute_2rdm_ab(std::vector<double>& rdm, bool alfa)
{

}

}}
