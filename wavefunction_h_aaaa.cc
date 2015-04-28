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
#include <libqt/qt.h>

#include "wavefunction.h"

//using namespace std;
//using namespace psi;
//using namespace boost;

namespace psi{ namespace libadaptive{

/**
 * Apply the same-spin two-particle Hamiltonian to the wave function
 * @param alfa flag for alfa or beta component, true = alfa, false = beta
 */
void FCIWfn::H2_aaaa2(FCIWfn& result, bool alfa)
{
    for(int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym){
        int beta_sym = alfa_sym ^ symmetry_;
        if(detpi_[alfa_sym] > 0){
            SharedMatrix C = alfa ? C_[alfa_sym] : C1;
            SharedMatrix Y = alfa ? result.C_[alfa_sym] : Y1;
            double** Ch = C->pointer();
            double** Yh = Y->pointer();

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
            // Loop over (p>q) == (p>q)
            for(int pq_sym = 0; pq_sym < nirrep_; ++pq_sym){
                size_t max_pq = lists_->pairpi(pq_sym);
                for(size_t pq = 0; pq < max_pq; ++pq){
                    const Pair& pq_pair = lists_->get_nn_list_pair(pq_sym,pq);
                    int p_abs = pq_pair.first;
                    int q_abs = pq_pair.second;
//                    double integral = alfa ? tei_aaaa(p_abs,p_abs,q_abs,q_abs)
//                                           : tei_bbbb(p_abs,p_abs,q_abs,q_abs); // Grab the integral
//                    integral -= alfa ? tei_aaaa(p_abs,q_abs,q_abs,p_abs)
//                                     : tei_bbbb(p_abs,q_abs,q_abs,p_abs); // Grab the integral

                    double integral = alfa ? tei_aaaa(p_abs,q_abs,p_abs,q_abs) : tei_bbbb(p_abs,q_abs,p_abs,q_abs);

                    std::vector<StringSubstitution>& OO = alfa ? lists_->get_alfa_oo_list(pq_sym,pq,alfa_sym)
                                                               : lists_->get_beta_oo_list(pq_sym,pq,beta_sym);

                    size_t maxss = OO.size();
                    for(size_t ss = 0; ss < maxss; ++ss)
                        C_DAXPY(maxL,static_cast<double>(OO[ss].sign) * integral, &(C->pointer()[OO[ss].I][0]), 1, &(Y->pointer()[OO[ss].J][0]), 1);
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

//                        double integral = alfa ? tei_aaaa(p_abs,r_abs,q_abs,s_abs)
//                                               : tei_bbbb(p_abs,r_abs,q_abs,s_abs); // Grab the integral
//                        integral -= alfa ? tei_aaaa(p_abs,s_abs,q_abs,r_abs)
//                                         : tei_bbbb(p_abs,s_abs,q_abs,r_abs); // Grab the integral

                        {
                            std::vector<StringSubstitution>& VVOO = alfa ? lists_->get_alfa_vvoo_list(p_abs,q_abs,r_abs,s_abs,alfa_sym)
                                                                         : lists_->get_beta_vvoo_list(p_abs,q_abs,r_abs,s_abs,beta_sym);
                            // TODO loop in a differen way
                            size_t maxss = VVOO.size();
                            for(size_t ss = 0; ss < maxss; ++ss)
                                C_DAXPY(maxL,static_cast<double>(VVOO[ss].sign) * integral, &(C->pointer()[VVOO[ss].I][0]), 1, &(Y->pointer()[VVOO[ss].J][0]), 1);
                        }
                        {
                            std::vector<StringSubstitution>& VVOO = alfa ? lists_->get_alfa_vvoo_list(r_abs,s_abs,p_abs,q_abs,alfa_sym)
                                                                         : lists_->get_beta_vvoo_list(r_abs,s_abs,p_abs,q_abs,beta_sym);
                            // TODO loop in a differen way
                            size_t maxss = VVOO.size();
                            for(size_t ss = 0; ss < maxss; ++ss)
                                C_DAXPY(maxL,static_cast<double>(VVOO[ss].sign) * integral, &(C->pointer()[VVOO[ss].I][0]), 1, &(Y->pointer()[VVOO[ss].J][0]), 1);
                        }
                    }
                }
            }
            if(!alfa){
                size_t maxIa = alfa_graph_->strpi(alfa_sym);
                size_t maxIb = beta_graph_->strpi(beta_sym);

                double** HC = result.C_[alfa_sym]->pointer();

                // Add Y1 transposed to Y
                for(size_t Ia = 0; Ia < maxIa; ++Ia)
                    for(size_t Ib = 0; Ib < maxIb; ++Ib)
                        HC[Ia][Ib] += Yh[Ib][Ia];
//                        result.coefficients[alfa_sym][Ia][Ib] += Y[Ib][Ia];
            }
        }
    } // End loop over h
}

}}
