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

#include "bitset_determinant.h"
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
    compute_2rdm_ab(tpdm_ab_);

//    rdm_test();
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

                    std::vector<StringSubstitution>& OO = alfa ? lists_->get_alfa_oo_list(pq_sym,pq,ha)
                                                               : lists_->get_beta_oo_list(pq_sym,pq,hb);

                    size_t maxss = OO.size();
                    for(size_t ss = 0; ss < maxss; ++ss){
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
                                    rdm[tei_index(p_abs,q_abs,r_abs,s_abs)] += 0.5 * c[L] * y[L] * H;
                                }
                            }
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
                                    rdm[tei_index(p_abs,q_abs,r_abs,s_abs)] += 0.5 * c[L] * y[L] * H;
                                }
                            }
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
                        outfile->Printf("\n  Lambda [%3lu][%3lu][%3lu][%3lu] = %18.12lf", p,q,r,s, rdm[tei_index(p,q,r,s)]);

                    }
                }
            }
        }
    }

    //    outfile->Printf("\n  Lambda [%3lu][%3lu][%3lu][%3lu] = %18.12lf", i, j, k, l, TwoPDC[i][j][k][l]);

}


/**
 * Compute the ab two-particle density matrix for a given wave function
 * @param alfa flag for alfa or beta component, true = aa, false = bb
 */
void FCIWfn::compute_2rdm_ab(std::vector<double>& rdm)
{
    rdm.assign(ncmo_ * ncmo_ * ncmo_ * ncmo_,0.0);

    // Loop over blocks of matrix C
    for(int Ia_sym = 0; Ia_sym < nirrep_; ++Ia_sym){
        size_t maxIa = alfa_graph_->strpi(Ia_sym);
        int Ib_sym = Ia_sym ^ symmetry_;
        double** C = C_[Ia_sym]->pointer();

        // Loop over all r,s
        for(int rs_sym = 0; rs_sym < nirrep_; ++rs_sym){
            int Ja_sym = Ia_sym ^ rs_sym;
            size_t maxJa = alfa_graph_->strpi(Ja_sym);
            double** Y = C_[Ja_sym]->pointer();
            for(int r_sym = 0; r_sym < nirrep_; ++r_sym){
                int s_sym = rs_sym ^ r_sym;

                for(int r_rel = 0; r_rel < cmopi_[r_sym]; ++r_rel){
                    for(int s_rel = 0; s_rel < cmopi_[s_sym]; ++s_rel){
                        int r_abs = r_rel + cmopi_offset_[r_sym];
                        int s_abs = s_rel + cmopi_offset_[s_sym];

                        // Grab list (r,s,Ib_sym)
                        std::vector<StringSubstitution>& vo_beta = lists_->get_beta_vo_list(r_abs,s_abs,Ib_sym);
                        size_t maxSSb = vo_beta.size();

                        // Loop over all p,q
                        int pq_sym = rs_sym;
                        for(int p_sym = 0; p_sym < nirrep_; ++p_sym){
                            int q_sym = pq_sym ^ p_sym;
                            for(int p_rel = 0; p_rel < cmopi_[p_sym]; ++p_rel){
                                int p_abs = p_rel + cmopi_offset_[p_sym];
                                for(int q_rel = 0; q_rel < cmopi_[q_sym]; ++q_rel){
                                    int q_abs = q_rel + cmopi_offset_[q_sym];

                                    std::vector<StringSubstitution>& vo_alfa = lists_->get_alfa_vo_list(p_abs,q_abs,Ia_sym);

                                    size_t maxSSa = vo_alfa.size();
                                    for(size_t SSa = 0; SSa < maxSSa; ++SSa){
                                        for(size_t SSb = 0; SSb < maxSSb; ++SSb){
                                            double V = static_cast<double>(vo_alfa[SSa].sign * vo_beta[SSb].sign);
                                            rdm[tei_index(p_abs,q_abs,r_abs,s_abs)] += Y[vo_alfa[SSa].J][vo_beta[SSb].J] * C[vo_alfa[SSa].I][vo_beta[SSb].I] * V;
                                        }
                                    }
                                }
                            }
                        } // End loop over p,q
                    }
                } // End loop over r_rel,s_rel
            }
        }
    }
    outfile->Printf("\n TPDM (ab):");
    for (int p = 0; p < ncmo_; ++p) {
        for (int q = 0; q < ncmo_; ++q) {
            for (int r = 0; r < ncmo_; ++r) {
                for (int s = 0; s < ncmo_; ++s) {
                    if (std::fabs(rdm[tei_index(p,q,r,s)]) > 1.0e-12){
                        outfile->Printf("\n  Lambda [%3lu][%3lu][%3lu][%3lu] = %18.12lf", p,q,r,s, rdm[tei_index(p,q,r,s)]);

                    }
                }
            }
        }
    }
}


/**
 * Compute the aaa three-particle density matrix for a given wave function
 * @param alfa flag for alfa or beta component, true = aa, false = bb
 */
void FCIWfn::compute_3rdm_aaa(std::vector<double>& rdm, bool alfa)
{

}

/**
 * Compute the aab three-particle density matrix for a given wave function
 * @param alfa flag for alfa or beta component, true = aa, false = bb
 */
void FCIWfn::compute_3rdm_aab(std::vector<double>& rdm, bool alfa)
{

}

void FCIWfn::rdm_test()
{
    bool* Ia = new bool[ncmo_];
    bool* Ib = new bool[ncmo_];

    // Generate the strings 1111100000
    //                      { k }{n-k}

    size_t na = lists_->na();
    size_t nb = lists_->nb();

    for(int i = 0; i < ncmo_ - na; ++i) Ia[i] = false; // 0
    for(int i = ncmo_ - na; i < ncmo_; ++i) Ia[i] = true;  // 1

    for(int i = 0; i < ncmo_ - nb; ++i) Ib[i] = false; // 0
    for(int i = ncmo_ - nb; i < ncmo_; ++i) Ib[i] = true;  // 1

    std::vector<BitsetDeterminant> dets;
    std::vector<double> C;
    std::vector<bool> a_occ(ncmo_);
    std::vector<bool> b_occ(ncmo_);
    do{
        for (int i = 0; i < ncmo_; ++i) a_occ[i] = Ia[i];
        do{
            for (int i = 0; i < ncmo_; ++i) b_occ[i] = Ib[i];
            if((alfa_graph_->sym(Ia) ^ beta_graph_->sym(Ib)) == symmetry_){
                dets.push_back(BitsetDeterminant(a_occ,b_occ));
                double c = C_[alfa_graph_->sym(Ia)]->get(alfa_graph_->rel_add(Ia),beta_graph_->rel_add(Ib));
                C.push_back(c);
            }
        } while (std::next_permutation(Ib,Ib + ncmo_));
    } while (std::next_permutation(Ia,Ia + ncmo_));


    BitsetDeterminant I,J;
    for(int pq_sym = 0; pq_sym < nirrep_; ++pq_sym){
        size_t max_pq = lists_->pairpi(pq_sym);
        for(size_t pq = 0; pq < max_pq; ++pq){
            const Pair& pq_pair = lists_->get_nn_list_pair(pq_sym,pq);
            int p = pq_pair.first;
            int q = pq_pair.second;
            double rdm = 0.0;
            for (size_t i = 0; i < dets.size(); ++i){
                I.copy(dets[i]);
                double sign = 1.0;
                sign *= I.destroy_alfa_bit(p);
                sign *= I.destroy_alfa_bit(q);
                sign *= I.create_alfa_bit(q);
                sign *= I.create_alfa_bit(p);
                rdm += sign * C[i] * C[i];
            }
            outfile->Printf("\n  Lambda [%3lu][%3lu][%3lu][%3lu] = %18.12lf", p,q,p,q,rdm);
        }
    }

    delete[] Ia;
    delete[] Ib;
}

}}
