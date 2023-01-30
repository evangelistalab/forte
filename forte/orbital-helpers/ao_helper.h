/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2022 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#ifndef _atomic_orbital_h_
#define _atomic_orbital_h_

#include "psi4/psi4-dec.h"
#include "psi4/lib3index/denominator.h"

namespace forte {

class AtomicOrbitalHelper {
  protected:
    psi::SharedMatrix CMO_;
    psi::SharedVector eps_rdocc_;
    psi::SharedVector eps_virtual_;
    psi::SharedVector eps_active_;

    std::vector<psi::SharedMatrix> LOcc_list_;
    std::vector<psi::SharedMatrix> LVir_list_;
    std::vector<psi::SharedMatrix> LAct_list_;
    
    std::vector<psi::SharedMatrix> POcc_list_;
    std::vector<psi::SharedMatrix> PVir_list_;

    std::vector<int> n_pseudo_occ_list_;
    std::vector<int> n_pseudo_vir_list_;
    std::vector<int> n_pseudo_act_list_;

    psi::SharedMatrix L_Occ_real_;
    //psi::SharedMatrix L_Vir_real_;
    psi::SharedMatrix POcc_real_;
    // psi::SharedMatrix PVir_real_;


    // LaplaceDenominator Laplace_;
    psi::SharedMatrix Occupied_Laplace_;
    psi::SharedMatrix Virtual_Laplace_;
    psi::SharedMatrix Active_Laplace_;
    double laplace_tolerance_ = 1e-10;

    int weights_;
    int nbf_;
    int nrdocc_;
    int nvir_;
    int nact_;
    /// How many orbitals does it take to go from occupied to virtual (ie should
    /// be active)
    int shift_;
    int nfrozen_;

  public:
    psi::SharedMatrix Occupied_Laplace() { return Occupied_Laplace_; }
    psi::SharedMatrix Virtual_Laplace() { return Virtual_Laplace_; }

    std::vector<psi::SharedMatrix> POcc_list() { return POcc_list_; }
    std::vector<psi::SharedMatrix> PVir_list() { return PVir_list_; }

    std::vector<psi::SharedMatrix> LOcc_list() { return LOcc_list_; }
    std::vector<psi::SharedMatrix> LVir_list() { return LVir_list_; }
    std::vector<psi::SharedMatrix> LAct_list() { return LAct_list_; }

    std::vector<int> n_pseudo_occ_list() { return n_pseudo_occ_list_; }
    std::vector<int> n_pseudo_vir_list() { return n_pseudo_vir_list_; }
    std::vector<int> n_pseudo_act_list() { return n_pseudo_act_list_; }
    
    psi::SharedMatrix L_Occ_real() { return L_Occ_real_; }

    psi::SharedMatrix POcc_real() { return POcc_real_; }
    
    int Weights() { return weights_; }

    AtomicOrbitalHelper(psi::SharedMatrix CMO, psi::SharedVector eps_occ, psi::SharedVector eps_vir,
                        double laplace_tolerance);
    
    AtomicOrbitalHelper(psi::SharedMatrix CMO, psi::SharedVector eps_occ, psi::SharedVector eps_vir,
                        double laplace_tolerance, int shift, int nfrozen);

    AtomicOrbitalHelper(psi::SharedMatrix CMO, psi::SharedVector eps_occ, psi::SharedVector eps_act,
                                         psi::SharedVector eps_vir, double laplace_tolerance,
                                         int shift, int nfrozen, bool cavv);

    void Compute_Cholesky_Density();
    void Compute_Cholesky_Pseudo_Density();
    void Compute_Cholesky_Pseudo_Density(psi::SharedMatrix RDM);
    //void Householder_QR();
    //void Compute_L_Directly();

    ~AtomicOrbitalHelper();
};
} // namespace forte

#endif
