/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifndef _reference_h_
#define _reference_h_

#include "psi4/libmints/wavefunction.h"

#include <ambit/tensor.h>

namespace psi {
namespace forte {

class Reference // : public Wavefunction
{
  protected:
    /// Reference energy = FCI_energy + frozen_core_energy + restricted_docc +
    /// nuclear_replusion
    double Eref_;
    /// Frozen_core_energy

    /// Density cumulants
    ambit::Tensor L1a_;
    ambit::Tensor L1b_;
    ambit::Tensor L2aa_;
    ambit::Tensor L2ab_;
    ambit::Tensor L2bb_;
    ambit::Tensor L3aaa_;
    ambit::Tensor L3aab_;
    ambit::Tensor L3abb_;
    ambit::Tensor L3bbb_;

    /// The 2-RDMs
    ambit::Tensor g2aa_;
    ambit::Tensor g2ab_;
    ambit::Tensor g2bb_;
    /// The Spin-free 2-RDM
    ambit::Tensor SFg2_;

  public:
    /// Default constructor
    Reference();

    /// Destructor
    ~Reference();

    /// Obtain reference energy
    double get_Eref() { return Eref_; }

    /// Obtain density cumulants
    ambit::Tensor L1a() { return L1a_; }
    ambit::Tensor L1b() { return L1b_; }
    ambit::Tensor L2aa() { return L2aa_; }
    ambit::Tensor L2ab() { return L2ab_; }
    ambit::Tensor L2bb() { return L2bb_; }
    ambit::Tensor L3aaa() { return L3aaa_; }
    ambit::Tensor L3aab() { return L3aab_; }
    ambit::Tensor L3abb() { return L3abb_; }
    ambit::Tensor L3bbb() { return L3bbb_; }

    /// Obtain 2-RDMs
    ambit::Tensor g2aa() { return g2aa_; }
    ambit::Tensor g2ab() { return g2ab_; }
    ambit::Tensor g2bb() { return g2bb_; }
    ambit::Tensor SFg2() { return SFg2_; }

    /// Set functions
    void set_Eref(double value) { Eref_ = value; }
    void set_L1a(ambit::Tensor L1a) { L1a_ = L1a; }
    void set_L1b(ambit::Tensor L1b) { L1b_ = L1b; }
    void set_L2aa(ambit::Tensor L2aa) { L2aa_ = L2aa; }
    void set_L2ab(ambit::Tensor L2ab) { L2ab_ = L2ab; }
    void set_L2bb(ambit::Tensor L2bb) { L2bb_ = L2bb; }
    void set_L3aaa(ambit::Tensor L3aaa) { L3aaa_ = L3aaa; }
    void set_L3aab(ambit::Tensor L3aab) { L3aab_ = L3aab; }
    void set_L3abb(ambit::Tensor L3abb) { L3abb_ = L3abb; }
    void set_L3bbb(ambit::Tensor L3bbb) { L3bbb_ = L3bbb; }

    /// Set the 2-RDMs
    void set_g2aa(ambit::Tensor g2aa) { g2aa_ = g2aa; }
    void set_g2ab(ambit::Tensor g2ab) { g2ab_ = g2ab; }
    void set_g2bb(ambit::Tensor g2bb) { g2bb_ = g2bb; }
    /// Spin-free 2-RDM
    void set_SFg2(ambit::Tensor SFg2) { SFg2_ = SFg2; }
};
}
} // End Namespaces

#endif // _reference_h_
