/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2019 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include <ambit/tensor.h>

namespace forte {

class ForteIntegrals;
class MOSpaceInfo;

class Reference {
  public:
    // ==> Class Interface <==

    /// 0-rdm constructor
    Reference();
    /// 1-rdm constructor
    Reference(ambit::Tensor g1a, ambit::Tensor g1b);
    /// 1- and 2- rdm constructor
    Reference(ambit::Tensor g1a, ambit::Tensor g1b, ambit::Tensor g2aa, ambit::Tensor g2ab,
              ambit::Tensor g2bb);
    /// 1-, 2- and 3- rdm constructor)
    Reference(ambit::Tensor g1a, ambit::Tensor g1b, ambit::Tensor g2aa, ambit::Tensor g2ab,
              ambit::Tensor g2bb, ambit::Tensor g3aaa, ambit::Tensor g3aab, ambit::Tensor g3abb,
              ambit::Tensor g3bbb);

    /// Obtain density cumulants
    ambit::Tensor L2aa() const { return L2aa_; }
    ambit::Tensor L2ab() const { return L2ab_; }
    ambit::Tensor L2bb() const { return L2bb_; }
    ambit::Tensor L3aaa() const { return L3aaa_; }
    ambit::Tensor L3aab() const { return L3aab_; }
    ambit::Tensor L3abb() const { return L3abb_; }
    ambit::Tensor L3bbb() const { return L3bbb_; }

    /// Obtain 2-RDMs
    ambit::Tensor g1a() const { return g1a_; }
    ambit::Tensor g1b() const { return g1b_; }
    ambit::Tensor g2aa() const { return g2aa_; }
    ambit::Tensor g2ab() const { return g2ab_; }
    ambit::Tensor g2bb() const { return g2bb_; }
    ambit::Tensor SFg2() const { return SFg2_; }

    /// Obtain 3-RDMs
    ambit::Tensor g3aaa() const { return g3aaa_; }
    ambit::Tensor g3aab() const { return g3aab_; }
    ambit::Tensor g3abb() const { return g3abb_; }
    ambit::Tensor g3bbb() const { return g3bbb_; }


  protected:
    // ==> Class Data <==

    /// Maximum RDM used to initialize this
    size_t max_rdm_ = 0;

    /// Density cumulants
    ambit::Tensor L2aa_;
    ambit::Tensor L2ab_;
    ambit::Tensor L2bb_;
    ambit::Tensor L3aaa_;
    ambit::Tensor L3aab_;
    ambit::Tensor L3abb_;
    ambit::Tensor L3bbb_;

    /// Reduced density matrices
    ambit::Tensor g1a_;
    ambit::Tensor g1b_;
    ambit::Tensor g2aa_;
    ambit::Tensor g2ab_;
    ambit::Tensor g2bb_;
    ambit::Tensor SFg2_; // Spin-free 2-RDM
    ambit::Tensor g3aaa_;
    ambit::Tensor g3aab_;
    ambit::Tensor g3abb_;
    ambit::Tensor g3bbb_;
};

double compute_Eref_from_reference(const Reference& ref, std::shared_ptr<ForteIntegrals> ints,
                                   std::shared_ptr<MOSpaceInfo> mo_space_info, double Enuc);
} // namespace forte

#endif // _reference_h_
