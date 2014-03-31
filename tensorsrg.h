/*
 *@BEGIN LICENSE
 *
 * Libadaptive: an ab initio quantum chemistry software package
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 *@END LICENSE
 */

#ifndef _tensorsrg_h_
#define _tensorsrg_h_

#include <fstream>

#include "methodbase.h"

namespace psi{ namespace libadaptive{

/**
 * @brief The TensorSRG class
 * This class implements Canonical Transformation (CT) theory
 * and the Similarity Renormalization Group (SRG) method using
 * the Tensor classes
 */
class TensorSRG : public MethodBase
{
protected:
    // => Tensors <= //

    /// The scalar component of the similarity-transformed Hamiltonian
    double Hbar0;

    /// The one-body component of the similarity-transformed Hamiltonian
    BlockedTensor Hbar1a;
    BlockedTensor Hbar1b;

    /// The two-body component of the similarity-transformed Hamiltonian
    BlockedTensor Hbar2aa;
    BlockedTensor Hbar2ab;
    BlockedTensor Hbar2bb;

    /// The one-body component of the flow generator
    BlockedTensor eta1a;
    BlockedTensor eta1b;

    /// The two-body component of the flow generator
    BlockedTensor eta2aa;
    BlockedTensor eta2ab;
    BlockedTensor eta2bb;

    /// An intermediate one-body component of the similarity-transformed Hamiltonian
    BlockedTensor O1a;
    BlockedTensor O1b;

    /// An intermediate two-body component of the similarity-transformed Hamiltonian
    BlockedTensor O2aa;
    BlockedTensor O2ab;
    BlockedTensor O2bb;

    /// An intermediate one-body component of the similarity-transformed Hamiltonian
    BlockedTensor C1a;
    BlockedTensor C1b;

    /// An intermediate two-body component of the similarity-transformed Hamiltonian
    BlockedTensor C2aa;
    BlockedTensor C2ab;
    BlockedTensor C2bb;

    /// The scalar component of the operator S
    double S0;

    /// The one-body component of the operator S
    BlockedTensor S1a;
    BlockedTensor S1b;

    /// The one-body component of the operator S
    BlockedTensor S2aa;
    BlockedTensor S2ab;
    BlockedTensor S2bb;

    // => Private member functions <= //

    /// Called in the constructor
    void startup();

    /// Called in the destructor
    void cleanup();

    /// Compute the MP2 energy and amplitudes
    double compute_mp2_guess();

    /// Compute the similarity transformed Hamiltonian using the
    /// single commutator recursive approximation
    double compute_hbar();

public:

    // => Constructors <= //

    /// Class constructor
    TensorSRG(boost::shared_ptr<Wavefunction> wfn, Options& options, ExplorerIntegrals* ints);

    /// Class destructor
    ~TensorSRG();

    /// Compute the SRG or CT energy
    double compute_energy();
};

}} // End Namespaces

#endif // _tensorsrg_h_
