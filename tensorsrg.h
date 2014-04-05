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

    /// The scalar component of the operator S
    double S0;

    /// The one-body component of the operator S
    BlockedTensor S1;
    /// The two-body component of the operator S
    BlockedTensor S2;

    /// The one-body residual
    BlockedTensor R1;
    /// The two-body residual
    BlockedTensor R2;

    /// An intermediate one-body component of the similarity-transformed Hamiltonian
    BlockedTensor C1;
    /// An intermediate two-body component of the similarity-transformed Hamiltonian
    BlockedTensor C2;

    /// An intermediate one-body component of the similarity-transformed Hamiltonian
    BlockedTensor O1;
    /// An intermediate two-body component of the similarity-transformed Hamiltonian
    BlockedTensor O2;

    /// The one-body component of the similarity-transformed Hamiltonian
    BlockedTensor Hbar1;
    /// The two-body component of the similarity-transformed Hamiltonian
    BlockedTensor Hbar2;

    /// An intermediate tensor
    BlockedTensor I_ioiv;



//    /// The one-body component of the operator S
//    TensSpin1 S1;
//    /// The one-body component of the operator S
//    TensSpin2 S2;

//    /// An intermediate one-body component of the similarity-transformed Hamiltonian
//    TensSpin1 O1;
//    /// An intermediate two-body component of the similarity-transformed Hamiltonian
//    TensSpin2 O2;

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

    /// Compute the canonical transformation theory energy
    double compute_ct_energy();

    /// Update the S1 amplitudes
    void update_S1();
    /// Update the S2 amplitudes
    void update_S2();

    void commutator_A_B_C(double factor,
                          BlockedTensor& A1,BlockedTensor& A2,
                          BlockedTensor& B1,BlockedTensor& B2,
                          double& C0,BlockedTensor& C1,BlockedTensor& C2);
    void commutator_A_B_C_fourth_order(double factor,
                                       BlockedTensor& A1,BlockedTensor& A2,
                                       BlockedTensor& B1,BlockedTensor& B2,
                                       double& C0,BlockedTensor& C1,BlockedTensor& C2);
    /// The numbers indicate the rank of each operator
    void commutator_A1_B1_C0(BlockedTensor& A,BlockedTensor& B,double sign,double& C);
    void commutator_A1_B1_C1(BlockedTensor& A,BlockedTensor& B,double sign,BlockedTensor& C);
    void commutator_A1_B2_C0(BlockedTensor& A,BlockedTensor& B,double sign,double& C);
    void commutator_A1_B2_C1(BlockedTensor& A,BlockedTensor& B,double sign,BlockedTensor& C);
    void commutator_A1_B2_C2(BlockedTensor& A,BlockedTensor& B,double sign,BlockedTensor& C);
    void commutator_A2_B2_C0(BlockedTensor& A,BlockedTensor& B,double sign,double& C);
    void commutator_A2_B2_C1(BlockedTensor& A,BlockedTensor& B,double sign,BlockedTensor& C);
    void commutator_A2_B2_C2(BlockedTensor& A,BlockedTensor& B,double sign,BlockedTensor& C);

    void print_timings();

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
