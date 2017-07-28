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

#ifndef _tensorsrg_h_
#define _tensorsrg_h_

#include <fstream>

#include "methodbase.h"
typedef ambit::BlockedTensor BlockedTensor;

namespace psi {
namespace forte {

/**
 * @brief The TensorSRG class
 * This class implements Canonical Transformation (CT) theory
 * and the Similarity Renormalization Group (SRG) method using
 * the Tensor classes
 */
class TensorSRG : public MethodBase {
    friend class TensorSRG_ODEInterface;

  protected:
    // => Tensors <= //

    /// The one-body component of the operator S
    ambit::BlockedTensor S1;
    /// The two-body component of the operator S
    ambit::BlockedTensor S2;

    /// The one-body component of the operator S
    ambit::BlockedTensor DS1;
    /// The two-body component of the operator S
    ambit::BlockedTensor DS2;

    /// The one-body residual
    ambit::BlockedTensor R1;
    /// The two-body residual
    ambit::BlockedTensor R2;

    double C0;
    /// An intermediate one-body component of the similarity-transformed
    /// Hamiltonian
    ambit::BlockedTensor C1;
    /// An intermediate two-body component of the similarity-transformed
    /// Hamiltonian
    ambit::BlockedTensor C2;

    /// An intermediate one-body component of the similarity-transformed
    /// Hamiltonian
    ambit::BlockedTensor O1;
    /// An intermediate two-body component of the similarity-transformed
    /// Hamiltonian
    ambit::BlockedTensor O2;

    /// The scalar component of the similarity-transformed Hamiltonian
    double Hbar0;
    /// The one-body component of the similarity-transformed Hamiltonian
    ambit::BlockedTensor Hbar1;
    /// The two-body component of the similarity-transformed Hamiltonian
    ambit::BlockedTensor Hbar2;

    /// The one-body denominator
    ambit::BlockedTensor D1;
    /// The two-body denominator
    ambit::BlockedTensor D2;

    /// The renormalized one-body denominator
    ambit::BlockedTensor RInvD1;
    /// The renormalized two-body denominator
    ambit::BlockedTensor RInvD2;

    ambit::BlockedTensor g1;
    ambit::BlockedTensor g2;
    ambit::BlockedTensor eta1;
    ambit::BlockedTensor eta2;

    /// An intermediate tensor
    ambit::BlockedTensor I_ioiv;

    double dsrg_power_;

    // => Private member functions <= //

    /// Called in the constructor
    void startup();

    /// Called in the destructor
    void cleanup();

    /// Compute the MP2 energy and amplitudes
    double compute_mp2_guess();

    /// Compute the SRG-PT2 energy and amplitudes
    double compute_mp2_guess_driven_srg();

    /// Compute the similarity transformed Hamiltonian using the
    /// single commutator recursive approximation
    double compute_hbar();

    /// Compute the canonical transformation theory energy
    double compute_ct_energy();

    /// Compute the similarity renormalization group energy
    double compute_srg_energy();

    /// Compute the derivative of the Hamiltonian
    void compute_srg_step();

    /// Update the S1 amplitudes
    void update_S1();
    /// Update the S2 amplitudes
    void update_S2();

    /// Update the S1 amplitudes for the DSRG
    void update_S1_dsrg();
    /// Update the S2 amplitudes for the DSRG
    void update_S2_dsrg();

    /// Update the S1 amplitudes for the DSRG
    void update_S1_dsrg_mp2();
    /// Update the S2 amplitudes for the DSRG
    void update_S2_dsrg_mp2();

    /// Save the similarity-transformed Hamiltonian to a file
    void save_hbar();

    /// Compute the commutator of a general two-body operator A with an
    /// excitation operator B
    /// B is assumed to have components B1 and B2 which span the "ov" and "oovv"
    /// spaces.
    void commutator_A_B_C(double factor, BlockedTensor& A1, BlockedTensor& A2, BlockedTensor& B1,
                          BlockedTensor& B2, double& C0, BlockedTensor& C1, BlockedTensor& C2,
                          int order);
    /// The commutator [A,B - B+]_1,2
    void commutator_A_B_C_SRC(double factor, BlockedTensor& A1, BlockedTensor& A2,
                              BlockedTensor& B1, BlockedTensor& B2, double& C0, BlockedTensor& C1,
                              BlockedTensor& C2);
    /// The commutator [A,B - B+]_1,2 with the [V,T2]_1 diagram weights modified
    void commutator_A_B_C_SRC_fourth_order(double factor, BlockedTensor& A1, BlockedTensor& A2,
                                           BlockedTensor& B1, BlockedTensor& B2, double& C0,
                                           BlockedTensor& C1, BlockedTensor& C2);
    /// The commutator [A,B - B+]_1,2 with the [F,T2]_2 diagram weights modified
    void commutator_A_B_C_SRC_fourth_order2(double factor, BlockedTensor& A1, BlockedTensor& A2,
                                            BlockedTensor& B1, BlockedTensor& B2, double& C0,
                                            BlockedTensor& C1, BlockedTensor& C2);
    /// The commutator [A,B - B+]_1,2 using the approximation of Tsukiyama
    void commutator_A_B_C_SRC_Tsukiyama(double factor, BlockedTensor& A1, BlockedTensor& A2,
                                        BlockedTensor& B1, BlockedTensor& B2, double& C0,
                                        BlockedTensor& C1, BlockedTensor& C2);

    /// The numbers indicate the rank of each operator
    void commutator_A1_B1_C0(BlockedTensor& A, BlockedTensor& B, double sign, double& C);
    void commutator_A1_B1_C1(BlockedTensor& A, BlockedTensor& B, double sign, BlockedTensor& C);
    void commutator_A1_B2_C0(BlockedTensor& A, BlockedTensor& B, double sign, double& C);
    void commutator_A1_B2_C1(BlockedTensor& A, BlockedTensor& B, double sign, BlockedTensor& C);
    void commutator_A1_B2_C2(BlockedTensor& A, BlockedTensor& B, double sign, BlockedTensor& C);
    void commutator_A2_B2_C0(BlockedTensor& A, BlockedTensor& B, double sign, double& C);
    void commutator_A2_B2_C1(BlockedTensor& A, BlockedTensor& B, double sign, BlockedTensor& C);
    void commutator_A2_B2_C2(BlockedTensor& A, BlockedTensor& B, double sign, BlockedTensor& C);
    void commutator_A2_B2_C1_simplified(BlockedTensor& A, BlockedTensor& B, double alpha,
                                        BlockedTensor& C);
    void commutator_A2_B2_C1_fo(BlockedTensor& A, BlockedTensor& B, double alpha, BlockedTensor& C);
    void commutator_A1_B2_C2_fo(BlockedTensor& A, BlockedTensor& B, double alpha, BlockedTensor& C);

    /// Compute the commutator of a general two-body operator A with an
    /// excitation operator B
    /// B is assumed to have components B1 and B2 which span the "ov" and "oovv"
    /// spaces.
    void full_commutator_A_B_C(double factor, BlockedTensor& A1, BlockedTensor& A2,
                               BlockedTensor& B1, BlockedTensor& B2, double& C0, BlockedTensor& C1,
                               BlockedTensor& C2);
    void full_commutator_A_B_C_SRC(double factor, BlockedTensor& A1, BlockedTensor& A2,
                                   BlockedTensor& B1, BlockedTensor& B2, double& C0,
                                   BlockedTensor& C1, BlockedTensor& C2);
    void full_commutator_A_B_C_SRC_fourth_order(double factor, BlockedTensor& A1, BlockedTensor& A2,
                                                BlockedTensor& B1, BlockedTensor& B2, double& C0,
                                                BlockedTensor& C1, BlockedTensor& C2);
    /// The numbers indicate the rank of each operator
    void full_commutator_A1_B1_C0(BlockedTensor& A, BlockedTensor& B, double sign, double& C);
    void full_commutator_A1_B1_C1(BlockedTensor& A, BlockedTensor& B, double sign,
                                  BlockedTensor& C);
    void full_commutator_A1_B2_C0(BlockedTensor& A, BlockedTensor& B, double sign, double& C);
    void full_commutator_A1_B2_C1(BlockedTensor& A, BlockedTensor& B, double sign,
                                  BlockedTensor& C);
    void full_commutator_A1_B2_C2(BlockedTensor& A, BlockedTensor& B, double sign,
                                  BlockedTensor& C);
    void full_commutator_A2_B2_C0(BlockedTensor& A, BlockedTensor& B, double sign, double& C);
    void full_commutator_A2_B2_C1(BlockedTensor& A, BlockedTensor& B, double sign,
                                  BlockedTensor& C);
    void full_commutator_A2_B2_C2(BlockedTensor& A, BlockedTensor& B, double sign,
                                  BlockedTensor& C);

    /// Compute the commutator of a general two-body operator A with an
    /// excitation operator B
    /// B is assumed to have components B1 and B2 which span the "ov" and "oovv"
    /// spaces.
    void hermitian_commutator_A_B_C(double factor, BlockedTensor& A1, BlockedTensor& A2,
                                    BlockedTensor& B1, BlockedTensor& B2, double& C0,
                                    BlockedTensor& C1, BlockedTensor& C2);
    void hermitian_commutator_A_B_C_SRC(double factor, BlockedTensor& A1, BlockedTensor& A2,
                                        BlockedTensor& B1, BlockedTensor& B2, double& C0,
                                        BlockedTensor& C1, BlockedTensor& C2);
    void hermitian_commutator_A_B_C_SRC_fourth_order(double factor, BlockedTensor& A1,
                                                     BlockedTensor& A2, BlockedTensor& B1,
                                                     BlockedTensor& B2, double& C0,
                                                     BlockedTensor& C1, BlockedTensor& C2);
    /// The numbers indicate the rank of each operator
    void hermitian_commutator_A1_B1_C0(BlockedTensor& A, BlockedTensor& B, double sign, double& C);
    void hermitian_commutator_A1_B1_C1(BlockedTensor& A, BlockedTensor& B, double sign,
                                       BlockedTensor& C);
    void hermitian_commutator_A1_B2_C0(BlockedTensor& A, BlockedTensor& B, double sign, double& C);
    void hermitian_commutator_A1_B2_C1(BlockedTensor& A, BlockedTensor& B, double sign,
                                       BlockedTensor& C);
    void hermitian_commutator_A1_B2_C2(BlockedTensor& A, BlockedTensor& B, double sign,
                                       BlockedTensor& C);
    void hermitian_commutator_A2_B2_C0(BlockedTensor& A, BlockedTensor& B, double sign, double& C);
    void hermitian_commutator_A2_B2_C1(BlockedTensor& A, BlockedTensor& B, double sign,
                                       BlockedTensor& C);
    void hermitian_commutator_A2_B2_C2(BlockedTensor& A, BlockedTensor& B, double sign,
                                       BlockedTensor& C);

    void modified_commutator_A_B_C(double factor, BlockedTensor& A1, BlockedTensor& A2,
                                   BlockedTensor& B1, BlockedTensor& B2, double& C0,
                                   BlockedTensor& C1, BlockedTensor& C2);
    void modified_commutator_A2_B2_C0(BlockedTensor& A, BlockedTensor& B, double sign, double& C);
    void modified_commutator_A2_B2_C2(BlockedTensor& A, BlockedTensor& B, double alpha,
                                      BlockedTensor& C);
    void print_timings();

  public:
    // => Constructors <= //

    /// Class constructor
    TensorSRG(std::shared_ptr<Wavefunction> wfn, Options& options,
              std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Class destructor
    ~TensorSRG();

    /// Compute the SRG or CT energy
    double compute_energy();

    /// Transfer the integrals
    void transfer_integrals();
};

/// The type of container used to hold the state vector used by boost::odeint
typedef std::vector<double> odeint_state_type;

/// This class helps interface the SRG class to the boost ODE integrator.
/// It stores a reference to the TensorSRG object and it is passed to the
/// ODE solver in boost so that it can compute the derivative of H and
/// update the value of the Hamiltonian.
class TensorSRG_ODEInterface {
  protected:
    // => Class data <= //

    TensorSRG& tensorsrg_obj_;
    int neval_;

  public:
    // => Constructors <= //

    TensorSRG_ODEInterface(TensorSRG& tensorsrg_obj) : tensorsrg_obj_(tensorsrg_obj), neval_(0) {}
    void operator()(const odeint_state_type& x, odeint_state_type& dxdt, const double t);
    int neval() { return neval_; }
};

double one_minus_exp_div_x(double s, double x, double power);
}
} // End Namespaces

#endif // _tensorsrg_h_
