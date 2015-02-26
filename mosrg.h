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

#ifndef _mosrg_h_
#define _mosrg_h_

#include <fstream>

#include "libdiis/diismanager.h"
#include "mobase.h"
#include "tensor_basic.h"
#include "tensor_labeled.h"
#include "tensor_product.h"

namespace psi{ namespace libadaptive{

/* The type of container used to hold the state vector  *
 * Used by boost::odeint                                */
typedef std::vector<double> odeint_state_type;

class MOSRG : public MOBase
{
    enum SRGCommutators {SRCommutators,MRNOCommutators,MRCommutators};
    enum SRGOperator  {SRGOpUnitary,SRGOpCC};
public:
    // Constructor and destructor
    MOSRG(Options &options, ExplorerIntegrals* ints, TwoIndex G1aa, TwoIndex G1bb);
    ~MOSRG();

    /// Updates the integrals with the CT or SRG matrix elements of the
    /// similiarity-transformed Hamiltonian
    ///
    /// This function should be called after running MOSRG
    void transfer_integrals();
private:
    /// The type of operator used
    SRGOperator srgop;
    /// The type of commutators used in the computations
    SRGCommutators srgcomm;
    /// The scalar component of the similarity-transformed Hamiltonian
    double Hbar0_;
    /// The one-body component of the similarity-transformed Hamiltonian
    MOTwoIndex Hbar1_;
    /// The two-body component of the similarity-transformed Hamiltonian
    MOFourIndex Hbar2_;
    /// The one-body component of the flow generator
    MOTwoIndex eta1_;
    /// The two-body component of the flow generator
    MOFourIndex eta2_;
    /// An intermediate one-body component of the similarity-transformed Hamiltonian
    MOTwoIndex O1_;
    /// An intermediate two-body component of the similarity-transformed Hamiltonian
    MOFourIndex O2_;
    /// An intermediate one-body component of the similarity-transformed Hamiltonian
    MOTwoIndex C1_;
    /// An intermediate two-body component of the similarity-transformed Hamiltonian
    MOFourIndex C2_;
    /// The scalar component of the operator S
    double S0_;
    /// The one-body component of the operator S
    MOTwoIndex S1_;
    /// The one-body component of the operator S
    MOFourIndex S2_;
    /// Use the Tensor class?
    bool use_tensor_class_;

    ::Tensor D_a;
    ::Tensor D_b;
    ::Tensor CD_a;
    ::Tensor CD_b;
    ::Tensor C_a;
    ::Tensor C_b;

    ::Tensor A4_aa;
    ::Tensor A4_ab;
    ::Tensor A4_bb;
    ::Tensor A4m_aa;
    ::Tensor A4m_ab;
    ::Tensor A4m_bb;
    ::Tensor B4_aa;
    ::Tensor B4_ab;
    ::Tensor B4_bb;
    ::Tensor B4m_aa;
    ::Tensor B4m_ab;
    ::Tensor B4m_bb;
    ::Tensor C4_aa;
    ::Tensor C4_ab;
    ::Tensor C4_bb;
    ::Tensor I4;

    void mosrg_startup();
    void mosrg_cleanup();

    /// The SRG routines
    void compute_similarity_renormalization_group();
    void compute_similarity_renormalization_group_step();

    void compute_canonical_transformation_energy();
    double compute_recursive_single_commutator();

    void compute_driven_srg_energy();
    /// The contributions to the one-body DSRG equations
    void one_body_driven_srg();
    /// The contributions to the two-body DSRG equations
    void two_body_driven_srg();

    void update_S1();
    void update_S2();
    void compute_mp2_guest_S2();


    /// Functions to compute commutators C += factor * [A,B]
    void commutator_A_B_C(double factor,
                          MOTwoIndex A1,MOFourIndex A2,
                          MOTwoIndex B1,MOFourIndex B2,
                          double& C0,MOTwoIndex C1,MOFourIndex C2);
    /// Functions to compute commutators C += factor * [A,B] but the term [A2,B2] -> C1
    /// contains a factor of two to recover the correct prefactor for the fourth-order term
    /// 1/2 [[V,T2],T2] -> R2
    void commutator_A_B_C_fourth_order(double factor,
                          MOTwoIndex A1,MOFourIndex A2,
                          MOTwoIndex B1,MOFourIndex B2,
                          double& C0,MOTwoIndex C1,MOFourIndex C2);
    /// Functions to compute commutators C += factor * [A,B] as done in the SRG(2) approximation
    void commutator_A_B_C_SRG2(double factor,
                          MOTwoIndex A1,MOFourIndex A2,
                          MOTwoIndex B1,MOFourIndex B2,
                          double& C0,MOTwoIndex C1,MOFourIndex C2);
    /// The numbers indicate the rank of each operator
    void commutator_A1_B1_C0(MOTwoIndex A,MOTwoIndex B,double sign,double& C);
    void commutator_A1_B1_C1(MOTwoIndex A,MOTwoIndex B,double sign,MOTwoIndex C);
    void commutator_A1_B2_C0(MOTwoIndex A,MOFourIndex B,double sign,double& C);
    void commutator_A1_B2_C1(MOTwoIndex A,MOFourIndex B,double sign,MOTwoIndex C);
    void commutator_A1_B2_C2(MOTwoIndex A,MOFourIndex B,double sign,MOFourIndex C);
    void commutator_A2_B2_C0(MOFourIndex A,MOFourIndex B,double sign,double& C);
    void commutator_A2_B2_C1(MOFourIndex A,MOFourIndex B,double sign,MOTwoIndex C);
    void commutator_A2_B2_C2(MOFourIndex A,MOFourIndex B,double sign,MOFourIndex C);
    void print_timings();

    friend class MOSRG_ODEInterface;
};

/// This class helps interface the SRG class to the boost ODE integrator
class MOSRG_ODEInterface {
    MOSRG& mosrg_obj_;
    int neval_;
public:
    MOSRG_ODEInterface(MOSRG& mosrg_obj) : mosrg_obj_(mosrg_obj), neval_(0) { }
    void operator() (const odeint_state_type& x,odeint_state_type& dxdt,const double t);
    int neval() {return neval_;}
};

}} // End Namespaces

#endif // _mosrg_h_
