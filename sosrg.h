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

#ifndef _sosrg_h_
#define _sosrg_h_

#include <fstream>

#include "sobase.h"

namespace psi{ namespace libadaptive{

enum SRGCommutators {SRCommutators,MRNOCommutators,MRCommutators};
enum SRGOperator  {SRGOpUnitary,SRGOpCC};

class SOSRG : public SOBase
{
public:
    // Constructor and destructor
    SOSRG(Options &options, ExplorerIntegrals* ints, TwoIndex G1);
    ~SOSRG();
private:
    /// The type of operator used
    SRGOperator srgop;
    /// The type of commutators used in the computations
    SRGCommutators srgcomm;
    /// The one-body component of the similarity-transformed Hamiltonian
    TwoIndex Hbar1_;
    /// The two-body component of the similarity-transformed Hamiltonian
    FourIndex Hbar2_;
    /// An intermediate one-body component of the similarity-transformed Hamiltonian
    TwoIndex O1_;
    /// An intermediate two-body component of the similarity-transformed Hamiltonian
    FourIndex O2_;
    /// An intermediate one-body component of the similarity-transformed Hamiltonian
    TwoIndex C1_;
    /// An intermediate two-body component of the similarity-transformed Hamiltonian
    FourIndex C2_;
    /// The one-body component of the operator S
    TwoIndex S1_;
    /// The one-body component of the operator S
    FourIndex S2_;

    void sosrg_startup(Options &options);
    void sosrg_cleanup();

    void compute_canonical_transformation_energy(Options &options);
    double compute_recursive_single_commutator();

    void update_S1();
    void update_S2();

    /// Functions to compute commutators C = [A,B]
    /// The numbers indicate the rank of each operator
    void commutator_A1_B1_C0(TwoIndex restrict A,TwoIndex restrict B,double sign,double& C);
    void commutator_A1_B1_C1(TwoIndex restrict A,TwoIndex restrict B,double sign,TwoIndex C);
    void commutator_A1_B2_C0(TwoIndex restrict A,FourIndex restrict B,double sign,double& C);
    void commutator_A1_B2_C1(TwoIndex restrict A,FourIndex restrict B,double sign,TwoIndex C);
    void commutator_A1_B2_C2(TwoIndex restrict A,FourIndex restrict B,double sign,FourIndex C);
    void commutator_A2_B2_C0(FourIndex restrict A,FourIndex restrict B,double sign,double& C);
    void commutator_A2_B2_C1(FourIndex restrict A,FourIndex restrict B,double sign,TwoIndex C);
    void commutator_A2_B2_C2(FourIndex restrict A,FourIndex restrict B,double sign,FourIndex C);
};


}} // End Namespaces

#endif // _sosrg_h_
