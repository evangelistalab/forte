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

#include "mobase.h"

namespace psi{ namespace libadaptive{

enum SRGCommutators {SRCommutators,MRNOCommutators,MRCommutators};
enum SRGOperator  {SRGOpUnitary,SRGOpCC};

class MOSRG : public MOBase
{
public:
    // Constructor and destructor
    MOSRG(Options &options, ExplorerIntegrals* ints, TwoIndex G1aa, TwoIndex G1bb);
    ~MOSRG();
private:
    /// The type of operator used
    SRGOperator srgop;
    /// The type of commutators used in the computations
    SRGCommutators srgcomm;
    /// The one-body component of the similarity-transformed Hamiltonian
    MOTwoIndex Hbar1_;
    /// The two-body component of the similarity-transformed Hamiltonian
    MOFourIndex Hbar2_;
    /// An intermediate one-body component of the similarity-transformed Hamiltonian
    MOTwoIndex O1_;
    /// An intermediate two-body component of the similarity-transformed Hamiltonian
    MOFourIndex O2_;
    /// An intermediate one-body component of the similarity-transformed Hamiltonian
    MOTwoIndex C1_;
    /// An intermediate two-body component of the similarity-transformed Hamiltonian
    MOFourIndex C2_;
    /// The one-body component of the operator S
    MOTwoIndex S1_;
    /// The one-body component of the operator S
    MOFourIndex S2_;

    void mosrg_startup(Options &options);
    void mosrg_cleanup();

    void compute_canonical_transformation_energy(Options &options);
    double compute_recursive_single_commutator();

    void update_S1();
    void update_S2();

    /// Functions to compute commutators C = [A,B]
    /// The numbers indicate the rank of each operator
    void commutator_A1_B1_C0(MOTwoIndex restrict A,MOTwoIndex restrict B,double sign,double& C);
    void commutator_A1_B1_C1(MOTwoIndex restrict A,MOTwoIndex restrict B,double sign,MOTwoIndex C);
    void commutator_A1_B2_C0(MOTwoIndex restrict A,MOFourIndex restrict B,double sign,double& C);
    void commutator_A1_B2_C1(MOTwoIndex restrict A,MOFourIndex restrict B,double sign,MOTwoIndex C);
    void commutator_A1_B2_C2(MOTwoIndex restrict A,MOFourIndex restrict B,double sign,MOFourIndex C);
    void commutator_A2_B2_C0(MOFourIndex restrict A,MOFourIndex restrict B,double sign,double& C);
    void commutator_A2_B2_C1(MOFourIndex restrict A,MOFourIndex restrict B,double sign,MOTwoIndex C);
    void commutator_A2_B2_C2(MOFourIndex restrict A,MOFourIndex restrict B,double sign,MOFourIndex C);
};


}} // End Namespaces

#endif // _mosrg_h_
