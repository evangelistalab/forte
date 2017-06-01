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

#ifndef _sobase_h_
#define _sobase_h_

#include <fstream>

#include <liboptions/liboptions.h>

#include "integrals.h"

namespace psi{ namespace forte{

typedef double* OneIndex;
typedef double** TwoIndex;
typedef double**** FourIndex;

/**
 * @brief The SOBase class
 * This class provides basic functions to write electronic structure
 * pilot codes in a spin orbital formalism
 */
class SOBase
{
public:
    // Constructor and destructor
    SOBase(Options &options, ForteIntegrals* ints, TwoIndex G1);
    ~SOBase();
protected:
    Options& options_;
    /// The number of irriducible representations
    int nirrep_;
    /// The wave function symmetry
    int wavefunction_symmetry_;
    /// The number of molecular orbitals
    int nmo_;
    /// The number of spin orbitals
    int nso_;
    /// The energy of the reference
    double E0_;

    std::vector<bool> so_spin;
    std::vector<int> so_mo;
    /// The number of alpha electrons
    int nalpha_;
    /// The number of beta electrons
    int nbeta_;
    /// The number of molecular orbitals per irrep
    Dimension nmopi_;
    /// The number of alpha electrons per irrep
    Dimension nalphapi_;
    /// The number of beta electrons per irrep
    Dimension nbetapi_;
    /// The nuclear repulsion energy
    double nuclear_repulsion_energy_;
    /// The molecular integrals required by SOBase
    ForteIntegrals* ints_;
    /// The reference occupation numbers
    OneIndex No_;
    /// The reference complementary occupation numbers (1 - No_)
    OneIndex Nv_;
    /// The reference one-particle density matrix
    TwoIndex G1_;
    /// The reference one-hole density matrix
    TwoIndex E1_;
    /// The reference two-particle density matrix
    FourIndex G2_;
    /// The reference two-particle cumulant matrix
    FourIndex L2_;
    /// The one-electron integrals Fock matrix
    TwoIndex H1_;
    /// The generalized Fock matrix
    TwoIndex F_;
    /// The two electron antisymmetrized integrals storeds as V[p][q][r][s] = <pq||rs>
    FourIndex V_;

    /// Starts the class
    void startup(TwoIndex G1);
    /// Routines to allocate and release memory
    void allocate();
    void release();
    void allocate(TwoIndex&  two_index);
    void allocate(FourIndex& four_index);
    void release(TwoIndex&  two_index);
    void release(FourIndex& four_index);
    void add(double fA, TwoIndex& A, double fB, TwoIndex& B);
    void add(double fA, FourIndex& A, double fB, FourIndex& B);
    void zero(TwoIndex& A);
    void zero(FourIndex& A);
    double norm(TwoIndex& A);
    double norm(FourIndex& A);

    /// Build the generalized Fock matrix using the one-particle density matrix
    void build_fock();

    void sort_integrals();
};

#define loop_p for(int p = 0; p < nso_; ++p)
#define loop_q for(int q = 0; q < nso_; ++q)
#define loop_r for(int r = 0; r < nso_; ++r)
#define loop_s for(int s = 0; s < nso_; ++s)
#define loop_t for(int t = 0; t < nso_; ++t)
#define loop_u for(int u = 0; u < nso_; ++u)
#define loop_v for(int v = 0; v < nso_; ++v)
#define loop_x for(int x = 0; x < nso_; ++x)
#define loop_y for(int y = 0; y < nso_; ++y)
#define loop_z for(int z = 0; z < nso_; ++z)

}} // End Namespaces

#endif // _sobase_h_
