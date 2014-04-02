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

#ifndef _mobase_h_
#define _mobase_h_

#include <fstream>

#include <liboptions/liboptions.h>

#include "integrals.h"

namespace psi{ namespace libadaptive{

typedef double* OneIndex;
typedef double** TwoIndex;
typedef double**** FourIndex;

struct MOOneIndex
{
    OneIndex a;
    OneIndex b;
};

struct MOTwoIndex
{
    TwoIndex aa;
    TwoIndex bb;
};

struct MOFourIndex
{
    FourIndex aaaa;
    FourIndex abab;
    FourIndex bbbb;
};

/**
 * @brief The MOBase class
 * This class provides basic functions to write electronic structure
 * pilot codes in a spin orbital formalism
 */
class MOBase
{
public:
    // Constructor and destructor
    MOBase(Options &options, ExplorerIntegrals* ints,TwoIndex G1aa,TwoIndex G1bb);
    ~MOBase();
protected:
    /// The print level
    int print_;
    /// A reference to the option object
    Options& options_;
    /// The number of irriducible representations
    int nirrep_;
    /// The wave function symmetry
    int wavefunction_symmetry_;
    /// The number of molecular orbitals
    size_t nmo_;
    /// The energy of the reference
    double E0_;

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
    ExplorerIntegrals* ints_;
    /// The reference occupation numbers
    MOOneIndex No_;
    /// The reference complementary occupation numbers (1 - No_)
    MOOneIndex Nv_;
    /// The reference one-particle density matrix
    MOTwoIndex G1_;
    /// The reference one-hole density matrix
    MOTwoIndex E1_;
    /// The reference two-particle density matrix
    MOFourIndex G2_;
    /// The reference two-particle cumulant matrix
    MOFourIndex L2_;
    /// The one-electron integrals Fock matrix
    MOTwoIndex H1_;
    /// The generalized Fock matrix
    MOTwoIndex F_;
    /// The two electron antisymmetrized integrals storeds as V[p][q][r][s] = <pq||rs>
    MOFourIndex V_;



    /// Starts the class
    void startup(TwoIndex G1aa,TwoIndex G1bb);
    /// Routines to allocate and release memory
    void allocate();
    void release();
    void allocate(MOTwoIndex&  two_index);
    void allocate(MOFourIndex& four_index);
    void release(MOTwoIndex&  two_index);
    void release(MOFourIndex& four_index);
    void add(double fA, MOTwoIndex& A, double fB, MOTwoIndex& B);
    void add(double fA, MOFourIndex& A, double fB, MOFourIndex& B);
    void zero(MOTwoIndex& A);
    void zero(MOFourIndex& A);
    double norm(MOTwoIndex& A);
    double norm(MOFourIndex& A);
    double norm(FourIndex& A);

    /// Build the generalized Fock matrix using the one-particle density matrix
    void build_fock();

    void sort_integrals();
};

#define loop_mo_p for(int p = 0; p < nmo_; ++p)
#define loop_mo_q for(int q = 0; q < nmo_; ++q)
#define loop_mo_r for(int r = 0; r < nmo_; ++r)
#define loop_mo_s for(int s = 0; s < nmo_; ++s)
#define loop_mo_t for(int t = 0; t < nmo_; ++t)
#define loop_mo_u for(int u = 0; u < nmo_; ++u)
#define loop_mo_v for(int v = 0; v < nmo_; ++v)
#define loop_mo_x for(int x = 0; x < nmo_; ++x)
#define loop_mo_y for(int y = 0; y < nmo_; ++y)
#define loop_mo_z for(int z = 0; z < nmo_; ++z)

}} // End Namespaces

#endif // _sobase_h_
