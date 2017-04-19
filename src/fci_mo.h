/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see LICENSE, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#ifndef _fci_mo_h_
#define _fci_mo_h_

#include <string>
#include <tuple>
#include <vector>

#include "ambit/tensor.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/mintshelper.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/sointegral_onebody.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libpsio/psio.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libqt/qt.h"
#include "psi4/physconst.h"

#include "active_dsrgpt2.h"
#include "ci_rdms.h"
#include "helpers.h"
#include "integrals/integrals.h"
#include "reference.h"
#include "sparse_ci_solver.h"
#include "stl_bitset_determinant.h"

using namespace std;

using d1 = vector<double>;
using d2 = vector<d1>;
using d3 = vector<d2>;
using d4 = vector<d3>;
using d5 = vector<d4>;
using d6 = vector<d5>;
using vecdet = vector<psi::forte::STLBitsetDeterminant>;

namespace psi {
namespace forte {

/// Set the FCI_MO options
void set_FCI_MO_options(ForteOptions& foptions);

class FCI_MO : public Wavefunction {
    friend class ACTIVE_DSRGPT2;

  public:
    /**
     * @brief FCI_MO Constructor
     * @param ref_wfn The reference wavefunction object
     * @param options PSI4 and FORTE options
     * @param ints ForteInegrals
     * @param mo_space_info MOSpaceInfo
     */
    FCI_MO(SharedWavefunction ref_wfn, Options& options,
           std::shared_ptr<ForteIntegrals> ints,
           std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Destructor
    ~FCI_MO();

    /// Compute CASCI energy
    double compute_energy();
    /// Compute semi-canonical CASCI energy
    double compute_canonical_energy();

    /// Returns the reference object
    Reference reference(const int& level = 3);

    /// Compute state-averaged CASCI energy
    double compute_sa_energy();
    /// Compute semi-canonical state-averaged CASCI energy
    double compute_canonical_sa_energy();
    /**
     * @brief Rotate the SA references such that <M|F|N> is diagonal
     * @param irrep The irrep of states M and N (same irrep)
     */
    void xms_rotate(const int& irrep);

    /// Set multiplicity
    void set_multiplicity(int multiplicity) { multi_ = multiplicity; }

    /// Set symmetry of the root
    void set_root_sym(int root_sym) { root_sym_ = root_sym; }

    /// Set number of roots
    void set_nroots(int nroot) { nroot_ = nroot; }

    /// Set which root is preferred
    void set_root(int root) { root_ = root; }

    /// Set active space type
    void set_active_space_type(string act) { active_space_type_ = act; }

    /// Set orbitals
    void set_orbs(SharedMatrix Ca, SharedMatrix Cb);

    /// Return the vector of determinants
    vecdet p_space() { return determinant_; }

    /// Return P spaces for states with different symmetry
    vector<vecdet> p_spaces() { return FCI_MO::p_spaces_; }

    /// Return the orbital extents of the current state
    vector<vector<vector<double>>> orb_extents() {
        return compute_orbital_extents();
    }

    /// Return the vector of eigen vectors and eigen values
    vector<pair<SharedVector, double>> eigen() { return eigen_; }

    /// Return the vector of eigen vectors and eigen values (used in
    /// state-average computation)
    vector<vector<pair<SharedVector, double>>> eigens() { return eigens_; }

    /// Return a vector of dominant determinant for each root
    vector<STLBitsetDeterminant> dominant_dets() { return dominant_dets_; }

    /// Quiet mode (no printing, for use with CASSCF)
    void set_quite_mode(bool quiet) { quiet_ = quiet; }

    /// Set false to skip Fock build in FCI_MO
    void set_form_Fock(bool form_fock) { form_Fock_ = form_fock; }

    /// Return indices (relative to active, not absolute) of active occupied
    /// orbitals
    vector<size_t> actv_occ() { return ah_; }

    /// Return indices (relative to active, not absolute) of active virtual
    /// orbitals
    vector<size_t> actv_uocc() { return ap_; }

    /// Return the T1 percentage in CISD computations
    vector<double> compute_T1_percentage();

  protected:
    /// Basic Preparation
    void startup();
    void read_options();
    void cleanup();

    /// Integrals
    std::shared_ptr<ForteIntegrals> integral_;
    std::string int_type_;
    std::shared_ptr<FCIIntegrals> fci_ints_;

    /// Reference Type
    std::string ref_type_;

    /// MO space info
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// Print Levels
    int print_;
    /// Quiet mode (Do not print anything in FCI)
    bool quiet_ = false;

    /// Nucear Repulsion Energy
    double e_nuc_;

    /// Convergence
    double econv_;
    double dconv_;

    /// Multiplicity
    int multi_;
    int ms_;

    /// Symmetry
    int nirrep_;             // number of irrep
    int root_sym_;           // root
    vector<int> sym_active_; // active MOs
    vector<int> sym_ncmo_;   // correlated MOs

    /// Molecular Orbitals
    size_t nmo_; // total MOs
    Dimension nmopi_;
    size_t ncmo_; // correlated MOs
    Dimension ncmopi_;
    Dimension frzcpi_; // frozen core
    Dimension frzvpi_; // frozen virtual
    size_t nfrzc_;
    size_t nfrzv_;
    Dimension core_; // core MOs
    size_t nc_;
    vector<size_t> idx_c_;
    Dimension active_; // active MOs
    size_t na_;
    vector<size_t> idx_a_;
    size_t nv_; // virtual MOs
    Dimension virtual_;
    vector<size_t> idx_v_;
    size_t nh_; // hole MOs
    vector<size_t> idx_h_;
    size_t npt_; // particle MOs
    vector<size_t> idx_p_;
    Dimension active_h_; // active hole for incomplete active space
    vector<size_t> ah_;
    Dimension active_p_; // active particle for incomplete active space
    vector<size_t> ap_;

    /// Compute IP or EA
    std::string ipea_;

    /// Number of Alpha and Beta Electrons
    long int nalfa_;
    long int nbeta_;

    /// Active Space Type: CAS, CIS, CISD
    string active_space_type_;

    /// Form Determiants Space
    void form_p_space();

    /// Determinants
    void form_det();
    void form_det_cis();
    void form_det_cisd();
    vecdet determinant_;
    vector<STLBitsetDeterminant> dominant_dets_;
    vector<vecdet> p_spaces_;

    /// Size of Singles Determinants
    size_t singles_size_;

    /// Orbital Strings
    vector<vector<vector<bool>>> Form_String(const int& active_elec,
                                             const bool& print = false);
    vector<bool> Form_String_Ref(const bool& print = false);
    vector<vector<vector<bool>>>
    Form_String_Singles(const vector<bool>& ref_string,
                        const bool& print = false);
    vector<vector<vector<bool>>>
    Form_String_Doubles(const vector<bool>& ref_string,
                        const bool& print = false);
    vector<vector<vector<bool>>> Form_String_IP(const vector<bool>& ref_string,
                                                const bool& print = false);
    vector<vector<vector<bool>>> Form_String_EA(const vector<bool>& ref_string,
                                                const bool& print = false);

    /// Choice of Roots
    int nroot_; // number of roots
    int root_;  // which root in nroot

    /// State Average Information (tuple of irrep, multi, nstates, weights)
    std::vector<std::tuple<int, int, int, std::vector<double>>> sa_info_;

    /// Eigen Values and Eigen Vectors of Certain Symmetry
    vector<pair<SharedVector, double>> eigen_;
    /// A List of Eigen Values and Vectors for State Average
    vector<vector<pair<SharedVector, double>>> eigens_;
    /// The algorithm for diagonalization
    std::string diag_algorithm_;

    /// Diagonalize the Hamiltonian
    void Diagonalize_H(const vecdet& P_space, const int& multi, const int& nroot,
                       vector<pair<SharedVector, double>>& eigen);
    /// Diagonalize the Hamiltonian without the HF determinant
    void Diagonalize_H_noHF(const vecdet& p_space, const int& multi,
                            const int& nroot,
                            vector<pair<SharedVector, double>>& eigen);

    /// Print the CI Vectors and Configurations (figure out the dominant
    /// determinants)
    void print_CI(const int& nroot, const double& CI_threshold,
                  const vector<pair<SharedVector, double>>& eigen,
                  const vecdet& det);

    /// Semi-canonicalize orbitals
    void semi_canonicalize();
    /// Use natural orbitals
    void nat_orbs();

    /// Density Matrix
    d2 Da_;
    d2 Db_;
    ambit::Tensor L1a; // only in active
    ambit::Tensor L1b; // only in active

    /// 2-Body Density Cumulant
    d4 L2aa_;
    d4 L2ab_;
    d4 L2bb_;
    ambit::Tensor L2aa;
    ambit::Tensor L2ab;
    ambit::Tensor L2bb;

    /// 3-Body Density Cumulant
    d6 L3aaa_;
    d6 L3aab_;
    d6 L3abb_;
    d6 L3bbb_;
    ambit::Tensor L3aaa;
    ambit::Tensor L3aab;
    ambit::Tensor L3abb;
    ambit::Tensor L3bbb;

    /// Print Functions
    void print_d2(const string& str, const d2& OnePD);
    void print2PDC(const string& str, const d4& TwoPDC, const int& PRINT);
    void print3PDC(const string& str, const d6& ThreePDC, const int& PRINT);

    /// Print Density Matrix (Active ONLY)
    void print_density(const string& spin, const d2& density);
    /// Form Density Matrix
    void FormDensity(CI_RDMS& ci_rdms, d2& A, d2& B);
    /// Check Density Matrix
    bool CheckDensity();
    /// Fill in L1a, L1b from Da_, Db_
    void fill_density();
    /// Fill in L1a, L1b, Da_, Db_ from the RDM Vectors
    void fill_density(vector<double>& opdm_a, vector<double>& opdm_b);

    /// Form 2-Particle Density Cumulant
    void FormCumulant2(CI_RDMS& ci_rdms, d4& AA, d4& AB, d4& BB);
    void FormCumulant2AA(const vector<double>& tpdm_aa,
                         const vector<double>& tpdm_bb, d4& AA, d4& BB);
    void FormCumulant2AB(const vector<double>& tpdm_ab, d4& AB);
    /// Fill in L2aa, L2ab and L2bb from L2aa_, L2ab_, and L2bb_
    void fill_cumulant2();
    /// Fill in L2aa, L2ab, L2bb from the 2RDMs (used in state average, L2aa_
    /// ... are not initialized)
    void compute_cumulant2(vector<double>& tpdm_aa, vector<double>& tpdm_ab,
                           vector<double>& tpdm_bb);

    /// Form 3-Particle Density Cumulant
    void FormCumulant3(CI_RDMS& ci_rdms, d6& AAA, d6& AAB, d6& ABB, d6& BBB,
                       string& DC);
    void FormCumulant3AAA(const vector<double>& tpdm_aaa,
                          const vector<double>& tpdm_bbb, d6& AAA, d6& BBB,
                          string& DC);
    void FormCumulant3AAB(const vector<double>& tpdm_aab,
                          const vector<double>& tpdm_abb, d6& AAB, d6& ABB,
                          string& DC);
    void FormCumulant3_DIAG(const vecdet& determinants, const int& root,
                            d6& AAA, d6& AAB, d6& ABB, d6& BBB);
    /// Fill in L3aaa, L3aab, L3abb, L3bbb from L3aaa_, L3aab_, L3abb_, L3bbb_
    void fill_cumulant3();
    /// Fill in L3aaa, L3aab, L3abb, L3bbb from the 3RDMs (used in state
    /// average, L3aaa_ ... are not initialized)
    void compute_cumulant3(vector<double>& tpdm_aaa, vector<double>& tpdm_aab,
                           vector<double>& tpdm_abb, vector<double>& tpdm_bbb);

    /// N-Particle Operator
    double OneOP(const STLBitsetDeterminant& J, STLBitsetDeterminant& Jnew,
                 const size_t& p, const bool& sp, const size_t& q,
                 const bool& sq);
    double TwoOP(const STLBitsetDeterminant& J, STLBitsetDeterminant& Jnew,
                 const size_t& p, const bool& sp, const size_t& q,
                 const bool& sq, const size_t& r, const bool& sr,
                 const size_t& s, const bool& ss);
    double ThreeOP(const STLBitsetDeterminant& J, STLBitsetDeterminant& Jnew,
                   const size_t& p, const bool& sp, const size_t& q,
                   const bool& sq, const size_t& r, const bool& sr,
                   const size_t& s, const bool& ss, const size_t& t,
                   const bool& st, const size_t& u, const bool& su);

    /// Fock Matrix
    d2 Fa_;
    d2 Fb_;
    bool form_Fock_ = true;
    void Form_Fock(d2& A, d2& B);
    void Check_Fock(const d2& A, const d2& B, const double& E, size_t& count);
    void Check_FockBlock(const d2& A, const d2& B, const double& E,
                         size_t& count, const size_t& dim,
                         const vector<size_t>& idx, const string& str);
    void BD_Fock(const d2& Fa, const d2& Fb, SharedMatrix& Ua, SharedMatrix& Ub,
                 const string& name);
    /// Print Fock Matrix in Blocks
    void print_Fock(const string& spin, const d2& Fock);

    /// Reference Energy
    double Eref_;

    /// Compute 2- and 3-cumulants
    void compute_ref();
    void compute_sa_ref();

    /// Orbital Extents
    /// returns a vector of irrep by # active orbitals in current irrep
    /// by orbital extents {xx, yy, zz}
    d3 compute_orbital_extents();
    size_t idx_diffused_;
    vector<size_t> diffused_orbs_;

    /// Compute permanent dipole moments
    void compute_permanent_dipole();
    /// Transition dipoles
    map<string, vector<double>> trans_dipole_;
    /// Compute transition dipole
    void compute_trans_dipole();

    /// Compute oscillator strength
    void compute_oscillator_strength();

    /**
     * @brief Return a vector of corresponding indices before the vector is
     * sorted
     * @typename T The data type of the sorted vector
     * @param v The sorted vector
     * @param decending Sort the vector v in decending order?
     * @return The vector of indices before sorting v
     */
    template <typename T>
    vector<size_t> sort_indexes(const vector<T>& v,
                                const bool& decend = false) {

        // initialize original index locations
        vector<size_t> idx(v.size());
        std::iota(idx.begin(), idx.end(), 0);

        // sort indexes based on comparing values in v
        if (decend) {
            sort(idx.begin(), idx.end(),
                 [&v](size_t i1, size_t i2) { return v[i1] > v[i2]; });
        } else {
            sort(idx.begin(), idx.end(),
                 [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
        }

        return idx;
    }

    /// Check Sign (inline functons)
    double CheckSign(const vector<bool>& I, const int& n) {
        timer_on("Check Sign");
        size_t count = 0;
        for (vector<bool>::const_iterator iter = I.begin();
             iter != I.begin() + n; ++iter) {
            if (*iter)
                ++count;
        }
        timer_off("Check Sign");
        return pow(-1.0, count % 2);
    }
    double CheckSign(bool* I, const int& n) {
        timer_on("Check Sign");
        size_t count = 0;
        for (int i = 0; i < n; ++i) {
            if (I[i])
                ++count;
        }
        timer_off("Check Sign");
        return pow(-1.0, count % 2);
    }

    /// Print Size of a Array with Irrep
    void print_irrep(const string& str, const Dimension& array) {
        outfile->Printf("\n    %-30s", str.c_str());
        outfile->Printf("[");
        for (int h = 0; h < nirrep_; ++h) {
            outfile->Printf(" %4d ", array[h]);
        }
        outfile->Printf("]");
    }

    /// Print Indices
    void print_idx(const string& str, const vector<size_t>& vec) {
        outfile->Printf("\n    %-30s", str.c_str());
        size_t c = 0;
        for (size_t x : vec) {
            outfile->Printf("%4zu ", x);
            ++c;
            if (c % 15 == 0)
                outfile->Printf("\n  %-32c", ' ');
        }
    }

    /// Print Determinants
    void print_det(const vecdet& dets) {
        outfile->Printf("\n\n  ==> Determinants |alpha|beta> <==\n");
        for (const STLBitsetDeterminant& x : dets) {
            outfile->Printf("  ");
            x.print();
        }
        outfile->Printf("\n");
    }

    /// Permutations for 3-PDC
    double P3DDD(const d2& Density, const size_t& p, const size_t& q,
                 const size_t& r, const size_t& s, const size_t& t,
                 const size_t& u) {
        double E = 0.0;
        int index[] = {0, 1, 2};
        size_t cop[] = {p, q, r};
        int count1 = 1;
        do {
            int count2 = count1 / 2;
            E += pow(-1.0, count2) * Density[cop[index[0]]][s] *
                 Density[cop[index[1]]][t] * Density[cop[index[2]]][u];
            ++count1;
        } while (std::next_permutation(index, index + 3));
        return E;
    }
    double P3DC(const d2& Density, const d4& Cumulant, const size_t& p,
                const size_t& q, const size_t& r, const size_t& s,
                const size_t& t, const size_t& u) {
        double E = 0.0;
        int idc[] = {0, 1, 2};    // creation index of cop[]
        int ida[] = {0, 1, 2};    // annihilation index of aop[]
        size_t cop[] = {p, q, r}; // abs. creation index
        size_t aop[] = {s, t, u}; // abs. annihilation index
        int a = 1;                // a and b decide the sign
        do {
            if (a % 2 == 0) {
                ++a;
                continue;
            }
            int count1 = a / 2;
            int b = 1;
            do {
                if (b % 2 == 0) {
                    ++b;
                    continue;
                }
                int count2 = b / 2;
                size_t Didx1 =
                    idx_a_[cop[idc[0]]]; // first index (creation) of denisty
                size_t Didx2 = idx_a_[aop[ida[0]]]; // second index
                                                    // (annihilation) of density
                double value = Density[Didx1][Didx2];
                value *= Cumulant[cop[idc[1]]][cop[idc[2]]][aop[ida[1]]]
                                 [aop[ida[2]]];
                E += pow(-1.0, (count1 + count2)) * value;
                ++b;
            } while (std::next_permutation(ida, ida + 3));
            ++a;
        } while (std::next_permutation(idc, idc + 3));
        return E;
    }
};
}
}

#endif // _fci_mo_h_
