/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER,
 * AUTHORS).
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
#include "fci/fci_integrals.h"
#include "stl_bitset_determinant.h"

using d1 = std::vector<double>;
using d2 = std::vector<d1>;
using d3 = std::vector<d2>;
using d4 = std::vector<d3>;
using d5 = std::vector<d4>;
using d6 = std::vector<d5>;
using vecdet = std::vector<psi::forte::STLBitsetDeterminant>;

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
    FCI_MO(SharedWavefunction ref_wfn, Options& options, std::shared_ptr<ForteIntegrals> ints,
           std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Destructor
    ~FCI_MO();

    /// Compute state-specific or state-averaged energy
    double compute_energy();

    /// Compute state-specific CASCI energy
    double compute_ss_energy();
    /// Compute state-averaged CASCI energy
    double compute_sa_energy();

    /// Returns the reference object
    Reference reference(const int& level = 3);

    /// Compute dipole moments with DSRG transformed MO dipole integrals
    /// This function is used for reference relaxation and SA-MRDSRG
    /// This function should be in RUN_DSRG
    std::map<std::string, std::vector<double>> compute_relaxed_dm(const std::vector<double>& dm0,
                                                                  std::vector<BlockedTensor>& dm1,
                                                                  std::vector<BlockedTensor>& dm2);

    /// Compute oscillator strengths using DSRG transformed MO dipole integrals
    /// This function is used for SA-MRDSRG
    /// This function should be in RUN_DSRG
    std::map<std::string, std::vector<double>> compute_relaxed_osc(std::vector<BlockedTensor>& dm1,
                                                                   std::vector<BlockedTensor>& dm2);

    /// Compute Fock (stored in ForteIntegal) using this->Da_
    void compute_Fock_ints();

    /**
     * @brief Rotate the SA references such that <M|F|N> is diagonal
     * @param irrep The irrep of states M and N (same irrep)
     */
    void xms_rotate(const int& irrep);

    /// Set fci_int_ pointer
    void set_fci_int(std::shared_ptr<FCIIntegrals> fci_ints) { fci_ints_ = fci_ints; }

    /// Set multiplicity
    void set_multiplicity(int multiplicity) { multi_ = multiplicity; }

    /// Set symmetry of the root
    void set_root_sym(int root_sym) { root_sym_ = root_sym; }

    /// Set number of roots
    void set_nroots(int nroot) { nroot_ = nroot; }

    /// Set which root is preferred
    void set_root(int root) { root_ = root; }

    //    /// Set active space type
    //    void set_active_space_type(string act) { active_space_type_ = act; }

    //    /// Set orbitals
    //    void set_orbs(SharedMatrix Ca, SharedMatrix Cb);

    /// Return fci_int_ pointer
    std::shared_ptr<FCIIntegrals> fci_ints() { return fci_ints_; }

    /// Return the vector of determinants
    vecdet p_space() { return determinant_; }

    /// Return P spaces for states with different symmetry
    std::vector<vecdet> p_spaces() { return p_spaces_; }

    /// Return the orbital extents of the current state
    std::vector<vector<vector<double>>> orb_extents() { return compute_orbital_extents(); }

    /// Return the vector of eigen vectors and eigen values
    std::vector<pair<SharedVector, double>> const eigen() { return eigen_; }

    /// Return the vector of eigen vectors and eigen values (used in
    /// state-average computation)
    std::vector<vector<pair<SharedVector, double>>> const eigens() { return eigens_; }

    /// Return a vector of dominant determinant for each root
    std::vector<STLBitsetDeterminant> dominant_dets() { return dominant_dets_; }

    /// Quiet mode (no printing, for use with CASSCF)
    void set_quite_mode(bool quiet) { quiet_ = quiet; }

    //    /// Set true to compute semi-canonical orbitals
    //    void set_semi(bool semi) { semi_ = semi; }

    //    /// Set false to skip Fock build in FCI_MO
    //    void set_form_Fock(bool form_fock) { form_Fock_ = form_fock; }

    /// Return indices (relative to active, not absolute) of active occupied orbitals
    std::vector<size_t> actv_occ() { return ah_; }

    /// Return indices (relative to active, not absolute) of active virtual orbitals
    std::vector<size_t> actv_uocc() { return ap_; }

    /// Return the Dimension of active occupied orbitals
    Dimension actv_docc() { return active_h_; }

    /// Return the Dimension of active virtual orbitals
    Dimension actv_virt() { return active_p_; }

    /// Return the T1 percentage in CISD computations
    std::vector<double> compute_T1_percentage();

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
    double fcheck_threshold_;

    /// Multiplicity
    int multi_;
    int twice_ms_;

    /// Symmetry
    int nirrep_;                  // number of irrep
    int root_sym_;                // root
    std::vector<int> sym_active_; // active MOs
    std::vector<int> sym_ncmo_;   // correlated MOs

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
    std::vector<size_t> idx_c_;
    Dimension active_; // active MOs
    size_t na_;
    std::vector<size_t> idx_a_;
    size_t nv_; // virtual MOs
    Dimension virtual_;
    std::vector<size_t> idx_v_;
    size_t nh_; // hole MOs
    std::vector<size_t> idx_h_;
    size_t npt_; // particle MOs
    std::vector<size_t> idx_p_;
    Dimension active_h_; // active hole for incomplete active space
    std::vector<size_t> ah_;
    Dimension active_p_; // active particle for incomplete active space
    std::vector<size_t> ap_;

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
    std::vector<STLBitsetDeterminant> dominant_dets_;
    std::vector<vecdet> p_spaces_;

    /// Size of Singles Determinants
    size_t singles_size_;

    /// Orbital Strings
    std::vector<vector<vector<bool>>> Form_String(const int& active_elec,
                                                  const bool& print = false);
    std::vector<bool> Form_String_Ref(const bool& print = false);
    std::vector<vector<vector<bool>>> Form_String_Singles(const std::vector<bool>& ref_string,
                                                          const bool& print = false);
    std::vector<vector<vector<bool>>> Form_String_Doubles(const std::vector<bool>& ref_string,
                                                          const bool& print = false);
    std::vector<vector<vector<bool>>> Form_String_IP(const std::vector<bool>& ref_string,
                                                     const bool& print = false);
    std::vector<vector<vector<bool>>> Form_String_EA(const std::vector<bool>& ref_string,
                                                     const bool& print = false);

    /// Choice of Roots
    int nroot_; // number of roots
    int root_;  // which root in nroot

    /// State Average Information (tuple of irrep, multi, nstates, weights)
    std::vector<std::tuple<int, int, int, std::vector<double>>> sa_info_;

    /// Eigen Values and Eigen Vectors of Certain Symmetry
    std::vector<pair<SharedVector, double>> eigen_;
    /// A List of Eigen Values and Vectors for State Average
    std::vector<vector<pair<SharedVector, double>>> eigens_;
    /// The algorithm for diagonalization
    std::string diag_algorithm_;

    /// Diagonalize the Hamiltonian
    void Diagonalize_H(const vecdet& P_space, const int& multi, const int& nroot,
                       std::vector<pair<SharedVector, double>>& eigen);
    /// Diagonalize the Hamiltonian without the HF determinant
    void Diagonalize_H_noHF(const vecdet& p_space, const int& multi, const int& nroot,
                            std::vector<pair<SharedVector, double>>& eigen);

    /// Print the CI Vectors and Configurations (figure out the dominant
    /// determinants)
    void print_CI(const int& nroot, const double& CI_threshold,
                  const std::vector<pair<SharedVector, double>>& eigen, const vecdet& det);

    /// Semi-canonicalize orbitals
    bool semi_;
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
    void fill_density(vector<double>& opdm_a, std::vector<double>& opdm_b);
    /// Fill in non-tensor quantities D1a_ and D1b_ using ambit tensors
    void fill_one_cumulant(ambit::Tensor& L1a, ambit::Tensor& L1b);

    /// Form 2-Particle Density Cumulant
    void FormCumulant2(CI_RDMS& ci_rdms, d4& AA, d4& AB, d4& BB);
    void FormCumulant2AA(const std::vector<double>& tpdm_aa, const std::vector<double>& tpdm_bb,
                         d4& AA, d4& BB);
    void FormCumulant2AB(const std::vector<double>& tpdm_ab, d4& AB);
    /// Fill in L2aa, L2ab and L2bb from L2aa_, L2ab_, and L2bb_
    void fill_cumulant2();
    /// Fill in L2aa, L2ab, L2bb from the 2RDMs (used in state average, L2aa_
    /// ... are not initialized)
    void compute_cumulant2(vector<double>& tpdm_aa, std::vector<double>& tpdm_ab,
                           std::vector<double>& tpdm_bb);
    /// Fill in non-tensor quantities L2aa_, L2ab_, and L2bb_ using ambit tensors
    void fill_two_cumulant(ambit::Tensor& L2aa, ambit::Tensor& L2ab, ambit::Tensor& L2bb);

    /// Form 3-Particle Density Cumulant
    void FormCumulant3(CI_RDMS& ci_rdms, d6& AAA, d6& AAB, d6& ABB, d6& BBB, string& DC);
    void FormCumulant3AAA(const std::vector<double>& tpdm_aaa, const std::vector<double>& tpdm_bbb,
                          d6& AAA, d6& BBB, string& DC);
    void FormCumulant3AAB(const std::vector<double>& tpdm_aab, const std::vector<double>& tpdm_abb,
                          d6& AAB, d6& ABB, string& DC);
    void FormCumulant3_DIAG(const vecdet& determinants, const int& root, d6& AAA, d6& AAB, d6& ABB,
                            d6& BBB);
    /// Fill in L3aaa, L3aab, L3abb, L3bbb from L3aaa_, L3aab_, L3abb_, L3bbb_
    void fill_cumulant3();
    /// Fill in L3aaa, L3aab, L3abb, L3bbb from the 3RDMs (used in state
    /// average, L3aaa_ ... are not initialized)
    void compute_cumulant3(vector<double>& tpdm_aaa, std::vector<double>& tpdm_aab,
                           std::vector<double>& tpdm_abb, std::vector<double>& tpdm_bbb);
    /// Fill in non-tensor quantities L3aaa_, L3aab_, L3abb_ and L3bbb_ using ambit tensors
    void fill_three_cumulant(ambit::Tensor& L3aaa, ambit::Tensor& L3aab, ambit::Tensor& L3abb,
                             ambit::Tensor& L3bbb);

    /// Fill in non-tensor cumulants used in the naive MR-DSRG-PT2 code
    void fill_naive_cumulants(Reference& ref, const int& level);

    /// N-Particle Operator
    double OneOP(const STLBitsetDeterminant& J, STLBitsetDeterminant& Jnew, const size_t& p,
                 const bool& sp, const size_t& q, const bool& sq);
    double TwoOP(const STLBitsetDeterminant& J, STLBitsetDeterminant& Jnew, const size_t& p,
                 const bool& sp, const size_t& q, const bool& sq, const size_t& r, const bool& sr,
                 const size_t& s, const bool& ss);
    double ThreeOP(const STLBitsetDeterminant& J, STLBitsetDeterminant& Jnew, const size_t& p,
                   const bool& sp, const size_t& q, const bool& sq, const size_t& r, const bool& sr,
                   const size_t& s, const bool& ss, const size_t& t, const bool& st,
                   const size_t& u, const bool& su);

    /// Fock Matrix
    d2 Fa_;
    d2 Fb_;
    //    bool form_Fock_ = true;
    void Form_Fock(d2& A, d2& B);
    void Check_Fock(const d2& A, const d2& B, const double& E, size_t& count);
    void Check_FockBlock(const d2& A, const d2& B, const double& E, size_t& count,
                         const size_t& dim, const std::vector<size_t>& idx, const string& str);
    void BD_Fock(const d2& Fa, const d2& Fb, SharedMatrix& Ua, SharedMatrix& Ub,
                 const string& name);
    /// Print Fock Matrix in Blocks
    void print_Fock(const string& spin, const d2& Fock);

    /// Reference Energy
    double Eref_;

    /// Compute 2- and 3-cumulants
    void compute_ref(const int& level);
    void compute_sa_ref(const int& level);

    /// Orbital Extents
    /// returns a vector of irrep by # active orbitals in current irrep
    /// by orbital extents {xx, yy, zz}
    d3 compute_orbital_extents();
    size_t idx_diffused_;
    std::vector<size_t> diffused_orbs_;

    /// Compute permanent dipole moments
    void compute_permanent_dipole();

    /// Transition dipoles
    std::map<std::string, std::vector<double>> trans_dipole_;
    /// Compute transition dipole of same symmetry
    void compute_transition_dipole();
    /// Compute oscillator strength of same symmetry
    void compute_oscillator_strength();

    /// Compute transition dipole when doing state averaging
    void compute_transition_dipole_sa();
    /// Compute oscillator strength when doing state averaging
    void compute_oscillator_strength_sa();

    /// Compute dipole (or transition dipole) using DSRG transformed MO dipole integrals (dm)
    /// and densities (or transition densities, D)
    double relaxed_dm_helper(const double& dm0, BlockedTensor& dm1, BlockedTensor& dm2,
                             BlockedTensor& D1, BlockedTensor& D2);

    /// Compute RDMs at given order and put into BlockedTensor format
    ambit::BlockedTensor compute_n_rdm(CI_RDMS& cirdm, const int& order);

    /**
     * @brief Return a vector of corresponding indices before the vector is
     * sorted
     * @typename T The data type of the sorted vector
     * @param v The sorted vector
     * @param decending Sort the vector v in decending order?
     * @return The vector of indices before sorting v
     */
    template <typename T>
    std::vector<size_t> sort_indexes(const std::vector<T>& v, const bool& decend = false) {

        // initialize original index locations
        std::vector<size_t> idx(v.size());
        std::iota(idx.begin(), idx.end(), 0);

        // sort indexes based on comparing values in v
        if (decend) {
            sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] > v[i2]; });
        } else {
            sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
        }

        return idx;
    }

    /// Check Sign (inline functons)
    double CheckSign(const std::vector<bool>& I, const int& n) {
        timer_on("Check Sign");
        size_t count = 0;
        for (vector<bool>::const_iterator iter = I.begin(); iter != I.begin() + n; ++iter) {
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
    void print_idx(const string& str, const std::vector<size_t>& vec) {
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
    double P3DDD(const d2& Density, const size_t& p, const size_t& q, const size_t& r,
                 const size_t& s, const size_t& t, const size_t& u) {
        double E = 0.0;
        int index[] = {0, 1, 2};
        size_t cop[] = {p, q, r};
        int count1 = 1;
        do {
            int count2 = count1 / 2;
            E += pow(-1.0, count2) * Density[cop[index[0]]][s] * Density[cop[index[1]]][t] *
                 Density[cop[index[2]]][u];
            ++count1;
        } while (std::next_permutation(index, index + 3));
        return E;
    }
    double P3DC(const d2& Density, const d4& Cumulant, const size_t& p, const size_t& q,
                const size_t& r, const size_t& s, const size_t& t, const size_t& u) {
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
                size_t Didx1 = idx_a_[cop[idc[0]]]; // first index (creation) of denisty
                size_t Didx2 = idx_a_[aop[ida[0]]]; // second index
                                                    // (annihilation) of density
                double value = Density[Didx1][Didx2];
                value *= Cumulant[cop[idc[1]]][cop[idc[2]]][aop[ida[1]]][aop[ida[2]]];
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
