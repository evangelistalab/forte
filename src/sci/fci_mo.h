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
#include <utility>
#include <unordered_set>

#include "ambit/tensor.h"

#include "ci_rdm/ci_rdms.h"
#include "helpers/mo_space_info.h"
#include "helpers/helpers.h"
#include "integrals/integrals.h"
#include "base_classes/reference.h"
#include "sparse_ci/sparse_ci_solver.h"
#include "fci/fci_integrals.h"
#include "sparse_ci/determinant.h"
#include "mrdsrg-spin-integrated/active_dsrgpt2.h"

using d1 = std::vector<double>;
using d2 = std::vector<d1>;
using d3 = std::vector<d2>;
using d4 = std::vector<d3>;
using d5 = std::vector<d4>;
using d6 = std::vector<d5>;
using vecdet = std::vector<psi::forte::Determinant>;

namespace psi {
namespace forte {

/// Set the FCI_MO options
void set_FCI_MO_options(ForteOptions& foptions);

class FCI_MO : public Wavefunction {

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

    /**
     * @brief FCI_MO Constructor
     * @param ref_wfn The reference wavefunction object
     * @param options PSI4 and FORTE options
     * @param ints ForteInegrals
     * @param mo_space_info MOSpaceInfo
     * @param fci_ints FCIInegrals
     */
    FCI_MO(SharedWavefunction ref_wfn, Options& options, std::shared_ptr<ForteIntegrals> ints,
           std::shared_ptr<MOSpaceInfo> mo_space_info, std::shared_ptr<FCIIntegrals> fci_ints);

    /// Destructor
    ~FCI_MO();

    /// Compute state-specific or state-averaged energy
    double compute_energy();

    /// Compute state-specific CASCI energy
    double compute_ss_energy();
    /// Compute state-averaged CASCI energy
    double compute_sa_energy();

    /// Return the reference object
    /// Return averaged cumulants if AVG_STATE is not empty
    Reference reference(const int& level = 3);

    /// Compute densities or transition densities
    /// root1, root2 -- the ket and bra roots of p_space and eigen
    /// multi_state -- grab p_spaces_ and eigens_ if true, otherwise p_space_ and eigen_
    /// entry -- symmetry entry of p_spaces_ and eigens_ (same entry as sa_info_)
    /// max_level -- max RDM level to be computed
    /// do_cumulant -- returned Reference is filled by cumulants (not RDMs) if true
    Reference transition_reference(int root1, int root2, bool multi_state, int entry = 0,
                                   int max_level = 3, bool do_cumulant = false, bool disk = true);

    /// Density files
    std::vector<std::string> density_filenames_generator(int rdm_level, int irrep, int multi,
                                                         int root1, int root2);
    bool check_density_files(int rdm_level, int irrep, int multi, int root1, int root2);
    void remove_density_files(int rdm_level, int irrep, int multi, int root1, int root2);

    /// Compute dipole moments with DSRG transformed MO dipole integrals
    /// This function is used for reference relaxation and SA-MRDSRG
    /// This function should be in RUN_DSRG
    std::map<std::string, std::vector<double>>
    compute_ref_relaxed_dm(const std::vector<double>& dm0, std::vector<ambit::BlockedTensor>& dm1,
                           std::vector<ambit::BlockedTensor>& dm2);
    std::map<std::string, std::vector<double>>
    compute_ref_relaxed_dm(const std::vector<double>& dm0, std::vector<ambit::BlockedTensor>& dm1,
                           std::vector<ambit::BlockedTensor>& dm2,
                           std::vector<ambit::BlockedTensor>& dm3);

    /// Compute oscillator strengths using DSRG transformed MO dipole integrals
    /// This function is used for SA-MRDSRG
    /// This function should be in RUN_DSRG
    std::map<std::string, std::vector<double>>
    compute_ref_relaxed_osc(std::vector<ambit::BlockedTensor>& dm1,
                            std::vector<ambit::BlockedTensor>& dm2);
    std::map<std::string, std::vector<double>>
    compute_ref_relaxed_osc(std::vector<ambit::BlockedTensor>& dm1,
                            std::vector<ambit::BlockedTensor>& dm2,
                            std::vector<ambit::BlockedTensor>& dm3);

    /// Compute Fock (stored in ForteIntegal) using this->Da_
    void compute_Fock_ints();

    /**
     * @brief Rotate the SA references such that <M|F|N> is diagonal
     * @param irrep The irrep of states M and N (same irrep)
     */
    void xms_rotate_civecs();

    /// Set if safe to read densities from files
    void set_safe_to_read_density_files(bool safe) { safe_to_read_density_files_ = safe; }

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

    /// Quiet mode (no printing, for use with CASSCF)
    void set_quite_mode(bool quiet) { quiet_ = quiet; }

    /// Set if localize orbitals
    void set_localize_actv(bool localize) { localize_actv_ = localize; }

    /// Set projected roots
    void project_roots(std::vector<std::vector<std::pair<size_t, double>>>& projected) {
        projected_roots_ = projected;
    }

    /// Set initial guess
    void set_initial_guess(std::vector<std::pair<size_t, double>>& guess) {
        initial_guess_ = guess;
    }

    /// Set SA infomation
    void set_sa_info(const std::vector<std::tuple<int, int, int, std::vector<double>>>& info);

    /// Set state-averaged eigen values and vectors
    void set_eigens(const std::vector<std::vector<std::pair<SharedVector, double>>>& eigens);

    /// Return fci_int_ pointer
    std::shared_ptr<FCIIntegrals> fci_ints() { return fci_ints_; }

    /// Return the vector of determinants
    vecdet p_space() { return determinant_; }

    /// Return P spaces for states with different symmetry
    std::vector<vecdet> p_spaces() { return p_spaces_; }

    /// Return the orbital extents of the current state
    std::vector<std::vector<std::vector<double>>> orb_extents() {
        return compute_orbital_extents();
    }

    /// Return the vector of eigen vectors and eigen values
    std::vector<std::pair<SharedVector, double>> const eigen() { return eigen_; }

    /// Return the vector of eigen vectors and eigen values (used in state-average computation)
    std::vector<std::vector<std::pair<SharedVector, double>>> const eigens() {
        return eigens_;
    }

    /// Return a vector of dominant determinant for each root
    std::vector<Determinant> dominant_dets() { return dominant_dets_; }

    /// Return indices (relative to active, not absolute) of active occupied orbitals
    std::vector<size_t> actv_occ() { return actv_hole_mos_; }

    /// Return indices (relative to active, not absolute) of active virtual orbitals
    std::vector<size_t> actv_uocc() { return actv_part_mos_; }

    /// Return the Dimension of active occupied orbitals
    Dimension actv_docc() { return actv_hole_dim_; }

    /// Return the Dimension of active virtual orbitals
    Dimension actv_virt() { return actv_part_dim_; }

    /// Return the T1 percentage in CISD computations
    std::vector<double> compute_T1_percentage();

    /// Return the parsed state-averaged info
    std::vector<std::tuple<int, int, int, std::vector<double>>> sa_info() { return sa_info_; }

  protected:
    /// Basic Preparation
    void startup();
    void read_options();
    void print_options();
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
    std::vector<std::string> multi_symbols_;

    /// Symmetry
    int nirrep_;                // number of irrep
    int root_sym_;              // root
    std::vector<int> sym_actv_; // active MOs
    std::vector<int> sym_ncmo_; // correlated MOs
    std::vector<std::string> irrep_symbols_;

    /// Molecular Orbitals
    size_t nmo_; // total MOs
    Dimension nmopi_;
    size_t ncmo_; // correlated MOs
    Dimension ncmopi_;
    Dimension frzc_dim_; // frozen core
    Dimension frzv_dim_; // frozen virtual
    size_t nfrzc_;
    size_t nfrzv_;
    Dimension core_dim_; // core MOs
    size_t ncore_;
    std::vector<size_t> core_mos_;
    Dimension actv_dim_; // active MOs
    size_t nactv_;
    std::vector<size_t> actv_mos_;
    size_t nvirt_; // virtual MOs
    Dimension virt_dim_;
    std::vector<size_t> virt_mos_;
    size_t nhole_; // hole MOs
    std::vector<size_t> hole_mos_;
    size_t npart_; // particle MOs
    std::vector<size_t> part_mos_;
    Dimension actv_hole_dim_; // active hole for incomplete active space
    std::vector<size_t> actv_hole_mos_;
    Dimension actv_part_dim_; // active particle for incomplete active space
    std::vector<size_t> actv_part_mos_;

    /// Compute IP or EA
    std::string ipea_;

    /// Number of Alpha and Beta Electrons
    long int nalfa_;
    long int nbeta_;

    /// Active Space Type: CAS, CIS, CISD
    std::string actv_space_type_;

    /// Form Determiants Space
    void form_p_space();

    /// Determinants
    void form_det();
    void form_det_cis();
    void form_det_cisd();
    vecdet determinant_;
    std::vector<Determinant> dominant_dets_;
    std::vector<vecdet> p_spaces_;

    /// Size of Singles Determinants
    size_t singles_size_;

    /// Orbital Strings
    std::vector<std::vector<std::vector<bool>>> Form_String(const int& active_elec,
                                                            const bool& print = false);
    std::vector<bool> Form_String_Ref(const bool& print = false);
    std::vector<std::vector<std::vector<bool>>>
    Form_String_Singles(const std::vector<bool>& ref_string, const bool& print = false);
    std::vector<std::vector<std::vector<bool>>>
    Form_String_Doubles(const std::vector<bool>& ref_string, const bool& print = false);
    std::vector<std::vector<std::vector<bool>>> Form_String_IP(const std::vector<bool>& ref_string,
                                                               const bool& print = false);
    std::vector<std::vector<std::vector<bool>>> Form_String_EA(const std::vector<bool>& ref_string,
                                                               const bool& print = false);

    /// Choice of Roots
    int nroot_; // number of roots
    int root_;  // which root in nroot

    /// State Average Information (tuple of irrep, multi, nstates, weights)
    std::vector<std::tuple<int, int, int, std::vector<double>>> sa_info_;

    /// Roots to be projected out in the diagonalization
    std::vector<std::vector<std::pair<size_t, double>>> projected_roots_;

    /// Initial guess vector
    std::vector<std::pair<size_t, double>> initial_guess_;

    /// Eigen Values and Eigen Vectors of Certain Symmetry
    std::vector<std::pair<SharedVector, double>> eigen_;
    /// A List of Eigen Values and Vectors for State Average
    std::vector<std::vector<std::pair<SharedVector, double>>> eigens_;
    /// The algorithm for diagonalization
    std::string diag_algorithm_;

    /// Diagonalize the Hamiltonian
    void Diagonalize_H(const vecdet& P_space, const int& multi, const int& nroot,
                       std::vector<std::pair<SharedVector, double>>& eigen);
    /// Diagonalize the Hamiltonian without the HF determinant
    void Diagonalize_H_noHF(const vecdet& p_space, const int& multi, const int& nroot,
                            std::vector<std::pair<SharedVector, double>>& eigen);

    /// Print the CI Vectors and Configurations (figure out the dominant determinants)
    void print_CI(const int& nroot, const double& CI_threshold,
                  const std::vector<std::pair<SharedVector, double>>& eigen, const vecdet& det);

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

    /// File Names of Densities Stored on Disk
    std::unordered_set<std::string> density_files_;
    bool safe_to_read_density_files_ = false;
//    std::vector<std::string> density_filenames_generator(int rdm_level, int irrep, int multi,
//                                                         int root1, int root2);
//    bool check_density_files(int rdm_level, int irrep, int multi, int root1, int root2);
//    void remove_density_files(int rdm_level, int irrep, int multi, int root1, int root2);
    void clean_all_density_files();

    std::vector<ambit::Tensor> compute_n_rdm(const vecdet& p_space, SharedMatrix evecs,
                                             int rdm_level, int root1, int root2, int irrep,
                                             int multi, bool disk);

    /// Print Functions
    void print2PDC(const std::string& str, const d4& TwoPDC, const int& PRINT);
    void print3PDC(const std::string& str, const d6& ThreePDC, const int& PRINT);

    /// Print Density Matrix (Active ONLY)
    void print_density(const std::string& spin, const d2& density);

    /// Fill in non-tensor cumulants used in the naive MR-DSRG-PT2 code
    void fill_naive_cumulants(Reference& ref, const int& level);
    /// Fill in non-tensor quantities D1a_ and D1b_ using ambit tensors
    void fill_one_cumulant(ambit::Tensor& L1a, ambit::Tensor& L1b);
    /// Fill in non-tensor quantities L2aa_, L2ab_, and L2bb_ using ambit tensors
    void fill_two_cumulant(ambit::Tensor& L2aa, ambit::Tensor& L2ab, ambit::Tensor& L2bb);
    /// Fill in non-tensor quantities L3aaa_, L3aab_, L3abb_ and L3bbb_ using ambit tensors
    void fill_three_cumulant(ambit::Tensor& L3aaa, ambit::Tensor& L3aab, ambit::Tensor& L3abb,
                             ambit::Tensor& L3bbb);

    /// Add wedge product of L1 to L2
    void add_wedge_cu2(const ambit::Tensor& L1a, const ambit::Tensor& L1b, ambit::Tensor& L2aa,
                       ambit::Tensor& L2ab, ambit::Tensor& L2bb);
    /// Add wedge product of L1 and L2 to L3
    void add_wedge_cu3(const ambit::Tensor& L1a, const ambit::Tensor& L1b,
                       const ambit::Tensor& L2aa, const ambit::Tensor& L2ab,
                       const ambit::Tensor& L2bb, ambit::Tensor& L3aaa, ambit::Tensor& L3aab,
                       ambit::Tensor& L3abb, ambit::Tensor& L3bbb);

    /// Fock Matrix
    d2 Fa_;
    d2 Fb_;
    /// Form Fock matrix
    void Form_Fock(d2& A, d2& B);
    /// Print Fock Matrix in Blocks
    void print_Fock(const std::string& spin, const d2& Fock);

    /// Rotate the given CI vectors by XMS
    SharedMatrix xms_rotate_this_civecs(const det_vec& p_space, SharedMatrix civecs,
                                        ambit::Tensor Fa, ambit::Tensor Fb);

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

    /// Reformat 1RDM from nactv x nactv vector to N x N SharedMatrix
    SharedMatrix reformat_1rdm(const std::string& name, const std::vector<double>& data, bool TrD);

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
    double ref_relaxed_dm_helper(const double& dm0, ambit::BlockedTensor& dm1,
                                 ambit::BlockedTensor& dm2, ambit::BlockedTensor& D1,
                                 ambit::BlockedTensor& D2);
    double ref_relaxed_dm_helper(const double& dm0, ambit::BlockedTensor& dm1,
                                 ambit::BlockedTensor& dm2, ambit::BlockedTensor& dm3,
                                 ambit::BlockedTensor& D1, ambit::BlockedTensor& D2,
                                 ambit::BlockedTensor& D3);

    /// Compute RDMs at given order and put into BlockedTensor format
    ambit::BlockedTensor compute_n_rdm(CI_RDMS& cirdm, const int& order);

    /// Localize active orbitals
    bool localize_actv_;
    void localize_actv_orbs();

    /// Print Determinants
    void print_det(const vecdet& dets);

    /// Print occupations of strings
    void
    print_occupation_strings_perirrep(std::string name,
                                      const std::vector<std::vector<std::vector<bool>>>& string);
};
}
}

#endif // _fci_mo_h_
