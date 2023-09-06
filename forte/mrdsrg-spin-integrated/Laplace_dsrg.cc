#include <algorithm>
#include <numeric>
#include <string>
#include <sstream>
#include <vector>
#include <math.h>

#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/lib3index/dftensor.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/vector.h"
#include "psi4/libpsio/psio.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libqt/qt.h"
#include "psi4/libmints/integral.h"
#include "psi4/psifiles.h"
#include "psi4/libfock/jk.h"

#include "orbital-helpers/ao_helper.h"
#include "helpers/blockedtensorfactory.h"
#include "helpers/printing.h"
#include "helpers/timer.h"
#include "fci/fci_solver.h"
#include "fci/fci_vector.h"
#include "sci/fci_mo.h"
#include "sci/aci.h"

#include "orbital-helpers/Laplace.h"
#include "Laplace_dsrg.h"

using namespace ambit;

using namespace psi;

namespace forte {
LaplaceDSRG::LaplaceDSRG(std::shared_ptr<ForteOptions> options,
                         std::shared_ptr<ForteIntegrals> ints,
                         std::shared_ptr<MOSpaceInfo> mo_space_info,
                         psi::SharedVector epsilon_rdocc, psi::SharedVector epsilon_virtual,
                         psi::SharedVector epsilon_active, psi::SharedMatrix Gamma1,
                         psi::SharedMatrix Eta1)
    : foptions_(options), ints_(ints), mo_space_info_(mo_space_info), eps_rdocc_(epsilon_rdocc),
      eps_virtual_(epsilon_virtual), eps_active_(epsilon_active), Gamma1_mat_(Gamma1),
      Eta1_mat_(Eta1) {

    /// Number of MO.
    nmo_ = mo_space_info_->dimension("ALL").sum();
    nthree_ = ints_->nthree();
    Cwfn_ = ints_->Ca();
    nfrozen_ = mo_space_info_->dimension("FROZEN").sum();
    ncore_ = eps_rdocc_->dim();
    nactive_ = eps_active_->dim();
    nvirtual_ = eps_virtual_->dim();
    vir_tol_ = foptions_->get_double("VIR_TOL");
    /// CCVV
    theta_NB_ = foptions_->get_double("THETA_NB");
    theta_NB_IAP_ = foptions_->get_double("THETA_NB_IAP");
    theta_ij_ = foptions_->get_double("THETA_IJ");
    Omega_ = foptions_->get_double("OMEGA");
    theta_schwarz_ = foptions_->get_double("THETA_SCHWARZ");
    laplace_threshold_ = foptions_->get_double("LAPLACE_THRESHOLD");
    theta_ij_sqrt_ = sqrt(theta_ij_);
    /// CAVV
    theta_NB_cavv_ = foptions_->get_double("THETA_NB_CAVV");
    theta_NB_IAP_cavv_ = foptions_->get_double("THETA_NB_IAP_CAVV");
    theta_ij_cavv_ = foptions_->get_double("THETA_IJ_CAVV");
    theta_schwarz_cavv_ = foptions_->get_double("THETA_SCHWARZ_CAVV");
    theta_ij_sqrt_cavv_ = sqrt(theta_ij_cavv_);
    theta_XNB_cavv_ = foptions_->get_double("THETA_XNB_CAVV");
    theta_NB_XAP_cavv_ = foptions_->get_double("THETA_NB_XAP_CAVV");
    /// CCAV
    theta_NB_ccav_ = foptions_->get_double("THETA_NB_CCAV");
    theta_NB_IAP_ccav_ = foptions_->get_double("THETA_NB_IAP_CCAV");
    theta_ij_ccav_ = foptions_->get_double("THETA_IJ_CCAV");
    theta_schwarz_ccav_ = foptions_->get_double("THETA_SCHWARZ_CCAV");
    theta_ij_sqrt_ccav_ = sqrt(theta_ij_ccav_);

    C_pq_ = erfc_metric(Omega_, ints_);

    outfile->Printf("\n    Done with C_pq");
    S_ = ints_->wfn()->S();
    outfile->Printf("\n    Done with S");

    primary_ = ints_->wfn()->basisset();
    auxiliary_ = ints_->wfn()->get_basisset("DF_BASIS_MP2");

    DiskDFJK disk_jk(primary_, auxiliary_);
    disk_jk.erfc_three_disk(Omega_);

    // std::cout << ints_.use_count() << std::endl;

    // ints_->~ForteIntegrals();

    // std::cout << ints_.use_count() << std::endl;

    print_header();
}

LaplaceDSRG::~LaplaceDSRG() { outfile->Printf("\n\n  Done with Laplace_DSRG class"); }

void LaplaceDSRG::print_header() {
    print_method_banner({"LAPLACE-DSRG-MRPT2", "Shuhang Li"});
    std::vector<std::pair<std::string, int>> calculation_info_int;
    std::vector<std::pair<std::string, std::string>> calculation_info_string;
    std::vector<std::pair<std::string, bool>> calculation_info_bool;
    std::vector<std::pair<std::string, double>> calculation_info_double{
        {"laplace_threshold", laplace_threshold_},
        {"Omega", Omega_},
        {"theta_NB", theta_NB_},
        {"theta_NB_IAP", theta_NB_IAP_},
        {"theta_ij", theta_ij_},
        {"theta_schwarz", theta_schwarz_},
        {"theta_NB_cavv", theta_NB_cavv_},
        {"theta_NB_IAP_cavv", theta_NB_IAP_cavv_},
        {"theta_XNB_cavv", theta_XNB_cavv_},
        {"theta_NB_XAP_cavv", theta_NB_XAP_cavv_},
        {"theta_ij_cavv", theta_ij_cavv_},
        {"theta_schwarz_cavv", theta_schwarz_cavv_},
        {"theta_NB_ccav", theta_NB_ccav_},
        {"theta_NB_IAP_ccav", theta_NB_IAP_ccav_},
        {"theta_ij_ccav", theta_ij_ccav_},
        {"theta_schwarz_ccav", theta_schwarz_ccav_},
        {"vir_tol", vir_tol_}};
    print_selected_options("Calculation Information", calculation_info_string,
                           calculation_info_bool, calculation_info_double, calculation_info_int);
}

// void LaplaceDSRG::prepare_cholesky_coeff(&psi::SharedMatrix C_wfn, &psi::SharedVector eps_rdocc,
// &psi::SharedVector eps_virtual,
//                                         &double laplace_threshold, &size_t nactive, &size_t
//                                         nfrozen, &double vir_tol) {

// }

void LaplaceDSRG::prepare_cholesky_coeff(std::shared_ptr<AtomicOrbitalHelper> ao_helper_ptr,
                                         const std::string& algorithm) {
    weights_ = ao_helper_ptr->Weights();
    vir_start_ = ao_helper_ptr->vir_start();
    ao_helper_ptr->Compute_Cholesky_Pseudo_Density();
    if (algorithm == "ccvv") {
        ao_helper_ptr->Compute_Cholesky_Density();
        Cholesky_Occ_ = ao_helper_ptr->L_Occ_real();
        Cholesky_Occ_abs_ =
            std::make_shared<psi::Matrix>("LOcc_abs", nmo_, Cholesky_Occ_->coldim());
        for (size_t u = 0; u < nmo_; u++) {
            for (size_t i = 0; i < Cholesky_Occ_->coldim(); i++) {
                Cholesky_Occ_abs_->set(u, i, std::abs(Cholesky_Occ_->get(u, i)));
            }
        }

    } else if (algorithm == "cavv") {
        // I assume that Cholesky_Occ_ has been initialized in the CCVV algorithm.
        ao_helper_ptr->Compute_Cholesky_Active_Density(Gamma1_mat_);
        Active_cholesky_ = ao_helper_ptr->LAct_list();
        Active_cholesky_abs_.resize(weights_);
    } else if (algorithm == "ccav") {
        ao_helper_ptr->Compute_Cholesky_Active_Density(Eta1_mat_);
        Active_cholesky_ = ao_helper_ptr->LAct_list();
        Active_cholesky_abs_.resize(weights_);
    }

    Occupied_cholesky_ = ao_helper_ptr->LOcc_list();
    Virtual_cholesky_ = ao_helper_ptr->LVir_list();
    Occupied_cholesky_abs_.resize(weights_);
    Virtual_cholesky_abs_.resize(weights_);

    for (size_t nweight = 0; nweight < weights_; nweight++) {
        Occupied_cholesky_abs_[nweight] =
            std::make_shared<psi::Matrix>("LOcc_abs", nmo_, Occupied_cholesky_[nweight]->coldim());
        Virtual_cholesky_abs_[nweight] =
            std::make_shared<psi::Matrix>("LVir_abs", nmo_, Virtual_cholesky_[nweight]->coldim());

        if (algorithm != "ccvv") {
            Active_cholesky_abs_[nweight] = std::make_shared<psi::Matrix>(
                "LAct_abs", nmo_, Active_cholesky_[nweight]->coldim());
        }

        for (size_t u = 0; u < nmo_; u++) {
            for (size_t i = 0; i < Occupied_cholesky_[nweight]->coldim(); i++) {
                Occupied_cholesky_abs_[nweight]->set(
                    u, i, std::abs(Occupied_cholesky_[nweight]->get(u, i)));
            }
            for (size_t a = 0; a < Virtual_cholesky_[nweight]->coldim(); a++) {
                Virtual_cholesky_abs_[nweight]->set(
                    u, a, std::abs(Virtual_cholesky_[nweight]->get(u, a)));
            }
            if (algorithm != "ccvv") {
                for (size_t v = 0; v < Active_cholesky_[nweight]->coldim(); v++) {
                    Active_cholesky_abs_[nweight]->set(
                        u, v, std::abs(Active_cholesky_[nweight]->get(u, v)));
                }
            }
        }
    }

    T_ibar_i_list_.clear();
    T_ibar_i_list_.resize(weights_);
    for (size_t nweight = 0; nweight < weights_; nweight++) {
        T_ibar_i_list_[nweight] = psi::linalg::triplet(Occupied_cholesky_[nweight], S_,
                                                       Cholesky_Occ_, true, false, false);
    }
}

void LaplaceDSRG::Load_int(double& theta_NB, const std::string& algorithm) {
    local_timer loadingTime;
    int file_unit = PSIF_DFSCF_BJ;
    size_t n_func_pairs = (nmo_ + 1) * nmo_ / 2;
    size_t max_rows = 1;
    std::shared_ptr<PSIO> psio(new PSIO());
    psi::SharedMatrix loadAOtensor =
        std::make_shared<psi::Matrix>("DiskDF: Load (A|mn) from DF-SCF", max_rows, n_func_pairs);
    psio->open(file_unit, PSIO_OPEN_OLD);

    std::vector<psi::SharedMatrix> amn;
    std::vector<psi::SharedMatrix> amn_new;

    ao_list_per_q_.clear();
    ao_list_per_q_.resize(nthree_);

    N_pu_ = std::make_shared<psi::Matrix>("N_pu_", nthree_, nmo_);
    N_pu_->zero();
    double* N_pu_p = N_pu_->get_pointer();
    outfile->Printf("\n    Start loading");

    P_xbar_u_cavv_.assign(weights_, std::vector<psi::SharedMatrix>(nthree_));
    P_xbar_u_ccav_.assign(weights_, std::vector<psi::SharedMatrix>(nthree_));

    for (int Q = 0; Q < nthree_; Q += max_rows) {
        int naux = (nthree_ - Q <= max_rows ? nthree_ - Q : max_rows);
        psio_address addr =
            psio_get_address(PSIO_ZERO, (Q * (size_t)n_func_pairs) * sizeof(double));
        psio->read(file_unit, "ERFC Integrals", (char*)(loadAOtensor->pointer()[0]),
                   sizeof(double) * naux * n_func_pairs, addr, &addr);

        double* add = loadAOtensor->get_pointer();

        amn.resize(naux);
        amn_new.resize(naux);

        psi::SharedMatrix N_pu_batch = std::make_shared<psi::Matrix>("N_pu_batch", naux, nmo_);
        N_pu_batch->zero();
        double* N_pu_batch_p = N_pu_batch->get_pointer();

        for (int q = 0; q < naux; q++) {
            amn[q] = std::make_shared<psi::Matrix>("(m * n)", nmo_, nmo_);
            /// Fill amn
            for (int u_idx = 0; u_idx < nmo_; u_idx++) {
                for (int v_idx = 0; v_idx <= u_idx; v_idx++) {
                    amn[q]->set(u_idx, v_idx, *add);
                    amn[q]->set(v_idx, u_idx, *add);
                    add++;
                }
            }
            /// Loop over amn
            double* row_add = amn[q]->get_pointer();
            for (int u = 0; u < nmo_; u++) {
                double max_per_uq = (double)0.0;
                for (int v = 0; v < nmo_; v++) {
                    double row_add_abs = std::abs(*row_add);
                    max_per_uq =
                        std::abs(max_per_uq) > row_add_abs ? std::abs(max_per_uq) : row_add_abs;
                    //*M_uv_p = std::abs(*M_uv_p) > row_add_abs ? std::abs(*M_uv_p) : row_add_abs;
                    row_add++;
                    // M_uv_p++;
                }
                if (std::abs(max_per_uq) >= theta_NB) {
                    ao_list_per_q_[Q + q].emplace_back(u);
                }
                *N_pu_p = std::abs(*N_pu_p) > max_per_uq ? std::abs(*N_pu_p) : max_per_uq;
                *N_pu_batch_p =
                    std::abs(*N_pu_batch_p) > max_per_uq ? std::abs(*N_pu_batch_p) : max_per_uq;
                N_pu_p++;
                N_pu_batch_p++;
            }
            /// Slice Amn -> Amn_new
            amn_new[q] =
                submatrix_rows_and_cols(*amn[q], ao_list_per_q_[Q + q], ao_list_per_q_[Q + q]);
        }
        if (algorithm == "ccvv") {
            P_iu_.resize(nthree_);
            SparseMap i_p_up(nthree_);
            i_p_.resize(nthree_);
            SparseMap i_p_for_i_up(nthree_);
            psi::SharedMatrix N_pi_batch =
                psi::linalg::doublet(N_pu_batch, Cholesky_Occ_abs_, false, false);
            double* N_pi_batch_p = N_pi_batch->get_pointer();
            for (int q = 0; q < naux; q++) {
                for (int i = 0; i < Cholesky_Occ_->coldim(); i++) {
                    if (std::abs(*N_pi_batch_p) >= theta_NB_) {
                        i_p_up[Q + q].emplace_back(i);
                    }
                    N_pi_batch_p++;
                }
                psi::SharedMatrix Cholesky_Occ_new =
                    submatrix_rows_and_cols(*Cholesky_Occ_, ao_list_per_q_[Q + q], i_p_up[Q + q]);
                P_iu_[Q + q] = psi::linalg::doublet(Cholesky_Occ_new, amn_new[q], true, false);
                for (int inew = 0; inew < i_p_up[Q + q].size(); inew++) {
                    for (int u = 0; u < ao_list_per_q_[Q + q].size(); u++) {
                        if (std::abs(P_iu_[Q + q]->get(inew, u)) >= theta_NB_) {
                            i_p_for_i_up[Q + q].emplace_back(inew);
                            i_p_[Q + q].emplace_back(i_p_up[Q + q][inew]);
                            break;
                        }
                    }
                }
                P_iu_[Q + q] = submatrix_rows(*P_iu_[Q + q], i_p_for_i_up[Q + q]);
            }
        } else if (algorithm == "cavv") {
            std::vector<SparseMap> xbar_p_up_cavv(weights_, SparseMap(nthree_));
            std::vector<SparseMap> xbar_u_p_for_xbar_up(weights_, SparseMap(nthree_));
            xbar_u_p_.assign(weights_, SparseMap(nthree_));

            for (int nweight = 0; nweight < weights_; nweight++) {
                psi::SharedMatrix N_px_cavv_batch =
                    psi::linalg::doublet(N_pu_batch, Active_cholesky_abs_[nweight]);
                double* N_px_batch_p = N_px_cavv_batch->get_pointer();
                for (int q = 0; q < naux; q++) {
                    for (int x = 0; x < Active_cholesky_abs_[nweight]->coldim(); x++) {
                        if (std::abs(*N_px_batch_p) >= theta_XNB_cavv_) {
                            xbar_p_up_cavv[nweight][Q + q].emplace_back(x);
                        }
                        N_px_batch_p++;
                    }
                    psi::SharedMatrix Active_cholesky_new =
                        submatrix_rows_and_cols(*Active_cholesky_[nweight], ao_list_per_q_[Q + q],
                                                xbar_p_up_cavv[nweight][Q + q]);
                    P_xbar_u_cavv_[nweight][Q + q] =
                        psi::linalg::doublet(Active_cholesky_new, amn_new[q], true, false);
                    for (int inew = 0; inew < xbar_p_up_cavv[nweight][Q + q].size(); inew++) {
                        for (int u = 0; u < ao_list_per_q_[Q + q].size(); u++) {
                            if (std::abs(P_xbar_u_cavv_[nweight][Q + q]->get(inew, u)) >=
                                theta_XNB_cavv_) {
                                xbar_u_p_for_xbar_up[nweight][Q + q].emplace_back(inew);
                                xbar_u_p_[nweight][Q + q].emplace_back(
                                    xbar_p_up_cavv[nweight][Q + q][inew]);
                                break;
                            }
                        }
                    }
                    P_xbar_u_cavv_[nweight][Q + q] = submatrix_rows(
                        *P_xbar_u_cavv_[nweight][Q + q], xbar_u_p_for_xbar_up[nweight][Q + q]);
                }
            }
        } else if (algorithm == "ccav") {
            std::vector<SparseMap> xbar_p_up_ccav(weights_, SparseMap(nthree_));
            std::vector<SparseMap> xbar_u_p_for_xbar_up(weights_, SparseMap(nthree_));
            std::vector<SparseMap> xbar_u_p_(weights_, SparseMap(nthree_));

            for (int nweight = 0; nweight < weights_; nweight++) {
                psi::SharedMatrix N_px_ccav_batch =
                    psi::linalg::doublet(N_pu_batch, Active_cholesky_abs_[nweight]);
                double* N_px_batch_p = N_px_ccav_batch->get_pointer();
                for (int q = 0; q < naux; q++) {
                    for (int x = 0; x < Active_cholesky_abs_[nweight]->coldim(); x++) {
                        if (std::abs(*N_px_batch_p) >= theta_NB) {
                            xbar_p_up_ccav[nweight][Q + q].emplace_back(x);
                        }
                        N_px_batch_p++;
                    }
                    psi::SharedMatrix Active_cholesky_new =
                        submatrix_rows_and_cols(*Active_cholesky_[nweight], ao_list_per_q_[Q + q],
                                                xbar_p_up_ccav[nweight][Q + q]);
                    P_xbar_u_ccav_[nweight][Q + q] =
                        psi::linalg::doublet(Active_cholesky_new, amn_new[q], true, false);
                    for (int inew = 0; inew < xbar_p_up_ccav[nweight][Q + q].size(); inew++) {
                        for (int u = 0; u < ao_list_per_q_[Q + q].size(); u++) {
                            if (std::abs(P_xbar_u_ccav_[nweight][Q + q]->get(inew, u)) >=
                                theta_NB) {
                                xbar_u_p_for_xbar_up[nweight][Q + q].emplace_back(inew);
                                xbar_u_p_[nweight][Q + q].emplace_back(
                                    xbar_p_up_ccav[nweight][Q + q][inew]);
                                break;
                            }
                        }
                    }
                    P_xbar_u_ccav_[nweight][Q + q] = submatrix_rows(
                        *P_xbar_u_ccav_[nweight][Q + q], xbar_u_p_for_xbar_up[nweight][Q + q]);
                }
            }
        }
    }
    psio->close(file_unit, 1);
    outfile->Printf("\n    End loading");
    outfile->Printf("\n\n  Loading takes %8.8f", loadingTime.get());
}

void LaplaceDSRG::clear_maps(const std::string& algorithm) {
    i_bar_p_up_.clear();
    a_bar_p_up_.clear();
    ibar_p_.clear();
    xbar_p_.clear();
    ibar_x_p_.clear();
    P_ibar_.clear();
    P_xbar_.clear();
    P_ibar_x_.clear();
    abar_ibar_.clear();
    abar_xbar_.clear();
    xbar_ibar_.clear();
    P_ibar_u_.clear();
    P_ibar_abar_.clear();
    P_xbar_abar_.clear();
    P_ibar_xbar_.clear();
    i_bar_a_bar_P_.clear();
    x_bar_a_bar_P_.clear();
    i_bar_x_bar_P_.clear();
    i_bar_a_bar_P_sliced_.clear();
    x_bar_a_bar_P_sliced_.clear();
    i_bar_x_bar_P_sliced_.clear();
    B_ia_Q_.clear();
    B_xa_Q_.clear();
    B_ix_Q_.clear();
    vir_intersection_per_ij_.clear();
    aux_intersection_per_ij_.clear();
    aux_in_B_i_.clear();
    aux_in_B_j_.clear();
    aux_in_B_ix_.clear();
    vir_intersection_per_ix_.clear();
    aux_intersection_per_ix_.clear();
    aux_intersection_per_ijx_.clear();
    aux_intersection_per_ixj_.clear();

    i_bar_p_up_.resize(nthree_);
    a_bar_p_up_.resize(nthree_);
    P_ibar_u_.resize(nthree_);
    P_ibar_abar_.resize(nthree_);
    ibar_p_.resize(nthree_);
    Z_pq_ = std::make_shared<psi::Matrix>("Z_pq", nthree_, nthree_);

    if (algorithm == "cavv") {
        P_xbar_abar_.resize(nthree_);
        xbar_p_.resize(nthree_);
        ZA_pq_ = std::make_shared<psi::Matrix>("ZA_pq", nthree_, nthree_);
    } else if (algorithm == "ccav") {
        P_ibar_xbar_.resize(nthree_);
        ibar_x_p_.resize(nthree_);
        ZA_pq_ = std::make_shared<psi::Matrix>("ZA_pq", nthree_, nthree_);
    }
}

void LaplaceDSRG::fill_maps(const std::string& algorithm, const int& nweight, const int& nocc,
                            const int& nact, const int& nvir) {
    double theta_NB;
    double theta_NB_IAP;

    if (algorithm == "ccvv") {
        theta_NB = theta_NB_;
        theta_NB_IAP = theta_NB_IAP_;
    } else if (algorithm == "cavv") {
        theta_NB = theta_NB_cavv_;
        theta_NB_IAP = theta_NB_IAP_cavv_;
    } else if (algorithm == "ccav") {
        theta_NB = theta_NB_ccav_;
        theta_NB_IAP = theta_NB_IAP_ccav_;
    }

    psi::SharedMatrix N_pi_bar =
        psi::linalg::doublet(N_pu_, Occupied_cholesky_abs_[nweight], false, false);
    double* N_pi_bar_p = N_pi_bar->get_pointer();
    psi::SharedMatrix N_pa_bar =
        psi::linalg::doublet(N_pu_, Virtual_cholesky_abs_[nweight], false, false);
    double* N_pa_bar_p = N_pa_bar->get_pointer();

    for (int qa = 0; qa < nthree_; qa++) {
        i_bar_p_up_[qa].clear();
        a_bar_p_up_[qa].clear();
        ibar_p_[qa].clear();
        for (int ibar = 0; ibar < nocc; ibar++) {
            if (std::abs(*N_pi_bar_p) >= theta_NB) {
                i_bar_p_up_[qa].emplace_back(ibar);
            }
            N_pi_bar_p++;
        }

        psi::SharedMatrix T_ibar_i_new =
            submatrix_rows_and_cols(*T_ibar_i_list_[nweight], i_bar_p_up_[qa], i_p_[qa]);
        P_ibar_u_[qa] = psi::linalg::doublet(T_ibar_i_new, P_iu_[qa], false, false);

        for (int abar = 0; abar < nvir; abar++) {
            if (std::abs(*N_pa_bar_p) >= theta_NB) {
                a_bar_p_up_[qa].emplace_back(abar);
            }
            N_pa_bar_p++;
        }

        psi::SharedMatrix Pseudo_Vir_Mo = submatrix_rows_and_cols(
            *Virtual_cholesky_[nweight], ao_list_per_q_[qa], a_bar_p_up_[qa]);
        P_ibar_abar_[qa] = psi::linalg::doublet(P_ibar_u_[qa], Pseudo_Vir_Mo, false, false);

        /// Construct {ibar}_p.
        for (int i = 0; i < P_ibar_abar_[qa]->rowdim(); i++) {
            for (int a = 0; a < P_ibar_abar_[qa]->coldim(); a++) {
                if (std::abs(P_ibar_abar_[qa]->get(i, a)) >= theta_NB_IAP) {
                    ibar_p_[qa].emplace_back(i_bar_p_up_[qa][i]);
                    break;
                }
            }
        }
        if (algorithm == "cavv") {
            xbar_p_[qa].clear();
            P_xbar_abar_[qa] =
                psi::linalg::doublet(P_xbar_u_cavv_[nweight][qa], Pseudo_Vir_Mo, false, false);
            /// Construct {xbar}_p.
            for (int x = 0; x < P_xbar_abar_[qa]->rowdim(); x++) {
                for (int a = 0; a < P_xbar_abar_[qa]->coldim(); a++) {
                    if (std::abs(P_xbar_abar_[qa]->get(x, a)) >= theta_NB_XAP_cavv_) {
                        xbar_p_[qa].emplace_back(xbar_u_p_[nweight][qa][x]);
                        break;
                    }
                }
            }
        } else if (algorithm == "ccav") {
            ibar_x_p_[qa].clear();
            psi::SharedMatrix Pseudo_Occ_Mo = submatrix_rows_and_cols(
                *Occupied_cholesky_[nweight], ao_list_per_q_[qa], i_bar_p_up_[qa]);
            P_ibar_xbar_[qa] =
                psi::linalg::doublet(Pseudo_Occ_Mo, P_xbar_u_ccav_[nweight][qa], true, true);
            /// Construct {ibar_x}_p.
            for (int i = 0; i < P_ibar_xbar_[qa]->rowdim(); i++) {
                for (int x = 0; x < P_ibar_xbar_[qa]->coldim(); x++) {
                    if (std::abs(P_ibar_xbar_[qa]->get(i, x)) >= theta_NB_IAP_ccav_) {
                        ibar_x_p_[qa].emplace_back(i_bar_p_up_[qa][i]);
                        break;
                    }
                }
            }
        }
    }
    /// Reorder. Construct (ibar abar|P). Use all ibar and all abar.
    i_bar_a_bar_P_.resize(nocc);
    i_bar_a_bar_P_sliced_.resize(nocc);
    if (algorithm == "ccav") {
        i_bar_x_bar_P_.resize(nocc);
        i_bar_x_bar_P_sliced_.resize(nocc);
        xbar_ibar_.resize(nocc);
    }

    for (int i = 0; i < nocc; i++) {
        i_bar_a_bar_P_[i] = std::make_shared<psi::Matrix>("(abar * P)", nvir, nthree_);
        i_bar_a_bar_P_[i]->zero();
        if (algorithm == "ccav") {
            i_bar_x_bar_P_[i] = std::make_shared<psi::Matrix>("(xbar * P)", nact, nthree_);
            i_bar_x_bar_P_[i]->zero();
        }
    }

    for (int q = 0; q < nthree_; q++) {
        for (int i = 0; i < P_ibar_abar_[q]->rowdim(); i++) {
            for (int a = 0; a < P_ibar_abar_[q]->coldim(); a++) {
                i_bar_a_bar_P_[i_bar_p_up_[q][i]]->set(a_bar_p_up_[q][a], q,
                                                       P_ibar_abar_[q]->get(i, a));
            }
            if (algorithm == "ccav") {
                for (int x = 0; x < P_ibar_xbar_[q]->coldim(); x++) {
                    i_bar_x_bar_P_[i_bar_p_up_[q][i]]->set(xbar_u_p_[nweight][q][x], q,
                                                           P_ibar_xbar_[q]->get(i, x));
                }
            }
        }
    }

    /// Construct {abar}_ibar.
    abar_ibar_.resize(nocc);
    for (int i = 0; i < nocc; i++) {
        abar_ibar_[i].clear();
        for (int a = 0; a < nvir; a++) {
            for (int q = 0; q < nthree_; q++) {
                if (std::abs(i_bar_a_bar_P_[i]->get(a, q)) >= theta_NB_IAP) {
                    abar_ibar_[i].emplace_back(a);
                    break;
                }
            }
        }
        /// Construct {xbar}_ibar.
        if (algorithm == "ccav") {
            xbar_ibar_[i].clear();
            for (int x = 0; x < nact; x++) {
                for (int q = 0; q < nthree_; q++) {
                    if (std::abs(i_bar_x_bar_P_[i]->get(x, q)) >= theta_NB_IAP) {
                        xbar_ibar_[i].emplace_back(x);
                        break;
                    }
                }
            }
        }
    }

    if (algorithm == "cavv") {
        x_bar_a_bar_P_.resize(nact);
        x_bar_a_bar_P_sliced_.resize(nact);

        for (int x = 0; x < nact; x++) {
            x_bar_a_bar_P_[x] = std::make_shared<psi::Matrix>("(abar * P)", nvir, nthree_);
            x_bar_a_bar_P_[x]->zero();
        }
        for (int q = 0; q < nthree_; q++) {
            for (int x = 0; x < P_xbar_abar_[q]->rowdim(); x++) {
                for (int a = 0; a < P_xbar_abar_[q]->coldim(); a++) {
                    x_bar_a_bar_P_[xbar_u_p_[nweight][q][x]]->set(a_bar_p_up_[q][a], q,
                                                                  P_xbar_abar_[q]->get(x, a));
                }
            }
        }
        /// Construct {abar}_xbar.
        abar_xbar_.resize(nact);
        for (int x = 0; x < nact; x++) {
            abar_xbar_[x].clear();
            for (int a = 0; a < nvir; a++) {
                for (int q = 0; q < nthree_; q++) {
                    if (std::abs(x_bar_a_bar_P_[x]->get(a, q)) >= theta_NB_XAP_cavv_) {
                        abar_xbar_[x].emplace_back(a);
                        break;
                    }
                }
            }
        }
    }
}

void LaplaceDSRG::compute_coulomb(const std::string& algorithm, const int& nocc, const int& nact,
                                  const int& nvir) {
    /// Construct Z_pq
    P_ibar_ = invert_map(ibar_p_, nocc);
    Z_pq_->zero();

    if (algorithm == "ccav") {
        P_ibar_x_ = invert_map(ibar_x_p_, nocc);
        ZA_pq_->zero();
    }

    for (int i = 0; i < nocc; i++) {
        i_bar_a_bar_P_sliced_[i] =
            submatrix_rows_and_cols(*i_bar_a_bar_P_[i], abar_ibar_[i], P_ibar_[i]);
        psi::SharedMatrix multi_per_i =
            psi::linalg::doublet(i_bar_a_bar_P_sliced_[i], i_bar_a_bar_P_sliced_[i], true, false);
        for (int p = 0; p < P_ibar_[i].size(); p++) {
            for (int q = 0; q < P_ibar_[i].size(); q++) {
                int row = P_ibar_[i][p];
                int col = P_ibar_[i][q];
                Z_pq_->add(row, col, multi_per_i->get(p, q));
            }
        }
        if (algorithm == "ccav") {
            i_bar_x_bar_P_sliced_[i] =
                submatrix_rows_and_cols(*i_bar_x_bar_P_[i], xbar_ibar_[i], P_ibar_x_[i]);
            psi::SharedMatrix multi_per_ix = psi::linalg::doublet(
                i_bar_x_bar_P_sliced_[i], i_bar_x_bar_P_sliced_[i], true, false);
            for (int px = 0; px < P_ibar_x_[i].size(); px++) {
                for (int qx = 0; qx < P_ibar_x_[i].size(); qx++) {
                    int row_x = P_ibar_x_[i][px];
                    int col_x = P_ibar_x_[i][qx];
                    ZA_pq_->add(row_x, col_x, multi_per_ix->get(px, qx));
                }
            }
        }
    }

    /// Construct D_pq
    psi::SharedMatrix D_pq = psi::linalg::doublet(Z_pq_, C_pq_, false, false);

    if (algorithm == "cavv") {
        P_xbar_ = invert_map(xbar_p_, nact);
        ZA_pq_->zero();
        for (int x = 0; x < nact; x++) {
            x_bar_a_bar_P_sliced_[x] =
                submatrix_rows_and_cols(*x_bar_a_bar_P_[x], abar_xbar_[x], P_xbar_[x]);
            psi::SharedMatrix multi_per_x = psi::linalg::doublet(
                x_bar_a_bar_P_sliced_[x], x_bar_a_bar_P_sliced_[x], true, false);
            for (int p = 0; p < P_xbar_[x].size(); p++) {
                for (int q = 0; q < P_xbar_[x].size(); q++) {
                    int row = P_xbar_[x][p];
                    int col = P_xbar_[x][q];
                    ZA_pq_->add(row, col, multi_per_x->get(p, q));
                }
            }
        }
    }

    /// Compute Coulomb Energy
    if (algorithm == "ccvv") {
        for (int q = 0; q < nthree_; q++) {
            E_J_ -= 2 * D_pq->get_row(0, q)->vector_dot(*D_pq->get_column(0, q));
        }
    } else {
        psi::SharedMatrix DA_pq = psi::linalg::doublet(ZA_pq_, C_pq_, false, false);
        for (int q = 0; q < nthree_; q++) {
            E_J_ -= 2 * D_pq->get_row(0, q)->vector_dot(*DA_pq->get_column(0, q));
        }
    }
}

void LaplaceDSRG::compute_exchange(const std::string& algorithm, const int& nocc, const int& nact,
                                   const int& nvir) {
    /// Exchange part
    B_ia_Q_.resize(nocc);
    Q_ia_ = std::make_shared<psi::Matrix>("Q_ia", nocc, nvir);
    Q_ia_->zero();

    if (algorithm == "ccav") {
        B_ix_Q_.resize(nocc);
        Q_ix_ = std::make_shared<psi::Matrix>("Q_ix", nocc, nact);
        Q_ix_->zero();
    } else if (algorithm == "cavv") {
        B_xa_Q_.resize(nact);
        Q_xa_ = std::make_shared<psi::Matrix>("Q_xa", nact, nvir);
        Q_xa_->zero();
    }

    for (int i = 0; i < nocc; i++) {
        psi::SharedMatrix sliced_C_pq = submatrix_rows_and_cols(*C_pq_, P_ibar_[i], P_ibar_[i]);
        B_ia_Q_[i] =
            psi::linalg::doublet(i_bar_a_bar_P_sliced_[i], sliced_C_pq, false, false); /// a*p

        if (algorithm == "ccvv") {
            /// Compute (i_bar a_bar|i_bar b_bar)
            psi::SharedMatrix iaib_per_i =
                psi::linalg::doublet(B_ia_Q_[i], i_bar_a_bar_P_sliced_[i], false, true);
            for (int abar = 0; abar < iaib_per_i->rowdim(); abar++) {
                E_K_ += iaib_per_i->get_row(0, abar)->vector_dot(*iaib_per_i->get_column(0, abar));
                double Q_value_2 = iaib_per_i->get(abar, abar);
                Q_ia_->set(i, abar_ibar_[i][abar], sqrt(Q_value_2));
            }
        } else if (algorithm == "ccav") {
            /// Cannot do (ia|ib) before screening. The reason is that P_ibar and P_ibar_x are
            /// different.
            for (int abar = 0; abar < B_ia_Q_[i]->rowdim(); abar++) {
                double Q_value_2 = B_ia_Q_[i]->get_row(0, abar)->vector_dot(
                    *i_bar_a_bar_P_sliced_[i]->get_row(0, abar));
                Q_ia_->set(i, abar_ibar_[i][abar], sqrt(Q_value_2));
            }
            psi::SharedMatrix sliced_C_pq_A =
                submatrix_rows_and_cols(*C_pq_, P_ibar_x_[i], P_ibar_x_[i]);
            B_ix_Q_[i] =
                psi::linalg::doublet(i_bar_x_bar_P_sliced_[i], sliced_C_pq_A, false, false); /// x*p
            for (int xbar = 0; xbar < B_ix_Q_[i]->rowdim(); xbar++) {
                double Q_value_A_2 = B_ix_Q_[i]->get_row(0, xbar)->vector_dot(
                    *i_bar_x_bar_P_sliced_[i]->get_row(0, xbar));
                Q_ix_->set(i, xbar_ibar_[i][xbar], sqrt(Q_value_A_2));
            }
        } else if (algorithm == "cavv") {
            for (int abar = 0; abar < B_ia_Q_[i]->rowdim(); abar++) {
                double Q_value_2 = B_ia_Q_[i]->get_row(0, abar)->vector_dot(
                    *i_bar_a_bar_P_sliced_[i]->get_row(0, abar));
                Q_ia_->set(i, abar_ibar_[i][abar], sqrt(Q_value_2));
            }
        }
    }

    if (algorithm == "cavv") {
        for (int x = 0; x < nact; x++) {
            psi::SharedMatrix sliced_C_pq_A =
                submatrix_rows_and_cols(*C_pq_, P_xbar_[x], P_xbar_[x]);
            B_xa_Q_[x] =
                psi::linalg::doublet(x_bar_a_bar_P_sliced_[x], sliced_C_pq_A, false, false);
            for (int abar = 0; abar < B_xa_Q_[x]->rowdim(); abar++) {
                double Q_value_A_2 = B_xa_Q_[x]->get_row(0, abar)->vector_dot(
                    *x_bar_a_bar_P_sliced_[x]->get_row(0, abar));
                Q_xa_->set(x, abar_xbar_[x][abar], sqrt(Q_value_A_2));
            }
        }
    }

    /// Start ij-prescreening.
    if (algorithm == "ccvv") {
        psi::SharedMatrix A_ij = psi::linalg::doublet(Q_ia_, Q_ia_, false, true);
        for (int i = 0; i < nocc; i++) {
            for (int j = 0; j < i; j++) {
                if (std::abs(A_ij->get(i, j)) >= theta_ij_sqrt_) {
                    vir_intersection_per_ij_.clear();
                    aux_intersection_per_ij_.clear();
                    std::set_intersection(P_ibar_[i].begin(), P_ibar_[i].end(), P_ibar_[j].begin(),
                                          P_ibar_[j].end(),
                                          std::back_inserter(aux_intersection_per_ij_));
                    aux_in_B_i_.clear();
                    std::set_intersection(abar_ibar_[i].begin(), abar_ibar_[i].end(),
                                          abar_ibar_[j].begin(), abar_ibar_[j].end(),
                                          std::back_inserter(vir_intersection_per_ij_));
                    for (auto aux : aux_intersection_per_ij_) {
                        int idx_aux_i =
                            binary_search_recursive(P_ibar_[i], aux, 0, P_ibar_[i].size() - 1);
                        aux_in_B_i_.emplace_back(idx_aux_i);
                    }
                    for (int a_idx = 0; a_idx < vir_intersection_per_ij_.size();
                         a_idx++) { // b <= a and j < i
                        for (int b_idx = 0; b_idx <= a_idx; b_idx++) {
                            int a = vir_intersection_per_ij_[a_idx];
                            int b = vir_intersection_per_ij_[b_idx];
                            double Schwarz = Q_ia_->get(i, a) * Q_ia_->get(j, b) *
                                             Q_ia_->get(i, b) * Q_ia_->get(j, a);
                            if (std::abs(Schwarz) >= theta_schwarz_) {
                                int a_idx_i = binary_search_recursive(abar_ibar_[i], a, 0,
                                                                      abar_ibar_[i].size() - 1);
                                int b_idx_i = binary_search_recursive(abar_ibar_[i], b, 0,
                                                                      abar_ibar_[i].size() - 1);

                                std::vector<int> vec_a_i{a_idx_i};
                                std::vector<int> vec_b_i{b_idx_i};
                                std::vector<int> vec_a_j{a};
                                std::vector<int> vec_b_j{b};

                                psi::SharedMatrix ia =
                                    submatrix_rows_and_cols(*B_ia_Q_[i], vec_a_i, aux_in_B_i_);
                                psi::SharedMatrix jb = submatrix_rows_and_cols(
                                    *i_bar_a_bar_P_[j], vec_b_j, aux_intersection_per_ij_);
                                psi::SharedMatrix ib =
                                    submatrix_rows_and_cols(*B_ia_Q_[i], vec_b_i, aux_in_B_i_);
                                psi::SharedMatrix ja = submatrix_rows_and_cols(
                                    *i_bar_a_bar_P_[j], vec_a_j, aux_intersection_per_ij_);

                                psi::SharedMatrix iajb_mat =
                                    psi::linalg::doublet(ia, jb, false, true);
                                double* iajb = iajb_mat->get_pointer();
                                psi::SharedMatrix ibja_mat =
                                    psi::linalg::doublet(ib, ja, false, true);
                                double* ibja = ibja_mat->get_pointer();

                                if (a == b) {
                                    E_K_ += 2 * (*iajb) * (*ibja);
                                } else {
                                    E_K_ += 4 * (*iajb) * (*ibja);
                                }
                            }
                        }
                    }
                }
            }
        }
    } else if (algorithm == "ccav") {
        psi::SharedMatrix Aa_ij = psi::linalg::doublet(Q_ix_, Q_ix_, false, true);
        psi::SharedMatrix A_ij = psi::linalg::doublet(Q_ia_, Q_ia_, false, true);
        for (int i = 0; i < nocc; i++) {
            for (int j = 0; j <= i; j++) {
                if (std::abs(Aa_ij->get(i, j)) * std::abs(A_ij->get(i, j)) >= theta_ij_ccav_) {
                    aux_intersection_per_ijx_.clear(); /// iajx
                    aux_intersection_per_ixj_.clear(); /// ixja
                    std::set_intersection(P_ibar_[i].begin(), P_ibar_[i].end(),
                                          P_ibar_x_[j].begin(), P_ibar_x_[j].end(),
                                          std::back_inserter(aux_intersection_per_ijx_));
                    std::set_intersection(P_ibar_[j].begin(), P_ibar_[j].end(),
                                          P_ibar_x_[i].begin(), P_ibar_x_[i].end(),
                                          std::back_inserter(aux_intersection_per_ixj_));
                    aux_in_B_i_.clear();
                    for (auto aux : aux_intersection_per_ijx_) {
                        int idx_aux_i =
                            binary_search_recursive(P_ibar_[i], aux, 0, P_ibar_[i].size() - 1);
                        aux_in_B_i_.emplace_back(idx_aux_i);
                    }
                    aux_in_B_ix_.clear();
                    for (auto aux : aux_intersection_per_ixj_) {
                        int idx_aux_i =
                            binary_search_recursive(P_ibar_x_[i], aux, 0, P_ibar_x_[i].size() - 1);
                        aux_in_B_ix_.emplace_back(idx_aux_i);
                    }
                    for (int a_idx = 0; a_idx < abar_ibar_[i].size(); a_idx++) {
                        for (int b_idx = 0; b_idx < xbar_ibar_[i].size(); b_idx++) {
                            int a = abar_ibar_[i][a_idx];
                            int b = xbar_ibar_[i][b_idx];
                            double Schwarz = Q_ia_->get(i, a) * Q_ix_->get(j, b) *
                                             Q_ix_->get(i, b) * Q_ia_->get(j, a);
                            if (std::abs(Schwarz) >= theta_schwarz_ccav_) {
                                int a_idx_i = binary_search_recursive(abar_ibar_[i], a, 0,
                                                                      abar_ibar_[i].size() - 1);
                                int b_idx_i = binary_search_recursive(xbar_ibar_[i], b, 0,
                                                                      xbar_ibar_[i].size() - 1);
                                std::vector<int> vec_a_i{a_idx_i};
                                std::vector<int> vec_b_i{b_idx_i};
                                std::vector<int> vec_a_j{a};
                                std::vector<int> vec_b_j{b};

                                psi::SharedMatrix ia =
                                    submatrix_rows_and_cols(*B_ia_Q_[i], vec_a_i, aux_in_B_i_);
                                psi::SharedMatrix jb = submatrix_rows_and_cols(
                                    *i_bar_x_bar_P_[j], vec_b_j, aux_intersection_per_ijx_);
                                psi::SharedMatrix ib =
                                    submatrix_rows_and_cols(*B_ix_Q_[i], vec_b_i, aux_in_B_ix_);
                                psi::SharedMatrix ja = submatrix_rows_and_cols(
                                    *i_bar_a_bar_P_[j], vec_a_j, aux_intersection_per_ixj_);

                                psi::SharedMatrix iajb_mat =
                                    psi::linalg::doublet(ia, jb, false, true);
                                double* iajb = iajb_mat->get_pointer();
                                psi::SharedMatrix ibja_mat =
                                    psi::linalg::doublet(ib, ja, false, true);
                                double* ibja = ibja_mat->get_pointer();

                                if (i == j) {
                                    E_K_ += (*iajb) * (*ibja);
                                } else {
                                    E_K_ += 2 * (*iajb) * (*ibja);
                                }
                            }
                        }
                    }
                }
            }
        }
    } else if (algorithm == "cavv") {
        /// Start ix-prescreening.
        psi::SharedMatrix Aa_ix = psi::linalg::doublet(Q_ia_, Q_xa_, false, true); // (occ * act)
        for (int i = 0; i < nocc; i++) {
            for (int x = 0; x < nact; x++) {
                if (std::abs(Aa_ix->get(i, x)) >= theta_ij_sqrt_cavv_) {
                    vir_intersection_per_ix_.clear();
                    aux_intersection_per_ix_.clear();
                    std::set_intersection(P_ibar_[i].begin(), P_ibar_[i].end(), P_xbar_[x].begin(),
                                          P_xbar_[x].end(),
                                          std::back_inserter(aux_intersection_per_ix_));
                    aux_in_B_i_.clear();
                    std::set_intersection(abar_ibar_[i].begin(), abar_ibar_[i].end(),
                                          abar_xbar_[x].begin(), abar_xbar_[x].end(),
                                          std::back_inserter(vir_intersection_per_ix_));
                    for (auto aux : aux_intersection_per_ix_) {
                        int idx_aux_i =
                            binary_search_recursive(P_ibar_[i], aux, 0, P_ibar_[i].size() - 1);
                        aux_in_B_i_.emplace_back(idx_aux_i);
                    }
                    for (int a_idx = 0; a_idx < vir_intersection_per_ix_.size(); a_idx++) {
                        for (int b_idx = 0; b_idx <= a_idx; b_idx++) {
                            int a = vir_intersection_per_ix_[a_idx];
                            int b = vir_intersection_per_ix_[b_idx];
                            double Schwarz = Q_ia_->get(i, a) * Q_xa_->get(x, b) *
                                             Q_ia_->get(i, b) * Q_xa_->get(x, a);
                            if (std::abs(Schwarz) >= theta_schwarz_cavv_) {
                                int a_idx_i = binary_search_recursive(abar_ibar_[i], a, 0,
                                                                      abar_ibar_[i].size() - 1);
                                int b_idx_i = binary_search_recursive(abar_ibar_[i], b, 0,
                                                                      abar_ibar_[i].size() - 1);
                                std::vector<int> vec_a_i{a_idx_i};
                                std::vector<int> vec_b_i{b_idx_i};
                                std::vector<int> vec_a_x{a};
                                std::vector<int> vec_b_x{b};

                                psi::SharedMatrix ia =
                                    submatrix_rows_and_cols(*B_ia_Q_[i], vec_a_i, aux_in_B_i_);
                                psi::SharedMatrix xb = submatrix_rows_and_cols(
                                    *x_bar_a_bar_P_[x], vec_b_x, aux_intersection_per_ix_);
                                psi::SharedMatrix ib =
                                    submatrix_rows_and_cols(*B_ia_Q_[i], vec_b_i, aux_in_B_i_);
                                psi::SharedMatrix xa = submatrix_rows_and_cols(
                                    *x_bar_a_bar_P_[x], vec_a_x, aux_intersection_per_ix_);

                                psi::SharedMatrix iaxb_mat =
                                    psi::linalg::doublet(ia, xb, false, true);
                                double* iaxb = iaxb_mat->get_pointer();
                                psi::SharedMatrix ibxa_mat =
                                    psi::linalg::doublet(ib, xa, false, true);
                                double* ibxa = ibxa_mat->get_pointer();

                                if (a == b) {
                                    E_K_ += (*iaxb) * (*ibxa);
                                } else {
                                    E_K_ += 2 * (*iaxb) * (*ibxa);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

double LaplaceDSRG::compute_ccvv() {
    local_timer loadingTime;
    E_J_ = 0.0;
    E_K_ = 0.0;
    std::shared_ptr<AtomicOrbitalHelper> ao_helper = std::make_shared<AtomicOrbitalHelper>(
        Cwfn_, eps_rdocc_, eps_virtual_, laplace_threshold_, nactive_, nfrozen_, vir_tol_);

    prepare_cholesky_coeff(ao_helper, "ccvv");
    Load_int(theta_NB_, "ccvv");
    clear_maps("ccvv");

    local_timer looptime;

    for (int nweight = 0; nweight < weights_; nweight++) {
        int nocc = Occupied_cholesky_[nweight]->coldim();
        int nvir = Virtual_cholesky_[nweight]->coldim();
        fill_maps("ccvv", nweight, nocc, 0, nvir);
        compute_coulomb("ccvv", nocc, 0, nvir);
        compute_exchange("ccvv", nocc, 0, nvir);
    }
    return (E_J_ + E_K_);
}

double LaplaceDSRG::compute_cavv() {
    E_J_ = 0.0;
    E_K_ = 0.0;

    std::shared_ptr<AtomicOrbitalHelper> ao_helper = std::make_shared<AtomicOrbitalHelper>(
        Cwfn_, eps_rdocc_, eps_active_, eps_virtual_, laplace_threshold_, nactive_, nfrozen_, true,
        vir_tol_);

    prepare_cholesky_coeff(ao_helper, "cavv");
    Load_int(theta_NB_cavv_, "cavv");
    clear_maps("cavv");

    for (int nweight = 0; nweight < weights_; nweight++) {
        int nocc = Occupied_cholesky_[nweight]->coldim();
        int nvir = Virtual_cholesky_[nweight]->coldim();
        int nact = Active_cholesky_[nweight]->coldim();
        fill_maps("cavv", nweight, nocc, nact, nvir);
        compute_coulomb("cavv", nocc, nact, nvir);
        compute_exchange("cavv", nocc, nact, nvir);
    }
    return (E_J_ + E_K_);
}

double LaplaceDSRG::compute_ccav() {
    E_J_ = 0.0;
    E_K_ = 0.0;
    std::shared_ptr<AtomicOrbitalHelper> ao_helper = std::make_shared<AtomicOrbitalHelper>(
        Cwfn_, eps_rdocc_, eps_active_, eps_virtual_, laplace_threshold_, nactive_, nfrozen_, false,
        vir_tol_);

    prepare_cholesky_coeff(ao_helper, "ccav");
    Load_int(theta_NB_ccav_, "ccav");
    clear_maps("ccav");

    for (int nweight = 0; nweight < weights_; nweight++) {
        int nocc = Occupied_cholesky_[nweight]->coldim();
        int nvir = Virtual_cholesky_[nweight]->coldim();
        int nact = Active_cholesky_[nweight]->coldim();
        fill_maps("ccav", nweight, nocc, nact, nvir);
        compute_coulomb("ccav", nocc, nact, nvir);
        compute_exchange("ccav", nocc, nact, nvir);
    }
    return (E_J_ + E_K_);
}

} // namespace forte