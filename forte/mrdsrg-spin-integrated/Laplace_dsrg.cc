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
        {"laplace_threshold", laplace_threshold_}, {"Omega", Omega_},
        {"theta_NB", theta_NB_},           {"theta_NB_IAP", theta_NB_IAP_},
        {"theta_ij", theta_ij_},           {"theta_schwarz", theta_schwarz_},
        {"theta_NB_cavv", theta_NB_cavv_}, {"theta_NB_IAP_cavv", theta_NB_IAP_cavv_},
        {"theta_NB_XAP_cavv", theta_NB_XAP_cavv_}, {"theta_ij_cavv", theta_ij_cavv_}, 
        {"theta_schwarz_cavv", theta_schwarz_cavv_}, {"theta_NB_ccav", theta_NB_ccav_}, 
        {"theta_NB_IAP_ccav", theta_NB_IAP_ccav_}, {"theta_ij_ccav", theta_ij_ccav_},
        {"theta_schwarz_ccav", theta_schwarz_ccav_}, {"vir_tol", vir_tol_}};
    print_selected_options("Calculation Information", calculation_info_string,
                           calculation_info_bool, calculation_info_double, calculation_info_int);
}

double LaplaceDSRG::compute_ccvv() {
    local_timer loadingTime;
    E_J_ = 0.0;
    E_K_ = 0.0;
    AtomicOrbitalHelper ao_helper(Cwfn_, eps_rdocc_, eps_virtual_, laplace_threshold_, nactive_,
                                  nfrozen_, vir_tol_);
    int weights = ao_helper.Weights();
    vir_start_ = ao_helper.vir_start();
    ao_helper.Compute_Cholesky_Pseudo_Density();
    ao_helper.Compute_Cholesky_Density();
    Occupied_cholesky_ = ao_helper.LOcc_list();
    Virtual_cholesky_ = ao_helper.LVir_list();
    Cholesky_Occ_ = ao_helper.L_Occ_real();

    /// |L_Occ|, |L_Vir| and |L_ChoMO|
    Occupied_cholesky_abs_.resize(weights);
    Virtual_cholesky_abs_.resize(weights);
    Cholesky_Occ_abs_ = std::make_shared<psi::Matrix>("LOcc_abs", nmo_, Cholesky_Occ_->coldim());

    for (int nweight = 0; nweight < weights; nweight++) {
        Occupied_cholesky_abs_[nweight] =
            std::make_shared<psi::Matrix>("LOcc_abs", nmo_, Occupied_cholesky_[nweight]->coldim());
        Virtual_cholesky_abs_[nweight] =
            std::make_shared<psi::Matrix>("LVir_abs", nmo_, Virtual_cholesky_[nweight]->coldim());
        for (int u = 0; u < nmo_; u++) {
            for (int i = 0; i < Occupied_cholesky_[nweight]->coldim(); i++) {
                Occupied_cholesky_abs_[nweight]->set(
                    u, i, std::abs(Occupied_cholesky_[nweight]->get(u, i)));
            }
            for (int a = 0; a < Virtual_cholesky_[nweight]->coldim(); a++) {
                Virtual_cholesky_abs_[nweight]->set(
                    u, a, std::abs(Virtual_cholesky_[nweight]->get(u, a)));
            }
        }
    }

    for (int u = 0; u < nmo_; u++) {
        for (int i = 0; i < Cholesky_Occ_->coldim(); i++) {
            Cholesky_Occ_abs_->set(u, i, std::abs(Cholesky_Occ_->get(u, i)));
        }
    }

    std::vector<psi::SharedMatrix> T_ibar_i_list(weights);
    for (int i_weight = 0; i_weight < weights; i_weight++) {
        T_ibar_i_list[i_weight] = psi::linalg::triplet(Occupied_cholesky_[i_weight], S_,
                                                       Cholesky_Occ_, true, false, false);
    }

    /// Construct (P|iu)
    P_iu.resize(nthree_); /// [([i]_p * ao_list_per_q), ...]

    /// Below is the code for loading data from disk.
    int file_unit = PSIF_DFSCF_BJ;
    int n_func_pairs = (nmo_ + 1) * nmo_ / 2;
    int max_rows = 1;
    std::shared_ptr<PSIO> psio(new PSIO());
    psi::SharedMatrix loadAOtensor =
        std::make_shared<psi::Matrix>("DiskDF: Load (A|mn) from DF-SCF", max_rows, n_func_pairs);
    psio->open(file_unit, PSIO_OPEN_OLD);

    std::vector<psi::SharedMatrix> amn;
    std::vector<psi::SharedMatrix> amn_new;

    SparseMap ao_list_per_q; ///{u}_P
    ao_list_per_q.resize(nthree_);

    SparseMap i_p_up(nthree_);
    i_p_.resize(nthree_);
    SparseMap i_p_for_i_up(nthree_);

    psi::SharedMatrix N_pu = std::make_shared<psi::Matrix>("N_pu", nthree_, nmo_);
    N_pu->zero();
    double* N_pu_p = N_pu->get_pointer();
    outfile->Printf("\n    Start loading");

    for (int Q = 0; Q < nthree_; Q += max_rows) {
        int naux = (nthree_ - Q <= max_rows ? nthree_ - Q : max_rows);
        psio_address addr = psio_get_address(PSIO_ZERO, (Q * (int)n_func_pairs) * sizeof(double));
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
                if (std::abs(max_per_uq) >= theta_NB_) {
                    ao_list_per_q[Q + q].push_back(u);
                }
                *N_pu_p = std::abs(*N_pu_p) > max_per_uq ? std::abs(*N_pu_p) : max_per_uq;
                *N_pu_batch_p =
                    std::abs(*N_pu_batch_p) > max_per_uq ? std::abs(*N_pu_batch_p) : max_per_uq;
                N_pu_p++;
                N_pu_batch_p++;
            }
            /// Slice Amn -> Amn_new
            amn_new[q] =
                submatrix_rows_and_cols(*amn[q], ao_list_per_q[Q + q], ao_list_per_q[Q + q]);
        }

        psi::SharedMatrix N_pi_batch =
            psi::linalg::doublet(N_pu_batch, Cholesky_Occ_abs_, false, false);
        double* N_pi_batch_p = N_pi_batch->get_pointer();
        for (int q = 0; q < naux; q++) {
            for (int i = 0; i < Cholesky_Occ_->coldim(); i++) {
                if (std::abs(*N_pi_batch_p) >= theta_NB_) {
                    i_p_up[Q + q].push_back(i);
                }
                N_pi_batch_p++;
            }
            psi::SharedMatrix Cholesky_Occ_new =
                submatrix_rows_and_cols(*Cholesky_Occ_, ao_list_per_q[Q + q], i_p_up[Q + q]);
            P_iu[Q + q] = psi::linalg::doublet(Cholesky_Occ_new, amn_new[q], true, false);
            for (int inew = 0; inew < i_p_up[Q + q].size(); inew++) {
                for (int u = 0; u < ao_list_per_q[Q + q].size(); u++) {
                    if (std::abs(P_iu[Q + q]->get(inew, u)) >= theta_NB_) {
                        i_p_for_i_up[Q + q].push_back(inew);
                        i_p_[Q + q].push_back(i_p_up[Q + q][inew]);
                        break;
                    }
                }
            }
            P_iu[Q + q] = submatrix_rows(*P_iu[Q + q], i_p_for_i_up[Q + q]);
        }
    }
    psio->close(file_unit, 1);
    loadAOtensor.reset();
    outfile->Printf("\n    End loading");

    //outfile->Printf("\n\n  Loading takes %8.8f", loadingTime.get());

    /// Construct [ibar]_p and [abar]_p
    SparseMap i_bar_p_up(nthree_);
    SparseMap a_bar_p_up(nthree_);

    /// Construct (P|ibar u)
    std::vector<psi::SharedMatrix> P_ibar_u(nthree_);

    /// Construct (P|ibar abar)
    std::vector<psi::SharedMatrix> P_ibar_abar(nthree_);

    /// Construct {ibar}_p
    SparseMap ibar_p(nthree_);

    /// Construct {P}_ibar.  Obtain this by "inversion" of {ibar}_p
    SparseMap P_ibar;

    /// Construct {abar}_p
    //SparseMap abar_p(nthree_);

    /// Construct {p}_abar.  Obtain this by "inversion" of {abar}_p
    //SparseMap P_abar;

    /// Construct (ibar abar|P)
    std::vector<psi::SharedMatrix> i_bar_a_bar_P;

    /// Constrcut Sliced (ibar abar|P);
    std::vector<psi::SharedMatrix> i_bar_a_bar_P_sliced;

    /// Construct {a_bar}_ibar
    SparseMap abar_ibar;

    /// Construct Z_pq
    psi::SharedMatrix Z_pq = std::make_shared<psi::Matrix>("Z_pq", nthree_, nthree_);

    /// Construct B_ia_Q
    std::vector<psi::SharedMatrix> B_ia_Q;

    /// Construc Q_ia square;
    psi::SharedMatrix Q_ia;

    std::vector<int> vir_intersection_per_ij;
    std::vector<int> aux_intersection_per_ij;
    std::vector<int> aux_in_B_i;
    std::vector<int> aux_in_B_j;

    local_timer looptime;

    for (int nweight = 0; nweight < weights; nweight++) {
        int nocc = Occupied_cholesky_[nweight]->coldim();
        int nvir = Virtual_cholesky_[nweight]->coldim();
        // keep track of ij prescreening
        int unselected_occ = 0;

        psi::SharedMatrix N_pi_bar =
            psi::linalg::doublet(N_pu, Occupied_cholesky_abs_[nweight], false, false);
        double* N_pi_bar_p = N_pi_bar->get_pointer();
        psi::SharedMatrix N_pa_bar =
            psi::linalg::doublet(N_pu, Virtual_cholesky_abs_[nweight], false, false);
        double* N_pa_bar_p = N_pa_bar->get_pointer();

        for (int qa = 0; qa < nthree_; qa++) {
            i_bar_p_up[qa].clear();
            a_bar_p_up[qa].clear();
            ibar_p[qa].clear();
            //abar_p[qa].clear();
            for (int ibar = 0; ibar < nocc; ibar++) {
                if (std::abs(*N_pi_bar_p) >= theta_NB_) {
                    i_bar_p_up[qa].push_back(ibar);
                }
                N_pi_bar_p++;
            }

            psi::SharedMatrix T_ibar_i_new =
                submatrix_rows_and_cols(*T_ibar_i_list[nweight], i_bar_p_up[qa], i_p_[qa]);
            P_ibar_u[qa] = psi::linalg::doublet(T_ibar_i_new, P_iu[qa], false, false);

            for (int abar = 0; abar < nvir; abar++) {
                if (std::abs(*N_pa_bar_p) >= theta_NB_) {
                    a_bar_p_up[qa].push_back(abar);
                }
                N_pa_bar_p++;
            }

            psi::SharedMatrix Pseudo_Vir_Mo = submatrix_rows_and_cols(
                *Virtual_cholesky_[nweight], ao_list_per_q[qa], a_bar_p_up[qa]);
            P_ibar_abar[qa] = psi::linalg::doublet(P_ibar_u[qa], Pseudo_Vir_Mo, false, false);

            /// Construct {ibar}_p.
            for (int i = 0; i < P_ibar_abar[qa]->rowdim(); i++) {
                for (int a = 0; a < P_ibar_abar[qa]->coldim(); a++) {
                    if (std::abs(P_ibar_abar[qa]->get(i, a)) >= theta_NB_IAP_) {
                        ibar_p[qa].push_back(i_bar_p_up[qa][i]);
                        break;
                    }
                }
            }

            /// Construct {abar}_p
            // for (int a = 0; a < P_ibar_abar[qa]->coldim(); a++) {
            //     for (int i = 0; i < P_ibar_abar[qa]->rowdim(); i++) {
            //         if (std::abs(P_ibar_abar[qa]->get(i, a)) > theta_NB) {
            //             abar_p[qa].push_back(a_bar_p_up[qa][a]);
            //             break;
            //         }
            //     }
            // }
        }

        /// Reorder. Construct (ibar abar|P). Use all ibar and all abar.
        i_bar_a_bar_P.resize(nocc);
        i_bar_a_bar_P_sliced.resize(nocc);
        for (int i = 0; i < nocc; i++) {
            i_bar_a_bar_P[i] = std::make_shared<psi::Matrix>("(abar * P)", nvir, nthree_);
            i_bar_a_bar_P[i]->zero();
        }

        for (int q = 0; q < nthree_; q++) {
            for (int i = 0; i < P_ibar_abar[q]->rowdim(); i++) {
                for (int a = 0; a < P_ibar_abar[q]->coldim(); a++) {
                    i_bar_a_bar_P[i_bar_p_up[q][i]]->set(a_bar_p_up[q][a], q,
                                                         P_ibar_abar[q]->get(i, a));
                }
            }
        }

        /// Construct {abar}_ibar.
        abar_ibar.resize(nocc);
        for (int i = 0; i < nocc; i++) {
            abar_ibar[i].clear();
            for (int a = 0; a < nvir; a++) {
                for (int q = 0; q < nthree_; q++) {
                    if (std::abs(i_bar_a_bar_P[i]->get(a, q)) >= theta_NB_IAP_) {
                        abar_ibar[i].push_back(a);
                        break;
                    }
                }
            }
        }

        /// Coulomb contribution.
        /// Construct Z_pq
        P_ibar = invert_map(ibar_p, nocc);
        // P_abar = invert_map(abar_p, nvir);
        Z_pq->zero();

        for (int i = 0; i < nocc; i++) {
            i_bar_a_bar_P_sliced[i] =
                submatrix_rows_and_cols(*i_bar_a_bar_P[i], abar_ibar[i], P_ibar[i]);
            psi::SharedMatrix multi_per_i =
                psi::linalg::doublet(i_bar_a_bar_P_sliced[i], i_bar_a_bar_P_sliced[i], true, false);
            for (int p = 0; p < P_ibar[i].size(); p++) {
                for (int q = 0; q < P_ibar[i].size(); q++) {
                    int row = P_ibar[i][p];
                    int col = P_ibar[i][q];
                    Z_pq->add(row, col, multi_per_i->get(p, q));
                }
            }
        }

        /// Construct D_pq
        psi::SharedMatrix D_pq = psi::linalg::doublet(Z_pq, C_pq_, false, false);

        /// Compute Coulomb Energy
        for (int q = 0; q < nthree_; q++) {
            E_J_ -= 2 * D_pq->get_row(0, q)->vector_dot(*D_pq->get_column(0, q));
        }

        //outfile->Printf("\n\n  Coulomb finished %8.8f", looptime.get());

        /// Exchange part
        B_ia_Q.resize(nocc);
        Q_ia = std::make_shared<psi::Matrix>("Q_ia", nocc, nvir);
        Q_ia->zero();

        for (int i = 0; i < nocc; i++) {
            psi::SharedMatrix sliced_C_pq = submatrix_rows_and_cols(*C_pq_, P_ibar[i], P_ibar[i]);
            B_ia_Q[i] =
                psi::linalg::doublet(i_bar_a_bar_P_sliced[i], sliced_C_pq, false, false); /// a*p

            /// Compute (i_bar a_bar|i_bar b_bar)
            psi::SharedMatrix iaib_per_i =
                psi::linalg::doublet(B_ia_Q[i], i_bar_a_bar_P_sliced[i], false, true);
            for (int abar = 0; abar < iaib_per_i->rowdim(); abar++) {
                E_K_ += iaib_per_i->get_row(0, abar)->vector_dot(*iaib_per_i->get_column(0, abar));
                double Q_value_2 = iaib_per_i->get(abar, abar);
                Q_ia->set(i, abar_ibar[i][abar], sqrt(Q_value_2));
            }
        }

        //outfile->Printf("\n\n  ii Exchange finished %8.8f", looptime.get());

        /// Start ij-prescreening.
        psi::SharedMatrix A_ij = psi::linalg::doublet(Q_ia, Q_ia, false, true);
        for (int i = 0; i < nocc; i++) {
            for (int j = 0; j < i; j++) {
                if (std::abs(A_ij->get(i, j)) >= theta_ij_sqrt_) {
                    vir_intersection_per_ij.clear();
                    aux_intersection_per_ij.clear();
                    std::set_intersection(P_ibar[i].begin(), P_ibar[i].end(), P_ibar[j].begin(),
                                          P_ibar[j].end(),
                                          std::back_inserter(aux_intersection_per_ij));
                    aux_in_B_i.clear();
                    std::set_intersection(abar_ibar[i].begin(), abar_ibar[i].end(),
                                          abar_ibar[j].begin(), abar_ibar[j].end(),
                                          std::back_inserter(vir_intersection_per_ij));
                    for (auto aux : aux_intersection_per_ij) {
                        int idx_aux_i =
                            binary_search_recursive(P_ibar[i], aux, 0, P_ibar[i].size() - 1);
                        aux_in_B_i.push_back(idx_aux_i);
                    }
                    for (int a_idx = 0; a_idx < vir_intersection_per_ij.size();
                         a_idx++) { // b <= a and j < i
                        for (int b_idx = 0; b_idx <= a_idx; b_idx++) {
                            int a = vir_intersection_per_ij[a_idx];
                            int b = vir_intersection_per_ij[b_idx];
                            double Schwarz = Q_ia->get(i, a) * Q_ia->get(j, b) * Q_ia->get(i, b) *
                                             Q_ia->get(j, a);
                            if (std::abs(Schwarz) >= theta_schwarz_) {
                                int a_idx_i = binary_search_recursive(abar_ibar[i], a, 0,
                                                                      abar_ibar[i].size() - 1);
                                int b_idx_i = binary_search_recursive(abar_ibar[i], b, 0,
                                                                      abar_ibar[i].size() - 1);
                                // int a_idx_j = binary_search_recursive(abar_ibar[j], a, 0,
                                //                                       abar_ibar[j].size() - 1);
                                // int b_idx_j = binary_search_recursive(abar_ibar[j], b, 0,
                                //                                       abar_ibar[j].size() - 1);

                                std::vector<int> vec_a_i{a_idx_i};
                                std::vector<int> vec_b_i{b_idx_i};
                                std::vector<int> vec_a_j{a};
                                std::vector<int> vec_b_j{b};

                                psi::SharedMatrix ia =
                                    submatrix_rows_and_cols(*B_ia_Q[i], vec_a_i, aux_in_B_i);
                                psi::SharedMatrix jb = submatrix_rows_and_cols(
                                    *i_bar_a_bar_P[j], vec_b_j, aux_intersection_per_ij);
                                psi::SharedMatrix ib =
                                    submatrix_rows_and_cols(*B_ia_Q[i], vec_b_i, aux_in_B_i);
                                psi::SharedMatrix ja = submatrix_rows_and_cols(
                                    *i_bar_a_bar_P[j], vec_a_j, aux_intersection_per_ij);

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
                } else {
                    unselected_occ += 1;
                }
            }
        }
        //outfile->Printf("\n\n  Exchange finished %8.8f", looptime.get());
        //outfile->Printf("  Number of unselected ij pairs: %d. \n", unselected_occ);
    }

    // std::cout << E_J_ << "\n";
    // std::cout << E_K_ << "\n";

    return (E_J_ + E_K_);
}

double LaplaceDSRG::compute_cavv() {
    E_J_ = 0.0;
    E_K_ = 0.0;
    AtomicOrbitalHelper ao_helper_cavv(Cwfn_, eps_rdocc_, eps_active_, eps_virtual_,
                                       laplace_threshold_, nactive_, nfrozen_, true, vir_tol_);

    int weights_cavv = ao_helper_cavv.Weights();
    vir_start_cavv_ = ao_helper_cavv.vir_start();
    ao_helper_cavv.Compute_Cholesky_Pseudo_Density();
    ao_helper_cavv.Compute_Cholesky_Active_Density(Gamma1_mat_);
    Occupied_cholesky_cavv_ = ao_helper_cavv.LOcc_list();
    Active_cholesky_cavv_ = ao_helper_cavv.LAct_list();
    Virtual_cholesky_cavv_ = ao_helper_cavv.LVir_list();
    Occupied_cholesky_cavv_abs_.resize(weights_cavv);
    Active_cholesky_cavv_abs_.resize(weights_cavv);
    Virtual_cholesky_cavv_abs_.resize(weights_cavv);

    for (int nweight = 0; nweight < weights_cavv; nweight++) {
        Occupied_cholesky_cavv_abs_[nweight] = std::make_shared<psi::Matrix>(
            "LOcc_abs", nmo_, Occupied_cholesky_cavv_[nweight]->coldim());
        Active_cholesky_cavv_abs_[nweight] = std::make_shared<psi::Matrix>(
            "LAct_abs", nmo_, Active_cholesky_cavv_[nweight]->coldim());
        Virtual_cholesky_cavv_abs_[nweight] = std::make_shared<psi::Matrix>(
            "LVir_abs", nmo_, Virtual_cholesky_cavv_[nweight]->coldim());
        for (int u = 0; u < nmo_; u++) {
            for (int i = 0; i < Occupied_cholesky_cavv_[nweight]->coldim(); i++) {
                Occupied_cholesky_cavv_abs_[nweight]->set(
                    u, i, std::abs(Occupied_cholesky_cavv_[nweight]->get(u, i)));
            }
            for (int v = 0; v < Active_cholesky_cavv_[nweight]->coldim(); v++) {
                Active_cholesky_cavv_abs_[nweight]->set(
                    u, v, std::abs(Active_cholesky_cavv_[nweight]->get(u, v)));
            }
            for (int a = 0; a < Virtual_cholesky_cavv_[nweight]->coldim(); a++) {
                Virtual_cholesky_cavv_abs_[nweight]->set(
                    u, a, std::abs(Virtual_cholesky_cavv_[nweight]->get(u, a)));
            }
        }
    }

    std::vector<psi::SharedMatrix> T_ibar_i_list(weights_cavv);
    for (int i_weight = 0; i_weight < weights_cavv; i_weight++) {
        T_ibar_i_list[i_weight] = psi::linalg::triplet(Occupied_cholesky_cavv_[i_weight], S_,
                                                       Cholesky_Occ_, true, false, false);
    }

    // std::vector<psi::SharedMatrix> N_px_cavv_batch_list(weights_cavv);
    std::vector<SparseMap> xbar_p_up_cavv(weights_cavv, SparseMap(nthree_));
    std::vector<std::vector<psi::SharedMatrix>> P_xbar_u_cavv(
        weights_cavv, std::vector<psi::SharedMatrix>(nthree_));
    std::vector<SparseMap> xbar_u_p_for_xbar_up(weights_cavv, SparseMap(nthree_));
    std::vector<SparseMap> xbar_u_p(weights_cavv, SparseMap(nthree_));

    /// Below is the code for loading data from disk.
    int file_unit = PSIF_DFSCF_BJ;
    int n_func_pairs = (nmo_ + 1) * nmo_ / 2;
    int max_rows = 1;
    std::shared_ptr<PSIO> psio(new PSIO());
    psi::SharedMatrix loadAOtensor =
        std::make_shared<psi::Matrix>("DiskDF: Load (A|mn) from DF-SCF", max_rows, n_func_pairs);
    psio->open(file_unit, PSIO_OPEN_OLD);

    std::vector<psi::SharedMatrix> amn;
    std::vector<psi::SharedMatrix> amn_new;

    SparseMap ao_list_per_q; ///{u}_P
    ao_list_per_q.resize(nthree_);

    psi::SharedMatrix N_pu = std::make_shared<psi::Matrix>("N_pu", nthree_, nmo_);
    N_pu->zero();
    double* N_pu_p = N_pu->get_pointer();
    outfile->Printf("\n    Start loading");

    for (int Q = 0; Q < nthree_; Q += max_rows) {
        int naux = (nthree_ - Q <= max_rows ? nthree_ - Q : max_rows);
        psio_address addr = psio_get_address(PSIO_ZERO, (Q * (int)n_func_pairs) * sizeof(double));
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
                if (std::abs(max_per_uq) >= theta_NB_cavv_) {
                    ao_list_per_q[Q + q].push_back(u);
                }
                *N_pu_p = std::abs(*N_pu_p) > max_per_uq ? std::abs(*N_pu_p) : max_per_uq;
                *N_pu_batch_p =
                    std::abs(*N_pu_batch_p) > max_per_uq ? std::abs(*N_pu_batch_p) : max_per_uq;
                N_pu_p++;
                N_pu_batch_p++;
            }
            /// Slice Amn -> Amn_new
            amn_new[q] =
                submatrix_rows_and_cols(*amn[q], ao_list_per_q[Q + q], ao_list_per_q[Q + q]);
        }

        for (int nweight = 0; nweight < weights_cavv; nweight++) {
            psi::SharedMatrix N_px_cavv_batch =
                psi::linalg::doublet(N_pu_batch, Active_cholesky_cavv_abs_[nweight]);
            double* N_px_batch_p = N_px_cavv_batch->get_pointer();
            for (int q = 0; q < naux; q++) {
                for (int x = 0; x < Active_cholesky_cavv_abs_[nweight]->coldim(); x++) {
                    if (std::abs(*N_px_batch_p) >= theta_NB_cavv_) {
                        xbar_p_up_cavv[nweight][Q + q].push_back(x);
                    }
                    N_px_batch_p++;
                }
                psi::SharedMatrix Active_cholesky_new =
                    submatrix_rows_and_cols(*Active_cholesky_cavv_[nweight], ao_list_per_q[Q + q],
                                            xbar_p_up_cavv[nweight][Q + q]);
                P_xbar_u_cavv[nweight][Q + q] =
                    psi::linalg::doublet(Active_cholesky_new, amn_new[q], true, false);
                for (int inew = 0; inew < xbar_p_up_cavv[nweight][Q + q].size(); inew++) {
                    for (int u = 0; u < ao_list_per_q[Q + q].size(); u++) {
                        if (std::abs(P_xbar_u_cavv[nweight][Q + q]->get(inew, u)) >= theta_NB_cavv_) {
                            xbar_u_p_for_xbar_up[nweight][Q + q].push_back(inew);
                            xbar_u_p[nweight][Q + q].push_back(
                                xbar_p_up_cavv[nweight][Q + q][inew]);
                            break;
                        }
                    }
                }
                P_xbar_u_cavv[nweight][Q + q] = submatrix_rows(
                    *P_xbar_u_cavv[nweight][Q + q], xbar_u_p_for_xbar_up[nweight][Q + q]);
            }
        }
    }
    psio->close(file_unit, 1);
    loadAOtensor.reset();
    outfile->Printf("\n    End loading");

    //////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////

    /// Construct [ibar]_p and [abar]_p
    SparseMap i_bar_p_up(nthree_);
    SparseMap a_bar_p_up(nthree_);

    /// Construct (P|ibar u)
    std::vector<psi::SharedMatrix> P_ibar_u(nthree_);

    /// Construct (P|ibar abar)
    std::vector<psi::SharedMatrix> P_ibar_abar(nthree_);

    /// Construct (P|xbar abar)
    std::vector<psi::SharedMatrix> P_xbar_abar(nthree_);

    /// Construct {ibar}_p
    SparseMap ibar_p(nthree_);

    /// Construct {xbar}_p
    SparseMap xbar_p(nthree_);

    /// Construct {P}_ibar.  Obtain this by "inversion" of {ibar}_p
    SparseMap P_ibar;

    /// Construct {abar}_p
    //SparseMap abar_p(nthree_);

    /// Construct {p}_abar.  Obtain this by "inversion" of {abar}_p
    //SparseMap P_abar;

    /// Construct {p}_xbar
    SparseMap P_xbar;

    /// Construct (ibar abar|P)
    std::vector<psi::SharedMatrix> i_bar_a_bar_P;

    /// Construct (xbar abar|P)
    std::vector<psi::SharedMatrix> x_bar_a_bar_P;

    /// Constrcut Sliced (ibar abar|P);
    std::vector<psi::SharedMatrix> i_bar_a_bar_P_sliced;

    /// Construct Sliced (xbar abar|P)
    std::vector<psi::SharedMatrix> x_bar_a_bar_P_sliced;

    /// Construct {a_bar}_ibar
    SparseMap abar_ibar;

    /// Construct {a_bar}_xbar
    SparseMap abar_xbar;

    /// Construct Z_pq
    psi::SharedMatrix Z_pq = std::make_shared<psi::Matrix>("Z_pq", nthree_, nthree_);

    /// Construct ZA_pq
    psi::SharedMatrix ZA_pq = std::make_shared<psi::Matrix>("ZA_pq", nthree_, nthree_);

    /// Construct B_xa_Q
    std::vector<psi::SharedMatrix> B_xa_Q;

    /// Construct B_ia_Q
    std::vector<psi::SharedMatrix> B_ia_Q;

    /// Construc Q_ia square;
    psi::SharedMatrix Q_ia;

    psi::SharedMatrix Q_xa;

    std::vector<int> vir_intersection_per_ix;
    std::vector<int> aux_intersection_per_ix;
    std::vector<int> aux_in_B_i;

    for (int nweight = 0; nweight < weights_cavv; nweight++) {
        int nocc = Occupied_cholesky_cavv_[nweight]->coldim();
        int nvir = Virtual_cholesky_cavv_[nweight]->coldim();
        int nact = Active_cholesky_cavv_[nweight]->coldim();
        psi::SharedMatrix N_pi_bar =
            psi::linalg::doublet(N_pu, Occupied_cholesky_cavv_abs_[nweight], false, false);
        double* N_pi_bar_p = N_pi_bar->get_pointer();
        psi::SharedMatrix N_pa_bar =
            psi::linalg::doublet(N_pu, Virtual_cholesky_cavv_abs_[nweight], false, false);
        double* N_pa_bar_p = N_pa_bar->get_pointer();

        for (int qa = 0; qa < nthree_; qa++) {
            i_bar_p_up[qa].clear();
            a_bar_p_up[qa].clear();
            ibar_p[qa].clear();
            //abar_p[qa].clear();
            xbar_p[qa].clear();
            for (int ibar = 0; ibar < nocc; ibar++) {
                if (std::abs(*N_pi_bar_p) >= theta_NB_cavv_) {
                    i_bar_p_up[qa].push_back(ibar);
                }
                N_pi_bar_p++;
            }

            psi::SharedMatrix T_ibar_i_new =
                submatrix_rows_and_cols(*T_ibar_i_list[nweight], i_bar_p_up[qa], i_p_[qa]);
            P_ibar_u[qa] = psi::linalg::doublet(T_ibar_i_new, P_iu[qa], false, false);
            for (int abar = 0; abar < nvir; abar++) {
                if (std::abs(*N_pa_bar_p) >= theta_NB_cavv_) {
                    a_bar_p_up[qa].push_back(abar);
                }
                N_pa_bar_p++;
            }

            psi::SharedMatrix Pseudo_Vir_Mo = submatrix_rows_and_cols(
                *Virtual_cholesky_cavv_[nweight], ao_list_per_q[qa], a_bar_p_up[qa]);
            P_ibar_abar[qa] = psi::linalg::doublet(P_ibar_u[qa], Pseudo_Vir_Mo, false, false);
            P_xbar_abar[qa] =
                psi::linalg::doublet(P_xbar_u_cavv[nweight][qa], Pseudo_Vir_Mo, false, false);

            /// Construct {ibar}_p.
            for (int i = 0; i < P_ibar_abar[qa]->rowdim(); i++) {
                for (int a = 0; a < P_ibar_abar[qa]->coldim(); a++) {
                    if (std::abs(P_ibar_abar[qa]->get(i, a)) >= theta_NB_IAP_cavv_) {
                        ibar_p[qa].push_back(i_bar_p_up[qa][i]);
                        break;
                    }
                }
            }

            /// Construct {xbar}_p.
            for (int x = 0; x < P_xbar_abar[qa]->rowdim(); x++) {
                for (int a = 0; a < P_xbar_abar[qa]->coldim(); a++) {
                    if (std::abs(P_xbar_abar[qa]->get(x, a)) >= theta_NB_IAP_cavv_) {
                        xbar_p[qa].push_back(xbar_u_p[nweight][qa][x]);
                        break;
                    }
                }
            }
        }
        /// Reorder. Construct (ibar abar|P). Use all ibar and all abar.
        /// Reorder. Construct (xbar abar|P). Use all xbar and all abar.
        i_bar_a_bar_P.resize(nocc);
        i_bar_a_bar_P_sliced.resize(nocc);
        x_bar_a_bar_P.resize(nact);
        x_bar_a_bar_P_sliced.resize(nact);

        for (int i = 0; i < nocc; i++) {
            i_bar_a_bar_P[i] = std::make_shared<psi::Matrix>("(abar * P)", nvir, nthree_);
            i_bar_a_bar_P[i]->zero();
        }

        for (int q = 0; q < nthree_; q++) {
            for (int i = 0; i < P_ibar_abar[q]->rowdim(); i++) {
                for (int a = 0; a < P_ibar_abar[q]->coldim(); a++) {
                    i_bar_a_bar_P[i_bar_p_up[q][i]]->set(a_bar_p_up[q][a], q,
                                                         P_ibar_abar[q]->get(i, a));
                }
            }
        }

        for (int x = 0; x < nact; x++) {
            x_bar_a_bar_P[x] = std::make_shared<psi::Matrix>("(abar * P)", nvir, nthree_);
            x_bar_a_bar_P[x]->zero();
        }

        for (int q = 0; q < nthree_; q++) {
            for (int x = 0; x < P_xbar_abar[q]->rowdim(); x++) {
                for (int a = 0; a < P_xbar_abar[q]->coldim(); a++) {
                    x_bar_a_bar_P[xbar_u_p[nweight][q][x]]->set(a_bar_p_up[q][a], q,
                                                                P_xbar_abar[q]->get(x, a));
                }
            }
        }

        /// Construct {abar}_ibar.
        abar_ibar.resize(nocc);
        for (int i = 0; i < nocc; i++) {
            abar_ibar[i].clear();
            for (int a = 0; a < nvir; a++) {
                for (int q = 0; q < nthree_; q++) {
                    if (std::abs(i_bar_a_bar_P[i]->get(a, q)) >= theta_NB_IAP_cavv_) {
                        abar_ibar[i].push_back(a);
                        break;
                    }
                }
            }
            //outfile->Printf("\n\n  Number for i %d", abar_ibar[i].size());
        }

        /// Construct {abar}_xbar.
        abar_xbar.resize(nact);
        for (int x = 0; x < nact; x++) {
            abar_xbar[x].clear();
            for (int a = 0; a < nvir; a++) {
                for (int q = 0; q < nthree_; q++) {
                    if (std::abs(x_bar_a_bar_P[x]->get(a, q)) >= theta_NB_XAP_cavv_) {
                        abar_xbar[x].push_back(a);
                        break;
                    }
                }
            }
            //outfile->Printf("\n\n  Number for x %d", abar_xbar[x].size());
        }
        /// Coulomb contribution.
        /// Construct Z_pq
        P_ibar = invert_map(ibar_p, nocc);
        P_xbar = invert_map(xbar_p, nact);
        // P_abar = invert_map(abar_p, nvir);
        Z_pq->zero();
        ZA_pq->zero();

        for (int i = 0; i < nocc; i++) {
            i_bar_a_bar_P_sliced[i] =
                submatrix_rows_and_cols(*i_bar_a_bar_P[i], abar_ibar[i], P_ibar[i]);
            psi::SharedMatrix multi_per_i =
                psi::linalg::doublet(i_bar_a_bar_P_sliced[i], i_bar_a_bar_P_sliced[i], true, false);
            for (int p = 0; p < P_ibar[i].size(); p++) {
                for (int q = 0; q < P_ibar[i].size(); q++) {
                    int row = P_ibar[i][p];
                    int col = P_ibar[i][q];
                    Z_pq->add(row, col, multi_per_i->get(p, q));
                }
            }
        }

        for (int x = 0; x < nact; x++) {
            x_bar_a_bar_P_sliced[x] =
                submatrix_rows_and_cols(*x_bar_a_bar_P[x], abar_xbar[x], P_xbar[x]);
            psi::SharedMatrix multi_per_x =
                psi::linalg::doublet(x_bar_a_bar_P_sliced[x], x_bar_a_bar_P_sliced[x], true, false);
            for (int p = 0; p < P_xbar[x].size(); p++) {
                for (int q = 0; q < P_xbar[x].size(); q++) {
                    int row = P_xbar[x][p];
                    int col = P_xbar[x][q];
                    ZA_pq->add(row, col, multi_per_x->get(p, q));
                }
            }
        }

        /// Construct D_pq
        psi::SharedMatrix D_pq = psi::linalg::doublet(Z_pq, C_pq_, false, false);
        /// Construct DA_pq
        psi::SharedMatrix DA_pq = psi::linalg::doublet(ZA_pq, C_pq_, false, false);

        /// Compute Coulomb Energy
        for (int q = 0; q < nthree_; q++) {
            E_J_ -= 2 * D_pq->get_row(0, q)->vector_dot(*DA_pq->get_column(0, q));
        }

        /// Exchange part
        B_ia_Q.resize(nocc);
        B_xa_Q.resize(nact);
        Q_ia = std::make_shared<psi::Matrix>("Q_ia", nocc, nvir);
        Q_xa = std::make_shared<psi::Matrix>("Q_xa", nact, nvir);
        Q_ia->zero();
        Q_xa->zero();

        for (int i = 0; i < nocc; i++) {
            psi::SharedMatrix sliced_C_pq = submatrix_rows_and_cols(*C_pq_, P_ibar[i], P_ibar[i]);
            B_ia_Q[i] =
                psi::linalg::doublet(i_bar_a_bar_P_sliced[i], sliced_C_pq, false, false); /// a*p
            for (int abar = 0; abar < B_ia_Q[i]->rowdim(); abar++) {
                double Q_value_2 = B_ia_Q[i]->get_row(0, abar)->vector_dot(
                    *i_bar_a_bar_P_sliced[i]->get_row(0, abar));
                Q_ia->set(i, abar_ibar[i][abar], sqrt(Q_value_2));
            }
        }

        for (int x = 0; x < nact; x++) {
            psi::SharedMatrix sliced_C_pq_A = submatrix_rows_and_cols(*C_pq_, P_xbar[x], P_xbar[x]);
            B_xa_Q[x] = psi::linalg::doublet(x_bar_a_bar_P_sliced[x], sliced_C_pq_A, false, false);
            for (int abar = 0; abar < B_xa_Q[x]->rowdim(); abar++) {
                double Q_value_A_2 = B_xa_Q[x]->get_row(0, abar)->vector_dot(
                    *x_bar_a_bar_P_sliced[x]->get_row(0, abar));
                Q_xa->set(x, abar_xbar[x][abar], sqrt(Q_value_A_2));
            }
        }

        /// Start ix-prescreening.
        psi::SharedMatrix Aa_ix = psi::linalg::doublet(Q_ia, Q_xa, false, true); // (occ * act)
        for (int i = 0; i < nocc; i++) {
            for (int x = 0; x < nact; x++) {
                if (std::abs(Aa_ix->get(i, x)) >= theta_ij_sqrt_cavv_) {
                    vir_intersection_per_ix.clear();
                    aux_intersection_per_ix.clear();
                    std::set_intersection(P_ibar[i].begin(), P_ibar[i].end(), P_xbar[x].begin(),
                                          P_xbar[x].end(),
                                          std::back_inserter(aux_intersection_per_ix));
                    aux_in_B_i.clear();
                    std::set_intersection(abar_ibar[i].begin(), abar_ibar[i].end(),
                                          abar_xbar[x].begin(), abar_xbar[x].end(),
                                          std::back_inserter(vir_intersection_per_ix));
                    for (auto aux : aux_intersection_per_ix) {
                        int idx_aux_i =
                            binary_search_recursive(P_ibar[i], aux, 0, P_ibar[i].size() - 1);
                        aux_in_B_i.push_back(idx_aux_i);
                    }
                    for (int a_idx = 0; a_idx < vir_intersection_per_ix.size(); a_idx++) {
                        for (int b_idx = 0; b_idx <= a_idx; b_idx++) {
                            int a = vir_intersection_per_ix[a_idx];
                            int b = vir_intersection_per_ix[b_idx];
                            double Schwarz = Q_ia->get(i, a) * Q_xa->get(x, b) * Q_ia->get(i, b) *
                                             Q_xa->get(x, a);
                            if (std::abs(Schwarz) >= theta_schwarz_cavv_) {
                                int a_idx_i = binary_search_recursive(abar_ibar[i], a, 0,
                                                                      abar_ibar[i].size() - 1);
                                int b_idx_i = binary_search_recursive(abar_ibar[i], b, 0,
                                                                      abar_ibar[i].size() - 1);
                                // int a_idx_x = binary_search_recursive(abar_xbar[x], a, 0,
                                //                                       abar_xbar[x].size() - 1);
                                // int b_idx_x = binary_search_recursive(abar_xbar[x], b, 0,
                                //                                       abar_xbar[x].size() - 1);
                                std::vector<int> vec_a_i{a_idx_i};
                                std::vector<int> vec_b_i{b_idx_i};
                                std::vector<int> vec_a_x{a};
                                std::vector<int> vec_b_x{b};

                                psi::SharedMatrix ia =
                                    submatrix_rows_and_cols(*B_ia_Q[i], vec_a_i, aux_in_B_i);
                                psi::SharedMatrix xb = submatrix_rows_and_cols(
                                    *x_bar_a_bar_P[x], vec_b_x, aux_intersection_per_ix);
                                psi::SharedMatrix ib =
                                    submatrix_rows_and_cols(*B_ia_Q[i], vec_b_i, aux_in_B_i);
                                psi::SharedMatrix xa = submatrix_rows_and_cols(
                                    *x_bar_a_bar_P[x], vec_a_x, aux_intersection_per_ix);

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

    // std::cout << "cavv" << E_J_ << "\n";
    // std::cout << "cavv" << E_K_ << "\n";

    return (E_J_ + E_K_);
}

double LaplaceDSRG::compute_ccav() {
    E_J_ = 0.0;
    E_K_ = 0.0;
    AtomicOrbitalHelper ao_helper_ccav(Cwfn_, eps_rdocc_, eps_active_, eps_virtual_,
                                       laplace_threshold_, nactive_, nfrozen_, false, vir_tol_);

    int weights_ccav = ao_helper_ccav.Weights();
    vir_start_ccav_ = ao_helper_ccav.vir_start();
    ao_helper_ccav.Compute_Cholesky_Pseudo_Density();
    ao_helper_ccav.Compute_Cholesky_Active_Density(Eta1_mat_);
    Occupied_cholesky_ccav_ = ao_helper_ccav.LOcc_list();
    Active_cholesky_ccav_ = ao_helper_ccav.LAct_list();
    Virtual_cholesky_ccav_ = ao_helper_ccav.LVir_list();
    Occupied_cholesky_ccav_abs_.resize(weights_ccav);
    Active_cholesky_ccav_abs_.resize(weights_ccav);
    Virtual_cholesky_ccav_abs_.resize(weights_ccav);

    for (int nweight = 0; nweight < weights_ccav; nweight++) {
        Occupied_cholesky_ccav_abs_[nweight] = std::make_shared<psi::Matrix>(
            "LOcc_abs", nmo_, Occupied_cholesky_ccav_[nweight]->coldim());
        Active_cholesky_ccav_abs_[nweight] = std::make_shared<psi::Matrix>(
            "LAct_abs", nmo_, Active_cholesky_ccav_[nweight]->coldim());
        Virtual_cholesky_ccav_abs_[nweight] = std::make_shared<psi::Matrix>(
            "LVir_abs", nmo_, Virtual_cholesky_ccav_[nweight]->coldim());
        for (int u = 0; u < nmo_; u++) {
            for (int i = 0; i < Occupied_cholesky_ccav_[nweight]->coldim(); i++) {
                Occupied_cholesky_ccav_abs_[nweight]->set(
                    u, i, std::abs(Occupied_cholesky_ccav_[nweight]->get(u, i)));
            }
            for (int v = 0; v < Active_cholesky_ccav_[nweight]->coldim(); v++) {
                Active_cholesky_ccav_abs_[nweight]->set(
                    u, v, std::abs(Active_cholesky_ccav_[nweight]->get(u, v)));
            }
            for (int a = 0; a < Virtual_cholesky_ccav_[nweight]->coldim(); a++) {
                Virtual_cholesky_ccav_abs_[nweight]->set(
                    u, a, std::abs(Virtual_cholesky_ccav_[nweight]->get(u, a)));
            }
        }
    }

    std::vector<psi::SharedMatrix> T_ibar_i_list(weights_ccav);
    for (int i_weight = 0; i_weight < weights_ccav; i_weight++) {
        T_ibar_i_list[i_weight] = psi::linalg::triplet(Occupied_cholesky_ccav_[i_weight], S_,
                                                       Cholesky_Occ_, true, false, false);
    }

    // std::vector<psi::SharedMatrix> N_px_ccav_batch_list(weights_ccav);
    std::vector<SparseMap> xbar_p_up_ccav(weights_ccav, SparseMap(nthree_));
    std::vector<std::vector<psi::SharedMatrix>> P_xbar_u_ccav(
        weights_ccav, std::vector<psi::SharedMatrix>(nthree_));
    std::vector<SparseMap> xbar_u_p_for_xbar_up(weights_ccav, SparseMap(nthree_));
    std::vector<SparseMap> xbar_u_p(weights_ccav, SparseMap(nthree_));

    /// Below is the code for loading data from disk.
    int file_unit = PSIF_DFSCF_BJ;
    int n_func_pairs = (nmo_ + 1) * nmo_ / 2;
    int max_rows = 1;
    std::shared_ptr<PSIO> psio(new PSIO());
    psi::SharedMatrix loadAOtensor =
        std::make_shared<psi::Matrix>("DiskDF: Load (A|mn) from DF-SCF", max_rows, n_func_pairs);
    psio->open(file_unit, PSIO_OPEN_OLD);

    std::vector<psi::SharedMatrix> amn;
    std::vector<psi::SharedMatrix> amn_new;

    SparseMap ao_list_per_q; ///{u}_P
    ao_list_per_q.resize(nthree_);

    psi::SharedMatrix N_pu = std::make_shared<psi::Matrix>("N_pu", nthree_, nmo_);
    N_pu->zero();
    double* N_pu_p = N_pu->get_pointer();
    outfile->Printf("\n    Start loading");

    for (int Q = 0; Q < nthree_; Q += max_rows) {
        int naux = (nthree_ - Q <= max_rows ? nthree_ - Q : max_rows);
        psio_address addr = psio_get_address(PSIO_ZERO, (Q * (int)n_func_pairs) * sizeof(double));
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
                if (std::abs(max_per_uq) >= theta_NB_ccav_) {
                    ao_list_per_q[Q + q].push_back(u);
                }
                *N_pu_p = std::abs(*N_pu_p) > max_per_uq ? std::abs(*N_pu_p) : max_per_uq;
                *N_pu_batch_p =
                    std::abs(*N_pu_batch_p) > max_per_uq ? std::abs(*N_pu_batch_p) : max_per_uq;
                N_pu_p++;
                N_pu_batch_p++;
            }
            /// Slice Amn -> Amn_new
            amn_new[q] =
                submatrix_rows_and_cols(*amn[q], ao_list_per_q[Q + q], ao_list_per_q[Q + q]);
        }

        for (int nweight = 0; nweight < weights_ccav; nweight++) {
            psi::SharedMatrix N_px_ccav_batch =
                psi::linalg::doublet(N_pu_batch, Active_cholesky_ccav_abs_[nweight]);
            double* N_px_batch_p = N_px_ccav_batch->get_pointer();
            for (int q = 0; q < naux; q++) {
                for (int x = 0; x < Active_cholesky_ccav_abs_[nweight]->coldim(); x++) {
                    if (std::abs(*N_px_batch_p) >= theta_NB_ccav_) {
                        xbar_p_up_ccav[nweight][Q + q].push_back(x);
                    }
                    N_px_batch_p++;
                }
                psi::SharedMatrix Active_cholesky_new =
                    submatrix_rows_and_cols(*Active_cholesky_ccav_[nweight], ao_list_per_q[Q + q],
                                            xbar_p_up_ccav[nweight][Q + q]);
                P_xbar_u_ccav[nweight][Q + q] =
                    psi::linalg::doublet(Active_cholesky_new, amn_new[q], true, false);
                for (int inew = 0; inew < xbar_p_up_ccav[nweight][Q + q].size(); inew++) {
                    for (int u = 0; u < ao_list_per_q[Q + q].size(); u++) {
                        if (std::abs(P_xbar_u_ccav[nweight][Q + q]->get(inew, u)) >= theta_NB_ccav_) {
                            xbar_u_p_for_xbar_up[nweight][Q + q].push_back(inew);
                            xbar_u_p[nweight][Q + q].push_back(
                                xbar_p_up_ccav[nweight][Q + q][inew]);
                            break;
                        }
                    }
                }
                P_xbar_u_ccav[nweight][Q + q] = submatrix_rows(
                    *P_xbar_u_ccav[nweight][Q + q], xbar_u_p_for_xbar_up[nweight][Q + q]);
            }
        }
    }
    psio->close(file_unit, 1);
    loadAOtensor.reset();
    outfile->Printf("\n    End loading");

    //////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////

    /// Construct [ibar]_p and [abar]_p
    SparseMap i_bar_p_up(nthree_);
    SparseMap a_bar_p_up(nthree_);

    /// Construct (P|ibar u)
    std::vector<psi::SharedMatrix> P_ibar_u(nthree_);

    /// Construct (P|ibar abar)
    std::vector<psi::SharedMatrix> P_ibar_abar(nthree_);

    /// Construct (P|ibar xbar)
    std::vector<psi::SharedMatrix> P_ibar_xbar(nthree_);

    /// Construct {ibar}_p
    SparseMap ibar_p(nthree_);

    /// Construct {ibar_x}_p
    SparseMap ibar_x_p(nthree_);

    /// Construct {P}_ibar.  Obtain this by "inversion" of {ibar}_p
    SparseMap P_ibar;

    /// Construct {abar}_p
    //SparseMap abar_p(nthree_);

    /// Construct {p}_abar.  Obtain this by "inversion" of {abar}_p
    //SparseMap P_abar;

    /// Construct {p}_ibar_x
    SparseMap P_ibar_x;

    /// Construct (ibar abar|P)
    std::vector<psi::SharedMatrix> i_bar_a_bar_P;

    /// Construct (ibar xbar|P)
    std::vector<psi::SharedMatrix> i_bar_x_bar_P;

    /// Constrcut Sliced (ibar abar|P);
    std::vector<psi::SharedMatrix> i_bar_a_bar_P_sliced;

    /// Construct Sliced (ibar xbar|P)
    std::vector<psi::SharedMatrix> i_bar_x_bar_P_sliced;

    /// Construct {a_bar}_ibar
    SparseMap abar_ibar;

    /// Construct {x_bar}_ibar
    SparseMap xbar_ibar;

    /// Construct Z_pq
    psi::SharedMatrix Z_pq = std::make_shared<psi::Matrix>("Z_pq", nthree_, nthree_);

    /// Construct ZA_pq
    psi::SharedMatrix ZA_pq = std::make_shared<psi::Matrix>("ZA_pq", nthree_, nthree_);

    /// Construct B_ix_Q
    std::vector<psi::SharedMatrix> B_ix_Q;

    /// Construct B_ia_Q
    std::vector<psi::SharedMatrix> B_ia_Q;

    /// Construc Q_ia square;
    psi::SharedMatrix Q_ia;

    psi::SharedMatrix Q_ix;

    // std::vector<int> vir_intersection_per_ix;
    std::vector<int> aux_intersection_per_ijx;
    std::vector<int> aux_intersection_per_ixj;
    std::vector<int> aux_in_B_i;
    std::vector<int> aux_in_B_ix;

    for (int nweight = 0; nweight < weights_ccav; nweight++) {
        int nocc = Occupied_cholesky_ccav_[nweight]->coldim();
        int nvir = Virtual_cholesky_ccav_[nweight]->coldim();
        int nact = Active_cholesky_ccav_[nweight]->coldim();

        psi::SharedMatrix N_pi_bar =
            psi::linalg::doublet(N_pu, Occupied_cholesky_ccav_abs_[nweight], false, false);
        double* N_pi_bar_p = N_pi_bar->get_pointer();
        psi::SharedMatrix N_pa_bar =
            psi::linalg::doublet(N_pu, Virtual_cholesky_ccav_abs_[nweight], false, false);
        double* N_pa_bar_p = N_pa_bar->get_pointer();

        for (int qa = 0; qa < nthree_; qa++) {
            i_bar_p_up[qa].clear();
            a_bar_p_up[qa].clear();
            ibar_p[qa].clear();
            //abar_p[qa].clear();
            ibar_x_p[qa].clear();
            for (int ibar = 0; ibar < nocc; ibar++) {
                if (std::abs(*N_pi_bar_p) >= theta_NB_ccav_) {
                    i_bar_p_up[qa].push_back(ibar);
                }
                N_pi_bar_p++;
            }

            psi::SharedMatrix T_ibar_i_new =
                submatrix_rows_and_cols(*T_ibar_i_list[nweight], i_bar_p_up[qa], i_p_[qa]);
            P_ibar_u[qa] = psi::linalg::doublet(T_ibar_i_new, P_iu[qa], false, false);

            for (int abar = 0; abar < nvir; abar++) {
                if (std::abs(*N_pa_bar_p) >= theta_NB_ccav_) {
                    a_bar_p_up[qa].push_back(abar);
                }
                N_pa_bar_p++;
            }

            psi::SharedMatrix Pseudo_Vir_Mo = submatrix_rows_and_cols(
                *Virtual_cholesky_ccav_[nweight], ao_list_per_q[qa], a_bar_p_up[qa]);
            psi::SharedMatrix Pseudo_Occ_Mo = submatrix_rows_and_cols(
                *Occupied_cholesky_ccav_[nweight], ao_list_per_q[qa], i_bar_p_up[qa]);
            P_ibar_abar[qa] = psi::linalg::doublet(P_ibar_u[qa], Pseudo_Vir_Mo, false, false);
            P_ibar_xbar[qa] =
                psi::linalg::doublet(Pseudo_Occ_Mo, P_xbar_u_ccav[nweight][qa], true, true);

            /// Construct {ibar}_p.
            for (int i = 0; i < P_ibar_abar[qa]->rowdim(); i++) {
                for (int a = 0; a < P_ibar_abar[qa]->coldim(); a++) {
                    if (std::abs(P_ibar_abar[qa]->get(i, a)) >= theta_NB_IAP_ccav_) {
                        ibar_p[qa].push_back(i_bar_p_up[qa][i]);
                        break;
                    }
                }
            }

            /// Construct {ibar_x}_p.
            for (int i = 0; i < P_ibar_xbar[qa]->rowdim(); i++) {
                for (int x = 0; x < P_ibar_xbar[qa]->coldim(); x++) {
                    if (std::abs(P_ibar_xbar[qa]->get(i, x)) >= theta_NB_IAP_ccav_) {
                        ibar_x_p[qa].push_back(i_bar_p_up[qa][i]);
                        break;
                    }
                }
            }
        }
        /// Reorder. Construct (ibar abar|P). Use all ibar and all abar.
        /// Reorder. Construct (ibar xbar|P). Use all ibar and all xbar.
        i_bar_a_bar_P.resize(nocc);
        i_bar_a_bar_P_sliced.resize(nocc);
        i_bar_x_bar_P.resize(nocc);
        i_bar_x_bar_P_sliced.resize(nocc);

        for (int i = 0; i < nocc; i++) {
            i_bar_a_bar_P[i] = std::make_shared<psi::Matrix>("(abar * P)", nvir, nthree_);
            i_bar_a_bar_P[i]->zero();
            i_bar_x_bar_P[i] = std::make_shared<psi::Matrix>("(xbar * P)", nact, nthree_);
            i_bar_x_bar_P[i]->zero();
        }

        for (int q = 0; q < nthree_; q++) {
            for (int i = 0; i < P_ibar_abar[q]->rowdim(); i++) {
                for (int a = 0; a < P_ibar_abar[q]->coldim(); a++) {
                    i_bar_a_bar_P[i_bar_p_up[q][i]]->set(a_bar_p_up[q][a], q,
                                                         P_ibar_abar[q]->get(i, a));
                }
                for (int x = 0; x < P_ibar_xbar[q]->coldim(); x++) {
                    i_bar_x_bar_P[i_bar_p_up[q][i]]->set(xbar_u_p[nweight][q][x], q,
                                                         P_ibar_xbar[q]->get(i, x));
                }
            }
        }

        /// Construct {abar}_ibar.
        /// Construct {xbar}_ibar.
        abar_ibar.resize(nocc);
        xbar_ibar.resize(nocc);

        for (int i = 0; i < nocc; i++) {
            abar_ibar[i].clear();
            xbar_ibar[i].clear();
            for (int a = 0; a < nvir; a++) {
                for (int q = 0; q < nthree_; q++) {
                    if (std::abs(i_bar_a_bar_P[i]->get(a, q)) >= theta_NB_IAP_ccav_) {
                        abar_ibar[i].push_back(a);
                        break;
                    }
                }
            }
            for (int x = 0; x < nact; x++) {
                for (int q = 0; q < nthree_; q++) {
                    if (std::abs(i_bar_x_bar_P[i]->get(x, q)) >= theta_NB_IAP_ccav_) {
                        xbar_ibar[i].push_back(x);
                        break;
                    }
                }
            }
        }

        /// Coulomb contribution.
        /// Construct Z_pq
        P_ibar = invert_map(ibar_p, nocc);
        P_ibar_x = invert_map(ibar_x_p, nocc);
        // P_abar = invert_map(abar_p, nvir);
        Z_pq->zero();
        ZA_pq->zero();

        for (int i = 0; i < nocc; i++) {
            i_bar_a_bar_P_sliced[i] =
                submatrix_rows_and_cols(*i_bar_a_bar_P[i], abar_ibar[i], P_ibar[i]);
            psi::SharedMatrix multi_per_i =
                psi::linalg::doublet(i_bar_a_bar_P_sliced[i], i_bar_a_bar_P_sliced[i], true, false);
            for (int p = 0; p < P_ibar[i].size(); p++) {
                for (int q = 0; q < P_ibar[i].size(); q++) {
                    int row = P_ibar[i][p];
                    int col = P_ibar[i][q];
                    Z_pq->add(row, col, multi_per_i->get(p, q));
                }
            }
            i_bar_x_bar_P_sliced[i] = submatrix_rows_and_cols(*i_bar_x_bar_P[i], xbar_ibar[i], P_ibar_x[i]);
            psi::SharedMatrix multi_per_ix = psi::linalg::doublet(i_bar_x_bar_P_sliced[i], i_bar_x_bar_P_sliced[i], true, false);
            for (int px = 0; px < P_ibar_x[i].size(); px++) {
                for (int qx = 0; qx < P_ibar_x[i].size(); qx++) {
                    int row_x = P_ibar_x[i][px];
                    int col_x = P_ibar_x[i][qx];
                    ZA_pq->add(row_x, col_x, multi_per_ix->get(px, qx));
                }
            }
        }

        /// Construct D_pq
        psi::SharedMatrix D_pq = psi::linalg::doublet(Z_pq, C_pq_, false, false);
        /// Construct DA_pq
        psi::SharedMatrix DA_pq = psi::linalg::doublet(ZA_pq, C_pq_, false, false);

        /// Compute Coulomb Energy
        for (int q = 0; q < nthree_; q++) {
            E_J_ -= 2 * D_pq->get_row(0, q)->vector_dot(*DA_pq->get_column(0, q));
        }

        /// Exchange part
        B_ia_Q.resize(nocc);
        B_ix_Q.resize(nocc);
        Q_ia = std::make_shared<psi::Matrix>("Q_ia", nocc, nvir);
        Q_ix = std::make_shared<psi::Matrix>("Q_ix", nocc, nact);
        Q_ia->zero();
        Q_ix->zero();

        for (int i = 0; i < nocc; i++) {
            psi::SharedMatrix sliced_C_pq = submatrix_rows_and_cols(*C_pq_, P_ibar[i], P_ibar[i]);
            psi::SharedMatrix sliced_C_pq_A = submatrix_rows_and_cols(*C_pq_, P_ibar_x[i], P_ibar_x[i]);
            B_ia_Q[i] =
                psi::linalg::doublet(i_bar_a_bar_P_sliced[i], sliced_C_pq, false, false); /// a*p
            B_ix_Q[i] = psi::linalg::doublet(i_bar_x_bar_P_sliced[i], sliced_C_pq_A, false, false); /// x*p

            /// Cannot do (ia|ib) before screening. The reason is that P_ibar and P_ibar_x are different.

            for (int abar = 0; abar < B_ia_Q[i]->rowdim(); abar++) {
                double Q_value_2 = B_ia_Q[i]->get_row(0, abar)->vector_dot(
                    *i_bar_a_bar_P_sliced[i]->get_row(0, abar));
                Q_ia->set(i, abar_ibar[i][abar], sqrt(Q_value_2));
            }
            for (int xbar = 0; xbar < B_ix_Q[i]->rowdim(); xbar++) {
                double Q_value_A_2 = B_ix_Q[i]->get_row(0, xbar)->vector_dot(
                    *i_bar_x_bar_P_sliced[i]->get_row(0, xbar));
                Q_ix->set(i, xbar_ibar[i][xbar], sqrt(Q_value_A_2));
            }
        }

        /// Start ix-prescreening.
        psi::SharedMatrix Aa_ij = psi::linalg::doublet(Q_ix, Q_ix, false, true); 
        psi::SharedMatrix A_ij = psi::linalg::doublet(Q_ia, Q_ia, false, true);
        

        for (int i = 0; i < nocc; i++) {
            for (int j = 0; j <= i; j++) {
                if (std::abs(Aa_ij->get(i, j)) * std::abs(A_ij->get(i, j)) >= theta_ij_ccav_) {
                    aux_intersection_per_ijx.clear(); /// iajx
                    aux_intersection_per_ixj.clear(); /// ixja
                    std::set_intersection(P_ibar[i].begin(), P_ibar[i].end(), P_ibar_x[j].begin(),
                                          P_ibar_x[j].end(),
                                          std::back_inserter(aux_intersection_per_ijx));
                    std::set_intersection(P_ibar[j].begin(), P_ibar[j].end(), P_ibar_x[i].begin(),
                                          P_ibar_x[i].end(),
                                          std::back_inserter(aux_intersection_per_ixj));
                    aux_in_B_i.clear();
                    for (auto aux : aux_intersection_per_ijx) {
                        int idx_aux_i =
                            binary_search_recursive(P_ibar[i], aux, 0, P_ibar[i].size() - 1);
                        aux_in_B_i.push_back(idx_aux_i);
                    }
                    aux_in_B_ix.clear();
                    for (auto aux : aux_intersection_per_ixj) {
                        int idx_aux_i = binary_search_recursive(P_ibar_x[i], aux, 0, P_ibar_x[i].size() -1);
                        aux_in_B_ix.push_back(idx_aux_i);
                    }
                    for (int a_idx = 0; a_idx < abar_ibar[i].size(); a_idx++) {
                        for (int b_idx = 0; b_idx < xbar_ibar[i].size(); b_idx++) {
                            int a = abar_ibar[i][a_idx];
                            int b = xbar_ibar[i][b_idx];
                            double Schwarz = Q_ia->get(i, a) * Q_ix->get(j, b) * Q_ix->get(i, b) *
                                             Q_ia->get(j, a);
                            if (std::abs(Schwarz) >= theta_schwarz_ccav_) {
                                int a_idx_i = binary_search_recursive(abar_ibar[i], a, 0,
                                                                      abar_ibar[i].size() - 1);
                                int b_idx_i = binary_search_recursive(xbar_ibar[i], b, 0,
                                                                      xbar_ibar[i].size() - 1);
                                // int a_idx_x = binary_search_recursive(abar_xbar[x], a, 0,
                                //                                       abar_xbar[x].size() - 1);
                                // int b_idx_x = binary_search_recursive(abar_xbar[x], b, 0,
                                //                                       abar_xbar[x].size() - 1);
                                std::vector<int> vec_a_i{a_idx_i};
                                std::vector<int> vec_b_i{b_idx_i};
                                std::vector<int> vec_a_j{a};
                                std::vector<int> vec_b_j{b};

                                psi::SharedMatrix ia =
                                    submatrix_rows_and_cols(*B_ia_Q[i], vec_a_i, aux_in_B_i);
                                psi::SharedMatrix jb = submatrix_rows_and_cols(
                                    *i_bar_x_bar_P[j], vec_b_j, aux_intersection_per_ijx);
                                psi::SharedMatrix ib =
                                    submatrix_rows_and_cols(*B_ix_Q[i], vec_b_i, aux_in_B_ix);
                                psi::SharedMatrix ja = submatrix_rows_and_cols(
                                    *i_bar_a_bar_P[j], vec_a_j, aux_intersection_per_ixj);

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
    }

    // std::cout << "ccav" << E_J_ << "\n";
    // std::cout << "ccav" << E_K_ << "\n";

    return (E_J_ + E_K_);
}

} // namespace forte