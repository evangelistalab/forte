#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsio/psio.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/sieve.h"
#include "psi4/lib3index/cholesky.h"
#include "psi4/libqt/qt.h"
#include "psi4/psifiles.h"
#include "psi4/lib3index/dftensor.h"

#include "helpers/timer.h"
#include "helpers/printing.h"
#include "helpers/memory.h"

#include "base_classes/forte_options.h"

#include "Laplace.h"

#include <tuple>
#include <algorithm>


using namespace psi;

namespace forte {

std::vector<int> merge_lists(const std::vector<int> &l1, const std::vector<int> &l2) {

    std::vector<int> l12;

    int i1 = 0, i2 = 0;
    while(i1 < l1.size() || i2 < l2.size()) {
        if(i1 == l1.size()) {
            l12.push_back(l2[i2]);
            ++i2;
        } else if(i2 == l2.size()) {
            l12.push_back(l1[i1]);
            ++i1;
        } else if(l1[i1] == l2[i2]) {
            l12.push_back(l1[i1]);
            ++i1;
            ++i2;
        } else if(l1[i1] < l2[i2]) {
            l12.push_back(l1[i1]);
            ++i1;
        } else {
            l12.push_back(l2[i2]);
            ++i2;
        }
    }

    return l12;

}

std::vector<int> contract_lists(const std::vector<int> &y, const std::vector<std::vector<int>> &A_to_y) {

    // TODO: runtime is proportional to A_to_y size (system size, O(N))
    // could maybe reduce to &y size (domain size, O(1)), probably doesn't matter
    std::vector<int> yA;

    for(int a = 0, y_ind = 0; a < A_to_y.size(); ++a) {

        bool is_a = false;
        for(auto y_val : A_to_y[a]) {
            if (y_ind < y.size() && y[y_ind] == y_val) {
                y_ind++;
                is_a = true;
            }
        }

        if(is_a) {
            for(auto y_val : A_to_y[a]) {
                yA.push_back(y_val);
            }
        }

    }

    return yA;

}

std::vector<int> block_list(const std::vector<int> &x_list, const std::vector<int> &x_to_y_map) {

    std::vector<int> y_list;

    for(int x_val : x_list) {
        int y_val = x_to_y_map[x_val];
        if(y_list.size() == 0) {
            y_list.push_back(y_val);
        } else if(y_list[y_list.size() - 1] != y_val) {
            y_list.push_back(y_val);
        }
    }

    return y_list;

}

std::vector<std::vector<int>> invert_map(const std::vector<std::vector<int>> &x_to_y, int ny) {

    int nx = x_to_y.size();
    std::vector<std::vector<int>> y_to_x(ny);

    for(int x = 0; x < nx; x++) {
        for(auto y : x_to_y[x]) {
            y_to_x[y].push_back(x);
        }
    }

    return y_to_x;

}

std::vector<std::vector<int>> chain_maps(const std::vector<std::vector<int>> &x_to_y, const std::vector<std::vector<int>> &y_to_z) {

    int nx = x_to_y.size();
    std::vector<std::vector<int>> x_to_z(nx);

    for(int x = 0; x < nx; x++) {
        for(auto y : x_to_y[x]) {
            for(auto z : y_to_z[y]) {
                x_to_z[x].push_back(z);
            }
        }
        std::sort(x_to_z[x].begin(), x_to_z[x].end());
        x_to_z[x].erase(std::unique(x_to_z[x].begin(), x_to_z[x].end()), x_to_z[x].end());
    }

    return x_to_z;

}

std::vector<std::vector<int>> extend_maps(const std::vector<std::vector<int>> &x_to_y, const std::vector<std::pair<int,int>> &xpairs) {

    int nx = x_to_y.size();
    std::vector<std::vector<int>> xext_to_y(nx);

    for(auto xpair : xpairs) {
        size_t x1, x2;
        std::tie(x1,x2) = xpair;
        xext_to_y[x1] = merge_lists(xext_to_y[x1], x_to_y[x2]);
    }

    return xext_to_y;

}

psi::SharedMatrix submatrix_rows(const psi::Matrix &mat, const std::vector<int> &row_inds) {

    psi::SharedMatrix mat_new = std::make_shared<psi::Matrix>(mat.name(), row_inds.size(), mat.colspi(0));
    double** mat_newp = mat_new->pointer();
    double** matp = mat.pointer();
    for(int r_new = 0; r_new < row_inds.size(); r_new++) {
        int r_old = row_inds[r_new];
        ::memcpy(&mat_newp[r_new][0], &matp[r_old][0], sizeof(double) * mat.colspi(0));
    }
    return mat_new;
}

psi::SharedMatrix submatrix_cols(const psi::Matrix &mat, const std::vector<int> &col_inds) {

    psi::SharedMatrix mat_new = std::make_shared<psi::Matrix>(mat.name(), mat.rowspi(0), col_inds.size());
    double** mat_newp = mat_new->pointer();
    double** matp = mat.pointer();
    for(int c_new = 0; c_new < col_inds.size(); c_new++) {
        int c_old = col_inds[c_new];
        C_DCOPY(mat.rowspi(0), &matp[0][c_old], mat.colspi(0), &mat_newp[0][c_new], col_inds.size());
    }
    return mat_new;
}

psi::SharedMatrix submatrix_rows_and_cols(const psi::Matrix &mat, const std::vector<int> &row_inds, const std::vector<int> &col_inds) {

    psi::SharedMatrix mat_new = std::make_shared<psi::Matrix>(mat.name(), row_inds.size(), col_inds.size());
    for(int r_new = 0; r_new < row_inds.size(); r_new++) {
        int r_old = row_inds[r_new];
        for(int c_new = 0; c_new < col_inds.size(); c_new++) {
            int c_old = col_inds[c_new];
            mat_new->set(r_new, c_new, mat.get(r_old, c_old));
        }
    }
    return mat_new;
}

psi::SharedMatrix load_Amn(const size_t A, const size_t mn) {
    std::shared_ptr<PSIO> psio(new PSIO());
    auto Amn = std::make_shared<psi::Matrix>("Load (A|mn) from DF-SCF", A, mn);
    outfile->Printf("\t Will attempt to load (A|mn) from file %d.\n\n", PSIF_DFSCF_BJ);
    double** Amnp = Amn->pointer();
    int file_unit = PSIF_DFSCF_BJ;
    psio->open(file_unit, PSIO_OPEN_OLD);
    psio->read_entry(file_unit, "ERFC Integrals", (char*)Amnp[0], sizeof(double) * A * mn);
    psio->close(file_unit, 1);
    return Amn;
}

psi::SharedMatrix load_Jinv_full(const size_t P, const size_t Q) {
    std::shared_ptr<PSIO> psio(new PSIO());
    auto Jinv_full = std::make_shared<psi::Matrix>("Load full inverse metric from DF-SCF", P, Q);
    outfile->Printf("\t Will attempt to load full inverse metric from file %d.\n\n", PSIF_DFSCF_BJ);
    double** Jinv_fullp = Jinv_full->pointer();
    int file_unit = PSIF_DFSCF_BJ;
    psio->open(file_unit, PSIO_OPEN_OLD);
    psio->read_entry(file_unit, "Jinv_full", (char*)Jinv_fullp[0], sizeof(double) * P * Q);
    psio->close(file_unit, 1);
    return Jinv_full;
}

psi::SharedMatrix initialize_erfc_integral(double Omega, int n_func_pairs, std::shared_ptr<ForteIntegrals> ints_forte) {
    std::shared_ptr<psi::BasisSet> zero = psi::BasisSet::zero_ao_basis_set();
    std::shared_ptr<psi::BasisSet> primary = ints_forte->wfn()->basisset();
    std::shared_ptr<psi::BasisSet> auxiliary = ints_forte->wfn()->get_basisset("DF_BASIS_MP2");

    std::shared_ptr<psi::IntegralFactory> integral = std::make_shared<IntegralFactory>(auxiliary,zero,primary,primary);
    std::shared_ptr<psi::TwoBodyAOInt> ints(integral->erf_complement_eri(Omega));
    ///std::shared_ptr<psi::TwoBodyAOInt> ints(integral->eri());

    int nthree = auxiliary->nbf();
    int nbf = primary->nbf();

    auto I = std::make_shared<psi::Matrix>("erfc integral", nthree, n_func_pairs);
    double **Ip = I->pointer();

    int numP, Pshell, MU, NU, P, PHI, mu, nu, nummu, numnu, omu, onu;

    for (MU = 0; MU < primary->nshell(); ++MU) {
        nummu = primary->shell(MU).nfunction();
        for (NU = 0; NU <= MU; ++NU) {
            numnu = primary->shell(NU).nfunction();
            for (Pshell = 0; Pshell < auxiliary->nshell(); ++Pshell) {
                numP = auxiliary->shell(Pshell).nfunction();
                ints->compute_shell(Pshell, 0, MU, NU);
                const double *buffer = ints->buffer();
                for (mu = 0; mu < nummu; ++mu) {
                    omu = primary->shell(MU).function_index() + mu;
                    for (nu = 0; nu < numnu; ++nu) {
                        onu = primary->shell(NU).function_index() + nu;
                        size_t addr = omu > onu ? omu * (omu + 1) / 2 + onu :
                                                      onu * (onu + 1) / 2 + omu;
                        for (P = 0; P < numP; ++P) {
                            PHI = auxiliary->shell(Pshell).function_index() + P;
                            Ip[PHI][addr] = buffer[P * nummu * numnu + mu * numnu + nu];
                        }
                    }
                }
            }
        }
    }
    return I;
}

psi::SharedMatrix erfc_metric (double Omega, std::shared_ptr<ForteIntegrals> ints_forte) {
    std::shared_ptr<psi::BasisSet> auxiliary = ints_forte->wfn()->get_basisset("DF_BASIS_MP2");
    auto Jinv = std::make_shared<psi::FittingMetric>(auxiliary, true);
    Jinv->form_full_eig_inverse_erfc(Omega, 1E-12);
    ///Jinv->form_full_eig_inverse(1E-12);
    psi::SharedMatrix Jinv_metric = Jinv->get_metric();
    return Jinv_metric;
}

int binary_search_recursive(std::vector<int> A, int key, int low, int high) {
    if (low > high) {
        return -1;
    }

    int mid = low + ((high - low) / 2);
    if (A[mid] == key) {
        return mid;
    } else if (key < A[mid]) {
        return binary_search_recursive(A, key, low, mid - 1);
    }

    return binary_search_recursive(A, key, mid + 1, high);
}

psi::SharedMatrix ambit_to_matrix(ambit::Tensor t) {
    size_t size1 = t.dim(0);
    size_t size2 = t.dim(1);
    auto M = std::make_shared<psi::Matrix>("M", size1, size2);
    t.iterate([&](const std::vector<size_t>& i, double& value) { M->set(i[0], i[1], value); });
    return M;
}

} // namespace forte