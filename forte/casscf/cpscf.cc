/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include "helpers/printing.h"
#include "helpers/lbfgs/lbfgs.h"
#include "helpers/lbfgs/lbfgs_param.h"

#include "cpscf.h"

namespace forte {

CPSCF_SOLVER::CPSCF_SOLVER(std::shared_ptr<ForteOptions> options, std::shared_ptr<psi::JK> JK,
                           psi::SharedMatrix C, psi::SharedMatrix b, psi::SharedVector edocc,
                           psi::SharedVector euocc)
    : options_(options), cpscf_(JK, C, b, edocc, euocc) {}

bool CPSCF_SOLVER::solve() {
    // recast Ax = b to minimization of 0.5 * x^T A x - b^T x

    auto lbfgs_param = std::make_shared<LBFGS_PARAM>();
    lbfgs_param->epsilon = options_->get_double("CPSCF_CONVERGENCE");
    lbfgs_param->maxiter = options_->get_int("CPSCF_MAXITER");
    lbfgs_param->print = options_->get_int("PRINT");
    lbfgs_param->max_dir = 0.2;
    lbfgs_param->step_length_method = LBFGS_PARAM::STEP_LENGTH_METHOD::MAX_CORRECTION;

    LBFGS lbfgs(lbfgs_param);

    print_h2("Solving CP-SCF Equation");

    x_ = std::make_shared<psi::Vector>("CPSCF x", cpscf_.vdimpi());
    lbfgs.minimize(cpscf_, x_);

    return lbfgs.converged();
}

psi::SharedMatrix CPSCF_SOLVER::x() { return cpscf_.vec_to_mat(x_); }

CPSCF::CPSCF(std::shared_ptr<psi::JK> JK, psi::SharedMatrix C, psi::SharedMatrix b,
             psi::SharedVector edocc, psi::SharedVector euocc)
    : JK_(JK), C_(C), edocc_(edocc), euocc_(euocc) {

    // set up basic stuff
    nirrep_ = C->nirrep();

    if (b->nirrep() != nirrep_)
        throw std::runtime_error("Inconsistent nirrep for b vector");
    if (edocc->nirrep() != nirrep_ or euocc->nirrep() != nirrep_)
        throw std::runtime_error("Inconsistent nirrep for orbital energies vectors");

    ndoccpi_ = edocc->dimpi();
    nuoccpi_ = euocc->dimpi();

    if (ndoccpi_ + nuoccpi_ != C->colspi())
        throw std::runtime_error("Inconsistent number of MOs in C and input orbital energies");

    Cdocc_ = std::make_shared<psi::Matrix>("C_docc (CPSCF)", C->rowspi(), ndoccpi_);
    Cuocc_ = std::make_shared<psi::Matrix>("C_uocc (CPSCF)", C->rowspi(), nuoccpi_);
    for (int h = 0; h < nirrep_; ++h) {
        for (int i = 0; i < ndoccpi_[h]; ++i) {
            Cdocc_->set_column(h, i, C->get_column(h, i));
        }
        for (int a = 0; a < nuoccpi_[h]; ++a) {
            Cuocc_->set_column(h, a, C->get_column(h, a + ndoccpi_[h]));
        }
    }

    std::vector<int> dims;
    for (int h = 0; h < nirrep_; ++h) {
        dims.push_back(ndoccpi_[h] * nuoccpi_[h]);
    }
    vdims_ = psi::Dimension(dims);

    b_ = mat_to_vec(b);
}

double CPSCF::evaluate(psi::SharedVector x, psi::SharedVector g, bool do_g) {
    auto X = vec_to_mat(x);

    // contract X with Roothaan-Bagus supermatrix
    // AX_{ai} = [4 * (ai|bj) - (ab|ji) - (aj|bi)] * X_{bj}
    auto Cdressed = psi::linalg::doublet(Cuocc_, X, false, false);

    JK_->set_do_K(true);
    std::vector<std::shared_ptr<psi::Matrix>>& Cls = JK_->C_left();
    std::vector<std::shared_ptr<psi::Matrix>>& Crs = JK_->C_right();
    Cls.clear();
    Crs.clear();
    Cls.push_back(Cdressed);
    Crs.push_back(Cdocc_);
    JK_->compute();

    auto J = JK_->J()[0];
    J->scale(4.0);
    J->subtract(JK_->K()[0]);
    J->subtract((JK_->K()[0])->transpose());

    auto AX = psi::linalg::triplet(Cuocc_, J, Cdocc_, true, false, false);

    // add contribution of orbital energy difference: AX_{ai} += (e_a - e_i) * X_{ai}
    for (int h = 0; h < nirrep_; ++h) {
        for (int a = 0; a < nuoccpi_[h]; ++a) {
            for (int i = 0; i < ndoccpi_[h]; ++i) {
                double value = (euocc_->get(h, a) - edocc_->get(h, i)) * X->get(h, a, i);
                AX->add(h, a, i, value);
            }
        }
    }

    // reshape Ax to vector
    auto ax = mat_to_vec(AX);

    // compute scalar: 0.5 * x^T A x - b^T x
    double fx = 0.5 * x->vector_dot(ax) - b_->vector_dot(x);

    if (do_g) {
        g->copy(*ax);
        g->subtract(b_);
    }

    return fx;
}

void CPSCF::hess_diag(psi::SharedVector, psi::SharedVector h0) {
    // just use orbital energy difference
    for (int h = 0; h < nirrep_; ++h) {
        for (int a = 0; a < nuoccpi_[h]; ++a) {
            for (int i = 0; i < ndoccpi_[h]; ++i) {
                double value = euocc_->get(h, a) - edocc_->get(h, i);
                h0->set(h, a * ndoccpi_[h] + i, value);
            }
        }
    }
}

void CPSCF::test_vec_dim(psi::SharedVector v) {
    if (v->dimpi() != vdims_) {
        psi::outfile->Printf("\n  Expected dimension:    ");
        vdims_.print();
        psi::outfile->Printf("\n  Input vector dimension:");
        (v->dimpi()).print();
        throw std::runtime_error("Invalid dimension of vector");
    }
}

void CPSCF::test_mat_dim(psi::SharedMatrix M) {
    if (M->rowspi() != nuoccpi_ or M->colspi() != ndoccpi_) {
        psi::outfile->Printf("\n  Expected dimension:\n");
        psi::outfile->Printf("    row:    ");
        nuoccpi_.print();
        psi::outfile->Printf("    colulmn:");
        ndoccpi_.print();
        psi::outfile->Printf("\n  Input matrix dimension:\n");
        psi::outfile->Printf("    row:    ");
        (M->rowspi()).print();
        psi::outfile->Printf("    colulmn:");
        (M->colspi()).print();
        throw std::runtime_error("Matrix dimension is not virtual by occupied!");
    }
}

psi::SharedVector CPSCF::mat_to_vec(psi::SharedMatrix M) {
    test_mat_dim(M);

    auto v = std::make_shared<psi::Vector>("V", vdims_);

    for (int h = 0; h < nirrep_; ++h) {
        for (int a = 0; a < nuoccpi_[h]; ++a) {
            for (int i = 0; i < ndoccpi_[h]; ++i) {
                v->set(h, a * ndoccpi_[h] + i, M->get(h, a, i));
            }
        }
    }

    return v;
}

psi::SharedMatrix CPSCF::vec_to_mat(psi::SharedVector v) {
    test_vec_dim(v);

    auto M = std::make_shared<psi::Matrix>("M", nuoccpi_, ndoccpi_);

    for (int h = 0; h < nirrep_; ++h) {
        for (int a = 0; a < nuoccpi_[h]; ++a) {
            for (int i = 0; i < ndoccpi_[h]; ++i) {
                M->set(h, a, i, v->get(h, a * ndoccpi_[h] + i));
            }
        }
    }

    return M;
}

} // namespace forte
