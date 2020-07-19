/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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
#include "psi4/libmints/vector.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include "lbfgs.h"

using namespace psi;

namespace forte {

LBFGS::LBFGS(int dim, int m, bool descent) : dim_(dim), m_(m), descent_(descent), counter_(0) {
    resize(m);

    h0_ = std::make_shared<psi::Vector>("Inverse diagonal Hessian", dim);
    guess_h0();
}

void LBFGS::guess_h0() {
    double v = gamma();
    for (int i = 0; i < dim_; ++i) {
        h0_->set(i, v);
    }
}

double LBFGS::gamma() {
    double value = 1.0;
    if (counter_) {
        int end = (counter_ - 1) % (std::min(counter_, m_));
        value = s_[end]->vector_dot(y_[end]) / y_[end]->norm();
    }
    return value;
}

psi::SharedVector LBFGS::compute_correction(psi::SharedVector x, psi::SharedVector g) {
    check_dim(h0_, x, "Invalid solution vector. Check dimension!");
    check_dim(h0_, g, "Invalid gradient vector. Check dimension!");

    auto z = std::make_shared<psi::Vector>(*g);

    if (counter_) {
        int m = std::min(counter_, m_);
        int end = (counter_ - 1) % m;

        // update differences vectors
        s_[end]->copy(*x);
        s_[end]->subtract(x_last_);
        s_[end]->set_name("s_" + std::to_string(end));

        y_[end]->copy(*g);
        y_[end]->subtract(g_last_);
        y_[end]->set_name("y_" + std::to_string(end));

        rho_[end] = 1.0 / y_[end]->vector_dot(s_[end]);

        auto temp = std::make_shared<psi::Vector>();

        // first loop
        for (int i = 0; i < m; ++i) {
            int k = (end - i + m) % m;

            alpha_[k] = rho_[k] * s_[k]->vector_dot(z);

            temp->copy(*(y_[k]));
            temp->scale(alpha_[k]);
            z->subtract(temp);
        }

        // Scale by initial Hessian
        if (hess_auto_) {
            guess_h0();
        }
        apply_h0(z);

        // second loop
        for (int i = 0; i < m; ++i) {
            const int k = (end + i + 1) % m;

            const double beta = rho_[k] * y_[k]->vector_dot(z);

            temp->copy(*(s_[k]));
            temp->scale(alpha_[k] - beta);
            z->add(temp);
        }
    } else {
        apply_h0(z);
    }

    if (descent_) {
        z->scale(-1.0);
    }

    // save current gradient
    g_last_->copy(*g);
    x_last_->copy(*x);

    counter_ += 1;

    return z;
}

void LBFGS::apply_h0(psi::SharedVector q) {
    for (int i = 0; i < dim_; ++i) {
        double v = q->get(i) * h0_->get(i);
        q->set(i, v);
    }
}

void LBFGS::set_size(int m) {
    if (m <= 0) {
        throw std::runtime_error("Invalid m value for L-BFGS.");
    }
    m_ = m;
    resize(m);
}

void LBFGS::resize(int m) {
    y_.resize(m);
    s_.resize(m);
    alpha_.resize(m);
    rho_.resize(m);
}

void LBFGS::set_hess_diag(psi::SharedVector hess_diag) {
    check_dim(h0_, hess_diag, "Invalid diagonal Hessian. Check dimension!");

    double g = gamma();
    for (int i = 0; i < dim_; ++i) {
        double v = 1.0 / hess_diag->get(i);
        if (v <= 0.0) {
            outfile->Printf("\n  Warning: negative inverse hess_diag[%d]: %.15f -> %.15f", i, v, g);
            v = g;
        }
        h0_->set(i, v);
    }
}

void LBFGS::check_dim(psi::SharedVector a, psi::SharedVector b, const std::string error_msg) {
    if (a->dimpi() != b->dimpi()) {
        throw std::runtime_error(error_msg);
    }
}

void LBFGS::reset() {
    resize(0);
}

} // namespace forte
