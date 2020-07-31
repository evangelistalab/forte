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

#include <cmath>

#include "psi4/psi4-dec.h"
#include "psi4/libmints/vector.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include "lbfgs.h"

using namespace psi;

namespace forte {

LBFGS::LBFGS(int dim, int m, bool descent) : dim_(dim), m_(m), descent_(descent), counter_(0) {
    resize(m);

    x_last_ = std::make_shared<psi::Vector>("Last Solution", dim);
    g_last_ = std::make_shared<psi::Vector>("Last Gradient", dim);

    h0_ = std::make_shared<psi::Vector>("Inverse Diagonal Hessian", dim);
    guess_h0();
}

LBFGS::LBFGS(const LBFGS_PARAM& param) : param_(param) { param_.check_param(); }

template <class Foo> double LBFGS::minimize(Foo& func, psi::SharedVector x) {
    nirrep_ = x->nirrep();
    dimpi_ = x->dimpi();

    g_ = std::make_shared<psi::Vector>("g", dimpi_);

    // compute the initial value and gradient
    double fx = func.evaluate(x, g_);
    double x_norm = x->norm();
    double g_norm = g_->norm();

    if (g_norm <= param_.epsilon * std::max(1.0, x_norm)) {
        return fx;
    }

    // initialize some vectors
    resize(param_.m);
    p_ = std::make_shared<psi::Vector>("p", dimpi_);
    x_last_ = std::make_shared<psi::Vector>(*x);
    g_last_ = std::make_shared<psi::Vector>(*g_);

    // use user defined initial Hessian
    if (not param_.auto_hess) {
        h0_ = std::make_shared<psi::Vector>("h0", dimpi_);
        func.hess_diag(h0_);
    }

    iter_ = 0;

    do {
        // compute direction vector
        update();
        double step = iter_ ? 1.0 : 1.0 / p_->norm();

        // line search to determine step length
        line_search(func, x, fx, step);

        // test convergence
        x_norm = x->norm();
        g_norm = g_->norm();
        if (g_norm <= param_.epsilon * std::max(1.0, x_norm)) {
            break;
        }

        // save history
        int index = iter_ % param_.m;

        if (iter_ < param_.m) {
            s_[index] = std::make_shared<psi::Vector>("s", dimpi_);
            y_[index] = std::make_shared<psi::Vector>("y", dimpi_);
        }

        s_[index]->copy(*x);
        s_[index]->subtract(x_last_);

        y_[index]->copy(*g_);
        y_[index]->subtract(g_last_);

        rho_[index] = 1.0 / y_[index]->vector_dot(s_[index]);

        x_last_->copy(*x);
        g_last_->copy(*g_);

        iter_ += 1;
    } while (iter_ < param_.maxiter);

    if (iter_ == param_.maxiter) {
        outfile->Printf("\n  Warning: L-BFGS did not converge in %d iterations", iter_);
    }

    return fx;
}

void LBFGS::update() {
    p_->copy(*g_);

    int m = std::min(iter_, param_.m);
    int end = (iter_ - 1) % m;

    // first loop
    for (int k = 0; k < m; ++k) {
        int i = (end - k + m) % m;
        alpha_[i] = rho_[i] * s_[i]->vector_dot(p_);
        p_->axpy(-alpha_[i], y_[i]);
    }

    // apply inverse diagonal Hessian
    apply_h0_new(p_);

    // second loop
    for (int k = 0; k < m; ++k) {
        int i = (end + k + 1) % m;
        double beta = rho_[i] * y_[i]->vector_dot(p_);
        p_->axpy(alpha_[i] - beta, s_[i]);
    }

    // for descent
    p_->scale(-1.0);
}

template <class Foo>
void LBFGS::line_search(Foo& func, psi::SharedVector x, double& fx, double& step) {
    double dg0 = g_->vector_dot(p_);
    double fx0 = fx;
    auto x0 = std::make_shared<psi::Vector>(*x);

    // need to restart because this is not a good direction
    if (dg0 >= 0) {
        outfile->Printf("\n  Warning: Direction increases the energy. Reset L-BFGS.");
        resize(param_.m);
        iter_ = 0;
    }

    // parameters Wolfe conditions
    double w1 = param_.c1 * dg0;
    double w2 = -param_.c2 * dg0;

    double fx_low = fx0, fx_high = fx0;
    double step_low = 0.0, step_high = 0.0;

    // braketing stage
    for (int i = 0; i < param_.maxiter_linesearch; ++i) {
        x->copy(*x0);
        x->axpy(step, p_);
        fx = func.evaluate(x, g_);

        if (fx - fx0 > w1 * step or (fx >= fx_low and i > 0)) {
            fx_high = fx;
            step_high = step;
            break;
        }

        double dg = g_->vector_dot(p_);

        if (std::fabs(dg) <= w2)
            return;

        fx_high = fx_low;
        step_high = step_low;

        fx_low = fx;
        step_low = step;

        if (dg >= 0)
            break;

        step *= 2.0;
    }
    if (param_.print > 1) {
        outfile->Printf("\n  Step lengths after bracketing stage: low = %.10f, high = %.10f",
                        step_low, step_high);
    }

    // zoom stage
    for (int i = 0; i < param_.maxiter_linesearch; ++i) {
        step = 0.5 * (step_low + step_high);

        x->copy(*x0);
        x->axpy(step, p_);
        fx = func.evaluate(x, g_);

        if (fx - fx0 > w1 * step or fx >= fx_low) {
            step_high = step;
            fx_high = fx;
        } else {
            double dg = g_->vector_dot(p_);

            if (std::fabs(dg) <= w2)
                return;

            if (dg * (step_high - step_low) >= 0) {
                step_high = step_low;
                fx_high = fx_low;
            }

            step_low = step;
            fx_low = fx;
        }
    }
    if (param_.print > 1) {
        outfile->Printf("\n  Step lengths after zooming stage: low = %.10f, high = %.10f", step_low,
                        step_high);
    }
}

void LBFGS::apply_h0_new(psi::SharedVector q) {
    if (param_.auto_hess) {
        double gamma = gamma_new();
        for (int h = 0; h < nirrep_; ++h) {
            for (int i = 0; i < dimpi_[h]; ++i) {
                q->set(h, i, q->get(h, i) * gamma);
            }
        }
    } else {
        for (int h = 0; h < nirrep_; ++h) {
            for (int i = 0; i < dimpi_[h]; ++i) {
                q->set(h, i, q->get(h, i) / h0_->get(h, i));
            }
        }
    }
}

double LBFGS::gamma_new() {
    double value = 1.0;
    if (iter_) {
        int end = (iter_ - 1) % (std::min(iter_, m_));
        value = s_[end]->vector_dot(y_[end]) / y_[end]->vector_dot(y_[end]);
    }
    return value;
}

void LBFGS::guess_h0_new() {
    double v = gamma_new();
    for (int h = 0; h < nirrep_; ++h) {
        for (int i = 0; i < dimpi_[h]; ++i) {
            h0_->set(h, i, v);
        }
    }
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
    z->set_name("Direction Vector");

    if (counter_) {
        int m = std::min(counter_, m_);
        int end = (counter_ - 1) % m;

        outfile->Printf("\n  m: %d, end: %d", m, end);

        // update differences vectors
        if (counter_ < m_) {
            s_[end] = std::make_shared<psi::Vector>(dim_);
            s_[end]->set_name("s_" + std::to_string(end));

            y_[end] = std::make_shared<psi::Vector>(dim_);
            y_[end]->set_name("y_" + std::to_string(end));
        }

        s_[end]->copy(*x);
        s_[end]->subtract(x_last_);
        s_[end]->print();

        y_[end]->copy(*g);
        y_[end]->subtract(g_last_);
        y_[end]->print();

        rho_[end] = 1.0 / y_[end]->vector_dot(s_[end]);

        auto temp = std::make_shared<psi::Vector>();

        // first loop
        for (int i = 0; i < m; ++i) {
            const int k = (end - i + m) % m;

            alpha_[k] = rho_[k] * s_[k]->vector_dot(z);

            temp->copy(*(y_[k]));
            temp->scale(alpha_[k]);
            z->subtract(temp);
        }

        // Scale by initial Hessian
        //        if (param_.auto_hess) {
        //            guess_h0();
        //        }
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

    outfile->Printf("\n gradient projected on direction: %.15f", z->vector_dot(g));

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
    y_ = std::vector<psi::SharedVector>(m, std::make_shared<psi::Vector>());
    s_ = std::vector<psi::SharedVector>(m, std::make_shared<psi::Vector>());
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
    resize(m_);
    counter_ = 0;
}

} // namespace forte
