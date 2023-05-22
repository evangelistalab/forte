/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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
#include "psi4/libqt/qt.h"

#include "helpers/printing.h"
#include "lbfgs.h"
#include "rosenbrock.h"
#include "casscf/casscf_orb_grad.h"
#include "casscf/cpscf.h"

using namespace psi;

namespace forte {

LBFGS::LBFGS(std::shared_ptr<LBFGS_PARAM> param) : param_(param), p_(psi::Vector(0)) {
    param_->check_param();
}

template <class Foo> double LBFGS::minimize(Foo& func, std::shared_ptr<psi::Vector> x) {
    nirrep_ = x->nirrep();
    dimpi_ = x->dimpi();

    g_ = std::make_shared<psi::Vector>("g", dimpi_);

    // compute the initial value and gradient
    double fx = func.evaluate(x, g_);
    double x_norm = x->norm();
    double g_norm = g_->norm();

    if (g_norm <= param_->epsilon * std::max(1.0, x_norm)) {
        converged_ = true;
        iter_ = 0;
        return fx;
    }

    // routine of diagonal Hessian
    auto compute_h0 = [&](std::shared_ptr<psi::Vector> x) {
        func.hess_diag(x, h0_);
        if (param_->print > 2) {
            print_h2("Diagonal Hessian at Iter. " + std::to_string(iter_));
            h0_->print();
        }
    };

    // initialize some vectors
    reset();
    p_ = psi::Vector("p", dimpi_);
    x_last_ = std::make_shared<psi::Vector>(*x);
    g_last_ = std::make_shared<psi::Vector>(*g_);
    if (param_->h0_freq >= 0) {
        h0_ = std::make_shared<psi::Vector>("h0", dimpi_);
        compute_h0(x);
    }

    // start iteration
    converged_ = false;
    do {
        // compute diagonal Hessian if needed
        if (iter_ != 0 and param_->h0_freq > 0) {
            if (iter_ % param_->h0_freq == 0) {
                compute_h0(x);
            }
        }

        // compute direction vector
        update();

        // determine step length
        double step = 1.0;
        next_step(func, x, fx, step);

        // print current iteration
        g_norm = g_->norm();
        if (param_->print > 0)
            outfile->Printf("\n    L-BFGS Iter:%3d; fx = %20.15f; g_norm = %12.6e; step = %9.3e",
                            iter_ + 1, fx, g_norm, step);

        // skip the rest if terminate
        if (++iter_ == param_->maxiter)
            break;

        // test convergence
        x_norm = x->norm();
        if (g_norm <= param_->epsilon * std::max(1.0, x_norm)) {
            converged_ = true;
            break;
        }

        // compute differences
        psi::Vector s(*x);
        s.subtract(*x_last_);

        psi::Vector y(*g_);
        y.subtract(*g_last_);

        double rho = y.vector_dot(s);

        // save history
        if (rho > 0) {
            int iter = iter_ - iter_shift_;
            int index = (iter - 1) % param_->m;

            if (iter <= param_->m) {
                s_[index] = std::make_shared<psi::Vector>("s", dimpi_);
                y_[index] = std::make_shared<psi::Vector>("y", dimpi_);
            }

            s_[index]->copy(s);
            y_[index]->copy(y);
            rho_[index] = 1.0 / rho;

            x_last_->copy(*x);
            g_last_->copy(*g_);
        } else {
            iter_shift_++;
            if (param_->print > 1) {
                outfile->Printf("\n  L-BFGS Warning: Skip this vector due to negative rho");
            }
        }
    } while (iter_ < param_->maxiter);

    if ((not converged_) and param_->print > 1) {
        outfile->Printf("\n  L-BFGS Warning: No convergence in %d iterations", iter_);
    }

    return fx;
}

void LBFGS::update() {
    p_.copy(*g_);

    int m = std::min(iter_ - iter_shift_, param_->m);
    int end = m ? (iter_ - iter_shift_ - 1) % m : 0; // skip for the very first iteration

    // first loop
    for (int k = 0; k < m; ++k) {
        int i = (end - k + m) % m;
        alpha_[i] = rho_[i] * s_[i]->vector_dot(p_);
        p_.axpy(-alpha_[i], *y_[i]);
    }

    // apply inverse diagonal Hessian
    apply_h0(p_);

    // second loop
    for (int k = 0; k < m; ++k) {
        int i = (end + k + 1) % m;
        double beta = rho_[i] * y_[i]->vector_dot(p_);
        p_.axpy(alpha_[i] - beta, *s_[i]);
    }

    // for descent
    p_.scale(-1.0);

    if (param_->print > 2)
        p_.print();
}

template <class Foo>
void LBFGS::next_step(Foo& foo, std::shared_ptr<psi::Vector> x, double& fx, double& step) {
    if (param_->step_length_method == LBFGS_PARAM::STEP_LENGTH_METHOD::MAX_CORRECTION) {
        scale_direction_vector(foo, x, fx, step);
    } else if (param_->step_length_method == LBFGS_PARAM::STEP_LENGTH_METHOD::LINE_BACKTRACKING) {
        line_search_backtracking(foo, x, fx, step);
    } else if (param_->step_length_method ==
               LBFGS_PARAM::STEP_LENGTH_METHOD::LINE_BRACKETING_ZOOM) {
        line_search_bracketing_zoom(foo, x, fx, step);
    } else {
        throw std::runtime_error("Unknown STEP_LENGTH_METHOD");
    }

    if (param_->print > 2)
        g_->print();
}

template <class Foo>
void LBFGS::scale_direction_vector(Foo& func, std::shared_ptr<psi::Vector> x, double& fx,
                                   double& step) {
    double p_max = 0.0;
    for (int h = 0; h < nirrep_; ++h) {
        for (int i = 0; i < dimpi_[h]; ++i) {
            double v = std::fabs(p_.get(h, i));
            if (v > p_max)
                p_max = v;
        }
    }
    step = (p_max > param_->max_dir) ? param_->max_dir / p_max : 1.0;

    x->axpy(step, p_);

    bool do_grad = iter_ - iter_shift_ + 1 < param_->maxiter;
    fx = func.evaluate(x, g_, do_grad);
}

template <class Foo>
void LBFGS::line_search_backtracking(Foo& func, std::shared_ptr<psi::Vector> x, double& fx,
                                     double& step) {
    double dg0 = g_->vector_dot(p_);
    double fx0 = fx;
    psi::Vector x0(*x);

    // need to restart because this is not a good direction
    if (dg0 >= 0) {
        outfile->Printf("\n  Warning: Direction increases the energy. Reset L-BFGS.");
        reset();
    }

    // backtracking for optimal step
    for (int i = 0; i < param_->maxiter_linesearch + 1; ++i) {
        x->copy(x0);
        x->axpy(step, p_);
        fx = func.evaluate(x, g_);

        if (i == param_->maxiter_linesearch)
            break;

        if (fx - fx0 > param_->c1 * dg0 * step) {
            step *= 0.5;
        } else {
            // Armijo condition is met here
            if (param_->line_search_condition == LBFGS_PARAM::LINE_SEARCH_CONDITION::ARMIJO)
                break;

            double dg = g_->vector_dot(p_);
            if (dg < param_->c2 * dg0) {
                step *= 2.05;
            } else {
                // Wolfe condition is met here
                if (param_->line_search_condition == LBFGS_PARAM::LINE_SEARCH_CONDITION::WOLFE)
                    break;

                if (std::fabs(dg) > -param_->c2 * dg0) {
                    step *= 0.5;
                } else {
                    // strong Wolfe condition is met here
                    break;
                }
            }
        }

        if (step > param_->max_step) {
            if (param_->print > 1)
                outfile->Printf("\n    Step length > max allowed value. Stopped line search.");
            step = param_->max_step;
            break;
        }

        if (step < param_->min_step) {
            if (param_->print > 1)
                outfile->Printf("\n    Step length < min allowed value. Stopped line search.");
            step = param_->min_step;
            break;
        }
    }
}

template <class Foo>
void LBFGS::line_search_bracketing_zoom(Foo& func, std::shared_ptr<psi::Vector> x, double& fx,
                                        double& step) {
    double dg0 = g_->vector_dot(p_);
    double fx0 = fx;
    psi::Vector x0(*x);

    // need to restart because this is not a good direction
    if (dg0 >= 0) {
        outfile->Printf("\n  Warning: Direction increases the energy. Reset L-BFGS.");
        reset();
    }

    // parameters Wolfe conditions
    double w1 = param_->c1 * dg0;
    double w2 = -param_->c2 * dg0;

    double fx_low = fx0; // fx_high = fx0;
    double step_low = 0.0, step_high = 0.0;

    // braketing stage
    for (int i = 0; i < param_->maxiter_linesearch; ++i) {
        x->copy(x0);
        x->axpy(step, p_);
        fx = func.evaluate(x, g_);

        if (fx - fx0 > w1 * step or (fx >= fx_low and i > 0)) {
            // fx_high = fx;
            step_high = step;
            break;
        }

        double dg = g_->vector_dot(p_);

        if (std::fabs(dg) <= w2) {
            if (param_->print > 2) {
                outfile->Printf("\n    Optimal step length from bracketing stage: %.15f", step);
            }
            return;
        }

        // fx_high = fx_low;
        step_high = step_low;

        fx_low = fx;
        step_low = step;

        if (dg >= 0)
            break;

        step *= 2.0;
    }
    if (param_->print > 2) {
        outfile->Printf("\n    Step lengths after bracketing stage: low = %.10f, high = %.10f",
                        step_low, step_high);
    }

    // zoom stage
    for (int i = 0; i < param_->maxiter_linesearch + 1; ++i) {
        step = 0.5 * (step_low + step_high);

        x->copy(x0);
        x->axpy(step, p_);
        fx = func.evaluate(x, g_);

        if (i == param_->maxiter_linesearch)
            break;

        if (fx - fx0 > w1 * step or fx >= fx_low) {
            step_high = step;
            // fx_high = fx;
        } else {
            double dg = g_->vector_dot(p_);

            if (std::fabs(dg) <= w2) {
                if (param_->print > 2) {
                    outfile->Printf("\n    Optimal step length from zooming stage: %.15f", step);
                }
                break;
            }

            if (dg * (step_high - step_low) >= 0) {
                step_high = step_low;
                // fx_high = fx_low;
            }

            step_low = step;
            fx_low = fx;
        }
    }
    if (param_->print > 2) {
        outfile->Printf("\n    Step lengths after zooming stage: low = %.10f, high = %.10f",
                        step_low, step_high);
    }

    if (step > param_->max_step) {
        if (param_->print > 1)
            outfile->Printf("\n    Step length > max allowed value. Use max allowed value.");
        step = param_->max_step;
    }

    if (step < param_->min_step) {
        if (param_->print > 1)
            outfile->Printf("\n    Step length < min allowed value. Use min allowed value.");
        step = param_->min_step;
    }
}

void LBFGS::apply_h0(psi::Vector& q) {
    if (param_->h0_freq < 0) {
        double gamma = compute_gamma();
        if (param_->print > 2)
            outfile->Printf("\n    gamma for H0: %.15f", gamma);
        q.scale(gamma);
    } else {
        for (int h = 0; h < nirrep_; ++h) {
            for (int i = 0; i < dimpi_[h]; ++i) {
                double vh = h0_->get(h, i);
                if (std::fabs(vh) > 1.0e-12) {
                    q.set(h, i, q.get(h, i) / vh);
                } else {
                    if (param_->print > 1) {
                        outfile->Printf("\n    Zero diagonal Hessian element (irrep: %d, i: %d)", h,
                                        i);
                    }
                }
            }
        }
    }
}

double LBFGS::compute_gamma() {
    double value = 1.0;
    if (iter_ - iter_shift_) {
        int end = (iter_ - iter_shift_ - 1) % (std::min(iter_ - iter_shift_, param_->m));
        value = s_[end]->vector_dot(*y_[end]) / y_[end]->vector_dot(*y_[end]);
    }
    return value;
}

void LBFGS::resize(int m) {
    y_ = std::vector<std::shared_ptr<psi::Vector>>(m, std::make_shared<psi::Vector>(0));
    s_ = std::vector<std::shared_ptr<psi::Vector>>(m, std::make_shared<psi::Vector>(0));
    alpha_.resize(m);
    rho_.resize(m);
}

void LBFGS::reset() {
    resize(param_->m);
    iter_ = 0;
    iter_shift_ = 0;
}

template double LBFGS::minimize(ROSENBROCK& func, std::shared_ptr<psi::Vector> x);
template double LBFGS::minimize(CASSCF_ORB_GRAD& func, std::shared_ptr<psi::Vector> x);
template double LBFGS::minimize(CPSCF& func, std::shared_ptr<psi::Vector> x);

} // namespace forte
