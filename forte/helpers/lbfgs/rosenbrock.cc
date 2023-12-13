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

#include "psi4/psi4-dec.h"
#include "psi4/libmints/vector.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "helpers/lbfgs/lbfgs_param.h"
#include "helpers/lbfgs/lbfgs.h"

#include "rosenbrock.h"

namespace forte {

ROSENBROCK::ROSENBROCK(int n) : n_(n) {
    if (n % 2 == 1) {
        std::runtime_error("Invalid size for Rosenbrock. Please use even number.");
    }
}

double ROSENBROCK::evaluate(std::shared_ptr<psi::Vector> x, std::shared_ptr<psi::Vector> g, bool) {
    check_dim(x);
    check_dim(g);

    // f(x) = sum_{i=0}^{n/2 - 1} [ 100 * (x_{2i}^2 - x_{2i + 1})^2 + (x_{2i} - 1)^2 ]
    double fx = 0.0;

    for (int i = 0; i < n_; i += 2) {
        double xi = x->get(i);
        double t1 = xi - 1.0;
        double t2 = 10 * (xi * xi - x->get(i + 1));
        g->set(i + 1, -20 * t2);
        g->set(i, 2.0 * (t1 - g->get(i + 1) * xi));
        fx += t1 * t1 + t2 * t2;
    }

    return fx;
}

void ROSENBROCK::hess_diag(std::shared_ptr<psi::Vector> x, std::shared_ptr<psi::Vector> h0) {
    check_dim(x);
    check_dim(h0);

    for (int i = 0; i < n_; i += 2) {
        double xi = x->get(i);
        double t2 = 10 * (xi * xi - x->get(i + 1));
        h0->set(i + 1, 200);
        h0->set(i, 2.0 * (1 + 400 * xi * xi + 20 * t2));
    }
}

void ROSENBROCK::check_dim(std::shared_ptr<psi::Vector> x) {
    if (x->dimpi().sum() != x->dim(0)) {
        std::runtime_error("Irrep not supported for Rosenbrock tests");
    }
    if (x->dim(0) != n_) {
        std::string msg = "Inconsistent dimension for " + x->name() + ", expected " +
                          std::to_string(n_) + " but get " + std::to_string(x->dim(0));
        std::runtime_error(msg.c_str());
    }
}

double test_lbfgs_rosenbrock(int n, int h0_freq) {
    // L-BFGS parameters
    auto param = std::make_shared<LBFGS_PARAM>();
    param->epsilon = 1.0e-6;
    param->maxiter = 100;
    param->h0_freq = h0_freq;
    param->print = 2;

    // L-BFGS solver
    LBFGS lbfgs_solver(param);

    // Rosenbrock function
    ROSENBROCK rosenbrock(n);

    // initial guess
    auto x = std::make_shared<psi::Vector>("x", n);

    double fx = lbfgs_solver.minimize(rosenbrock, x);

    // print final results
    psi::outfile->Printf("\n");
    psi::outfile->Printf("\n  L-BFGS converged in %d iterations.", lbfgs_solver.iter());
    psi::outfile->Printf("\n  Final function value f(x) = %.15f", fx);
    psi::outfile->Printf("\n  Optimized vector x:\n");
    x->print();

    return fx;
}
} // namespace forte
