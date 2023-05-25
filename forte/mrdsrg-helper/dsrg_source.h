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

#ifndef _dsrg_source_h_
#define _dsrg_source_h_

#include <cmath>
#include <stdexcept>

namespace forte {

class DSRG_SOURCE {
  public:
    /**
     * DSRG_SOURCE Constructor
     * @param s The flow parameter
     * @param taylor_threshold The threshold for Taylor expansion
     */
    DSRG_SOURCE(double s, double taylor_threshold);

    virtual ~DSRG_SOURCE() {}

    /// Bare effect of source operator
    virtual double compute_renormalized(const double& D) = 0;

    /// Regularize denominator
    virtual double compute_renormalized_denominator(const double& D) = 0;

    /// Partial components of the regularized denominator derivatives w.r.t. Møller-Plesset
    /// denominators
    /// TODO: probably not general and only useful for STD and LABS
    virtual double compute_regularized_denominator_derivR(const double& /*D*/) {
        throw std::runtime_error("Not implemented for this class! Please override this function!");
    }

  protected:
    /// Flow parameter
    double s_;
    /// Smaller than which we will do Taylor expansion
    double taylor_threshold_;
};

/// Standard source
class STD_SOURCE : public DSRG_SOURCE {
  public:
    /// Constructor
    STD_SOURCE(double s, double taylor_threshold);

    virtual ~STD_SOURCE() {}

    /// Return exp(-s * D^2)
    virtual double compute_renormalized(const double& D) { return std::exp(-s_ * D * D); }

    /// Return [1 - exp(-s * D^2)] / D
    virtual double compute_renormalized_denominator(const double& D) {
        double S = std::sqrt(s_);
        double Z = S * D;
        if (std::fabs(Z) < small_) {
            return Taylor_Exp(Z, taylor_order_) * S;
        } else {
            return (1.0 - std::exp(-Z * Z)) / D;
        }
    }

    /// Return [1 - exp(-s * D^2)] / D^2
    virtual double compute_regularized_denominator_derivR(const double& D) {
        double Z = std::sqrt(s_) * D;
        if (std::fabs(Z) < small_) {
            return Taylor_Exp(Z, taylor_order_, 2) * s_;
        } else {
            return (1.0 - std::exp(-Z * Z)) / (D * D);
        }
    }

  private:
    /// Order of the Taylor expansion
    int taylor_order_ = static_cast<int>(0.5 * (15.0 / taylor_threshold_ + 1)) + 1;

    /// Smaller than which will do Taylor expansion
    double small_ = std::pow(0.1, taylor_threshold_);

    /// Taylor Expansion of [1 - exp(- Z^2)] / Z^k for k = 1, 2
    double Taylor_Exp(const double Z, const int n, const int k = 1) {
        if (k < 1 or k > 2)
            throw std::runtime_error("Invalid power of denominator");
        if (n < 0)
            return 0.0;

        double value = (k == 1) ? Z : 1.0;
        double tmp = Z;

        for (int x = 0; x < n - 1; ++x) {
            tmp *= -1.0 * Z * Z / (x + 2);
            value += tmp;
        }
        return value;
    }
};

/// Linear absolute exponential source
class LABS_SOURCE : public DSRG_SOURCE {
  public:
    /// Constructor
    LABS_SOURCE(double s, double taylor_threshold);

    virtual ~LABS_SOURCE() {}

    /// Return exp(-s * |D|)
    virtual double compute_renormalized(const double& D) { return std::exp(-s_ * std::fabs(D)); }

    /// Return [1 - exp(-s * |D|)] / D
    virtual double compute_renormalized_denominator(const double& D) {
        double Z = s_ * D;
        if (std::fabs(Z) < small_) {
            return Taylor_Exp_Linear(Z, taylor_order_ * 2) * s_;
        } else {
            return (1.0 - std::exp(-s_ * std::fabs(D))) / D;
        }
    }

  private:
    /// Order of the Taylor expansion
    int taylor_order_ = static_cast<int>(15.0 / taylor_threshold_ + 1) + 1;

    /// Smaller than which will do Taylor expansion
    double small_ = std::pow(0.1, taylor_threshold_);

    /// Taylor Expansion of [1 - exp(-|Z|)] / Z
    double Taylor_Exp_Linear(const double& Z, const int& n) {
        double Zabs = std::fabs(Z);
        if (n > 0) {
            double value = 1.0, tmp = 1.0;
            for (int x = 0; x < n - 1; ++x) {
                tmp *= -1.0 * Zabs / (x + 2);
                value += tmp;
            }
            if (Z >= 0.0) {
                return value;
            } else {
                return -value;
            }
        } else {
            return 0.0;
        }
    }
};

/// Dyson source
class DYSON_SOURCE : public DSRG_SOURCE {
  public:
    /// Constructor
    DYSON_SOURCE(double s, double taylor_threshold);

    virtual ~DYSON_SOURCE() {}

    /// Return 1.0 / (1.0 + s * D^2)
    virtual double compute_renormalized(const double& D) { return 1.0 / (1.0 + s_ * D * D); }

    /// Return s * D / (1.0 + s * D^2)
    virtual double compute_renormalized_denominator(const double& D) {
        return s_ * D / (1.0 + s_ * D * D);
    }
};

/// MP2 denominator
class MP2_SOURCE : public DSRG_SOURCE {
  public:
    MP2_SOURCE(double s, double taylor_threshold);

    virtual ~MP2_SOURCE() {}

    virtual double compute_renormalized(const double&) { return 1.0; }

    virtual double compute_renormalized_denominator(const double& D) { return 1.0 / D; }
};
} // namespace forte

#endif // DSRG_SOURCE_H
