#ifndef COUPLINGCOEFFICIENTS_H
#define COUPLINGCOEFFICIENTS_H

#include <ambit/tensor.h>

namespace forte {

class CouplingCoefficients {
  public:
    CouplingCoefficients();

    /// @brief Construct using 1- and 2- coupling coefficients
    CouplingCoefficients(ambit::Tensor cc1a, ambit::Tensor cc1b, ambit::Tensor cc2aa,
                         ambit::Tensor cc2ab, ambit::Tensor cc2bb);

    /// @return the alpha 1-coupling coefficients
    ambit::Tensor cc1a() const { return cc1a_; }
    /// @return the beta 1-coupling coefficients
    ambit::Tensor cc1b() const { return cc1b_; }
    /// @return the alpha-alpha 2-coupling coefficients
    ambit::Tensor cc2aa() const { return cc2aa_; }
    /// @return the alpha-beta 2-coupling coefficients
    ambit::Tensor cc2ab() const { return cc2ab_; }
    /// @return the beta-beta 2-coupling coefficients
    ambit::Tensor cc2bb() const { return cc2bb_; }

  protected:
    /// 1-coupling coefficients for alpha spin
    ambit::Tensor cc1a_;
    /// 1-coupling coefficients for beta spin
    ambit::Tensor cc1b_;
    /// 2-coupling coefficients for alpha-alpha spin
    ambit::Tensor cc2aa_;
    /// 2-coupling coefficients for alpha-beta spin
    ambit::Tensor cc2ab_;
    /// 2-coupling coefficients for beta-beta spin
    ambit::Tensor cc2bb_;
};
} // namespace forte
#endif // COUPLINGCOEFFICIENTS_H
