#ifndef _coupling_coefficients_h_
#define _coupling_coefficients_h_

#include <ambit/tensor.h>

namespace forte {

class CICouplingCoefficients {
  public:
    CICouplingCoefficients();

    /// @brief Construct using 1 and 2 coupling coefficients
    CICouplingCoefficients(ambit::Tensor cc1a, ambit::Tensor cc1b, ambit::Tensor cc2aa,
                         ambit::Tensor cc2ab, ambit::Tensor cc2bb);

    /// @brief Construct using 1, 2, and 3 coupling coefficients
    CICouplingCoefficients(ambit::Tensor cc1a, ambit::Tensor cc1b, ambit::Tensor cc2aa,
                         ambit::Tensor cc2ab, ambit::Tensor cc2bb, ambit::Tensor cc3aaa,
                         ambit::Tensor cc3aab, ambit::Tensor cc3abb, ambit::Tensor cc3bbb);

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
    /// @return the alpha-alpha-alpha 3-coupling coefficients
    ambit::Tensor cc3aaa() const { return cc3aaa_; }
    /// @return the alpha-alpha-beta 3-coupling coefficients
    ambit::Tensor cc3aab() const { return cc3aab_; }
    /// @return the alpha-beta-beta 3-coupling coefficients
    ambit::Tensor cc3abb() const { return cc3abb_; }
    /// @return the beta-beta-beta 3-coupling coefficients
    ambit::Tensor cc3bbb() const { return cc3bbb_; }

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
    /// 3-coupling coefficients for alpha-alpha-alpha spin
    ambit::Tensor cc3aaa_;
    /// 3-coupling coefficients for alpha-alpha-beta spin
    ambit::Tensor cc3aab_;
    /// 3-coupling coefficients for alpha-beta-beta spin
    ambit::Tensor cc3abb_;
    /// 3-coupling coefficients for beta-beta-beta spin
    ambit::Tensor cc3bbb_;
};
} // namespace forte
#endif // _coupling_coefficients_h_
