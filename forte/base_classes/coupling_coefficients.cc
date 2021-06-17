#include "coupling_coefficients.h"

namespace forte {
CICouplingCoefficients::CICouplingCoefficients() {}
CICouplingCoefficients::CICouplingCoefficients(ambit::Tensor cc1a, ambit::Tensor cc1b,
                                           ambit::Tensor cc2aa, ambit::Tensor cc2ab,
                                           ambit::Tensor cc2bb)
    : cc1a_(cc1a), cc1b_(cc1b), cc2aa_(cc2aa), cc2ab_(cc2ab), cc2bb_(cc2bb) {}
CICouplingCoefficients::CICouplingCoefficients(ambit::Tensor cc1a, ambit::Tensor cc1b,
                                           ambit::Tensor cc2aa, ambit::Tensor cc2ab,
                                           ambit::Tensor cc2bb, ambit::Tensor cc3aaa,
                                           ambit::Tensor cc3aab, ambit::Tensor cc3abb,
                                           ambit::Tensor cc3bbb)
    : cc1a_(cc1a), cc1b_(cc1b), cc2aa_(cc2aa), cc2ab_(cc2ab), cc2bb_(cc2bb), cc3aaa_(cc3aaa),
      cc3aab_(cc3aab), cc3abb_(cc3abb), cc3bbb_(cc3bbb) {}
} // namespace forte
