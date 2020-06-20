#include "coupling_coefficients.h"

namespace forte {
CouplingCoefficients::CouplingCoefficients() {}
CouplingCoefficients::CouplingCoefficients(ambit::Tensor cc1a, ambit::Tensor cc1b, ambit::Tensor cc2aa,
                                           ambit::Tensor cc2ab, ambit::Tensor cc2bb)
    : cc1a_(cc1a), cc1b_(cc1b), cc2aa_(cc2aa), cc2ab_(cc2ab), cc2bb_(cc2bb) {}
} // namespace forte
