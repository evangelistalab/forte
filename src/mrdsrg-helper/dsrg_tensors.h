#ifndef _dsrg_tensors_h_
#define _dsrg_tensors_h_

#include "ambit/tensor.h"
#include "ambit/blocked_tensor.h"

namespace psi {
namespace forte {

class dsrg_tensors {
  public:
    dsrg_tensors();

    ambit::BlockedTensor Fock;
    ambit::BlockedTensor V;
    ambit::BlockedTensor T1, T2;
    ambit::BlockedTensor Hbar1, Hbar2, Hbar3;
    ambit::BlockedTensor Heff1, Heff2, Heff3;
};

struct dsrgHeff {
    double H0 = 0.0;
    ambit::BlockedTensor H1, H2, H3;
};

}
}
#endif // DSRG_TENSORS_H
