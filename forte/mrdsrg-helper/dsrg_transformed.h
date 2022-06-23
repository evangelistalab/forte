#ifndef _DSRG_TRANSFORMED_H_
#define _DSRG_TRANSFORMED_H_

#include "ambit/tensor.h"

#include "base_classes/rdms.h"

namespace forte {
class DressedQuantity {
  public:
    DressedQuantity();
    DressedQuantity(double scalar);
    DressedQuantity(double scalar, ambit::Tensor a, ambit::Tensor b);
    DressedQuantity(double scalar, ambit::Tensor a, ambit::Tensor b, ambit::Tensor aa,
                    ambit::Tensor ab, ambit::Tensor bb);
    DressedQuantity(double scalar, ambit::Tensor a, ambit::Tensor b, ambit::Tensor aa,
                    ambit::Tensor ab, ambit::Tensor bb, ambit::Tensor aaa, ambit::Tensor aab,
                    ambit::Tensor abb, ambit::Tensor bbb);
    double contract_with_rdms(std::shared_ptr<RDMs> rdms);

  private:
    size_t max_body_;
    double scalar_;
    ambit::Tensor a_;
    ambit::Tensor b_;
    ambit::Tensor aa_;
    ambit::Tensor ab_;
    ambit::Tensor bb_;
    ambit::Tensor aaa_;
    ambit::Tensor aab_;
    ambit::Tensor abb_;
    ambit::Tensor bbb_;
};
} // namespace forte
#endif // DSRG_TRANSFORMED_H
