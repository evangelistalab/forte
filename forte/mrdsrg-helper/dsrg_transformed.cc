#include "dsrg_transformed.h"

namespace forte {

DressedQuantity::DressedQuantity() : max_body_(0), scalar_(0.0) {}

DressedQuantity::DressedQuantity(double scalar) : max_body_(0), scalar_(scalar) {}

DressedQuantity::DressedQuantity(double scalar, ambit::Tensor a, ambit::Tensor b)
    : max_body_(1), scalar_(scalar), a_(a), b_(b) {}

DressedQuantity::DressedQuantity(double scalar, ambit::Tensor a, ambit::Tensor b, ambit::Tensor aa,
                                 ambit::Tensor ab, ambit::Tensor bb)
    : max_body_(2), scalar_(scalar), a_(a), b_(b), aa_(aa), ab_(ab), bb_(bb) {}

DressedQuantity::DressedQuantity(double scalar, ambit::Tensor a, ambit::Tensor b, ambit::Tensor aa,
                                 ambit::Tensor ab, ambit::Tensor bb, ambit::Tensor aaa,
                                 ambit::Tensor aab, ambit::Tensor abb, ambit::Tensor bbb)
    : max_body_(3), scalar_(scalar), a_(a), b_(b), aa_(aa), ab_(ab), bb_(bb), aaa_(aaa), aab_(aab),
      abb_(abb), bbb_(bbb) {}

double DressedQuantity::contract_with_rdms(std::shared_ptr<RDMs> rdms) {
    double out = scalar_;
    size_t max_rdm_level = rdms->max_rdm_level();

    if (max_rdm_level >= 1 and max_body_ >= 1) {
        out += a_("uv") * rdms->g1a()("vu");
        out += b_("uv") * rdms->g1b()("vu");
    }

    if (max_rdm_level >= 2 and max_body_ >= 2) {
        out += 0.25 * aa_("uvxy") * rdms->g2aa()("xyuv");
        out += 0.25 * bb_("uvxy") * rdms->g2bb()("xyuv");
        out += ab_("uvxy") * rdms->g2ab()("xyuv");
    }

    if (max_rdm_level >= 3 and max_body_ >= 3) {
        out += (1.0 / 36.0) * aaa_("uvwxyz") * rdms->g3aaa()("xyzuvw");
        out += (1.0 / 36.0) * bbb_("uvwxyz") * rdms->g3bbb()("xyzuvw");
        out += 0.25 * aab_("uvwxyz") * rdms->g3aab()("xyzuvw");
        out += 0.25 * abb_("uvwxyz") * rdms->g3abb()("xyzuvw");
    }

    return out;
}
} // namespace forte
