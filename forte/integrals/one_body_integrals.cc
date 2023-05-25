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

#include <algorithm>
#include <functional>

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/vector3.h"
#include "psi4/libmints/wavefunction.h"

#include "base_classes/mo_space_info.h"
#include "integrals/one_body_integrals.h"

namespace forte {

MultipoleIntegrals::MultipoleIntegrals(std::shared_ptr<ForteIntegrals> ints,
                                       std::shared_ptr<MOSpaceInfo> mo_space_info)
    : mo_space_info_(mo_space_info), cmotomo_(ints->cmotomo()), molecule_(ints->wfn()->molecule()) {
    dp_ints_ = ints->mo_dipole_ints();
    qp_ints_ = ints->mo_quadrupole_ints();
}

double MultipoleIntegrals::dp_ints_corr(int direction, size_t p, size_t q) const {
    return dp_ints_[direction]->get(cmotomo_[p], cmotomo_[q]);
}

double MultipoleIntegrals::dp_ints(int direction, size_t p, size_t q) const {
    return dp_ints_[direction]->get(p, q);
}

double MultipoleIntegrals::qp_ints_corr(int direction, size_t p, size_t q) const {
    return qp_ints_[direction]->get(cmotomo_[p], cmotomo_[q]);
}

double MultipoleIntegrals::qp_ints(int direction, size_t p, size_t q) const {
    return qp_ints_[direction]->get(p, q);
}

std::shared_ptr<psi::Vector> MultipoleIntegrals::nuclear_dipole(const psi::Vector3& origin) const {
    auto nuc = molecule_->nuclear_dipole(origin);
    auto out = std::make_shared<psi::Vector>(3);
    for (int i = 0; i < 3; ++i) {
        out->set(i, nuc[i]);
    }
    return out;
}

std::shared_ptr<psi::Vector>
MultipoleIntegrals::nuclear_quadrupole(const psi::Vector3& origin) const {
    auto nuc = std::make_shared<psi::Vector>(6);
    for (int ii = 0, address = 0; ii <= 2; ii++) {
        int lx = 2 - ii;
        for (int lz = 0; lz <= ii; lz++) {
            int ly = ii - lz;
            for (int atom = 0; atom < molecule_->natom(); ++atom) {
                auto geom = molecule_->xyz(atom) - origin;
                nuc->add(address, molecule_->Z(atom) * pow(geom[0], lx) * pow(geom[1], ly) *
                                      pow(geom[2], lz));
            }
            ++address;
        }
    }
    return nuc;
}

std::shared_ptr<psi::Vector> MultipoleIntegrals::dp_frozen_core() const {
    auto frzc_mos = mo_space_info_->absolute_mo("FROZEN_DOCC");
    auto dp_frzc = std::make_shared<psi::Vector>(3);
    for (int z = 0; z < 3; ++z) {
        for (const auto& p : frzc_mos) {
            dp_frzc->add(z, 2.0 * dp_ints(z, p, p));
        }
    }
    return dp_frzc;
}

std::shared_ptr<psi::Vector> MultipoleIntegrals::qp_frozen_core() const {
    auto frzc_mos = mo_space_info_->absolute_mo("FROZEN_DOCC");
    auto qp_frzc = std::make_shared<psi::Vector>(6);
    for (int z = 0; z < 6; ++z) {
        for (const auto& p : frzc_mos) {
            qp_frzc->add(z, 2.0 * qp_ints(z, p, p));
        }
    }
    return qp_frzc;
}

std::shared_ptr<MOSpaceInfo> MultipoleIntegrals::mo_space_info() const { return mo_space_info_; }

ActiveMultipoleIntegrals::ActiveMultipoleIntegrals(std::shared_ptr<MultipoleIntegrals> mpints)
    : mpints_(mpints), dp_many_body_level_(1), qp_many_body_level_(1) {

    auto mo_space_info = mpints->mo_space_info();
    nmo_ = mo_space_info->size("ACTIVE");
    nmo2_ = nmo_ * nmo_;
    nmo3_ = nmo_ * nmo2_;
    nmo4_ = nmo_ * nmo3_;

    // determine contributions from restricted_docc orbitals
    auto rdocc_mos = mo_space_info->corr_absolute_mo("CORE");
    dp0_rdocc_ = std::make_shared<psi::Vector>(3);
    for (int z = 0; z < 3; ++z) {
        for (const auto& i : rdocc_mos) {
            dp0_rdocc_->add(z, 2.0 * mpints->dp_ints_corr(z, i, i));
        }
    }

    qp0_rdocc_ = std::make_shared<psi::Vector>(6);
    for (int z = 0; z < 6; ++z) {
        for (const auto& i : rdocc_mos) {
            qp0_rdocc_->add(z, 2.0 * mpints->qp_ints_corr(z, i, i));
        }
    }

    // put active part to ambit Tensor form
    auto actv_mos = mo_space_info->corr_absolute_mo("ACTIVE");

    dp1_ints_.resize(3);
    dp2_ints_.resize(3);
    std::vector<std::string> dp_names{"X", "Y", "Z"};
    for (int z = 0; z < 3; ++z) {
        auto zints = ambit::Tensor::build(ambit::CoreTensor, "DIPOLE" + dp_names[z], {nmo_, nmo_});
        zints.iterate([&](const std::vector<size_t>& i, double& value) {
            value = mpints->dp_ints_corr(z, actv_mos[i[0]], actv_mos[i[1]]);
        });
        dp1_ints_[z] = zints;
    }

    qp1_ints_.resize(6);
    qp2_ints_.resize(6);
    std::vector<std::string> qp_names{"XX", "XY", "XZ", "YY", "YZ", "ZZ"};
    for (int z = 0; z < 6; ++z) {
        auto zints =
            ambit::Tensor::build(ambit::CoreTensor, "QUADRUPOLE" + qp_names[z], {nmo_, nmo_});
        zints.iterate([&](const std::vector<size_t>& i, double& value) {
            value = mpints->qp_ints_corr(z, actv_mos[i[0]], actv_mos[i[1]]);
        });
        qp1_ints_[z] = zints;
    }
}

int ActiveMultipoleIntegrals::dp_many_body_level() const { return dp_many_body_level_; }

int ActiveMultipoleIntegrals::qp_many_body_level() const { return qp_many_body_level_; }

std::shared_ptr<psi::Vector>
ActiveMultipoleIntegrals::compute_electronic_dipole(std::shared_ptr<RDMs> rdms, bool transition) {
    auto ndirs = 3;
    auto out = std::make_shared<psi::Vector>(ndirs);
    if (not transition)
        out = dp_scalars();
    psi::Vector actv(ndirs);

    // 1-body
    auto d1 = rdms->SF_G1();
    for (int z = 0; z < ndirs; ++z) {
        actv.add(z, d1("pq") * dp1_ints_[z]("pq"));
    }

    // 2-body
    if (dp_many_body_level_ > 1 and rdms->max_rdm_level() > 1) {
        if (dp2_ints_.size()) {
            auto d2 = rdms->SF_G2();
            for (int z = 0; z < ndirs; ++z) {
                actv.add(z, 0.5 * d2("pqrs") * dp2_ints_[z]("pqrs"));
            }
        } else {
            if (dp2_ints_aa_.size() and dp2_ints_ab_.size() and dp2_ints_bb_.size()) {
                auto d2aa = rdms->g2aa();
                auto d2ab = rdms->g2ab();
                auto d2bb = rdms->g2bb();
                for (int z = 0; z < ndirs; ++z) {
                    actv.add(z, 0.25 * dp2_ints_aa_[z]("uvxy") * d2aa("xyuv"));
                    actv.add(z, 0.25 * dp2_ints_bb_[z]("uvxy") * d2bb("xyuv"));
                    actv.add(z, dp2_ints_ab_[z]("uvxy") * d2ab("xyuv"));
                }
            }
        }
    }
    out->add(actv);
    return out;
}

std::shared_ptr<psi::Vector>
ActiveMultipoleIntegrals::compute_electronic_quadrupole(std::shared_ptr<RDMs> rdms,
                                                        bool transition) {
    if (rdms->dim() != nmo_) {
        throw std::runtime_error("Dimension mismatch between ActiveMultipoleIntegrals and RDMs!");
    }
    auto ndirs = 6;
    auto out = std::make_shared<psi::Vector>(ndirs);
    if (not transition)
        out = qp_scalars();
    psi::Vector actv(ndirs);

    // 1-body
    auto d1 = rdms->SF_G1();
    for (int z = 0; z < ndirs; ++z) {
        actv.add(z, d1("pq") * qp1_ints_[z]("pq"));
    }

    // 2-body
    if (qp_many_body_level_ > 1 and rdms->max_rdm_level() > 1) {
        if (qp2_ints_.size()) {
            auto d2 = rdms->SF_G2();
            for (int z = 0; z < ndirs; ++z) {
                actv.add(z, 0.5 * d2("pqrs") * qp2_ints_[z]("pqrs"));
            }
        } else {
            if (qp2_ints_aa_.size() and qp2_ints_ab_.size() and qp2_ints_bb_.size()) {
                auto d2aa = rdms->g2aa();
                auto d2ab = rdms->g2ab();
                auto d2bb = rdms->g2bb();
                for (int z = 0; z < ndirs; ++z) {
                    actv.add(z, 0.25 * qp2_ints_aa_[z]("uvxy") * d2aa("xyuv"));
                    actv.add(z, 0.25 * qp2_ints_bb_[z]("uvxy") * d2bb("xyuv"));
                    actv.add(z, qp2_ints_ab_[z]("uvxy") * d2ab("xyuv"));
                }
            }
        }
    }

    out->add(actv);
    return out;
}

std::shared_ptr<psi::Vector>
ActiveMultipoleIntegrals::nuclear_dipole(const psi::Vector3& origin) const {
    return mpints_->nuclear_dipole(origin);
}

std::shared_ptr<psi::Vector>
ActiveMultipoleIntegrals::nuclear_quadrupole(const psi::Vector3& origin) const {
    return mpints_->nuclear_quadrupole(origin);
}

std::shared_ptr<psi::Vector> ActiveMultipoleIntegrals::dp_scalars_fdocc() const {
    return mpints_->dp_frozen_core();
}

std::shared_ptr<psi::Vector> ActiveMultipoleIntegrals::qp_scalars_fdocc() const {
    return mpints_->qp_frozen_core();
}

std::shared_ptr<psi::Vector> ActiveMultipoleIntegrals::dp_scalars_rdocc() const {
    return dp0_rdocc_;
}

std::shared_ptr<psi::Vector> ActiveMultipoleIntegrals::qp_scalars_rdocc() const {
    return qp0_rdocc_;
}

std::shared_ptr<psi::Vector> ActiveMultipoleIntegrals::dp_scalars() const {
    auto out = std::make_shared<psi::Vector>(*dp_scalars_fdocc());
    out->add(*dp0_rdocc_);
    return out;
}

std::shared_ptr<psi::Vector> ActiveMultipoleIntegrals::qp_scalars() const {
    auto out = std::make_shared<psi::Vector>(*qp_scalars_fdocc());
    out->add(*qp0_rdocc_);
    return out;
}

void ActiveMultipoleIntegrals::set_dp_scalar_rdocc(int direction, double value) {
    dp0_rdocc_->set(direction, value);
}

void ActiveMultipoleIntegrals::set_qp_scalar_rdocc(int direction, double value) {
    qp0_rdocc_->set(direction, value);
}

void ActiveMultipoleIntegrals::set_dp1_ints(int direction, ambit::Tensor M1) {
    _test_tensor_dims(M1);
    dp1_ints_[direction] = M1;
}

void ActiveMultipoleIntegrals::set_qp1_ints(int direction, ambit::Tensor M1) {
    _test_tensor_dims(M1);
    qp1_ints_[direction] = M1;
}

void ActiveMultipoleIntegrals::set_dp2_ints(int direction, ambit::Tensor M2) {
    _test_tensor_dims(M2);
    dp2_ints_[direction] = M2;
    dp_many_body_level_ = 2;
}

void ActiveMultipoleIntegrals::set_qp2_ints(int direction, ambit::Tensor M2) {
    _test_tensor_dims(M2);
    qp2_ints_[direction] = M2;
    qp_many_body_level_ = 2;
}

void ActiveMultipoleIntegrals::set_dp2_ints(int direction, ambit::Tensor M2aa, ambit::Tensor M2ab,
                                            ambit::Tensor M2bb) {
    _test_tensor_dims(M2aa);
    _test_tensor_dims(M2ab);
    _test_tensor_dims(M2bb);
    dp2_ints_aa_[direction] = M2aa;
    dp2_ints_ab_[direction] = M2ab;
    dp2_ints_bb_[direction] = M2bb;
    dp_many_body_level_ = 2;
}

void ActiveMultipoleIntegrals::set_qp2_ints(int direction, ambit::Tensor M2aa, ambit::Tensor M2ab,
                                            ambit::Tensor M2bb) {
    _test_tensor_dims(M2aa);
    _test_tensor_dims(M2ab);
    _test_tensor_dims(M2bb);
    qp2_ints_aa_[direction] = M2aa;
    qp2_ints_ab_[direction] = M2ab;
    qp2_ints_bb_[direction] = M2bb;
    qp_many_body_level_ = 2;
}

void ActiveMultipoleIntegrals::_test_tensor_dims(ambit::Tensor T) {
    for (const auto i : T.dims()) {
        if (i != nmo_)
            throw std::runtime_error(
                "Wrong dimension of ambit::Tensor in ActiveMultipoleIntegrals");
    }
}

} // namespace forte