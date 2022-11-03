/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2022 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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
#include "psi4/libmints/multipoles.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/vector3.h"
#include "psi4/libmints/wavefunction.h"

#include "base_classes/mo_space_info.h"
#include "integrals/multipole_integrals.h"

namespace forte {

MultipoleIntegrals::MultipoleIntegrals(std::shared_ptr<ForteIntegrals> ints,
                                       std::shared_ptr<MOSpaceInfo> mo_space_info, int order)
    : mo_space_info_(mo_space_info), order_(order), cmotomo_(ints->cmotomo()),
      molecule_(ints->wfn()->molecule()) {
    if (order > 2 or order < 1)
        throw std::runtime_error("Input order: " + std::to_string(order) +
                                 " not available in MultipoleIntegrals");

    if (order == 1) {
        ndirs_ = 3;
        mp_ints_ = ints->mo_dipole_ints();
    } else {
        ndirs_ = 6;
        mp_ints_ = ints->mo_quadrupole_ints();
    }
}

double MultipoleIntegrals::mp_ints_corr(int direction, size_t p, size_t q) const {
    return mp_ints_[direction]->get(cmotomo_[p], cmotomo_[q]);
}

double MultipoleIntegrals::mp_ints(int direction, size_t p, size_t q) const {
    return mp_ints_[direction]->get(p, q);
}

psi::SharedVector MultipoleIntegrals::nuclear_contributions(const psi::Vector3& origin) const {
    auto nuc = std::make_shared<psi::Vector>(ndirs_);
    auto address = 0;
    for (int ii = 0; ii <= order_; ii++) {
        int lx = order_ - ii;
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

psi::SharedVector MultipoleIntegrals::mp_frozen_core() const {
    auto frzc_mos = mo_space_info_->absolute_mo("FROZEN_DOCC");
    auto mp_frzc = std::make_shared<psi::Vector>(ndirs_);
    for (int z = 0; z < ndirs_; ++z) {
        double v = 0.0;
        for (const auto& p : frzc_mos) {
            v += 2.0 * mp_ints(z, p, p);
        }
        mp_frzc->set(z, v);
    }
    return mp_frzc;
}

std::shared_ptr<MOSpaceInfo> MultipoleIntegrals::mo_space_info() const { return mo_space_info_; }

int MultipoleIntegrals::order() const { return order_; }

int MultipoleIntegrals::ndirs() const { return ndirs_; }

ActiveMultipoleIntegrals::ActiveMultipoleIntegrals(std::shared_ptr<MultipoleIntegrals> mpints)
    : mpints_(mpints), many_body_level_(1) {

    auto mo_space_info = mpints->mo_space_info();
    nmo_ = mo_space_info->size("ACTIVE");
    nmo2_ = nmo_ * nmo_;
    nmo3_ = nmo_ * nmo2_;
    nmo4_ = nmo_ * nmo3_;

    // determine contributions from restricted_docc orbitals
    auto rdocc_mos = mo_space_info->corr_absolute_mo("CORE");
    auto ndirs = mpints->ndirs();
    scalars_rdocc_ = std::make_shared<psi::Vector>(ndirs);
    for (int z = 0; z < ndirs; ++z) {
        double value = 0.0;
        for (const auto& i : rdocc_mos) {
            value += 2.0 * mpints->mp_ints_corr(z, i, i);
        }
        scalars_rdocc_->set(z, value);
    }

    // put active part to ambit Tensor form
    std::vector<std::string> dir_names;
    std::string prefix;
    if (mpints->order() == 1) {
        prefix = "DIPOLE";
        dir_names = {"X", "Y", "Z"};
    } else {
        prefix = "QUADRUPOLE";
        dir_names = {"XX", "XY", "XZ", "YY", "YZ", "ZZ"};
    }

    one_body_ints_.resize(ndirs);
    auto actv_mos = mo_space_info->corr_absolute_mo("ACTIVE");
    for (int z = 0; z < ndirs; ++z) {
        auto zints = ambit::Tensor::build(ambit::CoreTensor, prefix + dir_names[z], {nmo_, nmo_});
        zints.iterate([&](const std::vector<size_t>& i, double& value) {
            value = mpints->mp_ints_corr(z, actv_mos[i[0]], actv_mos[i[1]]);
        });
        one_body_ints_[z] = zints;
    }
}

int ActiveMultipoleIntegrals::order() const { return mpints_->order(); }

int ActiveMultipoleIntegrals::many_body_level() const { return many_body_level_; }

psi::SharedVector ActiveMultipoleIntegrals::compute_electronic_multipole(std::shared_ptr<RDMs> rdms,
                                                                         bool transition) {
    auto ndirs = mpints_->ndirs();
    auto out = std::make_shared<psi::Vector>(ndirs);
    if (not transition)
        out = scalars();
    psi::Vector actv(ndirs);

    // 1-body
    auto d1 = rdms->SF_G1();
    for (int z = 0; z < ndirs; ++z) {
        actv.add(z, d1("pq") * one_body_ints_[z]("pq"));
    }

    // 2-body
    if (rdms->max_rdm_level() > 1) {
        if (two_body_ints_.size()) {
            auto d2 = rdms->SF_G2();
            for (int z = 0; z < ndirs; ++z) {
                actv.add(z, 0.5 * d2("pqrs") * two_body_ints_[z]("pqrs"));
            }
        } else {
            if (two_body_ints_aa_.size() and two_body_ints_ab_.size() and
                two_body_ints_bb_.size()) {
                auto d2aa = rdms->g2aa();
                auto d2ab = rdms->g2ab();
                auto d2bb = rdms->g2bb();
                for (int z = 0; z < ndirs; ++z) {
                    actv.add(z, 0.25 * two_body_ints_aa_[z]("uvxy") * d2aa("xyuv"));
                    actv.add(z, 0.25 * two_body_ints_bb_[z]("uvxy") * d2bb("xyuv"));
                    actv.add(z, two_body_ints_ab_[z]("uvxy") * d2ab("xyuv"));
                }
            }
        }
    }

    out->add(actv);
    return out;
}

psi::SharedVector
ActiveMultipoleIntegrals::nuclear_contributions(const psi::Vector3& origin) const {
    return mpints_->nuclear_contributions(origin);
}

psi::SharedVector ActiveMultipoleIntegrals::scalars_fdocc() const {
    return mpints_->mp_frozen_core();
}

psi::SharedVector ActiveMultipoleIntegrals::scalars_rdocc() const { return scalars_rdocc_; }

psi::SharedVector ActiveMultipoleIntegrals::scalars() const {
    auto out = std::make_shared<psi::Vector>(*scalars_fdocc());
    out->add(*scalars_rdocc_);
    return out;
}

void ActiveMultipoleIntegrals::set_scalar_rdocc(int direction, double value) {
    scalars_rdocc_->set(direction, value);
}

void ActiveMultipoleIntegrals::set_1body(int direction, ambit::Tensor M1) {
    _test_tensor_dims(M1);
    one_body_ints_[direction] = M1;
}

void ActiveMultipoleIntegrals::set_2body(int direction, ambit::Tensor M2) {
    _test_tensor_dims(M2);
    two_body_ints_[direction] = M2;
    many_body_level_ = 2;
}

void ActiveMultipoleIntegrals::set_2body(int direction, ambit::Tensor M2aa, ambit::Tensor M2ab,
                                         ambit::Tensor M2bb) {
    _test_tensor_dims(M2aa);
    _test_tensor_dims(M2ab);
    _test_tensor_dims(M2bb);
    two_body_ints_aa_[direction] = M2aa;
    two_body_ints_ab_[direction] = M2ab;
    two_body_ints_bb_[direction] = M2bb;
    many_body_level_ = 2;
}

void ActiveMultipoleIntegrals::_test_tensor_dims(ambit::Tensor T) {
    for (const auto i : T.dims()) {
        if (i != nmo_)
            throw std::runtime_error(
                "Wrong dimension of ambit::Tensor in ActiveMultipoleIntegrals");
    }
}

} // namespace forte