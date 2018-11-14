/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "reference.h"

namespace psi {
namespace forte {

Reference::Reference() {}

Reference::~Reference() {}

double Reference::compute_Eref(std::shared_ptr<ForteIntegrals> ints,
                               std::shared_ptr<MOSpaceInfo> mo_space_info, double Enuc) {
    // similar to MASTER_DSRG::compute_reference_energy_from_ints (use Fock and cumulants)
    // here I form two density and directly use bare Hamiltonian

    double E = Enuc + ints->frozen_core_energy();

    std::vector<size_t> core_mos = mo_space_info->get_corr_abs_mo("RESTRICTED_DOCC");
    std::vector<size_t> actv_mos = mo_space_info->get_corr_abs_mo("ACTIVE");
    size_t ncore = core_mos.size();
    size_t nactv = actv_mos.size();

    // core 1-body: \sum_{m}^{C} h^{m}_{m}
    for (size_t m : core_mos) {
        E += ints->oei_a(m, m);
        E += ints->oei_b(m, m);
    }

    // active 1-body: \sum_{uv}^{A} h^{u}_{v} * G1^{v}_{u}
    ambit::Tensor Ha = ambit::Tensor::build(ambit::CoreTensor, "Ha", {nactv, nactv});
    ambit::Tensor Hb = ambit::Tensor::build(ambit::CoreTensor, "Hb", {nactv, nactv});
    for (size_t u = 0; u < nactv; ++u) {
        size_t nu = actv_mos[u];
        for (size_t v = 0; v < nactv; ++v) {
            size_t nv = actv_mos[v];
            Ha.data()[u * nactv + v] = ints->oei_a(nu, nv);
            Hb.data()[u * nactv + v] = ints->oei_b(nu, nv);
        }
    }
    E += Ha("uv") * L1a_("vu");
    E += Hb("uv") * L1b_("vu");

    // core-core 2-body: 0.5 * \sum_{mn}^{C} v^{mn}_{mn} in mini-batches
    // 4-index tensor of core-core-core-core could be large (> 600 electrons)

    ambit::Tensor I, Vtemp;
    I = ambit::Tensor::build(ambit::CoreTensor, "I", {1, ncore, 1, ncore});
    I.iterate([&](const std::vector<size_t>& i, double& value) {
        if (i[1] == i[3]) {
            value = 1.0;
        }
    });

    for (size_t m = 0; m < ncore; ++m) {
        size_t nm = core_mos[m];

        Vtemp = ints->aptei_aa_block({nm}, core_mos, {nm}, core_mos);
        E += 0.5 * Vtemp("pqrs") * I("pqrs");

        Vtemp = ints->aptei_ab_block({nm}, core_mos, {nm}, core_mos);
        E += 0.5 * Vtemp("pqrs") * I("pqrs");

        Vtemp = ints->aptei_ab_block(core_mos, {nm}, core_mos, {nm});
        E += 0.5 * Vtemp("pqrs") * I("qpsr");

        Vtemp = ints->aptei_bb_block({nm}, core_mos, {nm}, core_mos);
        E += 0.5 * Vtemp("pqrs") * I("pqrs");
    }

    // core-active 2-body: \sum_{m}^{C} \sum_{uv}^{A} v^{mu}_{mv} * G1^{v}_{u}

    I = ambit::Tensor::build(ambit::CoreTensor, "I", {ncore, ncore});
    I.iterate([&](const std::vector<size_t>& i, double& value) {
        if (i[0] == i[1]) {
            value = 1.0;
        }
    });

    Vtemp = ints->aptei_aa_block(core_mos, actv_mos, core_mos, actv_mos);
    E += Vtemp("munv") * I("mn") * L1a_("vu");

    Vtemp = ints->aptei_ab_block(core_mos, actv_mos, core_mos, actv_mos);
    E += Vtemp("munv") * I("mn") * L1b_("vu");

    Vtemp = ints->aptei_ab_block(actv_mos, core_mos, actv_mos, core_mos);
    E += Vtemp("umvn") * I("mn") * L1a_("vu");

    Vtemp = ints->aptei_bb_block(core_mos, actv_mos, core_mos, actv_mos);
    E += Vtemp("munv") * I("mn") * L1b_("vu");

    // active-active 2-body: 0.25 * \sum_{uvxy}^{A} * v^{uv}_{xy} * G2^{xy}_{uv}
    ambit::Tensor G2 = L2aa_.clone();
    G2("pqrs") += L1a_("pr") * L1a_("qs");
    G2("pqrs") -= L1a_("ps") * L1a_("qr");
    Vtemp = ints->aptei_aa_block(actv_mos, actv_mos, actv_mos, actv_mos);
    E += 0.25 * Vtemp("uvxy") * G2("uvxy");

    G2 = L2ab_.clone();
    G2("pqrs") += L1a_("pr") * L1b_("qs");
    Vtemp = ints->aptei_ab_block(actv_mos, actv_mos, actv_mos, actv_mos);
    E += Vtemp("uVxY") * G2("uVxY");

    G2 = L2bb_.clone();
    G2("pqrs") += L1b_("pr") * L1b_("qs");
    G2("pqrs") -= L1b_("ps") * L1b_("qr");
    Vtemp = ints->aptei_bb_block(actv_mos, actv_mos, actv_mos, actv_mos);
    E += 0.25 * Vtemp("UVXY") * G2("UVXY");

    return E;
}
}
}
