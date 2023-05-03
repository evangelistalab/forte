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
#include <memory>

#include "psi4/libpsi4util/process.h"
#include "psi4/libpsio/psio.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libmints/dimension.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include "mrdsrg-spin-adapted/sa_mrpt2.h"
#include "helpers/disk_io.h"
#include "helpers/printing.h"

#include "mrpt2_nos.h"

using namespace ambit;
using namespace psi;

namespace forte {

MRPT2_NOS::MRPT2_NOS(std::shared_ptr<RDMs> rdms, std::shared_ptr<SCFInfo> scf_info,
                     std::shared_ptr<ForteOptions> options, std::shared_ptr<ForteIntegrals> ints,
                     std::shared_ptr<MOSpaceInfo> mo_space_info)
    : OrbitalTransform(ints, mo_space_info), options_(options) {
    mrpt2_ = std::make_shared<SA_MRPT2>(rdms, scf_info, options, ints, mo_space_info);
}

psi::SharedMatrix MRPT2_NOS::compute_fno() {
    print_h2("DSRG-MRPT2 Frozen Natural Orbitals");

    D1v_ = mrpt2_->build_1rdm_unrelaxed_virt();
    psi::Process::environment.arrays["MRPT2 1RDM VV"] = D1v_;

    auto virt_mospi = D1v_->rowspi();
    psi::Vector D1v_evals("D1v_evals", virt_mospi);
    psi::Matrix D1v_evecs("D1v_evecs", virt_mospi, virt_mospi);
    D1v_->diagonalize(D1v_evecs, D1v_evals, descending);

    // build transformation matrix
    auto nmopi = mo_space_info_->dimension("ALL");
    auto ncmopi = mo_space_info_->dimension("CORRELATED");
    auto frzcpi = mo_space_info_->dimension("FROZEN_DOCC");
    auto doccpi = mo_space_info_->dimension("INACTIVE_DOCC");
    auto holepi = doccpi + mo_space_info_->dimension("ACTIVE");

    Ua_ = std::make_shared<psi::Matrix>("Ua", nmopi, nmopi);
    Ua_->identity();

    Slice slice_virt(holepi, frzcpi + ncmopi);
    Ua_->set_block(slice_virt, D1v_evecs);
    Ub_ = Ua_->clone();

    save_psi4_vector("NAT_OCC_VIRT", D1v_evals, holepi);
    return Ua_;
}

void MRPT2_NOS::compute_transformation() {
    // compute unrelaxed 1-RDMs for diagonal blocks
    mrpt2_->build_1rdm_unrelaxed(D1c_, D1v_, D1a_);

    psi::Process::environment.arrays["MRPT2 1RDM CC"] = D1c_;
    psi::Process::environment.arrays["MRPT2 1RDM VV"] = D1v_;
    psi::Process::environment.arrays["MRPT2 1RDM AA"] = D1a_;

    // diagonalize unrelaxed 1-RDM
    auto core_mospi = D1c_->rowspi();
    auto virt_mospi = D1v_->rowspi();
    auto actv_mospi = D1a_->rowspi();

    psi::Vector D1c_evals("D1c_evals", core_mospi);
    psi::Matrix D1c_evecs("D1c_evecs", core_mospi, core_mospi);
    D1c_->diagonalize(D1c_evecs, D1c_evals, descending);

    psi::Vector D1v_evals("D1v_evals", virt_mospi);
    psi::Matrix D1v_evecs("D1v_evecs", virt_mospi, virt_mospi);
    D1v_->diagonalize(D1v_evecs, D1v_evals, descending);

    psi::Vector D1a_evals("D1a_evals", actv_mospi);
    psi::Matrix D1a_evecs("D1a_evecs", actv_mospi, actv_mospi);
    D1a_->diagonalize(D1a_evecs, D1a_evals, descending);

    // print total number of electrons
    auto compute_ne = [&](psi::Vector& vec) {
        auto nirrep = vec.nirrep();
        auto nmopi = vec.dimpi();
        auto ne = 0.0;
        for (int h = 0; h < nirrep; ++h) {
            for (int p = 0; p < nmopi[h]; ++p)
                ne += vec.get(h, p);
        }
        return ne;
    };
    print_h2("Number of Electrons");
    outfile->Printf("\n    Number of electrons in core:    %12.6f", compute_ne(D1c_evals));
    outfile->Printf("\n    Number of electrons in active:  %12.6f", compute_ne(D1a_evals));
    outfile->Printf("\n    Number of electrons in virtual: %12.6f", compute_ne(D1v_evals));

    // print natural orbitals
    if (options_->get_bool("NAT_ORBS_PRINT")) {
        outfile->Printf("\n\n");
        D1a_evals.print();
        D1c_evals.print();
        D1v_evals.print();
    }

    // build transformation matrix
    auto nmopi = mo_space_info_->dimension("ALL");
    auto ncmopi = mo_space_info_->dimension("CORRELATED");
    auto frzcpi = mo_space_info_->dimension("FROZEN_DOCC");
    auto doccpi = mo_space_info_->dimension("INACTIVE_DOCC");
    auto holepi = doccpi + mo_space_info_->dimension("ACTIVE");

    Ua_ = std::make_shared<psi::Matrix>("Ua", nmopi, nmopi);
    Ua_->identity();

    Slice slice_core(frzcpi, doccpi);
    Ua_->set_block(slice_core, D1c_evecs);

    Slice slice_actv(doccpi, holepi);
    Ua_->set_block(slice_actv, D1a_evecs);

    Slice slice_virt(holepi, frzcpi + ncmopi);
    Ua_->set_block(slice_virt, D1v_evecs);

    auto S = std::make_shared<psi::Matrix>("Swap", nmopi, nmopi);
    S->identity();

    // suggest active space
    if (options_->get_bool("NAT_ACT")) {
        auto rot_pairs = suggest_active_space(D1c_evals, D1v_evals, D1a_evals);
        if (options_->get_bool("MRPT2NO_ACTV_ROTATE")) {
            for (int h = 0, nirrep = rot_pairs.size(); h < nirrep; ++h) {
                for (const auto& [i, j] : rot_pairs[h]) {
                    outfile->Printf("\n    irrep %d, swap orbitals: %3zu <-> %-3zu", h, i, j);
                    S->set(h, i, i, 0.0);
                    S->set(h, j, j, 0.0);
                    S->set(h, i, j, 1.0);
                    S->set(h, j, i, 1.0);
                }
            }
        }
    }

    auto US = psi::linalg::doublet(Ua_, S, false, false);
    Ua_->copy(US);

    Ub_ = Ua_->clone();

    // save natural occupations to disk
    save_psi4_vector("NAT_OCC_ACTV", D1a_evals, doccpi);
    save_psi4_vector("NAT_OCC_CORE", D1c_evals, frzcpi);
    save_psi4_vector("NAT_OCC_VIRT", D1v_evals, holepi);
}

std::vector<std::vector<std::pair<int, int>>>
MRPT2_NOS::suggest_active_space(const psi::Vector& D1c_evals, const psi::Vector& D1v_evals,
                                const psi::Vector& D1a_evals) {
    auto nirrep = mo_space_info_->nirrep();
    std::vector<std::vector<std::pair<int, int>>> out(nirrep);

    // print original active space
    print_h2("Original Occupation Information (User Input)");

    std::string dash(15 + 6 * nirrep + 7, '-');
    outfile->Printf("\n    %s", dash.c_str());
    outfile->Printf("\n    %15c", ' ');
    for (size_t h = 0; h < nirrep; ++h) {
        outfile->Printf(" %5s", mo_space_info_->irrep_label(h).c_str());
    }
    outfile->Printf("    Sum");
    outfile->Printf("\n    %s", dash.c_str());

    for (const std::string& space_name : mo_space_info_->space_names()) {
        auto dim = mo_space_info_->dimension(space_name);
        if (dim.sum() == 0)
            continue;
        outfile->Printf("\n    %15s", space_name.c_str());
        for (size_t h = 0; h < nirrep; ++h) {
            outfile->Printf(" %5d", dim[h]);
        }
        outfile->Printf(" %6d", dim.sum());
    }
    outfile->Printf("\n    %s", dash.c_str());

    // suggest new active space
    print_h2("Active Space Suggested by MRPT2 Natural Orbitals");

    auto core_cutoff = 2.0 * options_->get_double("PT2NO_OCC_THRESHOLD");
    auto virt_cutoff = 2.0 * options_->get_double("PT2NO_VIR_THRESHOLD");
    outfile->Printf("\n    RESTRICTED_DOCC Threshold (Spin-Summed): %10.4e", core_cutoff);
    outfile->Printf("\n    RESTRICTED_UOCC Threshold (Spin-Summed): %10.4e\n", virt_cutoff);

    auto dim_frzc = mo_space_info_->dimension("FROZEN_DOCC");
    auto dim_hole = mo_space_info_->dimension("GENERALIZED HOLE");
    auto dim_actv = mo_space_info_->dimension("ACTIVE");
    auto dim_core = mo_space_info_->dimension("RESTRICTED_DOCC");
    auto dim_virt = mo_space_info_->dimension("RESTRICTED_UOCC");

    psi::Dimension newdim_actv(nirrep, "new active");
    psi::Dimension newdim_rdocc(nirrep, "new rdocc");
    psi::Dimension newdim_ruocc(nirrep, "new ruocc");

    psi::Dimension dim_actv_i(nirrep, "active (core like)");
    psi::Dimension dim_actv_a(nirrep, "active (virtual like)");

    for (size_t h = 0; h < nirrep; ++h) {
        auto ndocc = 0, nactv = 0, nuocc = 0;

        for (int i = 0; i < dim_core[h]; ++i) {
            if (D1c_evals.get(h, i) < core_cutoff)
                nactv++;
            else
                ndocc++;
        }

        auto nactv_i = 0, nactv_a = 0;
        for (int u = 0; u < dim_actv[h]; ++u) {
            auto nu = D1a_evals.get(h, u);
            if (nu > core_cutoff)
                nactv_i++;
            if (nu < virt_cutoff)
                nactv_a++;
        }
        dim_actv_i[h] = nactv_i;
        dim_actv_a[h] = nactv_a;

        for (int a = 0; a < dim_virt[h]; ++a) {
            auto na = D1v_evals.get(h, a);
            if (na > virt_cutoff)
                nactv++;
            else
                nuocc++;
        }

        newdim_actv[h] = nactv;
        newdim_rdocc[h] = ndocc;
        newdim_ruocc[h] = nuocc;
    }

    // save occupation numbers to disk
    std::unordered_map<std::string, psi::Dimension> newdims;
    newdims["ACTIVE"] = dim_actv + newdim_actv - dim_actv_a - dim_actv_i;
    newdims["RESTRICTED_DOCC"] = newdim_rdocc + dim_actv_i;
    newdims["RESTRICTED_UOCC"] = newdim_ruocc + dim_actv_a;
    newdims["FROZEN_DOCC"] = mo_space_info_->dimension("FROZEN_DOCC");
    newdims["FROZEN_UOCC"] = mo_space_info_->dimension("FROZEN_UOCC");

    dump_occupations("mrpt2_nos_occ", newdims);

    // print occupation numbers considered to be active
    dash = std::string(12 + 14 + 3, '-');
    outfile->Printf("\n    %s", dash.c_str());
    outfile->Printf("\n    Orbital Idx.   Occ. Number");
    outfile->Printf("\n    %s", dash.c_str());

    for (size_t h = 0; h < nirrep; ++h) {
        if (newdims["ACTIVE"][h] == 0 and dim_actv_i[h] == 0 and dim_actv_a[h] == 0)
            continue;

        auto irrep_label = mo_space_info_->irrep_label(h);
        auto core_shift = dim_frzc[h];
        auto actv_shift = dim_frzc[h] + dim_core[h];
        auto virt_shift = dim_frzc[h] + dim_hole[h];

        for (int i = newdim_rdocc[h]; i < dim_core[h]; ++i)
            outfile->Printf("\n    %7zu%-5s  %12.6e  +", i + core_shift, irrep_label.c_str(),
                            D1c_evals.get(h, i));
        for (int u = 0; u < dim_actv_i[h]; ++u)
            outfile->Printf("\n    %7zu%-5s  %12.6e  -", u + actv_shift, irrep_label.c_str(),
                            D1a_evals.get(h, u));
        for (int u = dim_actv_i[h], limit = dim_actv[h] - dim_actv_a[h]; u < limit; ++u)
            outfile->Printf("\n    %7zu%-5s  %12.6e  .", u + actv_shift, irrep_label.c_str(),
                            D1a_evals.get(h, u));
        for (int u = dim_actv[h] - dim_actv_a[h]; u < dim_actv[h]; ++u)
            outfile->Printf("\n    %7zu%-5s  %12.6e  -", u + actv_shift, irrep_label.c_str(),
                            D1a_evals.get(h, u));
        for (int a = 0, limit = dim_virt[h] - newdim_ruocc[h]; a < limit; ++a)
            outfile->Printf("\n    %7zu%-5s  %12.6e  +", a + virt_shift, irrep_label.c_str(),
                            D1v_evals.get(h, a));
        outfile->Printf("\n    %s", dash.c_str());
    }
    outfile->Printf("\n    Orbital count starts from 0 within each irrep.");
    outfile->Printf("\n    .: original active orbitals");
    if (newdim_actv.sum())
        outfile->Printf("\n    +: to be added as active orbitals");
    if (dim_actv_a.sum() or dim_actv_i.sum())
        outfile->Printf("\n    -: to be removed from active orbitals");

    if (newdim_actv.sum() == 0 and dim_actv_a.sum() == 0 and dim_actv_i.sum() == 0) {
        outfile->Printf("\n\n    MRPT2 natural orbitals finds no additional active orbitals.");
        return out;
    }

    print_h2("Occupation Information Suggested by MRPT2 Natural Orbitals");

    dash = std::string(15 + 6 * nirrep + 7, '-');
    outfile->Printf("\n    %s", dash.c_str());
    outfile->Printf("\n    %15c", ' ');
    for (size_t h = 0; h < nirrep; ++h) {
        outfile->Printf(" %5s", mo_space_info_->irrep_label(h).c_str());
    }
    outfile->Printf("    Sum");
    outfile->Printf("\n    %s", dash.c_str());

    for (const auto& space_name :
         {"FROZEN_DOCC", "RESTRICTED_DOCC", "ACTIVE", "RESTRICTED_UOCC", "FROZEN_UOCC"}) {
        const auto& dim = newdims[space_name];
        outfile->Printf("\n    %15s", space_name);
        for (size_t h = 0; h < nirrep; ++h) {
            outfile->Printf(" %5d", dim[h]);
        }
        outfile->Printf(" %6d", dim.sum());
    }
    outfile->Printf("\n    %s", dash.c_str());

    // determine rotations outside the active space
    for (size_t h = 0; h < nirrep; ++h) {
        if (newdim_actv[h] == 0)
            continue;
        auto shift = dim_frzc[h] + dim_core[h];
        auto current_core = shift - 1;
        for (int u = 0; u < dim_actv_i[h]; ++u) {
            out[h].emplace_back(u + shift, current_core--);
        }

        auto current_virt = shift + dim_actv[h];
        shift += dim_actv[h] - dim_actv_a[h];
        for (int u = dim_actv_a[h] - 1; u >= 0; --u) {
            out[h].emplace_back(u + shift, current_virt++);
        }
    }
    return out;
}
} // namespace forte