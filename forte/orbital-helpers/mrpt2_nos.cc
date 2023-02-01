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

void MRPT2_NOS::compute_transformation() {
    // compute unrelaxed 1-RDMs for core and virtual blocks
    mrpt2_->build_1rdm_unrelaxed(D1c_, D1v_);

    psi::Process::environment.arrays["MRPT2 1RDM CC"] = D1c_;
    psi::Process::environment.arrays["MRPT2 1RDM VV"] = D1v_;

    // diagonalize unrelaxed 1-RDM
    auto core_mospi = D1c_->rowspi();
    auto virt_mospi = D1v_->rowspi();

    psi::Vector D1c_evals("D1c_evals", core_mospi);
    psi::Matrix D1c_evecs("D1c_evecs", core_mospi, core_mospi);
    D1c_->diagonalize(D1c_evecs, D1c_evals, descending);

    psi::Vector D1v_evals("D1v_evals", virt_mospi);
    psi::Matrix D1v_evecs("D1v_evecs", virt_mospi, virt_mospi);
    D1v_->diagonalize(D1v_evecs, D1v_evals, descending);

    // print natural orbitals
    if (options_->get_bool("NAT_ORBS_PRINT")) {
        D1c_evals.print();
        D1v_evals.print();
    }

    // suggest active space
    if (options_->get_bool("NAT_ACT")) {
        suggest_active_space(D1c_evals, D1v_evals);
    }

    // build transformation matrix
    auto nmopi = mo_space_info_->dimension("ALL");
    auto ncmopi = mo_space_info_->dimension("CORRELATED");
    auto frzcpi = mo_space_info_->dimension("FROZEN_DOCC");

    Ua_ = std::make_shared<psi::Matrix>("Ua", nmopi, nmopi);
    Ua_->identity();

    Slice slice_core(frzcpi, mo_space_info_->dimension("INACTIVE_DOCC"));
    Ua_->set_block(slice_core, D1c_evecs);

    Slice slice_virt(frzcpi + mo_space_info_->dimension("GENERALIZED HOLE"), frzcpi + ncmopi);
    Ua_->set_block(slice_virt, D1v_evecs);

    Ub_ = Ua_->clone();
}

void MRPT2_NOS::suggest_active_space(const psi::Vector& D1c_evals, const psi::Vector& D1v_evals) {
    // print original active space
    print_h2("Original Occupation Information (User Input)");

    auto nirrep = mo_space_info_->nirrep();
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

    auto dim_core = mo_space_info_->dimension("RESTRICTED_DOCC");
    auto dim_virt = mo_space_info_->dimension("RESTRICTED_UOCC");
    std::vector<int> newdim_rdocc(nirrep), newdim_actv(nirrep), newdim_ruocc(nirrep);

    for (size_t h = 0; h < nirrep; ++h) {
        auto ndocc = 0, nactv = 0, nuocc = 0;
        for (int i = 0; i < dim_core[h]; ++i) {
            if (D1c_evals.get(h, i) < core_cutoff)
                nactv++;
            else
                ndocc++;
        }
        for (int a = 0; a < dim_virt[h]; ++a) {
            if (D1v_evals.get(h, a) > virt_cutoff)
                nactv++;
            else
                nuocc++;
        }
        newdim_actv[h] = nactv;
        newdim_rdocc[h] = ndocc;
        newdim_ruocc[h] = nuocc;
    }

    if (psi::Dimension(newdim_actv).sum() == 0) {
        outfile->Printf("\n    MRPT2 natural orbitals finds no additional active orbitals.");
        return;
    }

    dash = std::string(5 + 9 + 14, '-');
    outfile->Printf("\n    %s", dash.c_str());
    outfile->Printf("\n    Irrep  Orbital   Occ. Number");
    outfile->Printf("\n    %s", dash.c_str());

    for (size_t h = 0; h < nirrep; ++h) {
        if (newdim_actv[h] == 0)
            continue;

        auto irrep_label = mo_space_info_->irrep_label(h);
        auto core_shift = dim_frzc[h];
        auto virt_shift = dim_frzc[h] + dim_hole[h];

        for (int i = newdim_rdocc[h]; i < dim_core[h]; ++i) {
            outfile->Printf("\n    %5s  %7zu  %12.6e", irrep_label.c_str(), i + core_shift,
                            D1c_evals.get(h, i));
        }

        for (int a = 0; a < dim_virt[h] - newdim_ruocc[h]; ++a) {
            outfile->Printf("\n    %5s  %7zu  %12.6e", irrep_label.c_str(), a + virt_shift,
                            D1v_evals.get(h, a));
        }

        outfile->Printf("\n    %s", dash.c_str());
    }

    std::unordered_map<std::string, psi::Dimension> newdims;
    newdims["ACTIVE"] = psi::Dimension(newdim_actv) + mo_space_info_->dimension("ACTIVE");
    newdims["RESTRICTED_DOCC"] = psi::Dimension(newdim_rdocc);
    newdims["RESTRICTED_UOCC"] = psi::Dimension(newdim_ruocc);
    newdims["FROZEN_DOCC"] = mo_space_info_->dimension("FROZEN_DOCC");
    newdims["FROZEN_UOCC"] = mo_space_info_->dimension("FROZEN_DOCC");

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

    // save occupation numbers to disk
    dump_occupations("mrpt2_nos_occ", newdims);
}

} // namespace forte