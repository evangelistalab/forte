/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER,
 * AUTHORS).
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


#include "psi4/libpsi4util/PsiOutStream.h"

#include "base_classes/scf_info.h"
#include "base_classes/state_info.h"
#include "base_classes/active_space_solver.h"
#include "casscf/casscf.h"
#include "mrdsrg-spin-integrated/mcsrgpt2_mo.h"

#include "helpers/timer.h"

#ifdef HAVE_CHEMPS2
#include "dmrg/dmrgscf.h"
#include "dmrg/dmrgsolver.h"
#endif

#ifdef HAVE_GA
#include <ga.h>
#include <macdecls.h>
#endif

#ifdef HAVE_MPI
#include <mpi.h>
#endif

using namespace psi;

namespace forte {

double forte_old_methods(psi::SharedWavefunction ref_wfn, std::shared_ptr<ForteOptions> options,
                         std::shared_ptr<ForteIntegrals> ints,
                         std::shared_ptr<MOSpaceInfo> mo_space_info) {
    timer method_timer("Method");
    //    if (options.get_str("ALTERNATIVE_CASSCF") == "FTHF") {
    //        auto FTHF = std::make_shared<FiniteTemperatureHF>(ref_wfn, options, mo_space_info);
    //        FTHF->compute_energy();
    //        ints->retransform_integrals();
    //    }

    double final_energy = 0.0;

    size_t nroot = options->get_int("NROOT");
    StateInfo state = make_state_info_from_psi_wfn(ref_wfn); // TODO move py-side
    auto scf_info = std::make_shared<SCFInfo>(ref_wfn);
    // generate a list of states with their own weights
    auto state_weights_map = make_state_weights_map(options, ref_wfn);
    auto state_map = to_state_nroots_map(state_weights_map);

    if (options->get_bool("CASSCF_REFERENCE") == true or options->get_str("JOB_TYPE") == "CASSCF") {
        auto as_ints = make_active_space_ints(mo_space_info, ints, "ACTIVE", {{"RESTRICTED_DOCC"}});
        auto casscf = std::make_shared<CASSCF>(state, nroot, std::make_shared<SCFInfo>(ref_wfn),
                                               options, mo_space_info, as_ints);
        final_energy = casscf->compute_energy();
        if (options->get_str("DERTYPE") == "FIRST") {
            casscf->compute_gradient();
        }
    }
    if (options->get_str("JOB_TYPE") == "MR-DSRG-PT2") {
        std::string cas_type = options->get_str("ACTIVE_SPACE_SOLVER");
        std::string actv_type = options->get_str("FCIMO_ACTV_TYPE");
        if (actv_type == "CIS" or actv_type == "CISD") {
            throw psi::PSIEXCEPTION("VCIS/VCISD is not supported for MR-DSRG-PT2");
        }
        int max_rdm_level = (options->get_str("THREEPDC") == "ZERO") ? 2 : 3;
        auto as_ints = make_active_space_ints(mo_space_info, ints, "ACTIVE", {{"RESTRICTED_DOCC"}});
        auto ci = make_active_space_solver(cas_type, state_map, scf_info, mo_space_info, as_ints,
                                           options);
        ci->compute_energy();

        RDMs rdms = ci->compute_average_rdms(state_weights_map, max_rdm_level);
        SemiCanonical semi(mo_space_info, ints, options);
        semi.semicanonicalize(rdms, max_rdm_level);

        MCSRGPT2_MO mcsrgpt2_mo(rdms, options, ints, mo_space_info);
        final_energy = mcsrgpt2_mo.compute_energy();
    }
    return final_energy;
}

/* THE FOLLOWING PROCEDURES ARE NOT TESTED
if (options.get_bool("USE_DMRGSCF")) {
#ifdef HAVE_CHEMPS2
        auto dmrg = std::make_shared<DMRGSCF>(state, std::make_shared<SCFInfo>(ref_wfn),
                                              options, ints, mo_space_info);
        dmrg->set_iterations(options.get_int("DMRGSCF_MAX_ITER"));
        final_energy = dmrg->compute_energy();
#else
        throw psi::PSIEXCEPTION("Did not compile with CHEMPS2 so DMRG will not work");
#endif
    }
    if (options.get_str("JOB_TYPE") == "DMRG") {
#ifdef HAVE_CHEMPS2
        auto dmrg = std::make_shared<DMRGSolver>(state, std::make_shared<SCFInfo>(ref_wfn),
                                                 options, ints, mo_space_info);
        dmrg->set_max_rdm(2);
        final_energy = dmrg->compute_energy();
#else
        throw psi::PSIEXCEPTION("Did not compile with CHEMPS2 so DMRG will not work");
#endif
    }
    if (options.get_str("JOB_TYPE") == "MRDSRG_SO") {
        std::string cas_type = options.get_str("ACTIVE_SPACE_SOLVER");
        auto as_ints = make_active_space_ints(mo_space_info, ints, "ACTIVE", {{"RESTRICTED_DOCC"}});
        auto ci = make_active_space_solver(cas_type, state_map, scf_info, mo_space_info, as_ints,
                                           options);
        RDMs rdms = ci->compute_average_rdms(state_weights_map, 3);
        if (options.get_bool("SEMI_CANONICAL")) {
            SemiCanonical semi(mo_space_info, ints, options);
            semi.semicanonicalize(rdms);
        }
        std::shared_ptr<MRDSRG_SO> mrdsrg(new MRDSRG_SO(rdms, options, ints, mo_space_info));
        final_energy = mrdsrg->compute_energy();
    }
    if (options.get_str("JOB_TYPE") == "ACTIVE-DSRGPT2") {
        ACTIVE_DSRGPT2 pt(std::make_shared<SCFInfo>(ref_wfn), options, ints, mo_space_info);
        final_energy = pt.compute_energy();
    }
    if (options.get_str("JOB_TYPE") == "DWMS-DSRGPT2") {
        DWMS_DSRGPT2 dwms(std::make_shared<SCFInfo>(ref_wfn), options, ints, mo_space_info);
        final_energy = dwms.compute_energy();
    }
    if (options.get_str("JOB_TYPE") == "DSRG_MRPT") {
        std::string cas_type = options.get_str("ACTIVE_SPACE_SOLVER");
        int max_rdm_level = (options.get_str("THREEPDC") == "ZERO") ? 2 : 3;
        auto as_ints = make_active_space_ints(mo_space_info, ints, "ACTIVE", {{"RESTRICTED_DOCC"}});
        auto ci = make_active_space_solver(cas_type, state_map, scf_info, mo_space_info, as_ints,
                                           options);

        ci->compute_energy();
        RDMs rdms = ci->compute_average_rdms(state_weights_map, 3);

        if (options.get_bool("SEMI_CANONICAL")) {
            SemiCanonical semi(mo_space_info, ints, options);
            semi.semicanonicalize(rdms, max_rdm_level);
        }

        std::shared_ptr<DSRG_MRPT> dsrg(
            new DSRG_MRPT(rdms, ref_wfn, options, ints, mo_space_info));
        if (options.get_str("RELAX_REF") == "NONE") {
            final_energy = dsrg->compute_energy();
        }
    }
    if (options.get_str("JOB_TYPE") == "SOMRDSRG") {
        std::string cas_type = options.get_str("ACTIVE_SPACE_SOLVER");
        int max_rdm_level = (options.get_str("THREEPDC") == "ZERO") ? 2 : 3;
        auto as_ints = make_active_space_ints(mo_space_info, ints, "ACTIVE", {{"RESTRICTED_DOCC"}});
        auto solver = make_active_space_solver(cas_type, state_map, scf_info, mo_space_info,
                                               as_ints, options);
        solver->compute_energy();
        RDMs rdms = solver->compute_average_rdms(state_weights_map, max_rdm_level);

        if (options.get_bool("SEMI_CANONICAL")) {
            SemiCanonical semi(mo_space_info, ints, options);
            semi.semicanonicalize(rdms, max_rdm_level);
        }
        std::shared_ptr<SOMRDSRG> somrdsrg(
            new SOMRDSRG(rdms, ref_wfn, options, ints, mo_space_info));
        final_energy = somrdsrg->compute_energy();
    }

    if (options.get_str("JOB_TYPE") == "MRCISD") {
        if (options.get_bool("ACI_NO")) {
            auto as_ints =
                make_active_space_ints(mo_space_info, ints, "ACTIVE", {{"RESTRICTED_DOCC"}});

            auto aci =
                std::make_shared<AdaptiveCI>(state, nroot, std::make_shared<SCFInfo>(ref_wfn),
                                             options, mo_space_info, as_ints);
            aci->compute_energy();
            aci->compute_nos();
        }
        auto as_ints = make_active_space_ints(mo_space_info, ints, "ACTIVE", {{"RESTRICTED_DOCC"}});
        auto aci = std::make_shared<AdaptiveCI>(state, nroot, std::make_shared<SCFInfo>(ref_wfn),
                                                options, mo_space_info, as_ints);
        aci->compute_energy();

        DeterminantHashVec reference = aci->get_PQ_space();
        auto mrci = std::make_shared<MRCI>(ref_wfn, options, ints, mo_space_info, reference);
        final_energy = mrci->compute_energy();
    }
*/

} // namespace forte
