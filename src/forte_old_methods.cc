/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2019 by its authors (see COPYING, COPYING.LESSER,
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

#include <cmath>
#include <memory>

#include "boost/format.hpp"

#include <ambit/tensor.h>

#include "psi4/libdpd/dpd.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libtrans/integraltransform.h"
#include "psi4/psi4-dec.h"
#include "psi4/psifiles.h"

#include "base_classes/reference.h"
#include "base_classes/scf_info.h"
#include "base_classes/mo_space_info.h"
#include "base_classes/state_info.h"
#include "base_classes/active_space_solver.h"

#include "integrals/integrals.h"
#include "integrals/active_space_integrals.h"

#include "casscf/casscf.h"

#include "orbital-helpers/localize.h"
#include "orbital-helpers/es-nos.h"
#include "orbital-helpers/mp2_nos.h"
#include "orbital-helpers/semi_canonicalize.h"

#include "sci/aci.h"
#include "sci/asci.h"
#include "sci/mrci.h"

#include "cc/cc.h"
#include "orbital-helpers/ci-no/ci-no.h"
#include "orbital-helpers/ci-no/mrci-no.h"
#include "mrdsrg-so/mrdsrg_so.h"
#include "mrdsrg-so/so-mrdsrg.h"
#include "mrdsrg-spin-adapted/dsrg_mrpt.h"
#include "mrdsrg-spin-integrated/active_dsrgpt2.h"
#include "mrdsrg-spin-integrated/dsrg_mrpt3.h"
#include "mrdsrg-spin-integrated/mcsrgpt2_mo.h"
#include "mrdsrg-spin-integrated/mrdsrg.h"
#include "mrdsrg-spin-integrated/dwms_mrpt2.h"
//#include "mrdsrg-spin-integrated/dsrg_mrpt2.h"
//#include "mrdsrg-spin-integrated/three_dsrg_mrpt2.h"
//#include "mrdsrg-spin-integrated/active_dsrgpt2.h"

#include "pci/ewci.h"
#include "pci/pci.h"
#include "pci/pci_hashvec.h"
#include "pci/pci_simple.h"

#include "v2rdm/v2rdm.h"
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

double forte_old_methods(psi::SharedWavefunction ref_wfn, psi::Options& options,
                         std::shared_ptr<ForteIntegrals> ints,
                         std::shared_ptr<MOSpaceInfo> mo_space_info) {
    timer method_timer("Method");
    //    if (options.get_str("ALTERNATIVE_CASSCF") == "FTHF") {
    //        auto FTHF = std::make_shared<FiniteTemperatureHF>(ref_wfn, options, mo_space_info);
    //        FTHF->compute_energy();
    //        ints->retransform_integrals();
    //    }

    double final_energy = 0.0;

    size_t nroot = options.get_int("NROOT");
    StateInfo state = make_state_info_from_psi_wfn(ref_wfn); // TODO move py-side
    auto scf_info = std::make_shared<SCFInfo>(ref_wfn);
    auto forte_options = std::make_shared<ForteOptions>(options);
    // generate a list of states with their own weights
    auto state_weights_list = make_state_weights_list(forte_options, ref_wfn);

    if (options.get_bool("CASSCF_REFERENCE") == true or options.get_str("JOB_TYPE") == "CASSCF") {
        auto as_ints = make_active_space_ints(mo_space_info, ints, "ACTIVE", {{"RESTRICTED_DOCC"}});
        auto casscf = std::make_shared<CASSCF>(state, nroot, std::make_shared<SCFInfo>(ref_wfn),
                                               forte_options, mo_space_info, as_ints);
        final_energy = casscf->compute_energy();
    }
    if (options.get_bool("MP2_NOS")) {
        auto mp2_nos = std::make_shared<MP2_NOS>(ref_wfn, options, ints, mo_space_info);
    }
    if (options.get_bool("CINO")) {
        auto cino = std::make_shared<CINO>(ref_wfn, options, ints, mo_space_info);
        cino->compute_energy();
    }
    if (options.get_bool("MRCINO")) {
        auto mrcino = std::make_shared<MRCINO>(ref_wfn, options, ints, mo_space_info);
        final_energy = mrcino->compute_energy();
    }
    if (options.get_bool("LOCALIZE")) {
        auto localize = std::make_shared<LOCALIZE>(ref_wfn, options, ints, mo_space_info);
        localize->split_localize();
    }

    if (options.get_str("JOB_TYPE") == "MR-DSRG-PT2") {
        if (std::string actv_type = options.get_str("FCIMO_ACTV_TYPE");
            actv_type == "CIS" or actv_type == "CISD") {
            throw psi::PSIEXCEPTION("VCIS/VCISD is not supported for MR-DSRG-PT2");
        }
        int max_rdm_level = (options.get_str("THREEPDC") == "ZERO") ? 2 : 3;
        auto as_ints = make_active_space_ints(mo_space_info, ints, "ACTIVE", {{"RESTRICTED_DOCC"}});
        FCI_MO fci_mo(state, nroot, scf_info, forte_options, mo_space_info, as_ints);
        fci_mo.compute_energy();
        fci_mo.set_max_rdm_level(max_rdm_level);

        Reference reference = fci_mo.get_reference();
        SemiCanonical semi(forte_options, ints, mo_space_info);
        semi.semicanonicalize(reference, max_rdm_level);

        MCSRGPT2_MO mcsrgpt2_mo(reference, forte_options, ints, mo_space_info);
        final_energy = mcsrgpt2_mo.compute_energy();
    }

    if (std::string cas_type = options.get_str("JOB_TYPE");
        (cas_type == "FCI") or (cas_type == "ACI") or (cas_type == "ASCI")) {
        auto as_ints = make_active_space_ints(mo_space_info, ints, "ACTIVE", {{"RESTRICTED_DOCC"}});
        auto solver = make_active_space_solver(cas_type, state_weights_list, scf_info,
                                               mo_space_info, as_ints, forte_options);
        const auto& state_energies_list = solver->compute_energy();
        double average_energy =
            compute_average_state_energy(state_energies_list, state_weights_list);
        final_energy = average_energy;
    }

    if (options.get_str("JOB_TYPE") == "PCI") {
        auto as_ints = make_active_space_ints(mo_space_info, ints, "ACTIVE", {{"RESTRICTED_DOCC"}});
        auto pci = std::make_shared<ProjectorCI>(state, options.get_int("NROOT"),
                                                 std::make_shared<SCFInfo>(ref_wfn), forte_options,
                                                 mo_space_info, as_ints);
        for (int n = 0; n < options.get_int("NROOT"); ++n) {
            final_energy = pci->compute_energy();
        }
    }
    if (options.get_str("JOB_TYPE") == "PCI_HASHVEC") {
        auto as_ints = make_active_space_ints(mo_space_info, ints, "ACTIVE", {{"RESTRICTED_DOCC"}});
        auto pci_hashvec = std::make_shared<ProjectorCI_HashVec>(
            state, options.get_int("NROOT"), std::make_shared<SCFInfo>(ref_wfn), forte_options,
            mo_space_info, as_ints);
        for (int n = 0; n < options.get_int("NROOT"); ++n) {
            final_energy = pci_hashvec->compute_energy();
        }
    }
    if (options.get_str("JOB_TYPE") == "PCI_SIMPLE") {
        auto as_ints = make_active_space_ints(mo_space_info, ints, "ACTIVE", {{"RESTRICTED_DOCC"}});
        auto pci_simple = std::make_shared<ProjectorCI_Simple>(
            state, options.get_int("NROOT"), std::make_shared<SCFInfo>(ref_wfn), forte_options,
            mo_space_info, as_ints);
        for (int n = 0; n < options.get_int("NROOT"); ++n) {
            final_energy = pci_simple->compute_energy();
        }
    }
    if (options.get_str("JOB_TYPE") == "EWCI") {
        auto as_ints = make_active_space_ints(mo_space_info, ints, "ACTIVE", {{"RESTRICTED_DOCC"}});
        auto ewci = std::make_shared<ElementwiseCI>(state, options.get_int("NROOT"),
                                                    std::make_shared<SCFInfo>(ref_wfn),
                                                    forte_options, mo_space_info, as_ints);
        for (int n = 0; n < options.get_int("NROOT"); ++n) {
            final_energy = ewci->compute_energy();
        }
    }
    // if (options.get_str("JOB_TYPE") == "FCI") {
    //     auto fci = make_active_space_method("FCI", state, nroot, scf_info, mo_space_info, ints,
    //                                         forte_options);
    //     final_energy = fci->compute_energy();
    // }
    if (options.get_bool("USE_DMRGSCF")) {
#ifdef HAVE_CHEMPS2
        auto dmrg = std::make_shared<DMRGSCF>(state, std::make_shared<SCFInfo>(ref_wfn),
                                              forte_options, ints, mo_space_info);
        dmrg->set_iterations(options.get_int("DMRGSCF_MAX_ITER"));
        final_energy = dmrg->compute_energy();
#else
        throw psi::PSIEXCEPTION("Did not compile with CHEMPS2 so DMRG will not work");
#endif
    }
    if (options.get_str("JOB_TYPE") == "DMRG") {
#ifdef HAVE_CHEMPS2
        auto dmrg = std::make_shared<DMRGSolver>(state, std::make_shared<SCFInfo>(ref_wfn),
                                                 forte_options, ints, mo_space_info);
        dmrg->set_max_rdm(2);
        final_energy = dmrg->compute_energy();
#else
        throw psi::PSIEXCEPTION("Did not compile with CHEMPS2 so DMRG will not work");
#endif
    }
    if (options.get_str("JOB_TYPE") == "MRDSRG") {
        std::string cas_type = options.get_str("CAS_TYPE");
        int max_rdm_level = (options.get_str("THREEPDC") == "ZERO") ? 2 : 3;

        size_t na = mo_space_info->get_dimension("ACTIVE").sum();
        ambit::Tensor Ua = ambit::Tensor::build(CoreTensor, "Uactv a", {na, na});
        ambit::Tensor Ub = ambit::Tensor::build(CoreTensor, "Uactv b", {na, na});
        Ua.iterate([&](const std::vector<size_t>& i, double& value) {
            if (i[0] == i[1])
                value = 1.0;
        });
        Ub.iterate([&](const std::vector<size_t>& i, double& value) {
            if (i[0] == i[1])
                value = 1.0;
        });

        auto as_ints = make_active_space_ints(mo_space_info, ints, "ACTIVE", {{"RESTRICTED_DOCC"}});
        auto ci = make_active_space_solver(cas_type, state_weights_list, scf_info, mo_space_info,
                                           as_ints, forte_options);
        ci->set_max_rdm_level(3);
        ci->compute_energy();
        Reference reference = ci->get_reference();

        std::string relax_mode = options.get_str("RELAX_REF");

        SemiCanonical semi(forte_options, ints, mo_space_info);
        if (options.get_bool("SEMI_CANONICAL")) {
            semi.semicanonicalize(reference, max_rdm_level);
            Ua = semi.Ua_t();
            Ub = semi.Ub_t();
        }

        auto mrdsrg = std::make_shared<MRDSRG>(reference, std::make_shared<SCFInfo>(ref_wfn),
                                               forte_options, ints, mo_space_info);
        mrdsrg->set_Uactv(Ua, Ub);
        final_energy = mrdsrg->compute_energy();

        std::vector<double> e_relax;
        std::vector<double> e_dsrg;

        e_dsrg.push_back(final_energy);
    
        size_t niter = 0;
        size_t maxiter = forte_options->get_int("MAXITER_RELAX_REF");
        double old_energy = 0.0;
        double e_conv = forte_options->get_double("RELAX_E_CONVERGENCE");
        while(niter < 10){

            if( relax_mode == "NONE"){
                break;
            }

            // grab the effective Hamiltonian in the active space
            auto fci_ints = mrdsrg->compute_Heff_actv();
            // make a solver and run it
            auto relaxed_solver = make_active_space_solver(cas_type, state_weights_list, scf_info,
                                                           mo_space_info, fci_ints, forte_options);

            relaxed_solver->set_max_rdm_level(3);
            final_energy = relaxed_solver->compute_energy()[0].second[0];

            e_relax.push_back(final_energy);

            niter++;
            if( relax_mode == "ONCE"){
                break;
            } else if( (relax_mode == "TWICE") and (niter == 2) ){
                break;
            } else if( std::fabs(old_energy - final_energy) <= e_conv ){
                psi::Process::environment.globals["FULLY RELAXED ENERGY"] = final_energy;
                break;
            }

            Reference rel_reference;
            // Rotate reference to original semicanonical basis
            {
                Reference tmp = relaxed_solver->get_reference();
                rel_reference = semi.transform_reference(Ua,Ub,tmp,max_rdm_level); 
            }

            // Semicanonicalize reference and transform ints
            if (options.get_bool("SEMI_CANONICAL")) {
                semi.semicanonicalize(rel_reference, max_rdm_level);
                Ua = semi.Ua_t();
                Ub = semi.Ub_t();
            }

            // Compute MRDSRG in this basis, save the energy
            mrdsrg = std::make_shared<MRDSRG>(rel_reference, std::make_shared<SCFInfo>(ref_wfn),
                                                   forte_options, ints, mo_space_info);
            mrdsrg->set_Uactv(Ua, Ub);
            final_energy = mrdsrg->compute_energy();

            e_dsrg.push_back(final_energy);
            old_energy = final_energy;       
        }

        // set energies to psi4 environment
        psi::Process::environment.globals["UNRELAXED ENERGY"] = e_dsrg[0];
        psi::Process::environment.globals["PARTIALLY RELAXED ENERGY"] = e_relax[0];
        if (niter > 1) {
            psi::Process::environment.globals["RELAXED ENERGY"] = e_dsrg[1];
        }
    }

    if (options.get_str("JOB_TYPE") == "MRDSRG_SO") {
        std::string cas_type = options.get_str("CAS_TYPE");
        auto as_ints = make_active_space_ints(mo_space_info, ints, "ACTIVE", {{"RESTRICTED_DOCC"}});
        auto ci = make_active_space_solver(cas_type, state_weights_list, scf_info, mo_space_info,
                                           as_ints, forte_options);
        Reference reference = ci->get_reference();
        if (options.get_bool("SEMI_CANONICAL")) {
            SemiCanonical semi(forte_options, ints, mo_space_info);
            semi.semicanonicalize(reference);
        }
        std::shared_ptr<MRDSRG_SO> mrdsrg(new MRDSRG_SO(reference, options, ints, mo_space_info));
        final_energy = mrdsrg->compute_energy();
    }
    if (options.get_str("JOB_TYPE") == "ACTIVE-DSRGPT2") {
        ACTIVE_DSRGPT2 pt(std::make_shared<SCFInfo>(ref_wfn), forte_options, ints, mo_space_info);
        final_energy = pt.compute_energy();
    }
    if (options.get_str("JOB_TYPE") == "DWMS-DSRGPT2") {
        DWMS_DSRGPT2 dwms(std::make_shared<SCFInfo>(ref_wfn), forte_options, ints, mo_space_info);
        final_energy = dwms.compute_energy();
    }
    if (options.get_str("JOB_TYPE") == "DSRG_MRPT") {
        std::string cas_type = options.get_str("CAS_TYPE");
        int max_rdm_level = (options.get_str("THREEPDC") == "ZERO") ? 2 : 3;
        auto as_ints = make_active_space_ints(mo_space_info, ints, "ACTIVE", {{"RESTRICTED_DOCC"}});
        auto ci = make_active_space_solver(cas_type, state_weights_list, scf_info, mo_space_info,
                                           as_ints, forte_options);

        ci->compute_energy();
        Reference reference = ci->get_reference();

        if (options.get_bool("SEMI_CANONICAL")) {
            SemiCanonical semi(forte_options, ints, mo_space_info);
            semi.semicanonicalize(reference, max_rdm_level);
        }

        std::shared_ptr<DSRG_MRPT> dsrg(
            new DSRG_MRPT(reference, ref_wfn, options, ints, mo_space_info));
        if (options.get_str("RELAX_REF") == "NONE") {
            final_energy = dsrg->compute_energy();
        }
    }
    if (options.get_str("JOB_TYPE") == "DSRG-MRPT2") {
        std::string cas_type = options.get_str("CAS_TYPE");
        std::string actv_type = options.get_str("FCIMO_ACTV_TYPE");
        int max_rdm_level = (options.get_str("THREEPDC") == "ZERO") ? 2 : 3;

        size_t na = mo_space_info->get_dimension("ACTIVE").sum();
        ambit::Tensor Ua = ambit::Tensor::build(CoreTensor, "Uactv a", {na, na});
        ambit::Tensor Ub = ambit::Tensor::build(CoreTensor, "Uactv b", {na, na});
        Ua.iterate([&](const std::vector<size_t>& i, double& value) {
            if (i[0] == i[1])
                value = 1.0;
        });
        Ub.iterate([&](const std::vector<size_t>& i, double& value) {
            if (i[0] == i[1])
                value = 1.0;
        });

        auto as_ints = make_active_space_ints(mo_space_info, ints, "ACTIVE", {{"RESTRICTED_DOCC"}});
        auto ci = make_active_space_solver(cas_type, state_weights_list, scf_info, mo_space_info,
                                           as_ints, forte_options);
        ci->set_max_rdm_level(3);
        ci->compute_energy();
        Reference reference = ci->get_reference();

        if (options.get_bool("SEMI_CANONICAL")) {
            SemiCanonical semi(forte_options, ints, mo_space_info);
            // from FCI_MO
            // if (actv_type == "CIS" || actv_type == "CISD") {
            //    semi.set_actv_dims(fci_mo->actv_docc(), fci_mo->actv_virt());
            //}
            semi.semicanonicalize(reference, max_rdm_level);
            Ua = semi.Ua_t();
            Ub = semi.Ub_t();
        }
        std::shared_ptr<DSRG_MRPT2> dsrg_mrpt2 = std::make_shared<DSRG_MRPT2>(
            reference, std::make_shared<SCFInfo>(ref_wfn), forte_options, ints, mo_space_info);
        dsrg_mrpt2->set_Uactv(Ua, Ub);
        final_energy = dsrg_mrpt2->compute_energy();

        bool multi_state = options["AVG_STATE"].size() != 0;
        bool ref_relax = options.get_str("RELAX_REF") != "NONE";

        psi::Process::environment.globals["UNRELAXED ENERGY"] = final_energy;

        if (ref_relax || multi_state) {
            // grab the effective Hamiltonian in the active space
            auto fci_ints = dsrg_mrpt2->compute_Heff_actv();
            // make a solver and run it
            auto relaxed_solver = make_active_space_solver(cas_type, state_weights_list, scf_info,
                                                           mo_space_info, fci_ints, forte_options);

            const auto& state_energies_list = relaxed_solver->compute_energy();
            double average_energy =
                compute_average_state_energy(state_energies_list, state_weights_list);
            final_energy = average_energy;

            // For some test cases
            psi::Process::environment.globals["PARTIALLY RELAXED ENERGY"] = final_energy;
        }
    }
    if (options.get_str("JOB_TYPE") == "THREE-DSRG-MRPT2") {
        local_timer all_three_dsrg_mrpt2;

        if (options.get_str("INT_TYPE") == "CONVENTIONAL") {
            outfile->Printf("\n THREE-DSRG-MRPT2 is designed for DF/CD integrals");
            throw psi::PSIEXCEPTION("Please set INT_TYPE  DF/CHOLESKY for THREE_DSRG");
        }

        bool multi_state = options["AVG_STATE"].size() != 0;
        bool ref_relax = options.get_str("RELAX_REF") != "NONE";
        std::string cas_type = options.get_str("CAS_TYPE");
        std::string actv_type = options.get_str("FCIMO_ACTV_TYPE");
        int max_rdm_level = (options.get_str("THREEPDC") == "ZERO") ? 2 : 3;

        size_t na = mo_space_info->get_dimension("ACTIVE").sum();
        ambit::Tensor Ua = ambit::Tensor::build(CoreTensor, "Uactv a", {na, na});
        ambit::Tensor Ub = ambit::Tensor::build(CoreTensor, "Uactv b", {na, na});
        Ua.iterate([&](const std::vector<size_t>& i, double& value) {
            if (i[0] == i[1])
                value = 1.0;
        });
        Ub.iterate([&](const std::vector<size_t>& i, double& value) {
            if (i[0] == i[1])
                value = 1.0;
        });

        auto as_ints = make_active_space_ints(mo_space_info, ints, "ACTIVE", {{"RESTRICTED_DOCC"}});
        auto solver = make_active_space_solver(cas_type, state_weights_list, scf_info,
                                               mo_space_info, as_ints, forte_options);
        solver->set_max_rdm_level(3);
        solver->compute_energy();
        Reference reference = solver->get_reference();

        SemiCanonical semi(forte_options, ints, mo_space_info);
        semi.semicanonicalize(reference, max_rdm_level);
        Ua = semi.Ua_t();
        Ub = semi.Ub_t();

        auto three_dsrg_mrpt2 = std::make_shared<THREE_DSRG_MRPT2>(
            reference, std::make_shared<SCFInfo>(ref_wfn), forte_options, ints, mo_space_info);

        three_dsrg_mrpt2->set_Uactv(Ua, Ub);
        final_energy = three_dsrg_mrpt2->compute_energy();

        if (ref_relax || multi_state) {
            // grab the effective Hamiltonian in the active space
            auto fci_ints = three_dsrg_mrpt2->compute_Heff_actv();

            // make a solver and run it
            auto relaxed_solver = make_active_space_solver(cas_type, state_weights_list, scf_info,
                                                           mo_space_info, fci_ints, forte_options);
            const auto& state_energies_list = relaxed_solver->compute_energy();
            double average_energy =
                compute_average_state_energy(state_energies_list, state_weights_list);
            final_energy = average_energy;
        }
        outfile->Printf("\n CD/DF DSRG-MRPT2 took %8.5f s.", all_three_dsrg_mrpt2.get());
    }
    if (options.get_str("JOB_TYPE") == "DSRG-MRPT3") {
        std::string cas_type = options.get_str("CAS_TYPE");
        int max_rdm_level = (options.get_str("THREEPDC") == "ZERO") ? 2 : 3;

        size_t na = mo_space_info->get_dimension("ACTIVE").sum();
        ambit::Tensor Ua = ambit::Tensor::build(CoreTensor, "Uactv a", {na, na});
        ambit::Tensor Ub = ambit::Tensor::build(CoreTensor, "Uactv b", {na, na});
        Ua.iterate([&](const std::vector<size_t>& i, double& value) {
            if (i[0] == i[1])
                value = 1.0;
        });
        Ub.iterate([&](const std::vector<size_t>& i, double& value) {
            if (i[0] == i[1])
                value = 1.0;
        });

        auto as_ints = make_active_space_ints(mo_space_info, ints, "ACTIVE", {{"RESTRICTED_DOCC"}});
        auto ci = make_active_space_solver(cas_type, state_weights_list, scf_info, mo_space_info,
                                           as_ints, forte_options);

        ci->set_max_rdm_level(3);
        ci->compute_energy();
        Reference reference = ci->get_reference();

        SemiCanonical semi(forte_options, ints, mo_space_info);
        semi.semicanonicalize(reference, max_rdm_level);
        Ua = semi.Ua_t();
        Ub = semi.Ub_t();
        auto dsrg_mrpt3 = std::make_shared<DSRG_MRPT3>(
            reference, std::make_shared<SCFInfo>(ref_wfn), forte_options, ints, mo_space_info);
        dsrg_mrpt3->set_Uactv(Ua, Ub);
        final_energy = dsrg_mrpt3->compute_energy();
        bool multi_state = options["AVG_STATE"].size() != 0;
        bool ref_relax = options.get_str("RELAX_REF") != "NONE";

        if (ref_relax || multi_state) {
            // grab the effective Hamiltonian in the active space
            auto fci_ints = dsrg_mrpt3->compute_Heff_actv();
            // make a solver and run it
            auto relaxed_solver = make_active_space_solver(cas_type, state_weights_list, scf_info,
                                                           mo_space_info, fci_ints, forte_options);
            const auto& state_energies_list = relaxed_solver->compute_energy();
            double average_energy =
                compute_average_state_energy(state_energies_list, state_weights_list);
            final_energy = average_energy;
        }
    }

    if (options.get_str("JOB_TYPE") == "SOMRDSRG") {
        std::string cas_type = options.get_str("CAS_TYPE");
        int max_rdm_level = (options.get_str("THREEPDC") == "ZERO") ? 2 : 3;
        auto as_ints = make_active_space_ints(mo_space_info, ints, "ACTIVE", {{"RESTRICTED_DOCC"}});
        auto solver = make_active_space_solver(cas_type, state_weights_list, scf_info,
                                               mo_space_info, as_ints, forte_options);
        solver->set_max_rdm_level(max_rdm_level);
        solver->compute_energy();
        Reference reference = solver->get_reference();

        if (options.get_bool("SEMI_CANONICAL")) {
            SemiCanonical semi(forte_options, ints, mo_space_info);
            semi.semicanonicalize(reference, max_rdm_level);
        }
        std::shared_ptr<SOMRDSRG> somrdsrg(
            new SOMRDSRG(reference, ref_wfn, options, ints, mo_space_info));
        final_energy = somrdsrg->compute_energy();
    }

    if (options.get_str("JOB_TYPE") == "CC") {
        auto cc = std::make_shared<CC>(ints, mo_space_info);
        final_energy = cc->compute_energy();
    }

    if (options.get_str("JOB_TYPE") == "MRCISD") {
        if (options.get_bool("ACI_NO")) {
            auto as_ints =
                make_active_space_ints(mo_space_info, ints, "ACTIVE", {{"RESTRICTED_DOCC"}});

            auto aci =
                std::make_shared<AdaptiveCI>(state, nroot, std::make_shared<SCFInfo>(ref_wfn),
                                             forte_options, mo_space_info, as_ints);
            aci->compute_energy();
            aci->compute_nos();
        }
        auto as_ints = make_active_space_ints(mo_space_info, ints, "ACTIVE", {{"RESTRICTED_DOCC"}});
        auto aci = std::make_shared<AdaptiveCI>(state, nroot, std::make_shared<SCFInfo>(ref_wfn),
                                                forte_options, mo_space_info, as_ints);
        aci->compute_energy();

        DeterminantHashVec reference = aci->get_wavefunction();
        auto mrci = std::make_shared<MRCI>(ref_wfn, options, ints, mo_space_info, reference);
        final_energy = mrci->compute_energy();
    }
    return final_energy;
}
} // namespace forte
