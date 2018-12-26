/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER,
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

#include "sci/aci.h"
#include "sci/asci.h"
#include "base_classes/reference.h"
#include "base_classes/scf_info.h"
#include "helpers/blockedtensorfactory.h"
#include "casscf/casscf.h"
#include "cc/cc.h"
#include "orbital-helpers/ci-no/ci-no.h"
#include "orbital-helpers/ci-no/mrci-no.h"
#include "sparse_ci/determinant_hashvector.h"
#include "fci/fci_solver.h"
#include "integrals/active_space_integrals.h"
#include "sci/fci_mo.h"
#include "finite_temperature/finite_temperature.h"
#include "base_classes/mo_space_info.h"
#include "sci/mrci.h"
#include "mrdsrg-so/mrdsrg_so.h"
#include "mrdsrg-so/so-mrdsrg.h"
#include "mrdsrg-spin-adapted/dsrg_mrpt.h"
#include "mrdsrg-spin-integrated/active_dsrgpt2.h"
#include "mrdsrg-spin-integrated/dsrg_mrpt2.h"
#include "mrdsrg-spin-integrated/dsrg_mrpt3.h"
#include "mrdsrg-spin-integrated/mcsrgpt2_mo.h"
#include "mrdsrg-spin-integrated/mrdsrg.h"
#include "mrdsrg-spin-integrated/three_dsrg_mrpt2.h"
#include "mrdsrg-spin-integrated/active_dsrgpt2.h"
#include "mrdsrg-spin-integrated/dwms_mrpt2.h"
#include "orbital-helpers/localize.h"
#include "orbital-helpers/es-nos.h"
#include "orbital-helpers/mp2_nos.h"
#include "orbital-helpers/semi_canonicalize.h"
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

    StateInfo state = make_state_info_from_psi_wfn(ref_wfn); // TODO move py-side
    auto scf_info = std::make_shared<SCFInfo>(ref_wfn);
    auto forte_options = std::make_shared<ForteOptions>(options);

    if (options.get_bool("CASSCF_REFERENCE") == true or options.get_str("JOB_TYPE") == "CASSCF") {
        auto casscf =
            std::make_shared<CASSCF>(state, std::make_shared<SCFInfo>(ref_wfn),
                                     std::make_shared<ForteOptions>(options), ints, mo_space_info);
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
        MCSRGPT2_MO mcsrgpt2_mo(std::make_shared<SCFInfo>(ref_wfn),
                                std::make_shared<ForteOptions>(options), ints, mo_space_info);
        final_energy = psi::Process::environment.globals["CURRENT ENERGY"];
    }
    if (options.get_str("JOB_TYPE") == "ASCI") {
        auto asci =
            make_active_space_solver("ASCI", state, scf_info, mo_space_info, ints, forte_options);
        final_energy = asci->compute_energy();
    }
    if (options.get_str("JOB_TYPE") == "ACI") {
        auto as_ints = make_active_space_ints(mo_space_info, ints, "ACTIVE", {{"RESTRICTED_DOCC"}});
        auto aci = std::make_shared<AdaptiveCI>(state, std::make_shared<SCFInfo>(ref_wfn),
                                                std::make_shared<ForteOptions>(options),
                                                mo_space_info, as_ints);
        final_energy = aci->compute_energy();
        if (options.get_bool("ACI_NO")) {
            aci->compute_nos();
        }
        if (options.get_bool("ACI_ADD_EXTERNAL_EXCITATIONS")) {
            DeterminantHashVec wfn = aci->get_wavefunction();
            aci->upcast_reference(wfn);
            aci->add_external_excitations(wfn);
        }
        if (options.get_bool("UNPAIRED_DENSITY")) {
            psi::SharedMatrix Ua;
            psi::SharedMatrix Ub;

            Ua = ref_wfn->Ca()->clone();
            Ub = ref_wfn->Ca()->clone();

            Ua->identity();
            Ub->identity();

            aci->unpaired_density(Ua, Ub);
        }
    }
    if (options.get_str("JOB_TYPE") == "PCI") {
        auto pci = std::make_shared<ProjectorCI>(state, std::make_shared<SCFInfo>(ref_wfn),
                                                 std::make_shared<ForteOptions>(options), ints,
                                                 mo_space_info);
        for (int n = 0; n < options.get_int("NROOT"); ++n) {
            final_energy = pci->compute_energy();
        }
    }
    if (options.get_str("JOB_TYPE") == "PCI_HASHVEC") {
        auto pci_hashvec = std::make_shared<ProjectorCI_HashVec>(
            state, std::make_shared<SCFInfo>(ref_wfn), std::make_shared<ForteOptions>(options),
            ints, mo_space_info);
        for (int n = 0; n < options.get_int("NROOT"); ++n) {
            final_energy = pci_hashvec->compute_energy();
        }
    }
    if (options.get_str("JOB_TYPE") == "PCI_SIMPLE") {
        auto pci_simple = std::make_shared<ProjectorCI_Simple>(
            state, std::make_shared<SCFInfo>(ref_wfn), std::make_shared<ForteOptions>(options),
            ints, mo_space_info);
        for (int n = 0; n < options.get_int("NROOT"); ++n) {
            final_energy = pci_simple->compute_energy();
        }
    }
    if (options.get_str("JOB_TYPE") == "EWCI") {
        auto ewci = std::make_shared<ElementwiseCI>(state, std::make_shared<SCFInfo>(ref_wfn),
                                                    std::make_shared<ForteOptions>(options), ints,
                                                    mo_space_info);
        for (int n = 0; n < options.get_int("NROOT"); ++n) {
            final_energy = ewci->compute_energy();
        }
    }
    if (options.get_str("JOB_TYPE") == "FCI") {
        auto fci =
            make_active_space_solver("FCI", state, scf_info, mo_space_info, ints, forte_options);
        final_energy = fci->compute_energy();
    }
    if (options.get_bool("USE_DMRGSCF")) {
#ifdef HAVE_CHEMPS2
        auto dmrg =
            std::make_shared<DMRGSCF>(state, std::make_shared<SCFInfo>(ref_wfn),
                                      std::make_shared<ForteOptions>(options), ints, mo_space_info);
        dmrg->set_iterations(options.get_int("DMRGSCF_MAX_ITER"));
        final_energy = dmrg->compute_energy();
#else
        throw psi::PSIEXCEPTION("Did not compile with CHEMPS2 so DMRG will not work");
#endif
    }
    if (options.get_str("JOB_TYPE") == "DMRG") {
#ifdef HAVE_CHEMPS2
        auto dmrg = std::make_shared<DMRGSolver>(state, std::make_shared<SCFInfo>(ref_wfn),
                                                 std::make_shared<ForteOptions>(options), ints,
                                                 mo_space_info);
        dmrg->set_max_rdm(2);
        final_energy = dmrg->compute_energy();
#else
        throw psi::PSIEXCEPTION("Did not compile with CHEMPS2 so DMRG will not work");
#endif
    }
    if (options.get_str("JOB_TYPE") == "CAS") {
        FCI_MO fci_mo(std::make_shared<SCFInfo>(ref_wfn), std::make_shared<ForteOptions>(options),
                      ints, mo_space_info);
        final_energy = fci_mo.compute_energy();
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

        if (cas_type == "CAS") {
            FCI_MO fci_mo(std::make_shared<SCFInfo>(ref_wfn),
                          std::make_shared<ForteOptions>(options), ints, mo_space_info);
            fci_mo.set_max_rdm_level(max_rdm_level);
            fci_mo.compute_energy();
            Reference reference = fci_mo.get_reference();

            if (options.get_bool("SEMI_CANONICAL")) {
                SemiCanonical semi(std::make_shared<ForteOptions>(options), ints, mo_space_info);
                semi.semicanonicalize(reference, max_rdm_level);
                Ua = semi.Ua_t();
                Ub = semi.Ub_t();
            }

            auto mrdsrg = std::make_shared<MRDSRG>(reference, std::make_shared<SCFInfo>(ref_wfn),
                                                   std::make_shared<ForteOptions>(options), ints,
                                                   mo_space_info);
            mrdsrg->set_Uactv(Ua, Ub);

            if (options["AVG_STATE"].size() != 0) {
                mrdsrg->set_p_spaces(fci_mo.p_spaces());
                mrdsrg->set_eigens(fci_mo.eigens());
                final_energy = mrdsrg->compute_energy_sa();
            } else {
                if (options.get_str("RELAX_REF") == "NONE") {
                    final_energy = mrdsrg->compute_energy();
                } else {
                    final_energy = mrdsrg->compute_energy_relaxed();
                }
            }
        } else if (cas_type == "FCI") {
            auto fci = make_active_space_solver("FCI", state, scf_info, mo_space_info, ints,
                                                forte_options);
            fci->set_max_rdm_level(max_rdm_level);
            fci->compute_energy();
            Reference reference = fci->get_reference();
            if (options.get_bool("SEMI_CANONICAL")) {
                SemiCanonical semi(std::make_shared<ForteOptions>(options), ints, mo_space_info);
                semi.semicanonicalize(reference, max_rdm_level);
                Ua = semi.Ua_t();
                Ub = semi.Ub_t();
            }

            auto mrdsrg = std::make_shared<MRDSRG>(reference, std::make_shared<SCFInfo>(ref_wfn),
                                                   std::make_shared<ForteOptions>(options), ints,
                                                   mo_space_info);
            mrdsrg->set_Uactv(Ua, Ub);

            if (options.get_str("RELAX_REF") == "NONE") {
                final_energy = mrdsrg->compute_energy();
            } else {
                final_energy = mrdsrg->compute_energy_relaxed();
            }
        } else if (cas_type == "ACI") {
            auto as_ints =
                make_active_space_ints(mo_space_info, ints, "ACTIVE", {{"RESTRICTED_DOCC"}});
            auto aci = std::make_shared<AdaptiveCI>(state, std::make_shared<SCFInfo>(ref_wfn),
                                                    std::make_shared<ForteOptions>(options),
                                                    mo_space_info, as_ints);
            aci->set_max_rdm(max_rdm_level);
            aci->compute_energy();
            Reference aci_reference = aci->get_reference();
            if (options.get_bool("ACI_NO")) {
                aci->compute_nos();
            }
            SemiCanonical semi(std::make_shared<ForteOptions>(options), ints, mo_space_info);
            semi.semicanonicalize(aci_reference, max_rdm_level);
            Ua = semi.Ua_t();
            Ub = semi.Ub_t();
            auto mrdsrg = std::make_shared<MRDSRG>(
                aci_reference, std::make_shared<SCFInfo>(ref_wfn),
                std::make_shared<ForteOptions>(options), ints, mo_space_info);
            mrdsrg->set_Uactv(Ua, Ub);
            if (options.get_str("RELAX_REF") == "NONE") {
                final_energy = mrdsrg->compute_energy();
            } else {
                final_energy = mrdsrg->compute_energy_relaxed();
            }
        }
    }
    if (options.get_str("JOB_TYPE") == "MRDSRG_SO") {
        FCI_MO fci_mo(std::make_shared<SCFInfo>(ref_wfn), std::make_shared<ForteOptions>(options),
                      ints, mo_space_info);
        fci_mo.compute_energy();
        Reference reference = fci_mo.get_reference();
        if (options.get_bool("SEMI_CANONICAL")) {
            SemiCanonical semi(std::make_shared<ForteOptions>(options), ints, mo_space_info);
            semi.semicanonicalize(reference);
        }
        std::shared_ptr<MRDSRG_SO> mrdsrg(new MRDSRG_SO(reference, options, ints, mo_space_info));
        final_energy = mrdsrg->compute_energy();
    }
    if (options.get_str("JOB_TYPE") == "ACTIVE-DSRGPT2") {
        ACTIVE_DSRGPT2 pt(std::make_shared<SCFInfo>(ref_wfn),
                          std::make_shared<ForteOptions>(options), ints, mo_space_info);
        final_energy = pt.compute_energy();
    }
    if (options.get_str("JOB_TYPE") == "DWMS-DSRGPT2") {
        DWMS_DSRGPT2 dwms(std::make_shared<SCFInfo>(ref_wfn),
                          std::make_shared<ForteOptions>(options), ints, mo_space_info);
        final_energy = dwms.compute_energy();
    }
    if (options.get_str("JOB_TYPE") == "DSRG_MRPT") {
        std::string cas_type = options.get_str("CAS_TYPE");
        int max_rdm_level = (options.get_str("THREEPDC") == "ZERO") ? 2 : 3;
        
        auto ci = make_active_space_solver(cas_type, state, scf_info, mo_space_info, ints,
                                           forte_options);
        ci->compute_energy();
        Reference reference = ci->get_reference();

        if (options.get_bool("SEMI_CANONICAL")) {
            SemiCanonical semi(std::make_shared<ForteOptions>(options), ints, mo_space_info);
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

        // Can't treat FCIMO the same way until we can get
        // actv_docc and actv_virt from base class
        Reference reference;
        if (cas_type == "CAS") {
            std::shared_ptr<FCI_MO> fci_mo = std::make_shared<FCI_MO>(
                std::make_shared<SCFInfo>(ref_wfn), std::make_shared<ForteOptions>(options), ints,
                mo_space_info);
            fci_mo->set_max_rdm_level(max_rdm_level);
            fci_mo->compute_energy();
            reference = fci_mo->get_reference();

            if (options.get_bool("SEMI_CANONICAL")) {
                SemiCanonical semi(std::make_shared<ForteOptions>(options), ints, mo_space_info);
                if (actv_type == "CIS" || actv_type == "CISD") {
                    semi.set_actv_dims(fci_mo->actv_docc(), fci_mo->actv_virt());
                }
                semi.semicanonicalize(reference, max_rdm_level);
                Ua = semi.Ua_t();
                Ub = semi.Ub_t();
            }
        } else {

            auto ci = make_active_space_solver(cas_type, state, scf_info, mo_space_info, ints,
                                               forte_options);
            ci->compute_energy();
            reference = ci->get_reference();

            if (options.get_bool("SEMI_CANONICAL")) {
                SemiCanonical semi(std::make_shared<ForteOptions>(options), ints, mo_space_info);
                semi.semicanonicalize(reference, max_rdm_level);
                Ua = semi.Ua_t();
                Ub = semi.Ub_t();
            }
        std::shared_ptr<DSRG_MRPT2> dsrg_mrpt2 = std::make_shared<DSRG_MRPT2>(
            reference, std::make_shared<SCFInfo>(ref_wfn),
            std::make_shared<ForteOptions>(options), ints, mo_space_info);
        dsrg_mrpt2->set_Uactv(Ua, Ub);
        if (options.get_str("RELAX_REF") != "NONE") {
            final_energy = dsrg_mrpt2->compute_energy_relaxed();
        } else {
            final_energy = dsrg_mrpt2->compute_energy();
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

        if (cas_type == "CAS") {

            if (options["AVG_STATE"].size() != 0) {
                std::string ms_type = options.get_str("DSRG_MULTI_STATE");
                if (ms_type != "SA_FULL") {
                    outfile->Printf("\n  SA_FULL is the ONLY available option "
                                    "in THREE-DSRG-MRPT2.");
                }
            }

            FCI_MO fci_mo(std::make_shared<SCFInfo>(ref_wfn),
                          std::make_shared<ForteOptions>(options), ints, mo_space_info);
            fci_mo.set_max_rdm_level(max_rdm_level);
            fci_mo.compute_energy();
            Reference reference = fci_mo.get_reference();

            if (options.get_bool("SEMI_CANONICAL")) {
                SemiCanonical semi(std::make_shared<ForteOptions>(options), ints, mo_space_info);
                if (actv_type == "CIS" || actv_type == "CISD") {
                    semi.set_actv_dims(fci_mo.actv_docc(), fci_mo.actv_virt());
                }
                semi.semicanonicalize(reference, max_rdm_level);
                Ua = semi.Ua_t();
                Ub = semi.Ub_t();
            }

            auto three_dsrg_mrpt2 = std::make_shared<THREE_DSRG_MRPT2>(
                reference, std::make_shared<SCFInfo>(ref_wfn),
                std::make_shared<ForteOptions>(options), ints, mo_space_info);
            three_dsrg_mrpt2->set_Uactv(Ua, Ub);

            if (actv_type == "CIS" || actv_type == "CISD") {
                three_dsrg_mrpt2->set_actv_occ(fci_mo.actv_occ());
                three_dsrg_mrpt2->set_actv_uocc(fci_mo.actv_uocc());
            }
            final_energy = three_dsrg_mrpt2->compute_energy();
            if (ref_relax || multi_state) {
                final_energy = three_dsrg_mrpt2->relax_reference_once();
            }

        } else if (cas_type == "V2RDM") {
            std::shared_ptr<V2RDM> v2rdm =
                std::make_shared<V2RDM>(ref_wfn, options, ints, mo_space_info);
            Reference reference = v2rdm->reference();
            auto dsrg_mrpt2 = std::make_shared<THREE_DSRG_MRPT2>(
                reference, std::make_shared<SCFInfo>(ref_wfn),
                std::make_shared<ForteOptions>(options), ints, mo_space_info);
            final_energy = dsrg_mrpt2->compute_energy();

        } else if (cas_type == "ACI") {

            Reference aci_reference;
            auto as_ints =
                make_active_space_ints(mo_space_info, ints, "ACTIVE", {{"RESTRICTED_DOCC"}});
            auto aci = std::make_shared<AdaptiveCI>(state, std::make_shared<SCFInfo>(ref_wfn),
                                                    std::make_shared<ForteOptions>(options),
                                                    mo_space_info, as_ints);
            aci->set_quiet(true);
            aci->set_max_rdm(max_rdm_level);
            aci->compute_energy();
            aci_reference = aci->get_reference();
            if (options.get_bool("ACI_NO")) {
                aci->compute_nos();
            }
            if (options.get_bool("ACI_ADD_EXTERNAL_EXCITATIONS")) {
                DeterminantHashVec wfn = aci->get_wavefunction();
                aci->upcast_reference(wfn);
            }
            if (options.get_bool("ESNOS")) {
                auto aci_wfn = aci->get_wavefunction();
                ESNO esno(ref_wfn, options, ints, mo_space_info, aci_wfn);
                esno.compute_nos();
                auto as_ints = make_active_space_ints(mo_space_info, ints, "ACTIVE",
                                                      {{"RESTRICTED_DOCC"}}); // maybe a BUG?
                auto aci2 = std::make_shared<AdaptiveCI>(state, std::make_shared<SCFInfo>(ref_wfn),
                                                         std::make_shared<ForteOptions>(options),
                                                         mo_space_info, as_ints);
                aci2->set_quiet(true);
                aci2->set_max_rdm(max_rdm_level);
                aci2->compute_energy();
                aci_reference = aci2->get_reference();
            }
            // if( options.get_bool("UNPAIRED_DENSITY")){
            //    aci->unpaired_density();
            //}
            SemiCanonical semi(std::make_shared<ForteOptions>(options), ints, mo_space_info);
            semi.semicanonicalize(aci_reference, max_rdm_level);
            Ua = semi.Ua_t();
            Ub = semi.Ub_t();
            auto three_dsrg_mrpt2 = std::make_shared<THREE_DSRG_MRPT2>(
                aci_reference, std::make_shared<SCFInfo>(ref_wfn),
                std::make_shared<ForteOptions>(options), ints, mo_space_info);

            three_dsrg_mrpt2->set_Uactv(Ua, Ub);
            final_energy = three_dsrg_mrpt2->compute_energy();
            if (ref_relax || multi_state) {
                if (options.get_bool("UNPAIRED_DENSITY")) {
                    psi::SharedMatrix Ua_f = semi.Ua();
                    psi::SharedMatrix Ub_f = semi.Ub();
                    three_dsrg_mrpt2->set_Ufull(Ua_f, Ub_f);
                }
                final_energy = three_dsrg_mrpt2->relax_reference_once();
            }

            //  if( options.get_bool("UNPAIRED_DENSITY")){
            //      psi::SharedMatrix Uam = semi.Ua();
            //      psi::SharedMatrix Ubm = semi.Ub();
            //      aci->unpaired_density(Uam, Ubm);
            //  }

        } else if (cas_type == "FCI") {
            auto fci = make_active_space_solver("FCI", state, scf_info, mo_space_info, ints,
                                                forte_options);
            fci->set_max_rdm_level(max_rdm_level);
            fci->compute_energy();
            Reference reference = fci->get_reference();
            if (options.get_bool("SEMI_CANONICAL")) {
                SemiCanonical semi(std::make_shared<ForteOptions>(options), ints, mo_space_info);
                semi.semicanonicalize(reference, max_rdm_level);
                Ua = semi.Ua_t();
                Ub = semi.Ub_t();
            }

            auto three_dsrg_mrpt2 = std::make_shared<THREE_DSRG_MRPT2>(
                reference, std::make_shared<SCFInfo>(ref_wfn),
                std::make_shared<ForteOptions>(options), ints, mo_space_info);
            three_dsrg_mrpt2->set_Uactv(Ua, Ub);
            final_energy = three_dsrg_mrpt2->compute_energy();
            if (ref_relax || multi_state) {
                final_energy = three_dsrg_mrpt2->relax_reference_once();
            }

        } else if (cas_type == "DMRG") {
#ifdef HAVE_CHEMPS2
            auto dmrg = std::make_shared<DMRGSolver>(state, std::make_shared<SCFInfo>(ref_wfn),
                                                     std::make_shared<ForteOptions>(options), ints,
                                                     mo_space_info);
            dmrg->set_max_rdm(max_rdm_level);
            dmrg->compute_energy();

            Reference dmrg_reference = dmrg->reference();
            if (options.get_bool("SEMI_CANONICAL")) {
                SemiCanonical semi(std::make_shared<ForteOptions>(options), ints, mo_space_info);
                semi.semicanonicalize(dmrg_reference, max_rdm_level);
            }

            std::shared_ptr<THREE_DSRG_MRPT2> three_dsrg_mrpt2(
                new THREE_DSRG_MRPT2(dmrg_reference, std::make_shared<SCFInfo>(ref_wfn),
                                     std::make_shared<ForteOptions>(options), ints, mo_space_info));
            final_energy = three_dsrg_mrpt2->compute_energy();
            if (ref_relax || multi_state) {
                final_energy = three_dsrg_mrpt2->relax_reference_once();
            }
#endif
        } else if (cas_type == "CASSCF") {
            auto casscf = std::make_shared<CASSCF>(state, std::make_shared<SCFInfo>(ref_wfn),
                                                   std::make_shared<ForteOptions>(options), ints,
                                                   mo_space_info);
            casscf->compute_energy();
            Reference casscf_reference = casscf->casscf_reference();

            if (options.get_bool("SEMI_CANONICAL")) {
                SemiCanonical semi(std::make_shared<ForteOptions>(options), ints, mo_space_info);
                semi.semicanonicalize(casscf_reference, max_rdm_level);
                Ua = semi.Ua_t();
                Ub = semi.Ub_t();
            }

            auto three_dsrg_mrpt2 = std::make_shared<THREE_DSRG_MRPT2>(
                casscf_reference, std::make_shared<SCFInfo>(ref_wfn),
                std::make_shared<ForteOptions>(options), ints, mo_space_info);
            three_dsrg_mrpt2->set_Uactv(Ua, Ub);
            final_energy = three_dsrg_mrpt2->compute_energy();
            if (ref_relax || multi_state) {
                final_energy = three_dsrg_mrpt2->relax_reference_once();
            }
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

        if (cas_type == "CAS") {
            auto fci_mo = std::make_shared<FCI_MO>(std::make_shared<SCFInfo>(ref_wfn),
                                                   std::make_shared<ForteOptions>(options), ints,
                                                   mo_space_info);
            fci_mo->set_max_rdm_level(max_rdm_level);
            fci_mo->compute_energy();
            Reference reference = fci_mo->get_reference();

            if (options.get_bool("SEMI_CANONICAL")) {
                SemiCanonical semi(std::make_shared<ForteOptions>(options), ints, mo_space_info);
                semi.semicanonicalize(reference, max_rdm_level);
                Ua = semi.Ua_t();
                Ub = semi.Ub_t();
            }

            auto dsrg_mrpt3 = std::make_shared<DSRG_MRPT3>(
                reference, std::make_shared<SCFInfo>(ref_wfn),
                std::make_shared<ForteOptions>(options), ints, mo_space_info);
            dsrg_mrpt3->set_Uactv(Ua, Ub);

            if (options["AVG_STATE"].size() != 0) {
                dsrg_mrpt3->set_p_spaces(fci_mo->p_spaces());
                dsrg_mrpt3->set_eigens(fci_mo->eigens());
                final_energy = dsrg_mrpt3->compute_energy_sa();
            } else {
                if (options.get_str("RELAX_REF") != "NONE") {
                    final_energy = dsrg_mrpt3->compute_energy_relaxed();
                } else {
                    final_energy = dsrg_mrpt3->compute_energy();
                }
            }
        } else if (cas_type == "ACI") {

            Reference aci_reference;
            {
                auto as_ints =
                    make_active_space_ints(mo_space_info, ints, "ACTIVE", {{"RESTRICTED_DOCC"}});
                auto aci = std::make_shared<AdaptiveCI>(state, std::make_shared<SCFInfo>(ref_wfn),
                                                        std::make_shared<ForteOptions>(options),
                                                        mo_space_info, as_ints);
                aci->set_quiet(true);
                aci->set_max_rdm(max_rdm_level);
                aci->compute_energy();
                aci_reference = aci->get_reference();
                if (options.get_bool("ACI_NO")) {
                    aci->compute_nos();
                }
                if (options.get_bool("ACI_ADD_EXTERNAL_EXCITATIONS")) {
                    DeterminantHashVec wfn = aci->get_wavefunction();
                    aci->upcast_reference(wfn);
                }
                if (options.get_bool("ESNOS")) {
                    auto aci_wfn = aci->get_wavefunction();
                    ESNO esno(ref_wfn, options, ints, mo_space_info, aci_wfn);
                    esno.compute_nos();
                    auto as_ints = make_active_space_ints(mo_space_info, ints, "ACTIVE",
                                                          {{"RESTRICTED_DOCC"}});
                    auto aci2 = std::make_shared<AdaptiveCI>(
                        state, std::make_shared<SCFInfo>(ref_wfn),
                        std::make_shared<ForteOptions>(options), mo_space_info, as_ints);
                    aci2->set_quiet(true);
                    aci2->set_max_rdm(max_rdm_level);
                    aci2->compute_energy();
                    aci_reference = aci2->get_reference();
                }
            }
            SemiCanonical semi(std::make_shared<ForteOptions>(options), ints, mo_space_info);
            semi.semicanonicalize(aci_reference, max_rdm_level);
            Ua = semi.Ua_t();
            Ub = semi.Ub_t();
            auto dsrg_mrpt3 = std::make_shared<DSRG_MRPT3>(
                aci_reference, std::make_shared<SCFInfo>(ref_wfn),
                std::make_shared<ForteOptions>(options), ints, mo_space_info);
            dsrg_mrpt3->set_Uactv(Ua, Ub);

            if (options.get_str("RELAX_REF") != "NONE") {
                final_energy = dsrg_mrpt3->compute_energy_relaxed();
            } else {
                final_energy = dsrg_mrpt3->compute_energy();
            }
        }
        if (cas_type == "FCI") {
            auto fci = make_active_space_solver("FCI", state, scf_info, mo_space_info, ints,
                                                forte_options);
            fci->set_max_rdm_level(max_rdm_level);
            fci->compute_energy();
            Reference reference = fci->get_reference();
            if (options.get_bool("SEMI_CANONICAL")) {
                SemiCanonical semi(std::make_shared<ForteOptions>(options), ints, mo_space_info);
                semi.semicanonicalize(reference, max_rdm_level);
                Ua = semi.Ua_t();
                Ub = semi.Ub_t();
            }

            auto dsrg_mrpt3 = std::make_shared<DSRG_MRPT3>(
                reference, std::make_shared<SCFInfo>(ref_wfn),
                std::make_shared<ForteOptions>(options), ints, mo_space_info);
            dsrg_mrpt3->set_Uactv(Ua, Ub);

            if (options.get_str("RELAX_REF") != "NONE") {
                final_energy = dsrg_mrpt3->compute_energy_relaxed();
            } else {
                final_energy = dsrg_mrpt3->compute_energy();
            }
        }
    }

    if (options.get_str("JOB_TYPE") == "SOMRDSRG") {
        int max_rdm_level = (options.get_str("THREEPDC") == "ZERO") ? 2 : 3;
        if (options.get_str("CAS_TYPE") == "CAS") {
            FCI_MO fci_mo(std::make_shared<SCFInfo>(ref_wfn),
                          std::make_shared<ForteOptions>(options), ints, mo_space_info);
            fci_mo.set_max_rdm_level(max_rdm_level);
            Reference reference = fci_mo.get_reference();
            if (options.get_bool("SEMI_CANONICAL")) {
                SemiCanonical semi(std::make_shared<ForteOptions>(options), ints, mo_space_info);
                semi.semicanonicalize(reference, max_rdm_level);
            }
            std::shared_ptr<SOMRDSRG> somrdsrg(
                new SOMRDSRG(reference, ref_wfn, options, ints, mo_space_info));
            final_energy = somrdsrg->compute_energy();
        }
        if (options.get_str("CAS_TYPE") == "FCI") {
            auto fci = make_active_space_solver("FCI", state, scf_info, mo_space_info, ints,
                                                forte_options);
            fci->set_max_rdm_level(max_rdm_level);
            fci->compute_energy();
            Reference reference = fci->get_reference();
            if (options.get_bool("SEMI_CANONICAL")) {
                SemiCanonical semi(std::make_shared<ForteOptions>(options), ints, mo_space_info);
                semi.semicanonicalize(reference, max_rdm_level);
            }
            std::shared_ptr<SOMRDSRG> somrdsrg(
                new SOMRDSRG(reference, ref_wfn, options, ints, mo_space_info));
            final_energy = somrdsrg->compute_energy();
        }
    }

    if (options.get_str("JOB_TYPE") == "CC") {
        auto cc = std::make_shared<CC>(ints, mo_space_info);
        final_energy = cc->compute_energy();
    }

    if (options.get_str("JOB_TYPE") == "MRCISD") {
        if (options.get_bool("ACI_NO")) {
            auto as_ints =
                make_active_space_ints(mo_space_info, ints, "ACTIVE", {{"RESTRICTED_DOCC"}});

            auto aci = std::make_shared<AdaptiveCI>(state, std::make_shared<SCFInfo>(ref_wfn),
                                                    std::make_shared<ForteOptions>(options),
                                                    mo_space_info, as_ints);
            aci->compute_energy();
            aci->compute_nos();
        }
        auto as_ints = make_active_space_ints(mo_space_info, ints, "ACTIVE", {{"RESTRICTED_DOCC"}});
        auto aci = std::make_shared<AdaptiveCI>(state, std::make_shared<SCFInfo>(ref_wfn),
                                                std::make_shared<ForteOptions>(options),
                                                mo_space_info, as_ints);
        aci->compute_energy();

        DeterminantHashVec reference = aci->get_wavefunction();
        auto mrci = std::make_shared<MRCI>(ref_wfn, options, ints, mo_space_info, reference);
        final_energy = mrci->compute_energy();
    }
    return final_energy;
}
} // namespace forte
