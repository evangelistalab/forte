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

#include "mini-boost/boost/format.hpp"
#include <ambit/tensor.h>

#include "psi4/libdpd/dpd.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libtrans/integraltransform.h"
#include "psi4/psi4-dec.h"
#include "psi4/psifiles.h"

#include "aci/aci.h"
#include "active_dsrgpt2.h"
#include "blockedtensorfactory.h"
#include "casscf.h"
#include "cc.h"
#include "ci-no/ci-no.h"
#include "determinant_hashvector.h"
#include "fci/fci.h"
#include "fci/fci_solver.h"
#include "fci/fci_integrals.h"
#include "fci_mo.h"
#include "fcimc.h"
#include "finite_temperature.h"
#include "helpers.h"
#include "mcsrgpt2_mo.h"
#include "mp2_nos.h"
#include "mrci.h"
#include "mrdsrg-so/mrdsrg_so.h"
#include "mrdsrg-so/so-mrdsrg.h"
#include "mrdsrg-spin-adapted/dsrg_mrpt.h"
#include "mrdsrg-spin-integrated/dsrg_mrpt2.h"
#include "mrdsrg-spin-integrated/dsrg_mrpt3.h"
#include "mrdsrg-spin-integrated/mrdsrg.h"
#include "mrdsrg-spin-integrated/three_dsrg_mrpt2.h"
#include "orbital-helper/localize.h"
#include "orbital-helper/es-nos.h"
#include "pci/ewci.h"
#include "pci/pci.h"
#include "pci/pci_hashvec.h"
#include "pci/pci_simple.h"
#include "reference.h"
#include "semi_canonicalize.h"
#include "sq.h"
#include "tensorsrg.h"
#include "v2rdm.h"

#ifdef HAVE_CHEMPS2
#include "dmrgscf.h"
#include "dmrgsolver.h"
#endif

#ifdef HAVE_GA
#include <ga.h>
#include <macdecls.h>
#endif

#ifdef HAVE_MPI
#include <mpi.h>
#endif

namespace psi {
namespace forte {

void forte_old_methods(SharedWavefunction ref_wfn, Options& options,
                       std::shared_ptr<ForteIntegrals> ints,
                       std::shared_ptr<MOSpaceInfo> mo_space_info, int my_proc) {
    timer method_timer("Method");
    if (options.get_str("ALTERNATIVE_CASSCF") == "FTHF") {
        auto FTHF = std::make_shared<FiniteTemperatureHF>(ref_wfn, options, mo_space_info);
        FTHF->compute_energy();
        ints->retransform_integrals();
    }

    if (options.get_bool("CASSCF_REFERENCE") == true or options.get_str("JOB_TYPE") == "CASSCF") {
        auto casscf = std::make_shared<CASSCF>(ref_wfn, options, ints, mo_space_info);
        casscf->compute_casscf();
    }
    if (options.get_bool("MP2_NOS")) {
        auto mp2_nos = std::make_shared<MP2_NOS>(ref_wfn, options, ints, mo_space_info);
    }
    if (options.get_bool("CINO")) {
        auto cino = std::make_shared<CINO>(ref_wfn, options, ints, mo_space_info);
        cino->compute_energy();
    }
    if (options.get_bool("LOCALIZE")) {
        auto localize = std::make_shared<LOCALIZE>(ref_wfn, options, ints, mo_space_info);
        localize->localize_orbitals();
    }

    if (options.get_str("JOB_TYPE") == "MR-DSRG-PT2") {
        MCSRGPT2_MO mcsrgpt2_mo(ref_wfn, options, ints, mo_space_info);
    }
    if (options.get_str("JOB_TYPE") == "FCIQMC") {
        auto fciqmc = std::make_shared<FCIQMC>(ref_wfn, options, ints, mo_space_info);
        fciqmc->compute_energy();
    }
    if (options.get_str("JOB_TYPE") == "ACI") {
        auto aci = std::make_shared<AdaptiveCI>(ref_wfn, options, ints, mo_space_info);
        aci->compute_energy();
        if (options.get_bool("ACI_NO")) {
            aci->compute_nos();
        }
        if (options.get_bool("ACI_ADD_EXTERNAL_SINGLES")) {
            DeterminantHashVec wfn = aci->get_wavefunction();
            aci->upcast_reference(wfn);
            aci->add_external_singles(wfn);
        }
    }
    if (options.get_str("JOB_TYPE") == "PCI") {
        auto pci = std::make_shared<ProjectorCI>(ref_wfn, options, ints, mo_space_info);
        for (int n = 0; n < options.get_int("NROOT"); ++n) {
            pci->compute_energy();
        }
    }
    if (options.get_str("JOB_TYPE") == "PCI_HASHVEC") {
        auto pci_hashvec =
            std::make_shared<ProjectorCI_HashVec>(ref_wfn, options, ints, mo_space_info);
        for (int n = 0; n < options.get_int("NROOT"); ++n) {
            pci_hashvec->compute_energy();
        }
    }
    if (options.get_str("JOB_TYPE") == "PCI_SIMPLE") {
        auto pci_simple =
            std::make_shared<ProjectorCI_Simple>(ref_wfn, options, ints, mo_space_info);
        for (int n = 0; n < options.get_int("NROOT"); ++n) {
            pci_simple->compute_energy();
        }
    }
    if (options.get_str("JOB_TYPE") == "EWCI") {
        auto ewci = std::make_shared<ElementwiseCI>(ref_wfn, options, ints, mo_space_info);
        for (int n = 0; n < options.get_int("NROOT"); ++n) {
            ewci->compute_energy();
        }
    }
    if (options.get_str("JOB_TYPE") == "FCI") {
        auto fci = std::make_shared<FCI>(ref_wfn, options, ints, mo_space_info);
        fci->compute_energy();
    }
    if (options.get_bool("USE_DMRGSCF")) {
#ifdef HAVE_CHEMPS2
        auto dmrg = std::make_shared<DMRGSCF>(ref_wfn, options, mo_space_info, ints);
        dmrg->set_iterations(options.get_int("DMRGSCF_MAX_ITER"));
        dmrg->compute_energy();
#else
        throw PSIEXCEPTION("Did not compile with CHEMPS2 so DMRG will not work");
#endif
    }
    if (options.get_str("JOB_TYPE") == "DMRG") {
#ifdef HAVE_CHEMPS2
        DMRGSolver dmrg(ref_wfn, options, mo_space_info, ints);
        dmrg.set_max_rdm(2);
        dmrg.compute_energy();
#else
        throw PSIEXCEPTION("Did not compile with CHEMPS2 so DMRG will not work");
#endif
    }
    if (options.get_str("JOB_TYPE") == "CAS") {
        FCI_MO fci_mo(ref_wfn, options, ints, mo_space_info);
        fci_mo.compute_energy();
    }
    if (options.get_str("JOB_TYPE") == "MRDSRG") {
        std::string cas_type = options.get_str("CAS_TYPE");
        int max_rdm_level = (options.get_str("THREEPDC") == "ZERO") ? 2 : 3;

        if (cas_type == "CAS") {
            FCI_MO fci_mo(ref_wfn, options, ints, mo_space_info);
            fci_mo.compute_energy();
            Reference reference = fci_mo.reference(max_rdm_level);

            if (options.get_bool("SEMI_CANONICAL")) {
                SemiCanonical semi(ref_wfn, ints, mo_space_info);
                semi.semicanonicalize(reference, max_rdm_level);
            }

            if (options["AVG_STATE"].size() != 0) {
                auto mrdsrg =
                    std::make_shared<MRDSRG>(reference, ref_wfn, options, ints, mo_space_info);
                mrdsrg->set_p_spaces(fci_mo.p_spaces());
                mrdsrg->set_eigens(fci_mo.eigens());
                mrdsrg->compute_energy_sa();
            } else {
                auto mrdsrg =
                    std::make_shared<MRDSRG>(reference, ref_wfn, options, ints, mo_space_info);
                if (options.get_str("RELAX_REF") == "NONE") {
                    mrdsrg->compute_energy();
                } else {
                    if (options.get_str("DSRG_TRANS_TYPE") == "CC") {
                        throw PSIEXCEPTION("Reference relaxation for CC-type DSRG "
                                           "transformation is not implemented yet.");
                    }
                    mrdsrg->compute_energy_relaxed();
                }
            }
        } else if (cas_type == "FCI") {
            auto fci = std::make_shared<FCI>(ref_wfn, options, ints, mo_space_info);
            fci->set_max_rdm_level(max_rdm_level);
            fci->compute_energy();
            Reference reference = fci->reference();
            if (options.get_bool("SEMI_CANONICAL")) {
                SemiCanonical semi(ref_wfn, ints, mo_space_info);
                semi.semicanonicalize(reference, max_rdm_level);
            }

            auto mrdsrg =
                std::make_shared<MRDSRG>(reference, ref_wfn, options, ints, mo_space_info);
            if (options.get_str("RELAX_REF") == "NONE") {
                mrdsrg->compute_energy();
            } else {
                if (options.get_str("DSRG_TRANS_TYPE") == "CC") {
                    throw PSIEXCEPTION("Reference relaxation for CC-type DSRG transformation "
                                       "is not implemented yet.");
                }
                mrdsrg->compute_energy_relaxed();
            }
        }
    }
    if (options.get_str("JOB_TYPE") == "MRDSRG_SO") {
        FCI_MO fci_mo(ref_wfn, options, ints, mo_space_info);
        fci_mo.compute_energy();
        Reference reference = fci_mo.reference();
        if (options.get_bool("SEMI_CANONICAL")) {
            SemiCanonical semi(ref_wfn, ints, mo_space_info);
            semi.semicanonicalize(reference);
        }
        std::shared_ptr<MRDSRG_SO> mrdsrg(new MRDSRG_SO(reference, options, ints, mo_space_info));
        mrdsrg->compute_energy();
    }
    if (options.get_str("JOB_TYPE") == "ACTIVE-DSRGPT2") {
        ACTIVE_DSRGPT2 pt(ref_wfn, options, ints, mo_space_info);
        pt.compute_energy();
    }
    if (options.get_str("JOB_TYPE") == "DSRG_MRPT") {
        std::string cas_type = options.get_str("CAS_TYPE");
        int max_rdm_level = (options.get_str("THREEPDC") == "ZERO") ? 2 : 3;

        if (cas_type == "CAS") {
            FCI_MO fci_mo(ref_wfn, options, ints, mo_space_info);
            fci_mo.compute_energy();
            Reference reference = fci_mo.reference(max_rdm_level);

            if (options.get_bool("SEMI_CANONICAL")) {
                SemiCanonical semi(ref_wfn, ints, mo_space_info);
                semi.semicanonicalize(reference, max_rdm_level);
            }

            std::shared_ptr<DSRG_MRPT> dsrg(
                new DSRG_MRPT(reference, ref_wfn, options, ints, mo_space_info));
            if (options.get_str("RELAX_REF") == "NONE") {
                dsrg->compute_energy();
            } else {
                //                dsrg->compute_energy_relaxed();
            }
        } else if (cas_type == "FCI") {
            auto fci = std::make_shared<FCI>(ref_wfn, options, ints, mo_space_info);
            fci->set_max_rdm_level(max_rdm_level);
            fci->compute_energy();
            Reference reference = fci->reference();
            if (options.get_bool("SEMI_CANONICAL")) {
                SemiCanonical semi(ref_wfn, ints, mo_space_info);
                semi.semicanonicalize(reference, max_rdm_level);
            }
            std::shared_ptr<DSRG_MRPT> dsrg(
                new DSRG_MRPT(reference, ref_wfn, options, ints, mo_space_info));
            if (options.get_str("RELAX_REF") == "NONE") {
                dsrg->compute_energy();
            } else {
                //                dsrg->compute_energy_relaxed();
            }
        }
    }
    if (options.get_str("JOB_TYPE") == "DSRG-MRPT2") {
        std::string cas_type = options.get_str("CAS_TYPE");
        std::string actv_type = options.get_str("FCIMO_ACTV_TYPE");
        int max_rdm_level = (options.get_str("THREEPDC") == "ZERO") ? 2 : 3;

        if (cas_type == "CAS") {
            std::shared_ptr<FCI_MO> fci_mo =
                std::make_shared<FCI_MO>(ref_wfn, options, ints, mo_space_info);
            fci_mo->compute_energy();
            Reference reference = fci_mo->reference(max_rdm_level);

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

            if (options.get_bool("SEMI_CANONICAL")) {
                SemiCanonical semi(ref_wfn, ints, mo_space_info);
                if (actv_type == "CIS" || actv_type == "CISD") {
                    semi.set_actv_dims(fci_mo->actv_docc(), fci_mo->actv_virt());
                }
                semi.semicanonicalize(reference, max_rdm_level);
                Ua = semi.Ua_t();
                Ub = semi.Ub_t();
            }

            std::shared_ptr<DSRG_MRPT2> dsrg_mrpt2 =
                std::make_shared<DSRG_MRPT2>(reference, ref_wfn, options, ints, mo_space_info);
            if (options["AVG_STATE"].size() != 0) {
                dsrg_mrpt2->set_p_spaces(fci_mo->p_spaces());
                dsrg_mrpt2->set_eigens(fci_mo->eigens());
                dsrg_mrpt2->set_Uactv(Ua, Ub);
                dsrg_mrpt2->compute_energy_multi_state();
            } else {
                if (options.get_str("RELAX_REF") != "NONE") {
                    dsrg_mrpt2->compute_energy_relaxed();
                } else {
                    if (actv_type == "CIS" || actv_type == "CISD") {
                        dsrg_mrpt2->set_actv_occ(fci_mo->actv_occ());
                        dsrg_mrpt2->set_actv_uocc(fci_mo->actv_uocc());
                    }
                    dsrg_mrpt2->compute_energy();
                }
            }

        } else if (cas_type == "FCI") {
            std::shared_ptr<FCI> fci = std::make_shared<FCI>(ref_wfn, options, ints, mo_space_info);
            fci->set_max_rdm_level(max_rdm_level);
            fci->compute_energy();
            Reference reference = fci->reference();
            if (options.get_bool("SEMI_CANONICAL")) {
                SemiCanonical semi(ref_wfn, ints, mo_space_info);
                semi.semicanonicalize(reference, max_rdm_level);
            }
            std::shared_ptr<DSRG_MRPT2> dsrg_mrpt2 =
                std::make_shared<DSRG_MRPT2>(reference, ref_wfn, options, ints, mo_space_info);
            if (options.get_str("RELAX_REF") != "NONE") {
                dsrg_mrpt2->compute_energy_relaxed();
            } else {
                dsrg_mrpt2->compute_energy();
            }

        } else if (cas_type == "V2RDM") {
            std::shared_ptr<V2RDM> v2rdm =
                std::make_shared<V2RDM>(ref_wfn, options, ints, mo_space_info);
            Reference reference = v2rdm->reference();
            std::shared_ptr<DSRG_MRPT2> dsrg_mrpt2 =
                std::make_shared<DSRG_MRPT2>(reference, ref_wfn, options, ints, mo_space_info);
            dsrg_mrpt2->compute_energy();

        } else if (cas_type == "ACI") {
            // Compute ACI wfn
            auto aci = std::make_shared<AdaptiveCI>(ref_wfn, options, ints, mo_space_info);
            aci->set_quiet(true);
            aci->set_max_rdm(max_rdm_level);
            aci->compute_energy();
            Reference aci_reference = aci->reference();
            if (options.get_bool("ACI_NO")) {
                aci->compute_nos();
            }

            // Transform integrals to semicanonical basis
            SemiCanonical semi(ref_wfn, ints, mo_space_info);
            semi.semicanonicalize(aci_reference, max_rdm_level);

            std::shared_ptr<DSRG_MRPT2> dsrg_mrpt2(
                new DSRG_MRPT2(aci_reference, ref_wfn, options, ints, mo_space_info));
            dsrg_mrpt2->compute_energy();

        } else if (cas_type == "DMRG") {
#ifdef HAVE_CHEMPS2
            DMRGSolver dmrg(ref_wfn, options, mo_space_info, ints);
            dmrg.set_max_rdm(max_rdm_level);
            dmrg.compute_energy();
            Reference dmrg_reference = dmrg.reference();
            // if (options.get_bool("SEMI_CANONICAL") and !options.get_bool("CASSCF_REFERENCE")) {
            if (options.get_bool("SEMI_CANONICAL")) {
                SemiCanonical semi(ref_wfn, options, ints, mo_space_info, dmrg_reference);
                semi.semicanonicalize(dmrg_reference, max_rdm_level);
            }
            std::shared_ptr<DSRG_MRPT2> dsrg_mrpt2(
                new DSRG_MRPT2(dmrg_reference, ref_wfn, options, ints, mo_space_info));
            dsrg_mrpt2->compute_energy();
#endif
        }
    }
    if (options.get_str("JOB_TYPE") == "THREE-DSRG-MRPT2") {
        Timer all_three_dsrg_mrpt2;

        if (options.get_str("INT_TYPE") == "CONVENTIONAL") {
            outfile->Printf("\n THREE-DSRG-MRPT2 is designed for DF/CD integrals");
            throw PSIEXCEPTION("Please set INT_TYPE  DF/CHOLESKY for THREE_DSRG");
        }

        std::string cas_type = options.get_str("CAS_TYPE");
        std::string actv_type = options.get_str("FCIMO_ACTV_TYPE");
        int max_rdm_level = (options.get_str("THREEPDC") == "ZERO") ? 2 : 3;

        if (cas_type == "CAS") {

            if (options["AVG_STATE"].size() != 0) {
                std::string ms_type = options.get_str("DSRG_MULTI_STATE");
                if (ms_type != "SA_FULL") {
                    outfile->Printf("\n  SA_FULL is the ONLY available option "
                                    "in THREE-DSRG-MRPT2.");
                }
            }

            FCI_MO fci_mo(ref_wfn, options, ints, mo_space_info);
            fci_mo.compute_energy();
            Reference reference = fci_mo.reference(max_rdm_level);

            if (options.get_bool("SEMI_CANONICAL")) {
                SemiCanonical semi(ref_wfn, ints, mo_space_info);
                if (actv_type == "CIS" || actv_type == "CISD") {
                    semi.set_actv_dims(fci_mo.actv_docc(), fci_mo.actv_virt());
                }
                semi.semicanonicalize(reference, max_rdm_level);
            }

            std::shared_ptr<THREE_DSRG_MRPT2> three_dsrg_mrpt2(
                new THREE_DSRG_MRPT2(reference, ref_wfn, options, ints, mo_space_info));

            if (actv_type == "CIS" || actv_type == "CISD") {
                three_dsrg_mrpt2->set_actv_occ(fci_mo.actv_occ());
                three_dsrg_mrpt2->set_actv_uocc(fci_mo.actv_uocc());
            }
            three_dsrg_mrpt2->compute_energy();

        } else if (cas_type == "V2RDM") {
            std::shared_ptr<V2RDM> v2rdm =
                std::make_shared<V2RDM>(ref_wfn, options, ints, mo_space_info);
            Reference reference = v2rdm->reference();
            std::shared_ptr<THREE_DSRG_MRPT2> dsrg_mrpt2 = std::make_shared<THREE_DSRG_MRPT2>(
                reference, ref_wfn, options, ints, mo_space_info);
            dsrg_mrpt2->compute_energy();

        } else if (cas_type == "ACI") {
            auto aci = std::make_shared<AdaptiveCI>(ref_wfn, options, ints, mo_space_info);
            aci->set_quiet(true);
            aci->set_max_rdm(max_rdm_level);
            aci->compute_energy();
            Reference aci_reference = aci->reference();
            if (options.get_bool("ACI_NO")) {
                aci->compute_nos();
            }
            if (options.get_bool("ACI_ADD_EXTERNAL_SINGLES")) {
                DeterminantHashVec wfn = aci->get_wavefunction();
                aci->upcast_reference(wfn);
            }
            if (options.get_bool("ESNOS")) {
                auto aci_wfn = aci->get_wavefunction();
                ESNO esno(ref_wfn, options, ints, mo_space_info, aci_wfn);
                esno.compute_nos();
                auto aci2 = std::make_shared<AdaptiveCI>(ref_wfn, options, ints, mo_space_info);
                aci2->set_quiet(true);
                aci2->set_max_rdm(max_rdm_level);
                aci2->compute_energy();
                aci_reference = aci2->reference();
            }

            SemiCanonical semi(ref_wfn, ints, mo_space_info);
            semi.semicanonicalize(aci_reference, max_rdm_level);
            std::shared_ptr<THREE_DSRG_MRPT2> three_dsrg_mrpt2(
                new THREE_DSRG_MRPT2(aci_reference, ref_wfn, options, ints, mo_space_info));
            three_dsrg_mrpt2->compute_energy();

        } else if (cas_type == "FCI") {
            auto fci = std::make_shared<FCI>(ref_wfn, options, ints, mo_space_info);
            fci->set_max_rdm_level(max_rdm_level);
            fci->compute_energy();
            Reference reference = fci->reference();
            if (options.get_bool("SEMI_CANONICAL")) {
                SemiCanonical semi(ref_wfn, ints, mo_space_info);
                semi.semicanonicalize(reference, max_rdm_level);
            }

            std::shared_ptr<THREE_DSRG_MRPT2> three_dsrg_mrpt2(
                new THREE_DSRG_MRPT2(reference, ref_wfn, options, ints, mo_space_info));
            three_dsrg_mrpt2->compute_energy();

        } else if (cas_type == "DMRG") {
#ifdef HAVE_CHEMPS2
            DMRGSolver dmrg(ref_wfn, options, mo_space_info, ints);
            dmrg.set_max_rdm(max_rdm_level);
            dmrg.compute_energy();

            Reference dmrg_reference = dmrg.reference();
            if (options.get_bool("SEMI_CANONICAL")) {
                SemiCanonical semi(ref_wfn, options, ints, mo_space_info, dmrg_reference);
                semi.semicanonicalize(dmrg_reference, max_rdm_level);
            }

            std::shared_ptr<THREE_DSRG_MRPT2> three_dsrg_mrpt2(
                new THREE_DSRG_MRPT2(dmrg_reference, ref_wfn, options, ints, mo_space_info));
            three_dsrg_mrpt2->compute_energy();
#endif
        }

        outfile->Printf("\n CD/DF DSRG-MRPT2 took %8.5f s.", all_three_dsrg_mrpt2.get());
    }
    if ((options.get_str("JOB_TYPE") == "TENSORSRG") or
        (options.get_str("JOB_TYPE") == "SR-DSRG")) {
        auto srg = std::make_shared<TensorSRG>(ref_wfn, options, ints, mo_space_info);
        srg->compute_energy();
    }
    if (options.get_str("JOB_TYPE") == "SR-DSRG-ACI") {
        {
            auto dsrg = std::make_shared<TensorSRG>(ref_wfn, options, ints, mo_space_info);
            dsrg->compute_energy();
            dsrg->transfer_integrals();
        }
        {
            auto aci = std::make_shared<AdaptiveCI>(ref_wfn, options, ints, mo_space_info);
            aci->compute_energy();
        }
    }
    if (options.get_str("JOB_TYPE") == "SR-DSRG-PCI") {
        {
            auto dsrg = std::make_shared<TensorSRG>(ref_wfn, options, ints, mo_space_info);
            dsrg->compute_energy();
            dsrg->transfer_integrals();
        }
        {
            auto pci = std::make_shared<ProjectorCI>(ref_wfn, options, ints, mo_space_info);
            pci->compute_energy();
        }
    }

    if (options.get_str("JOB_TYPE") == "DSRG-MRPT3") {
        std::string cas_type = options.get_str("CAS_TYPE");
        int max_rdm_level = (options.get_str("THREEPDC") == "ZERO") ? 2 : 3;

        if (cas_type == "CAS") {
            auto fci_mo = std::make_shared<FCI_MO>(ref_wfn, options, ints, mo_space_info);
            fci_mo->compute_energy();
            Reference reference = fci_mo->reference(max_rdm_level);

            if (options.get_bool("SEMI_CANONICAL")) {
                SemiCanonical semi(ref_wfn, ints, mo_space_info);
                semi.semicanonicalize(reference, max_rdm_level);
            }

            if (options["AVG_STATE"].size() != 0) {
                std::shared_ptr<DSRG_MRPT3> dsrg_mrpt3(
                    new DSRG_MRPT3(reference, ref_wfn, options, ints, mo_space_info));
                dsrg_mrpt3->set_p_spaces(fci_mo->p_spaces());
                dsrg_mrpt3->set_eigens(fci_mo->eigens());
                dsrg_mrpt3->compute_energy_sa();
            } else {
                std::shared_ptr<DSRG_MRPT3> dsrg_mrpt3(
                    new DSRG_MRPT3(reference, ref_wfn, options, ints, mo_space_info));
                if (options.get_str("RELAX_REF") != "NONE") {
                    dsrg_mrpt3->compute_energy_relaxed();
                } else {
                    dsrg_mrpt3->compute_energy();
                }
            }
        }

        if (cas_type == "FCI") {
            auto fci = std::make_shared<FCI>(ref_wfn, options, ints, mo_space_info);
            fci->set_max_rdm_level(max_rdm_level);
            fci->compute_energy();
            Reference reference = fci->reference();
            if (options.get_bool("SEMI_CANONICAL")) {
                SemiCanonical semi(ref_wfn, ints, mo_space_info);
                semi.semicanonicalize(reference, max_rdm_level);
            }

            std::shared_ptr<FCIWfn> fciwfn_ref = fci->get_FCIWfn();

            std::shared_ptr<DSRG_MRPT3> dsrg_mrpt3(
                new DSRG_MRPT3(reference, ref_wfn, options, ints, mo_space_info));
            dsrg_mrpt3->set_fciwfn0(fciwfn_ref);
            if (options.get_str("RELAX_REF") != "NONE") {
                dsrg_mrpt3->compute_energy_relaxed();
            } else {
                dsrg_mrpt3->compute_energy();
            }
        }
    }

    if (options.get_str("JOB_TYPE") == "SOMRDSRG") {
        int max_rdm_level = (options.get_str("THREEPDC") == "ZERO") ? 2 : 3;
        if (options.get_str("CAS_TYPE") == "CAS") {
            FCI_MO fci_mo(ref_wfn, options, ints, mo_space_info);
            Reference reference = fci_mo.reference(max_rdm_level);
            if (options.get_bool("SEMI_CANONICAL")) {
                SemiCanonical semi(ref_wfn, ints, mo_space_info);
                semi.semicanonicalize(reference, max_rdm_level);
            }
            std::shared_ptr<SOMRDSRG> somrdsrg(
                new SOMRDSRG(reference, ref_wfn, options, ints, mo_space_info));
            somrdsrg->compute_energy();
        }
        if (options.get_str("CAS_TYPE") == "FCI") {
            std::shared_ptr<FCI> fci = std::make_shared<FCI>(ref_wfn, options, ints, mo_space_info);
            fci->set_max_rdm_level(max_rdm_level);
            fci->compute_energy();
            Reference reference = fci->reference();
            if (options.get_bool("SEMI_CANONICAL")) {
                SemiCanonical semi(ref_wfn, ints, mo_space_info);
                semi.semicanonicalize(reference, max_rdm_level);
            }
            std::shared_ptr<SOMRDSRG> somrdsrg(
                new SOMRDSRG(reference, ref_wfn, options, ints, mo_space_info));
            somrdsrg->compute_energy();
        }
    }
    if (options.get_str("JOB_TYPE") == "SQ") {
        SqTest sqtest;
    }

    if (options.get_str("JOB_TYPE") == "CC") {
        auto cc = std::make_shared<CC>(ref_wfn, options, ints, mo_space_info);
        cc->compute_energy();
    }

    if (options.get_str("JOB_TYPE") == "MRCISD") {
        if (options.get_bool("ACI_NO")) {
            auto aci = std::make_shared<AdaptiveCI>(ref_wfn, options, ints, mo_space_info);
            aci->compute_energy();
            aci->compute_nos();
        }
        auto aci = std::make_shared<AdaptiveCI>(ref_wfn, options, ints, mo_space_info);
        aci->compute_energy();

        DeterminantHashVec reference = aci->get_wavefunction();
        auto mrci = std::make_shared<MRCI>(ref_wfn, options, ints, mo_space_info, reference);
        mrci->compute_energy();
    }
}
}
} // End Namespaces
