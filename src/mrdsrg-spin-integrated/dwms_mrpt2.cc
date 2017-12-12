#include <tuple>
#include <sstream>

#include "psi4/libmints/molecule.h"

#include "dwms_mrpt2.h"
#include "master_mrdsrg.h"
#include "../fci_mo.h"
#include "../semi_canonicalize.h"

namespace psi {
namespace forte {

void compute_dwms_mrpt2_energy(SharedWavefunction ref_wfn, Options& options,
                               std::shared_ptr<ForteIntegrals> ints,
                               std::shared_ptr<MOSpaceInfo> mo_space_info) {
    print_method_banner({"Dynamically Weighted Multi-State DSRG-PT2", "Chenyang Li"});

    print_h2("Implementation Note");
    outfile->Printf("\n  - State-average CASSCF/CASCI using user defined weights.");
    outfile->Printf("\n  - Loop over averaged states:");
    outfile->Printf("\n      - Compute the density weighted by CASCI energy difference.");
    outfile->Printf("\n      - Semicanonicalize orbitals using the weighted density.");
    outfile->Printf("\n      - Compute state-specific partially relaxed DSRG-MRPT2 energy.");
    outfile->Printf("\n  - Collect and print excitation energies.");

    // some preparation
    CharacterTable ct = Process::environment.molecule()->point_group()->char_table();
    std::vector<std::string> irrep_symbol;
    for (int h = 0, nirrep = ref_wfn->nirrep(); h < nirrep; ++h) {
        irrep_symbol.push_back(std::string(ct.gamma(h).symbol()));
    }
    std::vector<std::string> multi_label{"Singlet", "Doublet", "Triplet", "Quartet", "Quintet",
                                         "Sextet",  "Septet",  "Octet",   "Nonet",   "Decaet"};

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

    // State-average CASSCF/CASCI using user defined weights.
    std::shared_ptr<FCI_MO> fci_mo =
        std::make_shared<FCI_MO>(ref_wfn, options, ints, mo_space_info);
    fci_mo->compute_energy();
    std::string actv_type = options.get_str("FCIMO_ACTV_TYPE");

    std::vector<std::tuple<int, int, int, std::vector<double>>> sa_info = fci_mo->sa_info();
    int nentry = sa_info.size();

    std::vector<std::vector<double>> Eref0s(nentry, std::vector<double>());
    std::vector<std::vector<double>> Erefs(nentry, std::vector<double>());
    std::vector<std::vector<double>> Ept2s(nentry, std::vector<double>());

    // cache the original reference energies
    int total_nroots = 0;
    for (int n = 0; n < nentry; ++n) {
        int nroots;
        std::tie(std::ignore, std::ignore, nroots, std::ignore) = sa_info[n];
        for (int i = 0; i < nroots; ++i) {
            Eref0s[n].push_back(fci_mo->eigens()[n][i].second);
            total_nroots += 1;
        }
    }

    // loop over averaged states
    for (int n = 0, counter = 0; n < nentry; ++n) {
        int multi, irrep, nroots;
        std::tie(irrep, multi, nroots, std::ignore) = sa_info[n];

        for (int i = 0; i < nroots; ++i) {
            std::stringstream current_job;
            current_job << "Current Job: " << multi_label[multi - 1] << " " << irrep_symbol[irrep]
                        << ", root " << i;
            size_t title_size = current_job.str().size();
            outfile->Printf("\n\n  %s", std::string(title_size, '=').c_str());
            outfile->Printf("\n  %s", current_job.str().c_str());
            outfile->Printf("\n  %s\n", std::string(title_size, '=').c_str());

            // compute new weighted density
            fci_mo->set_target_dwms(n, i);
            Reference reference = fci_mo->reference(max_rdm_level);
            Erefs[n].push_back(reference.get_Eref());

            // semicanonicalize orbitals
            SemiCanonical semi(ref_wfn, ints, mo_space_info);
            if (actv_type == "CIS" || actv_type == "CISD") {
                semi.set_actv_dims(fci_mo->actv_docc(), fci_mo->actv_virt());
            }
            semi.semicanonicalize(reference, max_rdm_level);
            Ua = semi.Ua_t();
            Ub = semi.Ub_t();

            // state specific dsrg-mrpt2
            std::shared_ptr<MASTER_DSRG> dsrg_pt2;

            std::string int_type = options.get_str("INT_TYPE");
            if (int_type == "CHOLESKY" || int_type == "DF" || int_type == "DISKDF") {
                dsrg_pt2 = std::make_shared<THREE_DSRG_MRPT2>(reference, ref_wfn, options, ints,
                                                              mo_space_info);
            } else {
                dsrg_pt2 =
                    std::make_shared<DSRG_MRPT2>(reference, ref_wfn, options, ints, mo_space_info);
            }
            dsrg_pt2->set_Uactv(Ua, Ub);

            if (actv_type == "CIS" || actv_type == "CISD") {
                dsrg_pt2->set_actv_occ(fci_mo->actv_occ());
                dsrg_pt2->set_actv_uocc(fci_mo->actv_uocc());
            }

            dsrg_pt2->compute_energy();
            auto fci_ints = dsrg_pt2->compute_Heff();

            // rediagonalize the CAS Hamiltonian
            fci_mo->set_fci_int(fci_ints);
            fci_mo->set_nroots(nroots);
            fci_mo->set_root_sym(irrep);
            fci_mo->set_root(i);
            double Ept2 = fci_mo->compute_ss_energy();

            Ept2s[n].push_back(Ept2);

            // rotate integrals back to original basis (same as SA-CASSCF in step 1)
            counter += 1;
            if (counter < total_nroots) {
                semi.back_transform_ints();
            }
        }
    }

    // print summary
    print_h2("Dynamically Weighted Multi-State DSRG-PT2 Energy Summary");

    auto print_energy_summary = [&](const std::string& name,
                                    const std::vector<std::vector<double>>& energy,
                                    const bool& pass_process) {
        outfile->Printf("\n    Multi.  Irrep.  No.    %20s", name.c_str());
        std::string dash(43, '-');
        outfile->Printf("\n    %s", dash.c_str());

        for (int n = 0, counter = 0; n < nentry; ++n) {
            int irrep, multi, nroots;
            std::tie(irrep, multi, nroots, std::ignore) = sa_info[n];

            for (int i = 0; i < nroots; ++i) {
                outfile->Printf("\n     %3d     %3s    %2d   %22.14f", multi,
                                irrep_symbol[irrep].c_str(), i, energy[n][i]);
                if (pass_process) {
                    Process::environment.globals["ENERGY ROOT " + std::to_string(counter)] =
                        energy[n][i];
                }
                counter += 1;
            }
            outfile->Printf("\n    %s\n", dash.c_str());
        }
    };

    print_energy_summary("Old Ref. Energy", Eref0s, false);
    print_energy_summary("DW Avg. Ref. Energy", Erefs, false);
    print_energy_summary("DWMS-DSRGPT2 Energy", Ept2s, true);
}
}
}
