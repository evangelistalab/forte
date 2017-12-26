#include <tuple>
#include <sstream>

#include "psi4/libmints/molecule.h"

#include "dwms_mrpt2.h"
#include "master_mrdsrg.h"
#include "../fci_mo.h"
#include "../semi_canonicalize.h"

namespace psi {
namespace forte {

void set_DWMS_options(ForteOptions& foptions) {
    /*- Automatic weight switching -*/
    foptions.add_double("DWMS_ZETA", 0.0, "Gaussian width cutoff for the density weights");

    /*- DWMS algorithms
     *  - DWMS-0: weights computed using SA-CASCI energies, non-orthogonal final solutions
     *  - DWMS-1: weights computed using SA-CASCI energies, orthogonal final solutions
     *  - DWMS-AVG0: weights computed using SA-DSRG-PT2 energies, non-orthogonal final solutions
     *  - DWMS-AVG1: weights computed using SA-DSRG-PT2 energies, orthogonal final solutions -*/
    foptions.add_str("DWMS_ALGORITHM", "DWMS-0", {"DWMS-0", "DWMS-1", "DWMS-AVG0", "DWMS-AVG1"},
                     "DWMS algorithms");
}

void compute_dwms_mrpt2_energy(SharedWavefunction ref_wfn, Options& options,
                               std::shared_ptr<ForteIntegrals> ints,
                               std::shared_ptr<MOSpaceInfo> mo_space_info) {
    print_method_banner({"Dynamically Weighted Multi-State DSRG-PT2", "Chenyang Li"});

    print_h2("DWMS-DSRG-PT2 Options");
    double zeta = options.get_double("DWMS_ZETA");
    std::string algorithm = options.get_str("DWMS_ALGORITHM");
    outfile->Printf("\n    DWMS ZETA          %10.2e", zeta);
    outfile->Printf("\n    DWMS ALGORITHM     %10s", algorithm.c_str());

    print_h2("Implementation Note");
    outfile->Printf("\n  1. Perform SA-CASCI using user defined weights.");
    outfile->Printf("\n  2. Perform SA-DSRG-PT2/SA-CASCI using user defined weights.");
    outfile->Printf("\n  3. Loop over averaged states:");
    outfile->Printf("\n      3a Compute new density weights.");
    outfile->Printf("\n      3b Semicanonicalize orbitals.");
    outfile->Printf("\n      3c Form DSRG-PT2 transformed Hamiltonian.");
    outfile->Printf("\n      3d Perform CASCI on DSRG-PT2 Hamiltonian.");
    outfile->Printf("\n  4. Test orthogonality between CI vectors of step 3d.");
    outfile->Printf("\n  5. Collect and print excitation energies.");
    outfile->Printf("\n");

    outfile->Printf("\n  - DWMS-0:");
    outfile->Printf("\n    - Skip step 2.");
    outfile->Printf("\n    - Use energies of step 1 for step 3a.");
    outfile->Printf("\n    - CI vectors are not orthogonal in step 4.");
    outfile->Printf("\n  - DWMS-1:");
    outfile->Printf("\n    - Use energies of step 1 for step 3a.");
    outfile->Printf("\n    - Use CI vectors of step 2 as initial guess for step 3d.");
    outfile->Printf("\n    - Project out previous CI vectors of step 3d for current step 3d.");
    outfile->Printf("\n  - DWMS-AVG0:");
    outfile->Printf("\n    - Use energies of step 2 for step 3a.");
    outfile->Printf("\n    - CI vectors are not orthogonal in step 4.");
    outfile->Printf("\n  - DWMS-AVG1:");
    outfile->Printf("\n    - Use energies of step 2 for step 3a.");
    outfile->Printf("\n    - Use CI vectors of step 2 as initial guess for step 3d.");
    outfile->Printf("\n    - Project out previous CI vectors of step 3d for current step 3d.");

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

    // step 1: state-average CASCI using user defined weights.
    std::shared_ptr<FCI_MO> fci_mo =
        std::make_shared<FCI_MO>(ref_wfn, options, ints, mo_space_info);
    fci_mo->compute_energy();
    std::string actv_type = options.get_str("FCIMO_ACTV_TYPE");

    std::vector<std::tuple<int, int, int, std::vector<double>>> sa_info = fci_mo->sa_info();
    int nentry = sa_info.size();

    // energies
    std::vector<std::vector<double>> Eref0(nentry, std::vector<double>()); // CASCI of step 1
    std::vector<std::vector<double>> Ept20(nentry, std::vector<double>()); // SA-DSRG-PT2 of step 2
    std::vector<std::vector<double>> Erefs(nentry, std::vector<double>()); // MK vacuum energy
    std::vector<std::vector<double>> Ept2s(nentry, std::vector<double>()); // DWMS energy

    // cache the original reference energies
    int total_nroots = 0;
    for (int n = 0; n < nentry; ++n) {
        int nroots;
        std::tie(std::ignore, std::ignore, nroots, std::ignore) = sa_info[n];
        for (int i = 0; i < nroots; ++i) {
            Eref0[n].push_back(fci_mo->eigens()[n][i].second);
            total_nroots += 1;
        }
    }

    // step 2: compute SA-DSRG-PT2 energy
    std::vector<std::vector<SharedVector>> guesses(nentry, std::vector<SharedVector>());
    double Esa_corr = 0.0;
    if (algorithm != "DWMS-0") {
        Reference reference = fci_mo->reference(max_rdm_level);

        SemiCanonical semi(ref_wfn, ints, mo_space_info);
        if (actv_type == "CIS" || actv_type == "CISD") {
            semi.set_actv_dims(fci_mo->actv_docc(), fci_mo->actv_virt());
        }
        semi.semicanonicalize(reference, max_rdm_level);
        Ua = semi.Ua_t();
        Ub = semi.Ub_t();

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

        double Ept2 = dsrg_pt2->compute_energy();
        Esa_corr = Ept2 - reference.get_Eref();
        auto fci_ints = dsrg_pt2->compute_Heff();

        if (algorithm == "DWMS-1") {
            // create a new FCI_MO object
            FCI_MO fci(ref_wfn, options, ints, mo_space_info, fci_ints);
            fci.set_localize_actv(false);
            fci.compute_energy();
            auto eigens = fci.eigens();
            for (int n = 0; n < nentry; ++n) {
                int nroots;
                std::tie(std::ignore, std::ignore, nroots, std::ignore) = sa_info[n];
                for (int i = 0; i < nroots; ++i) {
                    guesses[n].push_back(eigens[n][i].first);
                    Ept20[n].push_back(eigens[n][i].second);
                }
            }
        } else {
            // use fci_mo and update its eigens_
            fci_mo->set_fci_int(fci_ints);
            fci_mo->set_localize_actv(false);
            fci_mo->compute_energy();
            auto eigens = fci_mo->eigens();
            for (int n = 0; n < nentry; ++n) {
                int nroots;
                std::tie(std::ignore, std::ignore, nroots, std::ignore) = sa_info[n];
                for (int i = 0; i < nroots; ++i) {
                    guesses[n].push_back(eigens[n][i].first);
                    Ept20[n].push_back(eigens[n][i].second);
                }
            }
        }
    }

    // step 3: loop over averaged states
    for (int n = 0, counter = 0; n < nentry; ++n) {
        int multi, irrep, nroots;
        std::tie(irrep, multi, nroots, std::ignore) = sa_info[n];

        // save the re-diagonalized eigen vectors
        std::vector<SharedVector> evecs_new;
        std::vector<std::vector<std::pair<size_t, double>>> projected_roots;

        for (int i = 0; i < nroots; ++i) {
            std::stringstream current_job;
            current_job << "Current Job: " << multi_label[multi - 1] << " " << irrep_symbol[irrep]
                        << ", root " << i;
            size_t title_size = current_job.str().size();
            outfile->Printf("\n\n  %s", std::string(title_size, '=').c_str());
            outfile->Printf("\n  %s", current_job.str().c_str());
            outfile->Printf("\n  %s\n", std::string(title_size, '=').c_str());

            // compute new weighted density using fci_mo->eigens()
            fci_mo->set_dwms_zeta(zeta);
            fci_mo->set_target_dwms(n, i);
            Reference reference = fci_mo->reference(max_rdm_level);
            double Eref = reference.get_Eref() - Esa_corr;
            reference.set_Eref(Eref);
            Erefs[n].push_back(Eref);

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
            fci_mo->set_root_sym(irrep);

            if (algorithm == "DWMS-1" || algorithm == "DWMS-AVG1") {
                fci_mo->set_nroots(nroots - i);
                fci_mo->set_root(0);

                // project out previous DWMS-DSRG-PT2 roots
                std::vector<std::pair<size_t, double>> projection;
                if (i != 0) {
                    // add last root to the projection list
                    outfile->Printf("\n\n    Project out previous DWMS-DSRG-PT2 roots.\n");
                    for (size_t I = 0, nI = evecs_new[i - 1]->dim(); I < nI; ++I) {
                        projection.push_back(std::make_pair(I, evecs_new[i - 1]->get(I)));
                    }
                    projected_roots.push_back(projection);
                }
                fci_mo->project_roots(projected_roots);

                // set initial guess to help SparseCISolver convergence
                std::vector<std::pair<size_t, double>> initial_guess;
                for (size_t I = 0, nI = guesses[n][i]->dim(); I < nI; ++I) {
                    initial_guess.push_back(std::make_pair(I, guesses[n][i]->get(I)));
                }
                fci_mo->set_initial_guess(initial_guess);
            } else {
                fci_mo->set_nroots(nroots);
                fci_mo->set_root(i);
            }

            double Ept2 = fci_mo->compute_ss_energy();

            Ept2s[n].push_back(Ept2);

            // since Heff is rotated to the original basis,
            // we can safely store the relaxed eigenvectors
            if (algorithm == "DWMS-1" || algorithm == "DWMS-AVG1") {
                evecs_new.push_back(fci_mo->eigen()[0].first);
            } else {
                evecs_new.push_back(fci_mo->eigen()[i].first);
            }

            // rotate integrals back to original basis (i.e., same as SA-CASSCF in step 1)
            counter += 1;
            if (counter < total_nroots) {
                semi.back_transform_ints();
            }
        }

        // overlap of rediagonalized wave functions of this symmetry
        std::string Sname = "Overlap of " + multi_label[multi - 1] + " " + irrep_symbol[irrep];
        print_h2(Sname);
        outfile->Printf("\n");
        SharedMatrix S(new Matrix("S", nroots, nroots));
        S->identity();
        for (int i = 0; i < nroots; ++i) {
            for (int j = i; j < nroots; ++j) {
                double Sij = evecs_new[i]->vector_dot(evecs_new[j]);
                S->set(i, j, Sij);
                S->set(j, i, Sij);
            }
        }
        S->print();
    }

    // print summary
    outfile->Printf("  ==> Dynamically Weighted Multi-State DSRG-PT2 Energy Summary <==\n");

    auto print_energy_summary = [&](const std::string& name,
                                    const std::vector<std::vector<double>>& energy,
                                    const bool& pass_process = false) {
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
            outfile->Printf("\n    %s", dash.c_str());
        }
        outfile->Printf("\n");
    };

    print_energy_summary("SA-CASCI Energy", Eref0);
    if (algorithm != "DWMS-0") {
        print_energy_summary("SA-DSRG-PT2 Energy", Ept20);
    }
    print_energy_summary("MK Vacuum Energy", Erefs);
    print_energy_summary("DWMS-DSRGPT2 Energy", Ept2s, true);
}
}
}
