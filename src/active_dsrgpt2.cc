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

#include <algorithm>
#include <iomanip>
#include <map>

#include "psi4/libmints/pointgrp.h"
#include "psi4/libmints/dipole.h"
#include "psi4/libmints/petitelist.h"
#include "psi4/physconst.h"

#include "fci_mo.h"
#include "active_dsrgpt2.h"
#include "mini-boost/boost/format.hpp"

namespace psi {
namespace forte {

ACTIVE_DSRGPT2::ACTIVE_DSRGPT2(SharedWavefunction ref_wfn, Options& options,
                               std::shared_ptr<ForteIntegrals> ints,
                               std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(options), ints_(ints), mo_space_info_(mo_space_info), total_nroots_(0) {
    // Copy the wavefunction information
    shallow_copy(ref_wfn);
    reference_wavefunction_ = ref_wfn;

    std::string description = "Wrapper for Multiple SS-DSRG-MRPT2 Computations";
    print_method_banner({"ACTIVE-DSRG-MRPT2", description, "Chenyang Li"});

    outfile->Printf("\n  Note: Orbitals are NOT optimized throughout the process.");
    outfile->Printf("\n  Reference selection criteria (CAS/CIS/CISD) "
                    "will NOT change.");
    outfile->Printf("\n  Each state uses its OWN semicanonical orbitals.");
    outfile->Printf("\n  Ground state is assumed to be a singlet.");
    outfile->Printf("\n  Otherwise, please run separate DSRG-MRPT2 jobs.");

    startup();
}

ACTIVE_DSRGPT2::~ACTIVE_DSRGPT2() {}

void ACTIVE_DSRGPT2::startup() {
    if (options_["NROOTPI"].size() == 0) {
        throw PSIEXCEPTION("Please specify NROOTPI for ACTIVE-DSRGPT2 jobs.");
    } else {
        std::shared_ptr<Molecule> molecule = Process::environment.molecule();
        multiplicity_ = molecule->multiplicity();
        if (options_["MULTIPLICITY"].has_changed()) {
            multiplicity_ = options_.get_int("MULTIPLICITY");
        }

        ref_type_ = options_.get_str("FCIMO_ACTV_TYPE");
        if (ref_type_ == "COMPLETE") {
            ref_type_ = "CAS";
        }

        int nirrep = this->nirrep();
        dominant_dets_ =
            vector<vector<STLBitsetDeterminant>>(nirrep, vector<STLBitsetDeterminant>());
        Uaorbs_ = vector<vector<SharedMatrix>>(nirrep, vector<SharedMatrix>());
        Uborbs_ = vector<vector<SharedMatrix>>(nirrep, vector<SharedMatrix>());
        ref_wfns_ = vector<SharedMatrix>(nirrep, SharedMatrix());

        // determined absolute orbitals indices in C1 symmetry
        Dimension nmopi = this->nmopi();
        Dimension frzcpi = mo_space_info_->get_dimension("FROZEN_DOCC");
        Dimension corepi = mo_space_info_->get_dimension("RESTRICTED_DOCC");
        Dimension actvpi = mo_space_info_->get_dimension("ACTIVE");
        Dimension virtpi = mo_space_info_->get_dimension("RESTRICTED_UOCC");

        coreIdxC1_ = vector<vector<size_t>>(nirrep, vector<size_t>());
        actvIdxC1_ = vector<vector<size_t>>(nirrep, vector<size_t>());
        virtIdxC1_ = vector<vector<size_t>>(nirrep, vector<size_t>());
        std::vector<std::tuple<double, int, int>> order;
        for (int h = 0; h < nirrep; ++h) {
            for (int i = 0; i < nmopi[h]; ++i) {
                order.push_back(std::tuple<double, int, int>(this->epsilon_a()->get(h, i), i, h));
            }
        }
        std::sort(order.begin(), order.end(), std::less<std::tuple<double, int, int>>());

        for (int idx = 0; idx < order.size(); ++idx) {
            int i = std::get<1>(order[idx]);
            int h = std::get<2>(order[idx]);

            int core_min = frzcpi[h];
            int core_max = core_min + corepi[h];
            int actv_max = core_max + actvpi[h];
            int virt_max = actv_max + virtpi[h];

            if (i >= core_min && i < core_max) {
                coreIdxC1_[h].push_back(static_cast<size_t>(idx));
            } else if (i >= core_max && i < actv_max) {
                actvIdxC1_[h].push_back(static_cast<size_t>(idx));
            } else if (i >= actv_max && i < virt_max) {
                virtIdxC1_[h].push_back(static_cast<size_t>(idx));
            }
            outfile->Printf("\n  h = %zu, i = %2zu, C1i = %2zu", h, i, idx);
        }

        CharacterTable ct = Process::environment.molecule()->point_group()->char_table();
        std::string cisd_noHF;
        if (ref_type_ == "CISD") {
            t1_percentage_ = vector<vector<double>>(nirrep, vector<double>());
            if (options_.get_bool("FCIMO_CISD_NOHF")) {
                cisd_noHF = "TURE";
            } else {
                cisd_noHF = "FALSE";
            }
        }

        for (int h = 0; h < nirrep; ++h) {
            nrootpi_.push_back(options_["NROOTPI"][h].to_integer());
            irrep_symbol_.push_back(std::string(ct.gamma(h).symbol()));
            total_nroots_ += nrootpi_[h];
        }

        // print request
        print_h2("Input Summary");
        std::vector<std::pair<std::string, std::string>> calculation_info_string{
            {"total roots requested (include S0)", std::to_string(total_nroots_)},
            {"multiplicity", std::to_string(multiplicity_)},
            {"reference space type", ref_type_}};
        if (ref_type_ == "CISD") {
            calculation_info_string.push_back({"separate HF in CISD", cisd_noHF});
        }
        std::string ipea = options_.get_str("FCIMO_IPEA");
        if (ipea != "NONE") {
            calculation_info_string.push_back({"IPEA type", ipea});
        }
        bool internals = (options_.get_str("INTERNAL_AMP") != "NONE");
        calculation_info_string.push_back(
            {"DSRG-MRPT2 internal amplitudes", options_.get_str("INTERNAL_AMP")});
        if (internals) {
            calculation_info_string.push_back(
                {"DSRG-MRPT2 internal type", options_.get_str("INTERNAL_AMP_SELECT")});
        }
        for (const auto& str_dim : calculation_info_string) {
            outfile->Printf("\n    %-40s %15s", str_dim.first.c_str(), str_dim.second.c_str());
        }

        print_h2("Roots Summary");
        int total_width = 4 + 6 + 6 * nirrep;
        outfile->Printf("\n      %s", std::string(6, ' ').c_str());
        for (int h = 0; h < nirrep; ++h)
            outfile->Printf(" %5s", irrep_symbol_[h].c_str());
        outfile->Printf("\n    %s", std::string(total_width, '-').c_str());
        outfile->Printf("\n      NROOTS");
        for (int h = 0; h < nirrep; ++h) {
            outfile->Printf("%6d", nrootpi_[h]);
        }
        outfile->Printf("\n    %s", std::string(total_width, '-').c_str());
    }
}

void ACTIVE_DSRGPT2::compute_modipole() {
    // obtain AO dipole from libmints
    std::shared_ptr<BasisSet> basisset = this->basisset();
    std::shared_ptr<IntegralFactory> ints = std::shared_ptr<IntegralFactory>(
        new IntegralFactory(basisset, basisset, basisset, basisset));
    int nbf = basisset->nbf();

    std::vector<SharedMatrix> aodipole_ints;
    for (const std::string& direction : {"X", "Y", "Z"}) {
        std::string name = "AO Dipole " + direction;
        aodipole_ints.push_back(SharedMatrix(new Matrix(name, nbf, nbf)));
    }
    std::shared_ptr<OneBodyAOInt> aodOBI(ints->ao_dipole());
    aodOBI->compute(aodipole_ints);

    modipole_ints_.clear();
    for (int i = 0; i < 3; ++i) {
        SharedMatrix modipole(aodipole_ints[i]->clone());
        modipole->set_name("MO Dipole " + std::to_string(i));
        modipole->transform(this->Ca_subset("AO"));
        modipole_ints_.push_back(modipole);
    }
}

void ACTIVE_DSRGPT2::precompute_energy() {
    print_h2("Precomputation of ACTIVE-DSRGPT2");
    outfile->Printf("\n  Note: Looping over all roots to");
    outfile->Printf("\n  1) determine excitation type;");
    outfile->Printf("\n  2) obtain original orbital extent;");
    outfile->Printf("\n  3) obtain unitary matrices that semicanonicalize"
                    " orbitals of each state;");
    outfile->Printf("\n  4) compute singlet CIS/CISD oscillator strength;");
    outfile->Printf("\n  5) determine %%T1 in CISD.");
    outfile->Printf("\n");

    CharacterTable ct = Process::environment.molecule()->point_group()->char_table();
    fci_mo_ = std::make_shared<FCI_MO>(reference_wavefunction_, options_, ints_, mo_space_info_);
    orb_extents_ = flatten_fci_orbextents(fci_mo_->orb_extents());

    int nirrep = this->nirrep();
    std::vector<std::pair<SharedVector, double>> eigen0;
    ref_wfns_.clear();

    // compute MO dipole integrals in the original basis
    compute_modipole();

    for (int h = 0; h < nirrep; ++h) {
        if (nrootpi_[h] == 0) {
            if (h == 0) {
                outfile->Printf("\n  Please change the nroot of %s to 1 for "
                                "the ground state.",
                                irrep_symbol_[0].c_str());
                throw PSIEXCEPTION("Please change NROOTPI to account for the ground state.");
            } else {
                continue; // move on to the next irrep
            }
        } else {
            std::string current = "Working on Irrep " + irrep_symbol_[h];
            print_h2(current);
            int nroot = nrootpi_[h];
            fci_mo_->set_root_sym(h);

            // set the ground state to singlet A1 when multiplicity is not 1
            if (multiplicity_ != 1 && h == 0) {
                outfile->Printf("\n  Set ground state to singlet %s.", irrep_symbol_[h].c_str());
                fci_mo_->set_nroots(1);
                fci_mo_->set_root(0);
                fci_mo_->set_multiplicity(1);

                fci_mo_->compute_ss_energy();
                dominant_dets_[h].push_back(fci_mo_->dominant_dets()[0]);
                if (ref_type_ == "CISD") {
                    t1_percentage_[h].push_back(fci_mo_->compute_T1_percentage()[0]);
                }

                // determine orbital rotations
                SharedMatrix Ua(new Matrix("Ua S0", fci_mo_->nmopi_, fci_mo_->nmopi_));
                SharedMatrix Ub(new Matrix("Ub S0", fci_mo_->nmopi_, fci_mo_->nmopi_));
                fci_mo_->BD_Fock(fci_mo_->Fa_, fci_mo_->Fb_, Ua, Ub, "Fock");
                Uaorbs_[0].push_back(Ua);
                Uborbs_[0].push_back(Ub);

                // set back to multiplicity
                fci_mo_->set_multiplicity(multiplicity_);
                outfile->Printf("\n  Set multiplicity back to %d.", multiplicity_);

                if (nrootpi_[0] - 1 > 0) {
                    nroot = nrootpi_[0] - 1;
                    fci_mo_->set_nroots(nroot);
                    fci_mo_->set_root(0);

                    fci_mo_->compute_ss_energy();
                } else {
                    continue; // move on to the next irrep
                }

            } else {
                fci_mo_->set_nroots(nroot);
                fci_mo_->set_root(0);

                fci_mo_->compute_ss_energy();
            }

            // fill in dominant_dets_
            std::vector<STLBitsetDeterminant> dominant_dets = fci_mo_->dominant_dets();
            for (int i = 0; i < nroot; ++i) {
                dominant_dets_[h].push_back(dominant_dets[i]);
            }

            // fill in %T1
            if (ref_type_ == "CISD") {
                std::vector<double> t1 = fci_mo_->compute_T1_percentage();
                for (int i = 0; i < nroot; ++i) {
                    t1_percentage_[h].push_back(t1[i]);
                }
            }

            // fill in the unitary matrices for semicanonical orbitals
            std::vector<std::pair<SharedVector, double>> eigen = fci_mo_->eigen();
            int eigen_size = eigen.size();
            if (eigen_size != nroot) {
                outfile->Printf("\n  Inconsistent nroot to eigen_size.");
                outfile->Printf("\n  There is a problem in FCI_MO.");
                throw PSIEXCEPTION("Inconsistent nroot to eigen_size.");
            }

            int dim = (eigen[0].first)->dim();
            SharedMatrix evecs(new Matrix("evecs", dim, eigen_size));
            for (int i = 0; i < eigen_size; ++i) {
                evecs->set_column(0, i, (eigen[i]).first);
            }

            for (int i = 0; i < eigen_size; ++i) {
                outfile->Printf("\n  Computing semicanonical orbital rotation "
                                "matrix for root %d.",
                                i);

                CI_RDMS ci_rdms(options_, fci_mo_->fci_ints_, fci_mo_->determinant_, evecs, i, i);

                fci_mo_->FormDensity(ci_rdms, fci_mo_->Da_, fci_mo_->Db_);
                fci_mo_->Form_Fock(fci_mo_->Fa_, fci_mo_->Fb_);

                std::string namea = "Ua " + irrep_symbol_[h] + " " + std::to_string(i);
                std::string nameb = "Ub " + irrep_symbol_[h] + " " + std::to_string(i);
                SharedMatrix Ua(new Matrix(namea, fci_mo_->nmopi_, fci_mo_->nmopi_));
                SharedMatrix Ub(new Matrix(nameb, fci_mo_->nmopi_, fci_mo_->nmopi_));
                fci_mo_->BD_Fock(fci_mo_->Fa_, fci_mo_->Fb_, Ua, Ub, "Fock");
                Uaorbs_[h].push_back(Ua);
                Uborbs_[h].push_back(Ub);
            }

            // compute oscillator strength (only for singlet)
            if (multiplicity_ == 1) {

                // save a copy of the ref. wfn. in the original basis
                ref_wfns_[h] = evecs;

                outfile->Printf("\n  Computing V%s reference oscillator strength 0%s -> n%s ... ",
                                ref_type_.c_str(), ct.gamma(0).symbol(), ct.gamma(h).symbol());

                std::vector<STLBitsetDeterminant> p_space1 = fci_mo_->p_space();
                if (h == 0) {
                    eigen0 = eigen;
                    p_space_g_ = fci_mo_->p_space();
                }
                compute_osc_ref(0, h, p_space_g_, p_space1, eigen0, eigen);
                outfile->Printf("Done.");
            }
        }
    }

    // print reference oscillator strength
    print_h2("V" + ref_type_ + " Transition Dipole Moment");
    for (const auto& tdp : tdipole_ref_) {
        const Vector4& td = tdp.second;
        outfile->Printf("\n  %s:  X: %7.4f  Y: %7.4f  Z: %7.4f  Total: %7.4f", tdp.first.c_str(),
                        td.x, td.y, td.z, td.t);
    }

    print_h2("V" + ref_type_ + " Oscillator Strength");
    for (const auto& fp : f_ref_) {
        const Vector4& f = fp.second;
        outfile->Printf("\n  %s:  X: %7.4f  Y: %7.4f  Z: %7.4f  Total: %7.4f", fp.first.c_str(),
                        f.x, f.y, f.z, f.t);
    }

    outfile->Printf("\n\n  ########## END OF ACTIVE-DSRGPT2 PRE-COMPUTATION ##########\n");
}

std::string ACTIVE_DSRGPT2::transition_type(const int& n0, const int& irrep0, const int& n1,
                                            const int& irrep1) {
    CharacterTable ct = Process::environment.molecule()->point_group()->char_table();
    std::string symbol = ct.symbol();
    int width = 2;
    if (symbol == "cs" || symbol == "d2h") {
        width = 3;
    } else if (symbol == "c1") {
        width = 1;
    }

    std::string S0_symbol = ct.gamma(irrep0).symbol();
    std::string Sn_symbol = ct.gamma(irrep1).symbol();

    std::stringstream name_ss;
    name_ss << std::setw(2) << n0 << " " << std::setw(width) << S0_symbol << " -> " << std::setw(2)
            << n1 << " " << std::setw(width) << Sn_symbol;
    return name_ss.str();
}

void ACTIVE_DSRGPT2::compute_osc_ref(const int& irrep0, const int& irrep1,
                                     const std::vector<STLBitsetDeterminant>& p_space0,
                                     const std::vector<STLBitsetDeterminant>& p_space1,
                                     const std::vector<std::pair<SharedVector, double>>& eigen0,
                                     const std::vector<std::pair<SharedVector, double>>& eigen1) {
    // some basic test
    size_t ndet0 = p_space0.size();
    size_t ndet1 = p_space1.size();
    if (ndet0 != static_cast<size_t>(eigen0[0].first->dim())) {
        std::string error = "Error from compute_ref_osc: size of p_space does not match the "
                            "dimension of eigen vector.";
        outfile->Printf("\n  %s", error.c_str());
        throw PSIEXCEPTION(error);
    }

    // determine if p_space0 and p_space1 are the same (even ordering)
    bool same = false;
    same = (p_space0 == p_space1) && (irrep0 == irrep1);

    // combined space of determinants
    size_t ndet = ndet0;
    std::vector<STLBitsetDeterminant> p_space(p_space0);
    if (!same) {
        ndet += ndet1;
        p_space.insert(p_space.end(), p_space1.begin(), p_space1.end());
    }

    // combined eigen values and vectors
    size_t nroot0 = eigen0.size();
    size_t nroot1 = eigen1.size();
    size_t nroot = nroot0;
    std::vector<double> evals(nroot, 0.0);
    SharedMatrix evecs(new Matrix("combined evecs", ndet, nroot));

    if (same) {
        for (int n = 0; n < nroot0; ++n) {
            evals[n] = eigen0[n].second;
            evecs->set_column(0, n, eigen0[n].first);
        }
    } else {
        nroot += nroot1;
        evals = std::vector<double>(nroot, 0.0);
        evecs = SharedMatrix(new Matrix("combined evecs", ndet, nroot));

        for (int n = 0; n < nroot0; ++n) {
            evals[n] = eigen0[n].second;

            SharedVector evec0 = eigen0[n].first;
            SharedVector evec(new Vector("combined evec0 " + std::to_string(n), ndet));
            for (size_t i = 0; i < ndet0; ++i) {
                evec->set(i, evec0->get(i));
            }
            evecs->set_column(0, n, evec);
        }

        for (int n = 0; n < nroot1; ++n) {
            evals[n + nroot0] = eigen1[n].second;

            SharedVector evec1 = eigen1[n].first;
            SharedVector evec(new Vector("combined evec1 " + std::to_string(n), ndet));
            for (size_t i = 0; i < ndet1; ++i) {
                evec->set(i + ndet0, evec1->get(i));
            }
            evecs->set_column(0, n + nroot0, evec);
        }
    }

    // compute oscillator strength for S0(sym0) -> Sn
    int start = 1, offset = 0;
    if (nroot != nroot0) {
        // different irrep
        start = nroot0;
        offset = nroot0;
    }

    for (int n = start; n < nroot; ++n) {
        Vector4 transD = compute_td_ref_root(fci_mo_->fci_ints_, p_space, evecs, 0, n);
        double Eexcited = evals[n] - evals[0];

        Vector4 osc;
        osc.x = 2.0 / 3.0 * Eexcited * transD.x * transD.x;
        osc.y = 2.0 / 3.0 * Eexcited * transD.y * transD.y;
        osc.z = 2.0 / 3.0 * Eexcited * transD.z * transD.z;
        osc.t = osc.x + osc.y + osc.z;

        std::string name = transition_type(0, irrep0, n - offset, irrep1);
        tdipole_ref_[name] = transD;
        f_ref_[name] = osc;
    }
}

Vector4 ACTIVE_DSRGPT2::compute_td_ref_root(std::shared_ptr<FCIIntegrals> fci_ints,
                                            const std::vector<STLBitsetDeterminant>& p_space,
                                            SharedMatrix evecs, const int& root0,
                                            const int& root1) {
    int nirrep = mo_space_info_->nirrep();
    Dimension nmopi = this->nmopi();
    Dimension actvpi = mo_space_info_->get_dimension("ACTIVE");
    size_t nactv = actvpi.sum();
    size_t nmo = nmopi.sum();

    // obtain MO transition density
    CI_RDMS ci_rdms(options_, fci_ints, p_space, evecs, root0, root1);
    vector<double> opdm_a(nactv * nactv, 0.0);
    vector<double> opdm_b(nactv * nactv, 0.0);
    ci_rdms.compute_1rdm(opdm_a, opdm_b);

    // prepare MO transition density (spin summed)
    SharedMatrix MOtransD(new Matrix("MO TransD", nmo, nmo));

    auto offset_irrep = [](const int& h, const Dimension& npi) -> size_t {
        int h_local = h;
        size_t offset = 0;
        while ((--h_local) >= 0) {
            offset += npi[h_local];
        }
        return offset;
    };

    for (int h0 = 0; h0 < nirrep; ++h0) {
        size_t offset_rdm_h0 = offset_irrep(h0, actvpi);

        for (int h1 = 0; h1 < nirrep; ++h1) {
            size_t offset_rdm_h1 = offset_irrep(h1, actvpi);

            for (size_t u = 0; u < actvpi[h0]; ++u) {
                size_t u_rdm = u + offset_rdm_h0;
                size_t u_all = actvIdxC1_[h0][u];

                for (size_t v = 0; v < actvpi[h1]; ++v) {
                    size_t v_rdm = v + offset_rdm_h1;
                    size_t v_all = actvIdxC1_[h1][v];

                    size_t idx = u_rdm * nactv + v_rdm;
                    MOtransD->set(u_all, v_all, opdm_a[idx] + opdm_b[idx]);
                }
            }
        }
    }

    // compute transition dipole
    Vector4 transD;
    transD.x = MOtransD->vector_dot(modipole_ints_[0]);
    transD.y = MOtransD->vector_dot(modipole_ints_[1]);
    transD.z = MOtransD->vector_dot(modipole_ints_[2]);
    transD.t = sqrt(transD.x * transD.x + transD.y * transD.y + transD.z * transD.z);

    return transD;
}

double ACTIVE_DSRGPT2::compute_energy() {
    if (total_nroots_ == 0) {
        outfile->Printf("\n  NROOTPI is zero. Did nothing.");
    } else {
        // precomputation
        precompute_energy();

        // save a copy of the original orbitals
        SharedMatrix Ca0(this->Ca()->clone());
        SharedMatrix Cb0(this->Cb()->clone());

        int nirrep = this->nirrep();
        ref_energies_ = vector<vector<double>>(nirrep, vector<double>());
        pt_energies_ = vector<vector<double>>(nirrep, vector<double>());

        double Tde_g = 0.0;
        std::vector<std::string> T1blocks{"aa", "AA", "av", "AV", "ca", "CA"};
        std::vector<std::string> T2blocks{"aaaa", "cava", "caaa", "aava", "AAAA",
                                          "CAVA", "CAAA", "AAVA", "aAaA", "cAvA",
                                          "aCaV", "cAaA", "aCaA", "aAvA", "aAaV"};

        //        std::map<std::string, ambit::Tensor> T1_block_g;
        //        std::map<std::string, ambit::Tensor> T2_block_g;

        //        std::map<std::string, ambit::Tensor> TDeff;
        //        std::vector<std::string> TDblocks{"aa", "AA", "av", "AV", "ca", "CA", "cv", "CV"};

        //        std::vector<double> opdm_a_g, opdm_b_g;

        //        // obtain absolute indices
        //        std::map<char, std::vector<std::pair<size_t, size_t>>> space_rel_idx;
        //        space_rel_idx['c'] = mo_space_info_->get_relative_mo("RESTRICTED_DOCC");
        //        space_rel_idx['a'] = mo_space_info_->get_relative_mo("ACTIVE");
        //        space_rel_idx['v'] = mo_space_info_->get_relative_mo("RESTRICTED_UOCC");

        //        std::map<char, std::vector<std::vector<size_t>>> space_C1_idx;
        //        space_C1_idx['c'] = coreIdxC1_;
        //        space_C1_idx['a'] = actvIdxC1_;
        //        space_C1_idx['v'] = virtIdxC1_;

        //        Dimension frzcDim = mo_space_info_->get_dimension("FROZEN_DOCC");
        //        Dimension coreDim = mo_space_info_->get_dimension("RESTRICTED_DOCC");
        //        Dimension actvDim = mo_space_info_->get_dimension("ACTIVE");

        // real computation
        for (int h = 0; h < nirrep; ++h) {
            if (nrootpi_[h] == 0) {
                continue;
            } else {
                int nroot = nrootpi_[h];
                fci_mo_->set_root_sym(h);

                // set the ground state to singlet A1 when multiplicity is not 1
                if (multiplicity_ != 1 && h == 0) {
                    outfile->Printf("\n  Set ground state to singlet %s.",
                                    irrep_symbol_[0].c_str());
                    outfile->Printf("\n\n  %s", std::string(35, '=').c_str());
                    outfile->Printf("\n    Current Job: %3s state, root %2d",
                                    irrep_symbol_[0].c_str(), 0);
                    outfile->Printf("\n  %s\n", std::string(35, '=').c_str());

                    // rotate to semicanonical orbitals
                    outfile->Printf("\n  Rotate to semicanonical orbitals.");
                    rotate_orbs(Ca0, Cb0, Uaorbs_[0][0], Uborbs_[0][0]);

                    // transform integrals
                    outfile->Printf("\n\n");
                    std::vector<size_t> idx_a = fci_mo_->idx_a_;
                    fci_mo_->integral_->retransform_integrals();
                    ambit::Tensor tei_active_aa =
                        fci_mo_->integral_->aptei_aa_block(idx_a, idx_a, idx_a, idx_a);
                    ambit::Tensor tei_active_ab =
                        fci_mo_->integral_->aptei_ab_block(idx_a, idx_a, idx_a, idx_a);
                    ambit::Tensor tei_active_bb =
                        fci_mo_->integral_->aptei_bb_block(idx_a, idx_a, idx_a, idx_a);
                    fci_mo_->fci_ints_->set_active_integrals(tei_active_aa, tei_active_ab,
                                                             tei_active_bb);
                    fci_mo_->fci_ints_->compute_restricted_one_body_operator();

                    // compute reference energy
                    fci_mo_->set_nroots(1);
                    fci_mo_->set_root(0);
                    fci_mo_->set_multiplicity(1);
                    double Eref = fci_mo_->compute_ss_energy();
                    ref_energies_[0].push_back(Eref);

                    // check Fock matrix
                    size_t count = 0;
                    fci_mo_->Check_Fock(fci_mo_->Fa_, fci_mo_->Fb_, fci_mo_->dconv_, count);

                    // compute 2- and 3-cumulants
                    Reference reference = fci_mo_->reference();

                    // compute DSRG-MRPT2 energy
                    double Ept2 = 0.0;
                    if (options_.get_str("INT_TYPE") == "CONVENTIONAL") {
                        std::shared_ptr<DSRG_MRPT2> dsrg = std::make_shared<DSRG_MRPT2>(
                            reference, reference_wavefunction_, options_, ints_, mo_space_info_);
                        dsrg->set_ignore_semicanonical(true);
                        dsrg->set_actv_occ(fci_mo_->actv_occ());
                        dsrg->set_actv_uocc(fci_mo_->actv_uocc());
                        Ept2 = dsrg->compute_energy();
                    } else {
                        std::shared_ptr<THREE_DSRG_MRPT2> dsrg = std::make_shared<THREE_DSRG_MRPT2>(
                            reference, reference_wavefunction_, options_, ints_, mo_space_info_);
                        dsrg->ignore_semicanonical(true);
                        dsrg->set_actv_occ(fci_mo_->actv_occ());
                        dsrg->set_actv_uocc(fci_mo_->actv_uocc());
                        Ept2 = dsrg->compute_energy();
                    }
                    pt_energies_[0].push_back(Ept2);

                    // set multiplicity back to original
                    fci_mo_->set_multiplicity(multiplicity_);

                    // minus nroot (just for irrep 0) by 1
                    nroot -= 1;
                }

                // -> START HERE: loop over nroot
                for (int i = 0; i < nroot; ++i) {
                    int i_real = i;
                    if (multiplicity_ != 1 && h == 0) {
                        i_real = i + 1;
                    }
                    outfile->Printf("\n\n  %s", std::string(35, '=').c_str());
                    outfile->Printf("\n    Current Job: %3s state, root %2d",
                                    irrep_symbol_[h].c_str(), i_real);
                    outfile->Printf("\n  %s\n", std::string(35, '=').c_str());

                    // rotate to semicanonical orbitals
                    outfile->Printf("\n  Rotate to semicanonical orbitals.");
                    rotate_orbs(Ca0, Cb0, Uaorbs_[h][i_real], Uborbs_[h][i_real]);

                    // transform integrals
                    outfile->Printf("\n\n");
                    std::vector<size_t> idx_a = fci_mo_->idx_a_;
                    fci_mo_->integral_->retransform_integrals();
                    ambit::Tensor tei_active_aa =
                        fci_mo_->integral_->aptei_aa_block(idx_a, idx_a, idx_a, idx_a);
                    ambit::Tensor tei_active_ab =
                        fci_mo_->integral_->aptei_ab_block(idx_a, idx_a, idx_a, idx_a);
                    ambit::Tensor tei_active_bb =
                        fci_mo_->integral_->aptei_bb_block(idx_a, idx_a, idx_a, idx_a);
                    fci_mo_->fci_ints_->set_active_integrals(tei_active_aa, tei_active_ab,
                                                             tei_active_bb);
                    fci_mo_->fci_ints_->compute_restricted_one_body_operator();

                    // compute reference energy
                    fci_mo_->set_nroots(nroot);
                    fci_mo_->set_root(i);
                    double Eref = fci_mo_->compute_ss_energy();
                    ref_energies_[h].push_back(Eref);
                    Reference reference = fci_mo_->reference();

                    // check Fock matrix
                    size_t count = 0;
                    fci_mo_->Check_Fock(fci_mo_->Fa_, fci_mo_->Fb_, fci_mo_->dconv_, count);

                    // obtain the name of transition type
                    std::string trans_name = transition_type(0, 0, i_real, h);

                    // decide whether to compute oscillator strength or not
                    bool do_osc = false;
                    if (f_ref_.find(trans_name) != f_ref_.end()) {
                        if (f_ref_[trans_name].t > 1.0e-6) {
                            do_osc = true;
                        }
                    }
                    bool gs = (h == 0) && (i_real == 0);

                    // Declare useful amplitudes outside dsrg-mrpt2 to avoid storage of multiple
                    // 3-density, since orbital ordering is identical for different states (although
                    // it is set in dsrg-mrpt2 for each state)
                    double Tde = 0.0;
                    ambit::BlockedTensor T1, T2;

                    // compute DSRG-MRPT2 energy
                    double Ept2 = 0.0;
                    if (options_.get_str("INT_TYPE") == "CONVENTIONAL") {
                        std::shared_ptr<DSRG_MRPT2> dsrg = std::make_shared<DSRG_MRPT2>(
                            reference, reference_wavefunction_, options_, ints_, mo_space_info_);
                        dsrg->set_ignore_semicanonical(true);
                        dsrg->set_actv_occ(fci_mo_->actv_occ());
                        dsrg->set_actv_uocc(fci_mo_->actv_uocc());
                        Ept2 = dsrg->compute_energy();

                        if (gs || do_osc) {
                            // obtain the scalar term of de-normal-ordered amplitudes
                            // before rotate amplitudes because the density will not be rotated
                            Tde = dsrg->Tamp_deGNO();

                            // rotate amplitudes from semicanonical to original basis
                            // PS: rotated T1, T1deGNO, and T2
                            dsrg->rotate_amp(Uaorbs_[h][i_real], Uborbs_[h][i_real], true, true);

                            // copy rotated amplitudes (T1deGNO, T2)
                            ambit::BlockedTensor T1temp = dsrg->T1deGNO(T1blocks);
                            ambit::BlockedTensor T2temp = dsrg->T2(T2blocks);

                            if (gs) {
                                Tde_g = Tde;
                                T1_g_ = ambit::BlockedTensor::build(ambit::CoreTensor, "T1_g",
                                                                    T1blocks);
                                T2_g_ = ambit::BlockedTensor::build(ambit::CoreTensor, "T2_g",
                                                                    T2blocks);
                                T1_g_["ia"] = T1temp["ia"];
                                T1_g_["IA"] = T1temp["IA"];
                                T2_g_["ijab"] = T2temp["ijab"];
                                T2_g_["iJaB"] = T2temp["iJaB"];
                                T2_g_["IJAB"] = T2temp["IJAB"];
                            } else {
                                T1 = ambit::BlockedTensor::build(ambit::CoreTensor, "T1_x",
                                                                 T1blocks);
                                T2 = ambit::BlockedTensor::build(ambit::CoreTensor, "T2_x",
                                                                 T2blocks);
                                T1["ia"] = T1temp["ia"];
                                T1["IA"] = T1temp["IA"];
                                T2["ijab"] = T2temp["ijab"];
                                T2["iJaB"] = T2temp["iJaB"];
                                T2["IJAB"] = T2temp["IJAB"];
                            }

                            //                            if (gs) {
                            //                                T1_block_g =
                            //                                dsrg->T1_blocks(T1blocks);
                            //                                T2_block_g =
                            //                                dsrg->T2_blocks(T2blocks);

                            //                                // recompute 1-density
                            //                                CI_RDMS ci_rdms(options_,
                            //                                fci_mo_->fci_ints_,
                            //                                fci_mo_->p_space(),
                            //                                                ref_wfns_[0], 0, 0);
                            //                                size_t na =
                            //                                mo_space_info_->size("ACTIVE");
                            //                                size_t na2 = na * na;
                            //                                opdm_a_g = vector<double>(na2, 0.0);
                            //                                opdm_b_g = vector<double>(na2, 0.0);
                            //                                ci_rdms.compute_1rdm(opdm_a_g,
                            //                                opdm_b_g);
                            //                            }
                        }

                        //                        if (h == 0 && i == 0) {
                        //                            Tde_g = Tde;
                        //                        } else {

                        //                            if (h == 0) {

                        //                            } else {
                        //                                std::vector<STLBitsetDeterminant>
                        //                                p_space(p_space_g_);
                        //                                std::vector<STLBitsetDeterminant> p_space1
                        //                                = fci_mo_->p_space();
                        //                                p_space.insert(p_space.end(),
                        //                                p_space1.begin(), p_space1.end());

                        //                                SharedMatrix evecs = combine_evecs(0, h);

                        //                                size_t na =
                        //                                mo_space_info_->size("ACTIVE");
                        //                                size_t na2 = na * na;
                        //                                size_t na4 = na2 * na2;
                        //                                size_t na6 = na4 * na2;
                        //                                vector<double> opdm_a(na2, 0.0);
                        //                                vector<double> opdm_b(na2, 0.0);
                        //                                vector<double> tpdm_aa(na4, 0.0);
                        //                                vector<double> tpdm_ab(na4, 0.0);
                        //                                vector<double> tpdm_bb(na4, 0.0);
                        //                                vector<double> tpdm_aaa(na6, 0.0);
                        //                                vector<double> tpdm_aab(na6, 0.0);
                        //                                vector<double> tpdm_abb(na6, 0.0);
                        //                                vector<double> tpdm_bbb(na6, 0.0);

                        //                                CI_RDMS ci_rdms(options_,
                        //                                fci_mo_->fci_ints_, p_space, evecs,
                        //                                                i + nrootpi_[0], 0);
                        //                                ci_rdms.compute_1rdm(opdm_a, opdm_b);
                        //                                ci_rdms.compute_2rdm(tpdm_aa, tpdm_ab,
                        //                                tpdm_bb);
                        //                                ci_rdms.compute_3rdm(tpdm_aaa, tpdm_aab,
                        //                                tpdm_abb, tpdm_bbb);

                        //                                dsrg->set_trans_dens(opdm_a, opdm_b,
                        //                                tpdm_aa, tpdm_ab, tpdm_bb,
                        //                                                     tpdm_aaa, tpdm_aab,
                        //                                                     tpdm_abb, tpdm_bbb);

                        //                                // compute (Ax)^+ * mu; Ax: excited state
                        //                                double mud_x =
                        //                                dsrg->compute_eff_trans_dens(true);
                        //                                for (const std::string& block : TDblocks)
                        //                                {
                        //                                    TDeff[block] =
                        //                                    dsrg->trans_dipole_dens_block(block);
                        //                                }

                        //                                // compute mu * Ag; Ag: ground state
                        //                                dsrg->set_T1_blocks(T1_block_g);
                        //                                dsrg->set_T2_blocks(T2_block_g);
                        //                                dsrg->set_gamma1(opdm_a_g, opdm_b_g);
                        //                                dsrg->Tamp_deGNO();
                        //                                dsrg->set_trans_dens(opdm_a, opdm_b,
                        //                                tpdm_aa, tpdm_ab, tpdm_bb,
                        //                                                     tpdm_aaa, tpdm_aab,
                        //                                                     tpdm_abb, tpdm_bbb);
                        //                                double mud_g =
                        //                                dsrg->compute_eff_trans_dens(false);
                        //                                for (const std::string& block : TDblocks)
                        //                                {
                        //                                    TDeff[block]("pq") +=
                        //                                    dsrg->trans_dipole_dens_block(block)("pq");
                        //                                }

                        //                                // put TDeff into SharedMatrix form (spin
                        //                                summed)
                        //                                size_t nmo = modipole_ints_[0]->nrow();
                        //                                SharedMatrix MOtransD(new Matrix("MO
                        //                                TransD", nmo, nmo));
                        //                                for (const std::string& block : TDblocks)
                        //                                {
                        //                                    char c0 = tolower(block[0]);
                        //                                    char c1 = tolower(block[1]);

                        //                                    std::vector<std::pair<size_t, size_t>>
                        //                                    relIdx0 =
                        //                                        space_rel_idx[c0];
                        //                                    std::vector<std::pair<size_t, size_t>>
                        //                                    relIdx1 =
                        //                                        space_rel_idx[c1];

                        //                                    TDeff[block].iterate(
                        //                                        [&](const std::vector<size_t>&
                        //                                        idx, double& value) {
                        //                                            size_t h0 =
                        //                                            relIdx0[idx[0]].first;
                        //                                            size_t h1 =
                        //                                            relIdx1[idx[1]].first;

                        //                                            size_t offset0 = 0.0, offset1
                        //                                            = 0.0;
                        //                                            if (c0 == 'c') {
                        //                                                offset0 -= frzcDim[h0];
                        //                                            } else if (c0 == 'a') {
                        //                                                offset0 -= frzcDim[h0] +
                        //                                                coreDim[h0];
                        //                                            } else {
                        //                                                offset0 -= frzcDim[h0] +
                        //                                                coreDim[h0] + actvDim[h0];
                        //                                            }

                        //                                            if (c1 == 'c') {
                        //                                                offset1 -= frzcDim[h1];
                        //                                            } else if (c1 == 'a') {
                        //                                                offset1 -= frzcDim[h1] +
                        //                                                coreDim[h1];
                        //                                            } else {
                        //                                                offset1 -= frzcDim[h1] +
                        //                                                coreDim[h1] + actvDim[h1];
                        //                                            }

                        //                                            size_t ri0 =
                        //                                            relIdx0[idx[0]].second +
                        //                                            offset0;
                        //                                            size_t ri1 =
                        //                                            relIdx1[idx[1]].second +
                        //                                            offset1;

                        //                                            size_t n0 =
                        //                                            space_C1_idx[c0][h0][ri0];
                        //                                            size_t n1 =
                        //                                            space_C1_idx[c1][h1][ri1];

                        //                                            MOtransD->add(n0, n1, value);
                        //                                        });
                        //                                }

                        //                                // contract with MO dipole integrals
                        //                                Vector4 transD;
                        //                                transD.x =
                        //                                MOtransD->vector_dot(modipole_ints_[0]);
                        //                                transD.y =
                        //                                MOtransD->vector_dot(modipole_ints_[1]);
                        //                                transD.z =
                        //                                MOtransD->vector_dot(modipole_ints_[2]);
                        //                                transD.t = 0.0;
                        //                                outfile->Printf("\n  sym: %d, root: %d,
                        //                                tdX : % .6f, tdY : %.6f, "
                        //                                                "tdZ: %.6f, tdT: %.6f",
                        //                                                h, i, transD.x, transD.y,
                        //                                                transD.z, transD.t);

                        //                                // compute diagonal contribution
                        //                                // sum_{m} mu^{m}_{m} * tc, where tc is a
                        //                                scalar from T * TD
                        //                                std::vector<double> mud_core(3, 0.0);
                        //                                for (int dir = 0; dir < 3; ++dir) {
                        //                                    double mu = 0.0;
                        //                                    for (const auto& p :
                        //                                    space_rel_idx['c']) {
                        //                                        size_t h = p.first;
                        //                                        size_t m = p.second - frzcDim[h];
                        //                                        size_t idx =
                        //                                        space_C1_idx['c'][h][m];
                        //                                        mu +=
                        //                                        modipole_ints_[dir]->get(idx,
                        //                                        idx);
                        //                                    }
                        //                                    mu *= mud_g + mud_x;

                        //                                    mud_core[dir] = mu;
                        //                                }
                        //                                transD.x += mud_core[0];
                        //                                transD.y += mud_core[1];
                        //                                transD.z += mud_core[2];

                        //                                // add zeroth-order transition density
                        //                                std::string name = transition_type(0, 0,
                        //                                i, h);
                        //                                transD.x += tdipole_ref_[name].x * (1.0 +
                        //                                Tde + Tde_g);
                        //                                transD.y += tdipole_ref_[name].y * (1.0 +
                        //                                Tde + Tde_g);
                        //                                transD.z += tdipole_ref_[name].z * (1.0 +
                        //                                Tde + Tde_g);
                        //                                outfile->Printf("\n%s, scale = %.12f",
                        //                                name.c_str(),
                        //                                                1.0 + Tde + Tde_g);

                        //                                transD.t = sqrt(transD.x * transD.x +
                        //                                transD.y * transD.y +
                        //                                                transD.z * transD.z);

                        //                                // compute oscillator strength
                        //                                double Eexcited = Ept2 -
                        //                                pt_energies_[0][0];
                        //                                Vector4 osc;
                        //                                osc.x = 2.0 / 3.0 * Eexcited * transD.x *
                        //                                transD.x;
                        //                                osc.y = 2.0 / 3.0 * Eexcited * transD.y *
                        //                                transD.y;
                        //                                osc.z = 2.0 / 3.0 * Eexcited * transD.z *
                        //                                transD.z;
                        //                                osc.t = osc.x + osc.y + osc.z;

                        //                                outfile->Printf("\n  sym: %d, root: %d,
                        //                                tdX : % .6f, tdY : %.6f, "
                        //                                                "tdZ: %.6f, tdT: %.6f",
                        //                                                h, i, transD.x, transD.y,
                        //                                                transD.z, transD.t);
                        //                                outfile->Printf("\n  sym: %d, root: %d,
                        //                                oscX : % .6f, oscY : %.6f, "
                        //                                                "oscZ: %.6f, oscT: %.6f",
                        //                                                h, i, osc.x, osc.y, osc.z,
                        //                                                osc.t);
                        //                            }
                        //                        }
                    } else {
                        std::shared_ptr<THREE_DSRG_MRPT2> dsrg = std::make_shared<THREE_DSRG_MRPT2>(
                            reference, reference_wavefunction_, options_, ints_, mo_space_info_);
                        dsrg->ignore_semicanonical(true);
                        dsrg->set_actv_occ(fci_mo_->actv_occ());
                        dsrg->set_actv_uocc(fci_mo_->actv_uocc());
                        Ept2 = dsrg->compute_energy();
                    }
                    pt_energies_[h].push_back(Ept2);

                    // if this state is ground state, copy amplitudes to private variables
                    //                    if (gs) {
                    //                        Tde_g = Tde;
                    //                        T1_g_ = ambit::BlockedTensor::build(ambit::CoreTensor,
                    //                        "T1_g", T1blocks);
                    //                        T2_g_ = ambit::BlockedTensor::build(ambit::CoreTensor,
                    //                        "T2_g", T2blocks);
                    //                        T1_g_["ia"] = T1["ia"];
                    //                        T1_g_["IA"] = T1["IA"];
                    //                        T2_g_["ijab"] = T2["ijab"];
                    //                        T2_g_["iJaB"] = T2["iJaB"];
                    //                        T2_g_["IJAB"] = T2["IJAB"];
                    //                    }

                    // if the reference oscillator strength is nonzero
                    if (do_osc) {
                        compute_osc_pt2(h, i_real, Tde, T1, T2);
                    }
                }
            }
        }
        print_osc();
        print_summary();

        // set the last energy to Process:environment
        for (int h = nirrep; h > 0; --h) {
            int n = nrootpi_[h - 1];
            if (n != 0) {
                Process::environment.globals["CURRENT ENERGY"] = pt_energies_[h - 1][n - 1];
                break;
            }
        }
    }
    return 0.0;
}

void ACTIVE_DSRGPT2::compute_osc_pt2(const int& irrep, const int& root, const double& T0_x,
                                     ambit::BlockedTensor& T1_x, ambit::BlockedTensor& T2_x) {
    // compute reference transition density
    // step 1: combine p_space and eigenvectors if needed
    int n = root;
    std::vector<STLBitsetDeterminant> p_space(p_space_g_);
    SharedMatrix evecs = ref_wfns_[0];

    if (irrep != 0) {
        n += ref_wfns_[0]->ncol();
        std::vector<STLBitsetDeterminant> p_space1 = fci_mo_->p_space();
        p_space.insert(p_space.end(), p_space1.begin(), p_space1.end());
        evecs = combine_evecs(0, irrep);
    }

    // step 2: use CI_RDMS to compute transition density
    size_t na = mo_space_info_->size("ACTIVE");
    size_t na2 = na * na;
    size_t na4 = na2 * na2;
    size_t na6 = na4 * na2;
    CI_RDMS ci_rdms(options_, fci_mo_->fci_ints_, p_space, evecs, n, 0); // what if swap n and 0

    std::vector<double> opdm_a(na2, 0.0);
    std::vector<double> opdm_b(na2, 0.0);
    ci_rdms.compute_1rdm(opdm_a, opdm_b);

    std::vector<double> tpdm_aa(na4, 0.0);
    std::vector<double> tpdm_ab(na4, 0.0);
    std::vector<double> tpdm_bb(na4, 0.0);
    ci_rdms.compute_2rdm(tpdm_aa, tpdm_ab, tpdm_bb);

    std::vector<double> tpdm_aaa(na6, 0.0);
    std::vector<double> tpdm_aab(na6, 0.0);
    std::vector<double> tpdm_abb(na6, 0.0);
    std::vector<double> tpdm_bbb(na6, 0.0);
    ci_rdms.compute_3rdm(tpdm_aaa, tpdm_aab, tpdm_abb, tpdm_bbb);

    // step 3: turn transition rdms into BlockedTensor format
    ambit::BlockedTensor TD1 =
        ambit::BlockedTensor::build(ambit::CoreTensor, "TD1", spin_cases({"aa"}));
    TD1.block("aa").data() = opdm_a;
    TD1.block("AA").data() = opdm_b;

    ambit::BlockedTensor TD2 =
        ambit::BlockedTensor::build(ambit::CoreTensor, "TD2", spin_cases({"aaaa"}));
    TD2.block("aaaa").data() = tpdm_aa;
    TD2.block("aAaA").data() = tpdm_ab;
    TD2.block("AAAA").data() = tpdm_bb;

    ambit::BlockedTensor TD3 =
        ambit::BlockedTensor::build(ambit::CoreTensor, "TD3", spin_cases({"aaaaaa"}));
    TD3.block("aaaaaa").data() = tpdm_aaa;
    TD3.block("aaAaaA").data() = tpdm_aab;
    TD3.block("aAAaAA").data() = tpdm_abb;
    TD3.block("AAAAAA").data() = tpdm_bbb;

    // compute first-order effective transition density
    // step 1: initialization
    ambit::BlockedTensor TDeff =
        ambit::BlockedTensor::build(ambit::CoreTensor, "TDeff", spin_cases({"hp"}));

    // step 2: compute TDeff from <ref_x| (A_x)^+ * mu |ref_g>
    double fc_x = compute_TDeff(T1_x, T2_x, TD1, TD2, TD3, TDeff, true);

    // step 3: compute TDeff from <ref_x| mu * A_g |ref_g>
    double fc_g = compute_TDeff(T1_g_, T2_g_, TD1, TD2, TD3, TDeff, false);

    // put TDeff into SharedMatrix format
    // step 1: setup orbital maps
    std::map<char, std::vector<std::pair<size_t, size_t>>> space_rel_idx;
    space_rel_idx['c'] = mo_space_info_->get_relative_mo("RESTRICTED_DOCC");
    space_rel_idx['a'] = mo_space_info_->get_relative_mo("ACTIVE");
    space_rel_idx['v'] = mo_space_info_->get_relative_mo("RESTRICTED_UOCC");

    std::map<char, std::vector<std::vector<size_t>>> space_C1_idx;
    space_C1_idx['c'] = coreIdxC1_;
    space_C1_idx['a'] = actvIdxC1_;
    space_C1_idx['v'] = virtIdxC1_;

    std::map<char, Dimension> space_offsets;
    space_offsets['c'] = mo_space_info_->get_dimension("FROZEN_DOCC");
    space_offsets['a'] = space_offsets['c'] + mo_space_info_->get_dimension("RESTRICTED_DOCC");
    space_offsets['v'] = space_offsets['a'] + mo_space_info_->get_dimension("ACTIVE");

    // step 2: copy data to SharedMatrix
    size_t nmo = modipole_ints_[0]->nrow();
    SharedMatrix MOtransD(new Matrix("MO TransD", nmo, nmo));
    for (const std::string& block : TDeff.block_labels()) {
        char c0 = tolower(block[0]);
        char c1 = tolower(block[1]);

        std::vector<std::pair<size_t, size_t>> relIdx0 = space_rel_idx[c0];
        std::vector<std::pair<size_t, size_t>> relIdx1 = space_rel_idx[c1];

        TDeff.block(block).citerate([&](const std::vector<size_t>& i, const double& value) {
            size_t h0 = relIdx0[i[0]].first;
            size_t h1 = relIdx1[i[1]].first;

            size_t offset0 = space_offsets[c0][h0];
            size_t offset1 = space_offsets[c1][h1];

            size_t ri0 = relIdx0[i[0]].second - offset0;
            size_t ri1 = relIdx1[i[1]].second - offset1;

            size_t n0 = space_C1_idx[c0][h0][ri0];
            size_t n1 = space_C1_idx[c1][h1][ri1];

            MOtransD->add(n0, n1, value);
        });
    }

    // contract with MO dipole integrals
    Vector4 transD;
    transD.x = MOtransD->vector_dot(modipole_ints_[0]);
    transD.y = MOtransD->vector_dot(modipole_ints_[1]);
    transD.z = MOtransD->vector_dot(modipole_ints_[2]);
    transD.t = 0.0;

    // add diagonal core contribution sum_{m} mu^{m}_{m} * tc, where tc is a scalar from T * TD
    std::vector<double> mud_core(3, 0.0);
    for (int dir = 0; dir < 3; ++dir) {
        double mu = 0.0;
        for (const auto& p : space_rel_idx['c']) {
            size_t h = p.first;
            size_t m = p.second - space_offsets['c'][h];
            size_t idx = space_C1_idx['c'][h][m];
            mu += modipole_ints_[dir]->get(idx, idx);
        }
        mu *= fc_g + fc_x;

        mud_core[dir] = mu;
    }
    transD.x += mud_core[0];
    transD.y += mud_core[1];
    transD.z += mud_core[2];

    // add zeroth-order transition density
    std::string name = transition_type(0, 0, root, irrep);
    double scale = 1.0 + T0_g_ + T0_x;
    transD.x += tdipole_ref_[name].x * scale;
    transD.y += tdipole_ref_[name].y * scale;
    transD.z += tdipole_ref_[name].z * scale;

    // save DSRG-PT2 transition density
    transD.t = sqrt(transD.x * transD.x + transD.y * transD.y + transD.z * transD.z);
    tdipole_pt_[name] = transD;

    // compute oscillator strength
    double Eexcited = pt_energies_[irrep][root] - pt_energies_[0][0];
    Vector4 osc;
    osc.x = 2.0 / 3.0 * Eexcited * transD.x * transD.x;
    osc.y = 2.0 / 3.0 * Eexcited * transD.y * transD.y;
    osc.z = 2.0 / 3.0 * Eexcited * transD.z * transD.z;
    osc.t = osc.x + osc.y + osc.z;
    f_pt_[name] = osc;
}

double ACTIVE_DSRGPT2::compute_TDeff(ambit::BlockedTensor& T1, ambit::BlockedTensor& T2,
                                     ambit::BlockedTensor& TD1, ambit::BlockedTensor& TD2,
                                     ambit::BlockedTensor& TD3, ambit::BlockedTensor& TDeff,
                                     const bool& transpose) {
    // initialization
    double scalar = 0.0;
    bool internal_amp = options_.get_str("INTERNAL_AMP") != "NONE";

    std::string uv = "uv", UV = "UV";
    std::string uvxy = "uvxy", uVxY = "uVxY", vUyX = "vUyX", UVXY = "UVXY";
    std::string uvwxyz = "uvwxyz", uvWxyZ = "uvWxyZ", uVWxYZ = "uVWxYZ";
    std::string vwUyzX = "vwUyzX", vUWyXZ = "vUWyXZ", UVWXYZ = "UVWXYZ";

    if (transpose) {
        uv = "vu";
        UV = "VU";

        uvxy = "xyuv";
        uVxY = "xYuV";
        vUyX = "yXvU";
        UVXY = "XYUV";

        uvwxyz = "xyzuvw";
        uvWxyZ = "xyZuvW";
        uVWxYZ = "xYZuVW";
        vwUyzX = "yzXvwU";
        vUWyXZ = "yXZvUW";
        UVWXYZ = "XYZUVW";
    }

    if (internal_amp) {
        scalar += T1["vu"] * TD1[uv];
        scalar += T1["VU"] * TD1[UV];

        scalar -= T1["uv"] * TD1[uv];
        scalar -= T1["UV"] * TD1[UV];

        scalar += 0.25 * T2["xyuv"] * TD2[uvxy];
        scalar += 0.25 * T2["XYUV"] * TD2[UVXY];
        scalar += T2["xYuV"] * TD2[uVxY];

        scalar -= 0.25 * T2["uvxy"] * TD2[uvxy];
        scalar -= 0.25 * T2["UVXY"] * TD2[UVXY];
        scalar -= T2["uVxY"] * TD2[uVxY];

        TDeff["ux"] += T1["vx"] * TD1[uv];
        TDeff["UX"] += T1["VX"] * TD1[UV];

        TDeff["ux"] -= T1["xv"] * TD1[uv];
        TDeff["UX"] -= T1["XV"] * TD1[UV];

        TDeff["ux"] += T1["yv"] * TD2[uvxy];
        TDeff["ux"] += T1["YV"] * TD2[uVxY];
        TDeff["UX"] += T1["yv"] * TD2[vUyX];
        TDeff["UX"] += T1["YV"] * TD2[UVXY];

        TDeff["ux"] -= T1["vy"] * TD2[uvxy];
        TDeff["ux"] -= T1["VY"] * TD2[uVxY];
        TDeff["UX"] -= T1["vy"] * TD2[vUyX];
        TDeff["UX"] -= T1["VY"] * TD2[UVXY];

        TDeff["uz"] += 0.5 * T2["xyzv"] * TD2[uvxy];
        TDeff["uz"] += T2["xYzV"] * TD2[uVxY];
        TDeff["UZ"] += T2["yXvZ"] * TD2[vUyX];
        TDeff["UZ"] += 0.5 * T2["XYZV"] * TD2[UVXY];

        TDeff["uz"] -= 0.5 * T2["zvxy"] * TD2[uvxy];
        TDeff["uz"] -= T2["zVxY"] * TD2[uVxY];
        TDeff["UZ"] -= T2["vZyX"] * TD2[vUyX];
        TDeff["UZ"] -= 0.5 * T2["ZVXY"] * TD2[UVXY];

        TDeff["ux"] += 0.25 * T2["yzvw"] * TD3[uvwxyz];
        TDeff["ux"] += T2["yZvW"] * TD3[uvWxyZ];
        TDeff["ux"] += 0.25 * T2["YZVW"] * TD3[uVWxYZ];
        TDeff["UX"] += 0.25 * T2["yzvw"] * TD3[vwUyzX];
        TDeff["UX"] += T2["yZvW"] * TD3[vUWyXZ];
        TDeff["UX"] += 0.25 * T2["YZVW"] * TD3[UVWXYZ];

        TDeff["ux"] -= 0.25 * T2["vwyz"] * TD3[uvwxyz];
        TDeff["ux"] -= T2["vWyZ"] * TD3[uvWxyZ];
        TDeff["ux"] -= 0.25 * T2["VWYZ"] * TD3[uVWxYZ];
        TDeff["UX"] -= 0.25 * T2["vwyz"] * TD3[vwUyzX];
        TDeff["UX"] -= T2["vWyZ"] * TD3[vUWyXZ];
        TDeff["UX"] -= 0.25 * T2["VWYZ"] * TD3[UVWXYZ];
    }

    TDeff["ue"] += T1["ve"] * TD1[uv];
    TDeff["UE"] += T1["VE"] * TD1[UV];

    TDeff["mv"] -= T1["mu"] * TD1[uv];
    TDeff["MV"] -= T1["MU"] * TD1[UV];

    TDeff["ma"] += T2["mvau"] * TD1[uv];
    TDeff["ma"] += T2["mVaU"] * TD1[UV];
    TDeff["MA"] += T2["vMuA"] * TD1[uv];
    TDeff["MA"] += T2["MVAU"] * TD1[UV];

    TDeff["ue"] += 0.5 * T2["xyev"] * TD2[uvxy];
    TDeff["ue"] += T2["xYeV"] * TD2[uVxY];
    TDeff["UE"] += T2["yXvE"] * TD2[vUyX];
    TDeff["UE"] += 0.5 * T2["XYEV"] * TD2[UVXY];

    TDeff["mx"] -= 0.5 * T2["myuv"] * TD2[uvxy];
    TDeff["mx"] -= T2["mYuV"] * TD2[uVxY];
    TDeff["MX"] -= T2["yMvU"] * TD2[vUyX];
    TDeff["MX"] -= 0.5 * T2["MYUV"] * TD2[UVXY];

    return scalar;
}

SharedMatrix ACTIVE_DSRGPT2::combine_evecs(const int& h0, const int& h1) {
    SharedMatrix evecs0 = ref_wfns_[h0];
    SharedMatrix evecs1 = ref_wfns_[h1];

    int nroot0 = evecs0->ncol();
    int nroot1 = evecs1->ncol();
    int nroot = nroot0 + nroot1;

    size_t ndet0 = evecs0->nrow();
    size_t ndet1 = evecs1->nrow();
    size_t ndet = ndet0 + ndet1;

    SharedMatrix evecs(new Matrix("combined evecs", ndet, nroot));

    for (int n = 0; n < nroot0; ++n) {
        SharedVector evec0 = evecs0->get_column(0, n);
        SharedVector evec(new Vector("combined evec0 " + std::to_string(n), ndet));
        for (size_t i = 0; i < ndet0; ++i) {
            evec->set(i, evec0->get(i));
        }
        evecs->set_column(0, n, evec);
    }

    for (int n = 0; n < nroot1; ++n) {
        SharedVector evec1 = evecs1->get_column(0, n);
        SharedVector evec(new Vector("combined evec1 " + std::to_string(n), ndet));
        for (size_t i = 0; i < ndet1; ++i) {
            evec->set(i + ndet0, evec1->get(i));
        }
        evecs->set_column(0, n + nroot0, evec);
    }

    return evecs;
}

void ACTIVE_DSRGPT2::rotate_orbs(SharedMatrix Ca0, SharedMatrix Cb0, SharedMatrix Ua,
                                 SharedMatrix Ub) {
    SharedMatrix Ca_new(Ca0->clone());
    SharedMatrix Cb_new(Cb0->clone());
    Ca_new->gemm(false, false, 1.0, Ca0, Ua, 0.0);
    Cb_new->gemm(false, false, 1.0, Cb0, Ub, 0.0);

    // copy to wavefunction
    SharedMatrix Ca = this->Ca();
    SharedMatrix Cb = this->Cb();
    Ca->copy(Ca_new);
    Cb->copy(Cb_new);

    // overlap of original and semicanonical orbitals
    SharedMatrix MOoverlap = Matrix::triplet(Ca0, this->S(), Ca_new, true, false, false);
    MOoverlap->set_name("MO overlap");

    // test active orbital ordering
    int nirrep = this->nirrep();
    Dimension ncmopi = mo_space_info_->get_dimension("CORRELATED");
    Dimension frzcpi = mo_space_info_->get_dimension("FROZEN_DOCC");
    Dimension corepi = mo_space_info_->get_dimension("RESTRICTED_DOCC");
    Dimension actvpi = mo_space_info_->get_dimension("ACTIVE");

    for (int h = 0; h < nirrep; ++h) {
        int actv_start = frzcpi[h] + corepi[h];
        int actv_end = actv_start + actvpi[h];

        std::map<int, int> indexmap;
        std::vector<int> idx_0;
        for (int i = actv_start; i < actv_end; ++i) {
            int ii = 0; // corresponding index in semicanonical basis
            double smax = 0.0;

            for (int j = actv_start; j < actv_end; ++j) {
                double s = MOoverlap->get(h, i, j);
                if (fabs(s) > smax) {
                    smax = fabs(s);
                    ii = j;
                }
            }

            if (ii != i) {
                indexmap[i] = ii;
                idx_0.push_back(i);
            }
        }

        // find orbitals to swap if the loop is closed
        std::vector<int> idx_swap;
        for (const int& x : idx_0) {
            // if index x is already in the to-be-swapped index, then continue
            if (std::find(idx_swap.begin(), idx_swap.end(), x) != idx_swap.end()) {
                continue;
            }

            std::vector<int> temp;
            int local = x;

            while (indexmap.find(indexmap[local]) != indexmap.end()) {
                if (std::find(temp.begin(), temp.end(), local) == temp.end()) {
                    temp.push_back(local);
                } else {
                    // a loop found
                    break;
                }

                local = indexmap[local];
            }

            // start from the point that has the value of "local" and copy to
            // idx_swap
            int pos = std::find(temp.begin(), temp.end(), local) - temp.begin();
            for (int i = pos; i < temp.size(); ++i) {
                if (std::find(idx_swap.begin(), idx_swap.end(), temp[i]) == idx_swap.end()) {
                    idx_swap.push_back(temp[i]);
                }
            }
        }

        // remove the swapped orbitals from the vector of orginal orbitals
        idx_0.erase(std::remove_if(idx_0.begin(), idx_0.end(),
                                   [&](int i) {
                                       return std::find(idx_swap.begin(), idx_swap.end(), i) !=
                                              idx_swap.end();
                                   }),
                    idx_0.end());

        // swap orbitals
        for (const int& x : idx_swap) {
            int h_local = h;
            size_t ni = x - frzcpi_[h];
            size_t nj = indexmap[x] - frzcpi_[h];
            while ((--h_local) >= 0) {
                ni += ncmopi[h_local];
                nj += ncmopi[h_local];
            }
            outfile->Printf("\n  Orbital ordering changed due to "
                            "semicanonicalization. Swapped orbital %3zu back "
                            "to %3zu.",
                            nj, ni);
            Ca->set_column(h, x, Ca_new->get_column(h, indexmap[x]));
            Cb->set_column(h, x, Cb_new->get_column(h, indexmap[x]));
        }

        // throw warnings when inconsistency is detected
        for (const int& x : idx_0) {
            int h_local = h;
            size_t ni = x - frzcpi_[h];
            size_t nj = indexmap[x] - frzcpi_[h];
            while ((--h_local) >= 0) {
                ni += ncmopi[h_local];
                nj += ncmopi[h_local];
            }
            outfile->Printf("\n  Orbital %3zu may have changed to "
                            "semicanonical orbital %3zu. Please interpret "
                            "orbitals with caution.",
                            ni, nj);
        }
    }
}

void ACTIVE_DSRGPT2::print_osc() {
    // print reference transition dipole
    print_h2("V" + ref_type_ + " Transition Dipole Moment");
    for (const auto& tdpair : tdipole_ref_) {
        const Vector4& td = tdpair.second;
        outfile->Printf("\n  %s:  X: %7.4f  Y: %7.4f  Z: %7.4f  Total: %7.4f", tdpair.first.c_str(),
                        td.x, td.y, td.z, td.t);
    }

    // print reference oscillator strength
    print_h2("V" + ref_type_ + " Oscillator Strength");
    for (const auto& fp : f_ref_) {
        const Vector4& f = fp.second;
        outfile->Printf("\n  %s:  X: %7.4f  Y: %7.4f  Z: %7.4f  Total: %7.4f", fp.first.c_str(),
                        f.x, f.y, f.z, f.t);
    }

    // print DSRG-PT2 transition dipole
    print_h2("V" + ref_type_ + "-DSRG-PT2 Transition Dipole Moment");
    for (const auto& tdpair : tdipole_pt_) {
        const Vector4& td = tdpair.second;
        outfile->Printf("\n  %s:  X: %7.4f  Y: %7.4f  Z: %7.4f  Total: %7.4f", tdpair.first.c_str(),
                        td.x, td.y, td.z, td.t);
    }

    // print DSRG-PT2 oscillator strength
    print_h2("V" + ref_type_ + "-DSRG-PT2 Oscillator Strength");
    for (const auto& fp : f_pt_) {
        const Vector4& f = fp.second;
        outfile->Printf("\n  %s:  X: %7.4f  Y: %7.4f  Z: %7.4f  Total: %7.4f", fp.first.c_str(),
                        f.x, f.y, f.z, f.t);
    }
}

void ACTIVE_DSRGPT2::print_summary() {
    print_h2("ACTIVE-DSRG-MRPT2 Summary");

    int nirrep = this->nirrep();

    // print raw data
    if (ref_type_ == "CISD") {
        int total_width = 4 + 4 + 4 + 18 + 18 + 6 + 2 * 5;
        outfile->Printf("\n    %4s  %4s  %4s  %18s  %18s  %6s", "2S+1", "Sym.", "ROOT",
                        ref_type_.c_str(), "DSRG-MRPT2", "% T1");
        outfile->Printf("\n    %s", std::string(total_width, '-').c_str());

        for (int h = 0; h < nirrep; ++h) {
            if (nrootpi_[h] != 0) {
                std::string sym = irrep_symbol_[h];

                for (int i = 0; i < nrootpi_[h]; ++i) {
                    if (h == 0 && multiplicity_ != 1 && i == 0) {
                        outfile->Printf("\n    %4d  %4s  %4d  %18.10f  %18.10f  %6.2f", 1,
                                        sym.c_str(), i, ref_energies_[h][i], pt_energies_[h][i],
                                        t1_percentage_[h][i]);
                    } else {
                        outfile->Printf("\n    %4d  %4s  %4d  %18.10f  %18.10f  %6.2f",
                                        multiplicity_, sym.c_str(), i, ref_energies_[h][i],
                                        pt_energies_[h][i], t1_percentage_[h][i]);
                    }
                }
                outfile->Printf("\n    %s", std::string(total_width, '-').c_str());
            }
        }
    } else {
        int total_width = 4 + 4 + 4 + 18 + 18 + 2 * 4;
        outfile->Printf("\n    %4s  %4s  %4s  %18s  %18s", "2S+1", "Sym.", "ROOT",
                        ref_type_.c_str(), "DSRG-MRPT2");
        outfile->Printf("\n    %s", std::string(total_width, '-').c_str());

        for (int h = 0; h < nirrep; ++h) {
            if (nrootpi_[h] != 0) {
                std::string sym = irrep_symbol_[h];

                for (int i = 0; i < nrootpi_[h]; ++i) {
                    if (h == 0 && multiplicity_ != 1 && i == 0) {
                        outfile->Printf("\n    %4d  %4s  %4d  %18.10f  %18.10f", 1, sym.c_str(), i,
                                        ref_energies_[h][i], pt_energies_[h][i]);
                    } else {
                        outfile->Printf("\n    %4d  %4s  %4d  %18.10f  %18.10f", multiplicity_,
                                        sym.c_str(), i, ref_energies_[h][i], pt_energies_[h][i]);
                    }
                }
                outfile->Printf("\n    %s", std::string(total_width, '-').c_str());
            }
        }
    }

    // print excitation energies in eV
    if (total_nroots_ > 1) {
        print_h2("Relative Energy WRT Totally Symmetric Ground State (eV)");

        double ev = pc_hartree2ev;
        if (ref_type_ == "CAS") {
            int width = 4 + 4 + 4 + 8 + 8 + 2 * 4;
            outfile->Printf("\n    %4s  %4s  %4s  %8s  %8s", "2S+1", "Sym.", "ROOT",
                            ref_type_.c_str(), "DSRG-PT2");
            outfile->Printf("\n    %s", std::string(width, '-').c_str());

            for (int h = 0; h < nirrep; ++h) {
                if (nrootpi_[h] != 0) {
                    std::string sym = irrep_symbol_[h];

                    for (int i = 0; i < nrootpi_[h]; ++i) {
                        if (h == 0 && i == 0) {
                            continue;
                        }

                        double Eref = ev * (ref_energies_[h][i] - ref_energies_[0][0]);
                        double Ept2 = ev * (pt_energies_[h][i] - pt_energies_[0][0]);

                        outfile->Printf("\n    %4d  %4s  %4d  %8.3f  %8.3f", multiplicity_,
                                        sym.c_str(), i, Eref, Ept2);
                    }
                    if (h != 0 || nrootpi_[0] != 1)
                        outfile->Printf("\n    %s", std::string(width, '-').c_str());
                }
            }
        } else {
            int width = 4 + 4 + 4 + 8 + 8 + 40 + 2 * 5;
            outfile->Printf("\n    %4s  %4s  %4s  %8s  %8s  %40s", "2S+1", "Sym.", "ROOT",
                            ref_type_.c_str(), "DSRG-PT2", "Excitation Type");
            outfile->Printf("\n    %s", std::string(width, '-').c_str());

            for (int h = 0; h < nirrep; ++h) {
                if (nrootpi_[h] != 0) {
                    std::string sym = irrep_symbol_[h];

                    for (int i = 0; i < nrootpi_[h]; ++i) {
                        if (h == 0 & i == 0) {
                            continue;
                        }

                        double Eref = ev * (ref_energies_[h][i] - ref_energies_[0][0]);
                        double Ept2 = ev * (pt_energies_[h][i] - pt_energies_[0][0]);

                        std::string ex_type =
                            compute_ex_type(dominant_dets_[h][i], dominant_dets_[0][0]);

                        outfile->Printf("\n    %4d  %4s  %4d  %8.3f  %8.3f  %40s", multiplicity_,
                                        sym.c_str(), i, Eref, Ept2, ex_type.c_str());
                    }
                    if (h != 0 || nrootpi_[0] != 1)
                        outfile->Printf("\n    %s", std::string(width, '-').c_str());
                }
            }
            outfile->Printf("\n    Notes on excitation type:");
            outfile->Printf("\n    General format: mAH -> nAP (<r^2>) (S/D)");
            outfile->Printf("\n      mAH:   Mulliken symbol of m-th Active Hole orbital");
            outfile->Printf("\n      nAP:   Mulliken symbol of n-th Active Particle orbital");
            outfile->Printf("\n      <r^2>: orbital extent of the nAP orbital in a.u.");
            outfile->Printf("\n      S/D:   single/double excitation");
            outfile->Printf("\n    NOTE: m and n are ZERO based ACTIVE indices (NO core orbitals)!");
        }
    }
}

std::string ACTIVE_DSRGPT2::compute_ex_type(const STLBitsetDeterminant& det,
                                            const STLBitsetDeterminant& ref_det) {
    Dimension active = mo_space_info_->get_dimension("ACTIVE");
    int nirrep = this->nirrep();
    std::vector<std::string> sym_active;
    for (int h = 0; h < nirrep; ++h) {
        for (int i = 0; i < active[h]; ++i) {
            sym_active.push_back(std::to_string(i) + irrep_symbol_[h]);
        }
    }

    // compare alpha occ
    std::vector<int> occA_ref(ref_det.get_alfa_occ());
    std::vector<int> occA_det(det.get_alfa_occ());
    std::vector<int> commonA;
    std::set_intersection(occA_ref.begin(), occA_ref.end(), occA_det.begin(), occA_det.end(),
                          back_inserter(commonA));
    occA_ref.erase(std::set_difference(occA_ref.begin(), occA_ref.end(), commonA.begin(),
                                       commonA.end(), occA_ref.begin()),
                   occA_ref.end());
    occA_det.erase(std::set_difference(occA_det.begin(), occA_det.end(), commonA.begin(),
                                       commonA.end(), occA_det.begin()),
                   occA_det.end());

    // compare beta occ
    std::vector<int> occB_ref(ref_det.get_beta_occ());
    std::vector<int> occB_det(det.get_beta_occ());
    std::vector<int> commonB;
    std::set_intersection(occB_ref.begin(), occB_ref.end(), occB_det.begin(), occB_det.end(),
                          back_inserter(commonB));
    occB_ref.erase(std::set_difference(occB_ref.begin(), occB_ref.end(), commonB.begin(),
                                       commonB.end(), occB_ref.begin()),
                   occB_ref.end());
    occB_det.erase(std::set_difference(occB_det.begin(), occB_det.end(), commonB.begin(),
                                       commonB.end(), occB_det.begin()),
                   occB_det.end());

    // output string
    std::string output;
    size_t A = occA_ref.size();
    size_t B = occB_ref.size();

    // same as reference
    if (A + B == 0) {
        output = "same as reference (?)";
    }

    // CIS
    if (A + B == 1) {
        int idx_ref, idx_det;
        if (A == 1 && B == 0) {
            idx_ref = occA_ref[0];
            idx_det = occA_det[0];
        } else if (A == 0 && B == 1) {
            idx_ref = occB_ref[0];
            idx_det = occB_det[0];
        }
        double orbex_det = orb_extents_[idx_det];
        std::string r2_str =
            (orbex_det > 1.0e6 ? " (Diffuse) " : str(boost::format(" (%7.2f) ") % orbex_det));

        output = sym_active[idx_ref] + " -> " + sym_active[idx_det] + r2_str + "(S)";
    }

    // CISD
    if (A + B == 2) {
        if (A == 1 && B == 1) {
            int i_ref = occA_ref[0], j_ref = occB_ref[0];
            int i_det = occA_det[0], j_det = occB_det[0];
            if (i_ref == j_ref && i_det == j_det) {
                double orbex_det = orb_extents_[i_det];
                std::string r2_str =
                    (orbex_det > 1.0e6 ? " (Diffuse) "
                                       : str(boost::format(" (%7.2f) ") % orbex_det));
                output = sym_active[i_ref] + " -> " + sym_active[i_det] + r2_str + "(D)";
            } else {
                double orbex_i_det = orb_extents_[i_det];
                double orbex_j_det = orb_extents_[j_det];
                std::string r2_str_i =
                    (orbex_i_det > 1.0e6 ? " (Diffuse) "
                                         : str(boost::format(" (%7.2f)") % orbex_i_det));
                std::string r2_str_j =
                    (orbex_j_det > 1.0e6 ? " (Diffuse) "
                                         : str(boost::format(" (%7.2f)") % orbex_j_det));

                output = sym_active[i_ref] + "," + sym_active[j_ref] + " -> " + sym_active[i_det] +
                         r2_str_i + "," + sym_active[j_det] + r2_str_j;
            }
        } else {
            int i_ref, j_ref, i_det, j_det;
            if (A == 2) {
                i_ref = occA_ref[0], j_ref = occA_ref[1];
                i_det = occA_det[0], j_det = occA_det[1];
            } else if (B == 2) {
                i_ref = occB_ref[0], j_ref = occB_ref[1];
                i_det = occB_det[0], j_det = occB_det[1];
            }

            double orbex_i_det = orb_extents_[i_det];
            double orbex_j_det = orb_extents_[j_det];
            std::string r2_str_i =
                (orbex_i_det > 1.0e6 ? " (Diffuse) "
                                     : str(boost::format(" (%7.2f)") % orbex_i_det));
            std::string r2_str_j =
                (orbex_j_det > 1.0e6 ? " (Diffuse) "
                                     : str(boost::format(" (%7.2f)") % orbex_j_det));

            output = sym_active[i_ref] + "," + sym_active[j_ref] + " -> " + sym_active[i_det] +
                     r2_str_i + "," + sym_active[j_det] + r2_str_j;
        }
    }

    return output;
}

std::vector<double> ACTIVE_DSRGPT2::flatten_fci_orbextents(
    const std::vector<std::vector<std::vector<double>>>& fci_orb_extents) {
    std::vector<double> out;

    size_t nirrep = fci_orb_extents.size();
    for (size_t h = 0; h < nirrep; ++h) {
        size_t nmo = fci_orb_extents[h].size();
        for (size_t i = 0; i < nmo; ++i) {
            double r2 =
                fci_orb_extents[h][i][0] + fci_orb_extents[h][i][1] + fci_orb_extents[h][i][2];
            out.push_back(r2);
        }
    }

    return out;
}
}
}
