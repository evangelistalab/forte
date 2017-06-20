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

#include "ambit/tensor.h"
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

        // determined absolute indices of active orbitals in C1 symmetry
        Dimension nmopi = this->nmopi();
        Dimension frzcpi = mo_space_info_->get_dimension("FROZEN_DOCC");
        Dimension corepi = mo_space_info_->get_dimension("RESTRICTED_DOCC");
        Dimension actvpi = mo_space_info_->get_dimension("ACTIVE");

        actvIdxC1_ = vector<vector<size_t>>(nirrep, vector<size_t>());
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

            int actv_min = frzcpi[h] + corepi[h];
            int actv_max = actv_min + actvpi[h];
            if (i >= actv_min && i < actv_max) {
                actvIdxC1_[h].push_back(static_cast<size_t>(idx));
            }
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
    std::vector<STLBitsetDeterminant> p_space0;
    std::vector<std::pair<SharedVector, double>> eigen0;

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
            if (multiplicity_ == 1 && h == 0) {
                eigen0 = eigen;
                p_space0 = fci_mo_->p_space();
                outfile->Printf("\n  Computing %s reference oscillator strength 0%s -> n%s ... ",
                                ref_type_.c_str(), ct.gamma(0).symbol(), ct.gamma(0).symbol());
                compute_oscillator_strength(0, h, p_space0, p_space0, eigen, eigen, true);
                outfile->Printf("Done.");

            } else if (multiplicity_ == 1 && h != 0) {

                std::vector<STLBitsetDeterminant> p_space1 = fci_mo_->p_space();
                outfile->Printf("\n  Computing %s reference oscillator strength 0%s -> n%s ... ",
                                ref_type_.c_str(), ct.gamma(0).symbol(), ct.gamma(h).symbol());
                compute_oscillator_strength(0, h, p_space0, p_space1, eigen0, eigen);
                outfile->Printf("Done.");
            }
        }
    }
}

void ACTIVE_DSRGPT2::compute_oscillator_strength(
    const int& irrep0, const int& irrep1, const std::vector<STLBitsetDeterminant>& p_space0,
    const std::vector<STLBitsetDeterminant>& p_space1,
    const std::vector<std::pair<SharedVector, double>>& eigen0,
    const std::vector<std::pair<SharedVector, double>>& eigen1, const bool& same) {

    // obtain AO dipole from libmints
    std::shared_ptr<BasisSet> basisset = this->basisset();
    std::shared_ptr<IntegralFactory> ints = std::shared_ptr<IntegralFactory>(
        new IntegralFactory(basisset, basisset, basisset, basisset));

    std::vector<SharedMatrix> aodipole_ints;
    for (const std::string& direction : {"X", "Y", "Z"}) {
        std::string name = "AO Dipole" + direction;
        aodipole_ints.push_back(SharedMatrix(new Matrix(name, basisset->nbf(), basisset->nbf())));
    }
    std::shared_ptr<OneBodyAOInt> aodOBI(ints->ao_dipole());
    aodOBI->compute(aodipole_ints);

    // SO to AO transformer
    boost::shared_ptr<PetiteList> pet(new PetiteList(basisset, ints));
    SharedMatrix sotoao = pet->sotoao();

    // prepare FCIIntegrals to initialize CI_RDM
    std::shared_ptr<FCIIntegrals> fci_ints =
        std::make_shared<FCIIntegrals>(ints_, mo_space_info_->get_corr_abs_mo("ACTIVE"),
                                       mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC"));

    // character table
    CharacterTable ct = Process::environment.molecule()->point_group()->char_table();

    // combined space of determinants
    size_t ndet0 = p_space0.size();
    size_t ndet1 = p_space1.size();
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
    std::string S0_symbol = ct.gamma(irrep0).symbol();
    for (int n = 1; n < nroot; ++n) {
        std::string Sn_symbol;
        int nn;
        if (n < nroot0) {
            Sn_symbol = ct.gamma(irrep0).symbol();
            nn = n;
        } else {
            Sn_symbol = ct.gamma(irrep1).symbol();
            nn = n - nroot0;
        }

        std::stringstream name_ss;
        int w = nn > 99 ? 3 : 2;
        name_ss << 0 << " " << std::setw(3) << S0_symbol << " -> " << std::setw(w) << nn << " "
                << std::setw(3) << Sn_symbol;
        std::string name = name_ss.str();

        if (f_ref_.find(name) == f_ref_.end()) {
            Vector4 transD =
                compute_root_trans_dipole(aodipole_ints, sotoao, fci_ints, p_space, evecs, 0, n);
            double Eexcited = evals[n] - evals[0];

            Vector4 osc;
            osc.x = 2.0 / 3.0 * Eexcited * transD.x * transD.x;
            osc.y = 2.0 / 3.0 * Eexcited * transD.y * transD.y;
            osc.z = 2.0 / 3.0 * Eexcited * transD.z * transD.z;
            osc.t = osc.x + osc.y + osc.z;

            f_ref_[name] = osc;
        }
    }
}

Vector4 ACTIVE_DSRGPT2::compute_root_trans_dipole(const std::vector<SharedMatrix>& aodipole_ints,
                                                  SharedMatrix sotoao,
                                                  std::shared_ptr<FCIIntegrals> fci_ints,
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

    // prepare AO transition density (spin summed)
    SharedMatrix AOtransD(new Matrix("AO TransD", nmo, nmo));

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
                    AOtransD->set(u_all, v_all, opdm_a[idx] + opdm_b[idx]);
                    //                    if (fabs(opdm_a[idx]) > 0.00000001) {
                    //                        outfile->Printf(
                    //                            "\n  rdm i: %2zu, rdm j: %2zu, all i: %2zu, all j:
                    //                            %2zu, alpha: %.12f",
                    //                            u_rdm, v_rdm, u_all, v_all, opdm_a[idx]);
                    //                    }
                }
            }
        }
    }

    AOtransD->back_transform(this->Ca_subset("AO"));

    // compute transition dipole
    Vector4 transD;
    transD.x = AOtransD->vector_dot(aodipole_ints[0]);
    transD.y = AOtransD->vector_dot(aodipole_ints[1]);
    transD.z = AOtransD->vector_dot(aodipole_ints[2]);
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

        // real computation
        int nirrep = this->nirrep();
        ref_energies_ = vector<vector<double>>(nirrep, vector<double>());
        pt_energies_ = vector<vector<double>>(nirrep, vector<double>());

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

                // loop over nroot
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
                    pt_energies_[h].push_back(Ept2);
                }
            }
        }
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

void ACTIVE_DSRGPT2::print_summary() {
    int nirrep = this->nirrep();

    // print oscillator strength
    print_h2(ref_type_ + " Oscillator Strength");
    for (const auto& fp : f_ref_) {
        const Vector4& f = fp.second;
        outfile->Printf("\n  %s:  X: %7.4f  Y: %7.4f  Z: %7.4f  Total: %7.4f", fp.first.c_str(),
                        f.x, f.y, f.z, f.t);
    }

    print_h2("ACTIVE-DSRG-MRPT2 Summary");

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
            outfile->Printf("\n    Excitation type: orbitals are zero-based "
                            "(active only).");
            outfile->Printf("\n    <r^2> (in a.u.) is given in parentheses. "
                            "\"Diffuse\" when <r^2> > 1.0e6.");
            outfile->Printf("\n    (S) for singles; (D) for doubles.");
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
