/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see LICENSE, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#include <map>
#include <algorithm>

#include "psi4/physconst.h"
#include "psi4/libmints/pointgrp.h"

#include "mini-boost/boost/format.hpp"
#include "active_dsrgpt2.h"

namespace psi {
namespace forte {

ACTIVE_DSRGPT2::ACTIVE_DSRGPT2(SharedWavefunction ref_wfn, Options& options,
                               std::shared_ptr<ForteIntegrals> ints,
                               std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(options), ints_(ints), mo_space_info_(mo_space_info),
      total_nroots_(0) {
    // Copy the wavefunction information
    shallow_copy(ref_wfn);
    reference_wavefunction_ = ref_wfn;

    code_name_ = "ACTIVE-DSRG" + options_.get_str("CORR_LEVEL");

    print_method_banner({code_name_, "Chenyang Li"});
    outfile->Printf("\n    The orbitals are fixed throughout the process.");
    outfile->Printf("\n    If different orbitals (or reference) are desired "
                    "for different roots,");
    outfile->Printf("\n    you need to run those separately using the regular "
                    "DSRG-MRPT2 / MRPT3 (or DF/CD) code.\n");
    startup();
}

ACTIVE_DSRGPT2::~ACTIVE_DSRGPT2() {}

void ACTIVE_DSRGPT2::startup() {
    if (options_["NROOTPI"].size() == 0) {
        throw PSIEXCEPTION(
            "Please specify NROOTPI for ACTIVE-DSRGPT2 / PT3 jobs.");
    } else {
        int nirrep = this->nirrep();
        ref_energies_ = vector<vector<double>>(nirrep, vector<double>());
        pt_energies_ = vector<vector<double>>(nirrep, vector<double>());
        dominant_dets_ = vector<vector<STLBitsetDeterminant>>(
            nirrep, vector<STLBitsetDeterminant>());
        orb_extents_ =
            vector<vector<vector<double>>>(nirrep, vector<vector<double>>());

        CharacterTable ct =
            Process::environment.molecule()->point_group()->char_table();
        if (options_.get_str("ACTIVE_SPACE_TYPE") == "CIS") {
            t1_percentage_ = vector<vector<pair<int, double>>>(
                nirrep, vector<pair<int, double>>());
        }

        for (int h = 0; h < nirrep; ++h) {
            nrootpi_.push_back(options_["NROOTPI"][h].to_integer());
            irrep_symbol_.push_back(std::string(ct.gamma(h).symbol()));
        }

        // print request
        int total_width = 4 + 6 + 6 * nirrep;
        outfile->Printf("\n      %s", std::string(6, ' ').c_str());
        for (int h = 0; h < nirrep; ++h)
            outfile->Printf(" %5s", irrep_symbol_[h].c_str());
        outfile->Printf("\n    %s", std::string(total_width, '-').c_str());
        outfile->Printf("\n      NROOTS");
        for (int h = 0; h < nirrep; ++h) {
            outfile->Printf("%6d", nrootpi_[h]);
            total_nroots_ += nrootpi_[h];
        }
        outfile->Printf("\n    %s", std::string(total_width, '-').c_str());
    }
}

double ACTIVE_DSRGPT2::compute_energy() {
    if (total_nroots_ == 0) {
        outfile->Printf("\n  NROOTPI is zero. Did nothing.");
    } else {
        FCI_MO fci_mo(reference_wavefunction_, options_, ints_, mo_space_info_);
        int nirrep = nrootpi_.size();

//        // save HF orbitals
//        SharedMatrix Ca_hf (this->Ca()->clone());
//        SharedMatrix Cb_hf (this->Cb()->clone());

        // before real computation, we will do CI over all states to determine
        // the excitation type
        outfile->Printf(
            "\n    Looping over all roots to determine excitation type.");
        for (int h = 0; h < nirrep; ++h) {
            if (nrootpi_[h] == 0)
                continue;
            else {
                fci_mo.set_root_sym(h);

                // when the ground state is only a single determinant
                if (options_.get_str("ACTIVE_SPACE_TYPE") == "CIS" ||
                    options_.get_bool("CISD_EX_NO_HF")) {
                    if (h == 0) {
                        // get ground state
                        fci_mo.set_nroots(1);
                        fci_mo.set_root(0);
                        fci_mo.compute_energy();
                        vector<STLBitsetDeterminant> dominant_dets =
                            fci_mo.dominant_dets();
                        dominant_dets_[h].push_back(dominant_dets[0]);
                        orb_extents_[h].push_back(
                            flatten_fci_orbextents(fci_mo.orb_extents()));
                    }
                }

                fci_mo.set_nroots(nrootpi_[h]);
                fci_mo.set_root(nrootpi_[h] - 1);

                fci_mo.compute_energy();
                vector<pair<SharedVector, double>> eigen = fci_mo.eigen();
                vector<STLBitsetDeterminant> dominant_dets =
                    fci_mo.dominant_dets();

                for (int i = 0; i < nrootpi_[h]; ++i) {
                    dominant_dets_[h].push_back(dominant_dets[i]);
                    orb_extents_[h].push_back(
                        flatten_fci_orbextents(fci_mo.orb_extents()));
                }

                // figure out the overlap <CIS|CISD>
                if (options_.get_str("ACTIVE_SPACE_TYPE") == "CIS" &&
                    options_.get_bool("CIS_CISD_OVERLAP")) {
                    std::string step_name =
                        "Computing Overlap <CISD|CIS> of Irrep " +
                        irrep_symbol_[h];
                    print_h2(step_name);

                    std::vector<STLBitsetDeterminant> vecdet_cis =
                        fci_mo.p_space();

                    int cisd_nroot =
                        nrootpi_[h] < 5 ? 2 * nrootpi_[h] : nrootpi_[h] + 5;
                    fci_mo.set_active_space_type("CISD");
                    fci_mo.form_p_space();
                    std::vector<STLBitsetDeterminant> vecdet_cisd =
                        fci_mo.p_space();
                    size_t p_space_size = vecdet_cisd.size();
                    if (cisd_nroot >= p_space_size) {
                        cisd_nroot = p_space_size;
                    }

                    // setup overlap matrix of CISD and CIS
                    std::string matrix_name =
                        "<CISD|CIS> in Irrep " + irrep_symbol_[h];
                    Matrix overlap(matrix_name, cisd_nroot,
                                   nrootpi_[h]); // CISD by CIS

                    // compute CISD
                    outfile->Printf("\n    Compute %d roots for CISD in Irrep "
                                    "%s for <CIS|CISD>.\n\n",
                                    cisd_nroot, irrep_symbol_[h].c_str());
                    std::vector<SharedVector> cisd_evecs;
                    fci_mo.set_nroots(cisd_nroot);
                    fci_mo.set_root(cisd_nroot - 1);
                    fci_mo.compute_energy();
                    for (int i = 0; i < cisd_nroot; ++i) {
                        cisd_evecs.push_back(fci_mo.eigen()[i].first);
                    }

                    // set back to CIS
                    fci_mo.set_active_space_type("CIS");

                    // printing
                    for (int i = 0; i < nrootpi_[h]; ++i) {
                        for (int j = 0; j < cisd_nroot; ++j) {
                            for (int s = 0; s < vecdet_cis.size(); ++s) {
                                for (int sd = 0; sd < vecdet_cisd.size();
                                     ++sd) {
                                    if (vecdet_cis[s] == vecdet_cisd[sd]) {
                                        double value = cisd_evecs[j]->get(sd) *
                                                       eigen[i].first->get(s);
                                        overlap.add(j, i, value);
                                    }
                                }
                            }
                        }
                    }
                    overlap.print();
                    for (int i = 0; i < overlap.ncol(); ++i) {
                        double max = std::fabs(overlap.get(i, i));
                        int root = i;
                        for (int j = 0; j < overlap.nrow(); ++j) {
                            double value = std::fabs(overlap.get(j, i));
                            if (max < value) {
                                max = value;
                                root = j;
                            }
                        }
                        t1_percentage_[h].push_back(
                            make_pair(root, 100.0 * max));
                    }
                } // end of <CIS|CISD>
            }
        }

        // real computation
        for (int h = 0; h < nirrep; ++h) {
            if (nrootpi_[h] == 0)
                continue;
            else {
                fci_mo.set_root_sym(h);

                for (int i = 0; i < nrootpi_[h]; ++i) {
                    // CI routine
                    outfile->Printf("\n\n  %s", std::string(35, '=').c_str());
                    outfile->Printf("\n    Current Job: %3s state, root %2d",
                                    irrep_symbol_[h].c_str(), i);
                    outfile->Printf("\n  %s\n", std::string(35, '=').c_str());
                    fci_mo.set_nroots(i + 1);
                    fci_mo.set_root(i);
                    ref_energies_[h].push_back(
                        fci_mo.compute_canonical_energy());
                    Reference reference = fci_mo.reference();
                    //                    dominant_dets_[h].push_back(fci_mo.dominant_det());
                    //                    orb_extents_[h].push_back(flatten_fci_orbextents(fci_mo.orb_extents()));

                    // PT2 or PT3 routine
                    double Ept = 0.0;
                    if (options_.get_str("CORR_LEVEL") == "PT2") {
                        if (options_.get_str("INT_TYPE") == "CONVENTIONAL") {
                            std::shared_ptr<DSRG_MRPT2> dsrg =
                                std::make_shared<DSRG_MRPT2>(
                                    reference, reference_wavefunction_,
                                    options_, ints_, mo_space_info_);
                            dsrg->set_ignore_semicanonical(true);
                            dsrg->set_actv_occ(fci_mo.actv_occ());
                            dsrg->set_actv_uocc(fci_mo.actv_uocc());
                            Ept = dsrg->compute_energy();
                        } else {
                            std::shared_ptr<THREE_DSRG_MRPT2> dsrg =
                                std::make_shared<THREE_DSRG_MRPT2>(
                                    reference, reference_wavefunction_,
                                    options_, ints_, mo_space_info_);
                            dsrg->ignore_semicanonical(true);
                            dsrg->set_actv_occ(fci_mo.actv_occ());
                            dsrg->set_actv_uocc(fci_mo.actv_uocc());
                            Ept = dsrg->compute_energy();
                        }
                    }
                    if (options_.get_str("CORR_LEVEL") == "PT3") {
                        auto dsrg = std::make_shared<DSRG_MRPT3>(
                            reference, reference_wavefunction_, options_, ints_,
                            mo_space_info_);
                        dsrg->ignore_semicanonical(true);
                        Ept = dsrg->compute_energy();
                    }
                    pt_energies_[h].push_back(Ept);

//                    // set back to HF orbitals
//                    fci_mo.set_orbs(Ca_hf, Cb_hf);
                }
            }
        }
        print_summary();

        // set the last energy to Process:environment
        for (int h = nirrep; h > 0; --h) {
            int n = nrootpi_[h - 1];
            if (n != 0) {
                Process::environment.globals["CURRENT ENERGY"] =
                    pt_energies_[h - 1][n - 1];
                break;
            }
        }
    }
    return 0.0;
}

void ACTIVE_DSRGPT2::print_summary() {
    std::string h2 = code_name_ + " Summary";
    print_h2(h2);

    std::string ref_type = options_.get_str("ACTIVE_SPACE_TYPE");
    if (ref_type == "COMPLETE")
        ref_type = std::string("CAS");

    int nirrep = nrootpi_.size();
    if (ref_type == "CIS" && options_.get_bool("CIS_CISD_OVERLAP")) {
        int total_width = 4 + 6 + 18 + 18 + 11 + 4 * 2;
        outfile->Printf("\n    %4s  %6s  %11s%7s  %11s%7s  %11s", "Sym.",
                        "ROOT", ref_type.c_str(), std::string(7, ' ').c_str(),
                        options_.get_str("CORR_LEVEL").c_str(),
                        std::string(7, ' ').c_str(), "% in CISD ");
        outfile->Printf("\n    %s", std::string(total_width, '-').c_str());
        for (int h = 0; h < nirrep; ++h) {
            if (nrootpi_[h] != 0) {
                for (int i = nrootpi_[h]; i > 0; --i) {
                    std::string sym(4, ' ');
                    if (i == 1)
                        sym = irrep_symbol_[h];
                    outfile->Printf(
                        "\n    %4s  %6d  %18.10f  %18.10f  %5.1f (%3d)",
                        sym.c_str(), i - 1, ref_energies_[h][i - 1],
                        pt_energies_[h][i - 1], t1_percentage_[h][i - 1].second,
                        t1_percentage_[h][i - 1].first);
                }
                outfile->Printf("\n    %s",
                                std::string(total_width, '-').c_str());
            }
        }
    } else {
        int total_width = 4 + 6 + 18 + 18 + 3 * 2;
        outfile->Printf("\n    %4s  %6s  %11s%7s  %11s", "Sym.", "ROOT",
                        ref_type.c_str(), std::string(7, ' ').c_str(),
                        options_.get_str("CORR_LEVEL").c_str());
        outfile->Printf("\n    %s", std::string(total_width, '-').c_str());
        for (int h = 0; h < nirrep; ++h) {
            if (nrootpi_[h] != 0) {
                for (int i = nrootpi_[h]; i > 0; --i) {
                    std::string sym(4, ' ');
                    if (i == 1)
                        sym = irrep_symbol_[h];
                    outfile->Printf("\n    %4s  %6d  %18.10f  %18.10f",
                                    sym.c_str(), i - 1, ref_energies_[h][i - 1],
                                    pt_energies_[h][i - 1]);
                }
                outfile->Printf("\n    %s",
                                std::string(total_width, '-').c_str());
            }
        }
    }

    if (nrootpi_[0] > 0 && total_nroots_ > 1) {
        print_h2("Relative Energy WRT Totally Symmetric Ground State (eV)");

        double ev = pc_hartree2ev;
        if (ref_type == "CAS") {
            int width = 4 + 6 + 8 + 8 + 3 * 2;
            outfile->Printf("\n    %4s  %6s  %6s%2s  %6s", "Sym.", "ROOT",
                            ref_type.c_str(), std::string(2, ' ').c_str(),
                            options_.get_str("CORR_LEVEL").c_str());
            outfile->Printf("\n    %s", std::string(width, '-').c_str());
            for (int h = 0; h < nirrep; ++h) {
                if (nrootpi_[h] != 0) {
                    for (int i = nrootpi_[h]; i > 0; --i) {
                        std::string sym(4, ' ');
                        if (h == 0 && i == 1)
                            continue;
                        if (h == 0 && i == 2)
                            sym = irrep_symbol_[h];
                        if (i == 1)
                            sym = irrep_symbol_[h];

                        double Eci =
                            (ref_energies_[h][i - 1] - ref_energies_[0][0]) *
                            ev;
                        double Ept =
                            (pt_energies_[h][i - 1] - pt_energies_[0][0]) * ev;
                        outfile->Printf("\n    %4s  %6d  %8.3f  %8.3f",
                                        sym.c_str(), i - 1, Eci, Ept);
                    }
                    if (h != 0 || nrootpi_[0] != 1)
                        outfile->Printf("\n    %s",
                                        std::string(width, '-').c_str());
                }
            }
        } else {
            int width = 4 + 6 + 8 + 8 + 40 + 4 * 2;
            outfile->Printf("\n    %4s  %6s  %6s%2s  %6s%2s  %40s", "Sym.",
                            "ROOT", ref_type.c_str(), "  ",
                            options_.get_str("CORR_LEVEL").c_str(), "  ",
                            "Excitation Type");
            outfile->Printf("\n    %s", std::string(width, '-').c_str());
            for (int h = 0; h < nirrep; ++h) {
                if (nrootpi_[h] != 0) {
                    for (int i = nrootpi_[h]; i > 0; --i) {
                        std::string sym(4, ' ');
                        if (h == 0 && i == 1)
                            continue;
                        if (h == 0 && i == 2)
                            sym = irrep_symbol_[h];
                        if (i == 1)
                            sym = irrep_symbol_[h];

                        double Eci =
                            (ref_energies_[h][i - 1] - ref_energies_[0][0]) *
                            ev;
                        double Ept =
                            (pt_energies_[h][i - 1] - pt_energies_[0][0]) * ev;
                        current_orb_extents_ = orb_extents_[h][i - 1];
                        std::string ex_type = compute_ex_type(
                            dominant_dets_[h][i - 1], dominant_dets_[0][0]);
                        outfile->Printf("\n    %4s  %6d  %8.3f  %8.3f  %40s",
                                        sym.c_str(), i - 1, Eci, Ept,
                                        ex_type.c_str());
                    }
                    if (h != 0 || nrootpi_[0] != 1)
                        outfile->Printf("\n    %s",
                                        std::string(width, '-').c_str());
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

std::string
ACTIVE_DSRGPT2::compute_ex_type(const STLBitsetDeterminant& det,
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
    std::set_intersection(occA_ref.begin(), occA_ref.end(), occA_det.begin(),
                          occA_det.end(), back_inserter(commonA));
    occA_ref.erase(std::set_difference(occA_ref.begin(), occA_ref.end(),
                                       commonA.begin(), commonA.end(),
                                       occA_ref.begin()),
                   occA_ref.end());
    occA_det.erase(std::set_difference(occA_det.begin(), occA_det.end(),
                                       commonA.begin(), commonA.end(),
                                       occA_det.begin()),
                   occA_det.end());

    // compare beta occ
    std::vector<int> occB_ref(ref_det.get_beta_occ());
    std::vector<int> occB_det(det.get_beta_occ());
    std::vector<int> commonB;
    std::set_intersection(occB_ref.begin(), occB_ref.end(), occB_det.begin(),
                          occB_det.end(), back_inserter(commonB));
    occB_ref.erase(std::set_difference(occB_ref.begin(), occB_ref.end(),
                                       commonB.begin(), commonB.end(),
                                       occB_ref.begin()),
                   occB_ref.end());
    occB_det.erase(std::set_difference(occB_det.begin(), occB_det.end(),
                                       commonB.begin(), commonB.end(),
                                       occB_det.begin()),
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
        double orbex_det = current_orb_extents_[idx_det];
        std::string r2_str =
            (orbex_det > 1.0e6 ? " (Diffuse) "
                               : str(boost::format(" (%7.2f) ") % orbex_det));

        output =
            sym_active[idx_ref] + " -> " + sym_active[idx_det] + r2_str + "(S)";
    }

    // CISD
    if (A + B == 2) {
        if (A == 1 && B == 1) {
            int i_ref = occA_ref[0], j_ref = occB_ref[0];
            int i_det = occA_det[0], j_det = occB_det[0];
            if (i_ref == j_ref && i_det == j_det) {
                double orbex_det = current_orb_extents_[i_det];
                std::string r2_str =
                    (orbex_det > 1.0e6
                         ? " (Diffuse) "
                         : str(boost::format(" (%7.2f) ") % orbex_det));
                output = sym_active[i_ref] + " -> " + sym_active[i_det] +
                         r2_str + "(D)";
            } else {
                double orbex_i_det = current_orb_extents_[i_det];
                double orbex_j_det = current_orb_extents_[j_det];
                std::string r2_str_i =
                    (orbex_i_det > 1.0e6
                         ? " (Diffuse) "
                         : str(boost::format(" (%7.2f)") % orbex_i_det));
                std::string r2_str_j =
                    (orbex_j_det > 1.0e6
                         ? " (Diffuse) "
                         : str(boost::format(" (%7.2f)") % orbex_j_det));

                output = sym_active[i_ref] + "," + sym_active[j_ref] + " -> " +
                         sym_active[i_det] + r2_str_i + "," +
                         sym_active[j_det] + r2_str_j;
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

            double orbex_i_det = current_orb_extents_[i_det];
            double orbex_j_det = current_orb_extents_[j_det];
            std::string r2_str_i =
                (orbex_i_det > 1.0e6
                     ? " (Diffuse) "
                     : str(boost::format(" (%7.2f)") % orbex_i_det));
            std::string r2_str_j =
                (orbex_j_det > 1.0e6
                     ? " (Diffuse) "
                     : str(boost::format(" (%7.2f)") % orbex_j_det));

            output = sym_active[i_ref] + "," + sym_active[j_ref] + " -> " +
                     sym_active[i_det] + r2_str_i + "," + sym_active[j_det] +
                     r2_str_j;
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
            double r2 = fci_orb_extents[h][i][0] + fci_orb_extents[h][i][1] +
                        fci_orb_extents[h][i][2];
            out.push_back(r2);
        }
    }

    return out;
}
}
}
