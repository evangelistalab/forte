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

#include <cmath>

#include "mini-boost/boost/timer.hpp"

#include <libmints/molecule.h>
#include <libmints/pointgrp.h>
#include <libmints/wavefunction.h>
#include <libpsio/psio.hpp>

#include "lambda-ci.h"

using namespace std;
using namespace psi;

namespace psi {
namespace forte {

LambdaCI::LambdaCI(Options& options, ForteIntegrals* ints)
    : options_(options), ints_(ints), min_energy_(0.0), pt2_energy_correction_(0.0) {
    ForteTimer t;

    // Read data and allocate member objects
    startup(options);

    std::string energy_type = options.get_str("ENERGY_TYPE");

    if (energy_type == "FACTORIZED_CI") {
        factorized_ci(options);
    } else {

        if (options.get_str("EXPLORER_ALGORITHM") == "DENOMINATORS") {
            explore_original(options);
        } else if (options.get_str("EXPLORER_ALGORITHM") == "SINGLES") {
            explore_singles(options);
        }

        // Optionally diagonalize a small Hamiltonian
        if (options.get_bool("COMPUTE_ENERGY")) {
            if (energy_type == "SELECT") {
                diagonalize_selected_space(options);
            } else if (energy_type == "FULL") {
                // Lambda-CI (store the full Hamiltonian)
                diagonalize_p_space(options);
            } else if (energy_type == "SPARSE") {
                // Lambda-CI (store only the non-zero elements of the Hamiltonian)
                diagonalize_p_space_direct(options);
            } else if ((energy_type == "LMRCISD") or (energy_type == "LMRCISD_SPARSE")) {
                // Lambda+SD-CI
                lambda_mrcisd(options);
            } else if ((energy_type == "LMRCIS") or (energy_type == "LMRCIS_SPARSE")) {
                // Lambda+S-CI
                lambda_mrcis(options);
            } else if (energy_type == "LOWDIN") {
                diagonalize_p_space_lowdin(options);
            } else if (energy_type == "RENORMALIZE") {
                diagonalize_renormalized_space(options);
            } else if (energy_type == "RENORMALIZE_FIXED") {
                diagonalize_renormalized_fixed_space(options);
            } else if ((energy_type == "IMRCISD") or (energy_type == "IMRCISD_SPARSE")) {
                //            iterative_adaptive_mrcisd(options);
                iterative_adaptive_mrcisd_bitset(options);
            }
        }
    }
    outfile->Printf("\n  Explorer ran in %f s", t.elapsed());
}

LambdaCI::~LambdaCI() {}

void LambdaCI::startup(Options& options) {
    read_info(options);

    screen_mos();

    // Connect the integrals to the determinant class
    StringDeterminant::set_ints(ints_);
    DynamicBitsetDeterminant::set_ints(ints_);

    // Build the reference determinant and compute its energy
    std::vector<int> occupation(2 * ncmo_, 0);
    int cumidx = 0;
    for (int h = 0; h < nirrep_; ++h) {
        for (int i = 0; i < nalphapi_ref_[h]; ++i) {
            occupation[i + cumidx] = 1;
        }
        for (int i = 0; i < nbetapi_ref_[h]; ++i) {
            occupation[ncmo_ + i + cumidx] = 1;
        }
        cumidx += ncmopi_[h];
    }
    reference_determinant_ = StringDeterminant(occupation);

    min_energy_ = ref_energy_ = reference_determinant_.energy() + nuclear_repulsion_energy_;
    min_energy_determinant_ = reference_determinant_;
    outfile->Printf("\n  The tentative reference determinant is:");
    reference_determinant_.print();
    outfile->Printf("\n  and its energy: %.12f Eh", min_energy_);

    max_energy_ = min_energy_;

    ints_->make_fock_matrix(reference_determinant_.get_alfa_bits(),
                            reference_determinant_.get_beta_bits());

    outfile->Printf("\n\n  Starting Explorer.\n\n");
}

void LambdaCI::read_info(Options& options) {
    // Now we want the reference (SCF) wavefunction
    boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();
    nirrep_ = wfn->nirrep();
    SharedVector wfn_eps_a_ = wfn->epsilon_a();
    SharedVector wfn_eps_b_ = wfn->epsilon_b();

    // The number of correlated molecular orbitals
    ncmo_ = ints_->ncmo();
    ncmopi_ = ints_->ncmopi();

    // Frozen orbitals read from the integral class
    frzcpi_ = ints_->frzcpi();
    frzvpi_ = ints_->frzvpi();

    ref_eps_a_ = SharedVector(new Vector("ref_eps_a", ncmopi_));
    ref_eps_b_ = SharedVector(new Vector("ref_eps_b", ncmopi_));
    // Sort the orbital energies
    for (int h = 0; h < nirrep_; ++h) {
        for (int p = 0; p < ncmopi_[h]; ++p) {
            ref_eps_a_->set(h, p, wfn_eps_a_->get(h, frzcpi_[h] + p));
            ref_eps_b_->set(h, p, wfn_eps_b_->get(h, frzcpi_[h] + p));
        }
    }

    // Read the restricted orbitals
    actvpi_ = Dimension(nirrep_, "Active MOs");
    rdoccpi_ = Dimension(nirrep_, "Restricted doubly occupied MOs");
    ruoccpi_ = Dimension(nirrep_, "Restricted unoccupied MOs");
    if (options["RESTRICTED_DOCC"].has_changed()) {
        if (options["RESTRICTED_DOCC"].size() == nirrep_) {
            for (int h = 0; h < nirrep_; ++h) {
                rdoccpi_[h] = options["RESTRICTED_DOCC"][h].to_integer();
            }
        } else {
            outfile->Printf("\n\n  The input array RESTRICTED_DOCC has information for %d irreps, "
                            "this does not match the total number of irreps %d",
                            options["RESTRICTED_DOCC"].size(), nirrep_);
            outfile->Printf("\n  Exiting the program.\n");
            printf("  The input array RESTRICTED_DOCC has information for %d irreps, this does not "
                   "match the total number of irreps %d",
                   options["RESTRICTED_DOCC"].size(), nirrep_);
            printf("\n  Exiting the program.\n");

            exit(Failure);
        }
    }
    if (options["RESTRICTED_UOCC"].has_changed()) {
        if (options["RESTRICTED_UOCC"].size() == nirrep_) {
            for (int h = 0; h < nirrep_; ++h) {
                ruoccpi_[h] = options["RESTRICTED_UOCC"][h].to_integer();
                actvpi_[h] = ncmopi_[h] - rdoccpi_[h] - ruoccpi_[h];
            }
        } else {
            outfile->Printf("\n\n  The input array RESTRICTED_UOCC has information for %d irreps, "
                            "this does not match the total number of irreps %d",
                            options["RESTRICTED_UOCC"].size(), nirrep_);
            outfile->Printf("\n  Exiting the program.\n");
            printf("  The input array RESTRICTED_UOCC has information for %d irreps, this does not "
                   "match the total number of irreps %d",
                   options["RESTRICTED_UOCC"].size(), nirrep_);
            printf("\n  Exiting the program.\n");
            exit(Failure);
        }
    }

    // Determine the active orbitals as the difference

    if (options["ACTIVE"].has_changed() and options["RESTRICTED_UOCC"].has_changed()) {
        outfile->Printf("\n\n  Cannot provide both ACTIVE and RESTRICTED_UOCC arrays");
        outfile->Printf("\n  Exiting the program.\n");
        printf("\n\n  Cannot provide both ACTIVE and RESTRICTED_UOCC arrays");
        printf("\n  Exiting the program.\n");
        exit(Failure);
    } else if (options["ACTIVE"].has_changed()) {
        if (options["ACTIVE"].size() == nirrep_) {
            for (int h = 0; h < nirrep_; ++h) {
                actvpi_[h] = options["ACTIVE"][h].to_integer();
                ruoccpi_[h] = ncmopi_[h] - rdoccpi_[h] - actvpi_[h];
            }
        } else {
            outfile->Printf("\n\n  The input array ACTIVE has information for %d irreps, this does "
                            "not match the total number of irreps %d",
                            options["ACTIVE"].size(), nirrep_);
            outfile->Printf("\n  Exiting the program.\n");
            printf("  The input array ACTIVE has information for %d irreps, this does not match "
                   "the total number of irreps %d",
                   options["ACTIVE"].size(), nirrep_);
            printf("\n  Exiting the program.\n");
            exit(Failure);
        }
    } else {
        actvpi_ = ncmopi_ - rdoccpi_ - ruoccpi_;
    }

    // Print a summary
    std::vector<std::pair<std::string, Dimension>> mo_space_info{
        {"Frozen doubly occupied", frzcpi_},
        {"Restricted doubly occupied", rdoccpi_},
        {"Active", actvpi_},
        {"Restricted unoccupied", rdoccpi_},
        {"Frozen unoccupied", frzvpi_}};

    CharacterTable ct = Process::environment.molecule()->point_group()->char_table();

    // Print some information
    outfile->Printf("\n  ==> Active Space Information <==\n");
    outfile->Printf("\n  %s", string(31 + (nirrep_ + 1) * 6, '-').c_str());
    outfile->Printf("\n%32c", ' ');
    for (int h = 0; h < nirrep_; ++h)
        outfile->Printf(" %5s", ct.gamma(h).symbol());
    outfile->Printf("  Total");
    outfile->Printf("\n  %s", string(31 + (nirrep_ + 1) * 6, '-').c_str());
    for (auto& str_dim : mo_space_info) {
        outfile->Printf("\n  %-30s", str_dim.first.c_str());
        for (int h = 0; h < nirrep_; ++h) {
            outfile->Printf(" %5d", str_dim.second[h]);
        }
        outfile->Printf(" %6d", str_dim.second.sum());
    }
    outfile->Printf("\n  %s", string(31 + (nirrep_ + 1) * 6, '-').c_str());

    // Create the vectors of frozen orbitals (in the Pitzer ordering)
    for (int h = 0, p = 0; h < nirrep_; ++h) {
        for (int i = 0; i < rdoccpi_[h]; ++i) {
            rdocc.push_back(p + i);
        }
        p += ncmopi_[h];
        for (int i = 0; i < ruoccpi_[h]; ++i) {
            ruocc.push_back(p - ruoccpi_[h] + i);
        }
    }

    // Create the array with mo symmetry
    for (int h = 0; h < nirrep_; ++h) {
        for (int p = 0; p < ncmopi_[h]; ++p) {
            mo_symmetry_.push_back(h);
        }
    }

    wavefunction_symmetry_ = 0;
    if (options["ROOT_SYM"].has_changed()) {
        wavefunction_symmetry_ = options.get_int("ROOT_SYM");
    }
    int charge = Process::environment.molecule()->molecular_charge();
    int multiplicity = Process::environment.molecule()->multiplicity();
    int nel = 0;

    // If the charge has changed, recompute the number of electrons
    // Or if you cannot find the number of electrons
    if ((nel == 0) or options["CHARGE"].has_changed()) {
        charge = options.get_int("CHARGE");
        nel = 0;
        int natom = Process::environment.molecule()->natom();
        for (int i = 0; i < natom; i++) {
            nel += static_cast<int>(Process::environment.molecule()->Z(i));
        }
        nel -= charge;
    }

    if (options["MULTIPLICITY"].has_changed()) {
        multiplicity = options.get_int("MULTIPLICITY");
    }

    if (((nel + 1 - multiplicity) % 2) != 0)
        throw PSIEXCEPTION("\n\n  MOInfoBase: Wrong multiplicity.\n\n");
    nel -= 2 * frzcpi_.sum();
    nalpha_ = (nel + multiplicity - 1) / 2;
    nbeta_ = nel - nalpha_;

    // Print a summary
    std::vector<std::pair<std::string, int>> calculation_info{
        {"Charge", charge},
        {"Multiplicity", multiplicity},
        {"Symmetry", wavefunction_symmetry_},
        {"Number of electrons", nel},
        {"Number of correlated alpha electrons", nalpha_},
        {"Number of correlated beta electrons", nbeta_},
        {"Number of restricted docc electrons", rdoccpi_.sum()},
        {"Number of active alpha electrons", nalpha_ - rdoccpi_.sum()},
        {"Number of beta alpha electrons", nbeta_ - rdoccpi_.sum()}};

    // Print some information
    outfile->Printf("\n\n  ==> Calculation Information <==\n");
    outfile->Printf("\n  %s", string(52, '-').c_str());
    for (auto& str_dim : calculation_info) {
        outfile->Printf("\n    %-40s   %5d", str_dim.first.c_str(), str_dim.second);
    }
    outfile->Printf("\n  %s", string(52, '-').c_str());

    Da_.assign(ncmo_, 0.0);
    Db_.assign(ncmo_, 0.0);

    boost::shared_ptr<Molecule> molecule_ = wfn->molecule();
    nuclear_repulsion_energy_ = molecule_->nuclear_repulsion_energy();

    determinant_threshold_ = options.get_double("DET_THRESHOLD");
    if (options["DEN_THRESHOLD"].has_changed()) {
        denominator_threshold_ = options.get_double("DEN_THRESHOLD");
    } else {
        denominator_threshold_ = 2.0 * determinant_threshold_;
    }
    space_m_threshold_ = options.get_double("SPACE_M_THRESHOLD");
    space_i_threshold_ = options.get_double("SPACE_I_THRESHOLD");
    if (space_m_threshold_ > determinant_threshold_) {
        space_m_threshold_ = determinant_threshold_;
        space_i_threshold_ = determinant_threshold_;
        outfile->Printf("\n  The model space comprises all the determinants.\n  Modifying the "
                        "model and intermediate space thresholds.\n");
    }
    if (space_m_threshold_ > space_i_threshold_) {
        space_i_threshold_ = space_m_threshold_;
    }
    if (space_i_threshold_ > determinant_threshold_) {
        space_i_threshold_ = determinant_threshold_;
        outfile->Printf("\n  Changing the value of the intermediate space threshold.\n");
    }

    t2_threshold_ = options.get_double("T2_THRESHOLD");

    if (options.get_str("SCREENING_TYPE") == "MP") {
        mp_screening_ = true;
    } else {
        mp_screening_ = false;
    }

    outfile->Printf("\n  Nuclear repulsion energy     = %20.12f a.u.", nuclear_repulsion_energy_);
    outfile->Printf("\n  Scalar energy contribution   = %20.12f a.u.", ints_->scalar());
    outfile->Printf("\n  Determinant threshold        = %.3f (Eh)", determinant_threshold_);
    outfile->Printf("\n  Denominator threshold        = %.3f (Eh)", denominator_threshold_);
    outfile->Printf("\n  Model space threshold        = %.3f (Eh)", space_m_threshold_);
    outfile->Printf("\n  Intermediate space threshold = %.3f (Eh)", space_i_threshold_);
    outfile->Printf("\n  Coupling threshold           = %.3f (muEh)", t2_threshold_ * 1000000.0);

    outfile->Printf("\n  String screening: %s (%s)",
                    mp_screening_ ? "Moller-Plesset denominators" : "excited determinants",
                    options.get_str("SCREENING_TYPE").c_str());
}

bool compare_tuples(const boost::tuple<double, int, int>& t1,
                    const boost::tuple<double, int, int>& t2) {
    if (t1.get<0>() != t2.get<0>()) {
        return (t1.get<0>() < t2.get<0>());
    } else if (t1.get<1>() != t2.get<1>()) {
        return (t1.get<1>() < t2.get<1>());
    }
    return (t1.get<2>() < t2.get<2>());
}

void LambdaCI::screen_mos() {
    // Determine the best occupation using the orbital energies
    std::vector<boost::tuple<double, int, int>> sorted_ea;
    std::vector<boost::tuple<double, int, int>> sorted_eb;
    for (int h = 0, sump = 0; h < nirrep_; ++h) {
        for (int p = 0; p < ncmopi_[h]; ++p, ++sump) {
            sorted_ea.push_back(boost::make_tuple(ref_eps_a_->get(h, p), h, sump));
            sorted_eb.push_back(boost::make_tuple(ref_eps_b_->get(h, p), h, sump));
        }
    }

    std::sort(sorted_ea.begin(), sorted_ea.end(), compare_tuples);
    std::sort(sorted_eb.begin(), sorted_eb.end(), compare_tuples);

    nalphapi_ref_ = Dimension(nirrep_);
    nbetapi_ref_ = Dimension(nirrep_);
    minalphapi_ = Dimension(nirrep_);
    minbetapi_ = Dimension(nirrep_);
    maxalphapi_ = Dimension(nirrep_);
    maxbetapi_ = Dimension(nirrep_);

    outfile->Printf("\n\n  ==> Molecular orbitals <==\n");
    outfile->Printf("\n  ====================================================");
    outfile->Printf("\n     MO         alpha                  beta");
    outfile->Printf("\n           irrep    energy  occ   irrep    energy  occ");
    outfile->Printf("\n  ----------------------------------------------------");
    for (int p = 0; p < ncmo_; ++p) {
        double ea = sorted_ea[p].get<0>();
        double eb = sorted_eb[p].get<0>();
        int ha = sorted_ea[p].get<1>();
        int hb = sorted_eb[p].get<1>();
        int pa = sorted_ea[p].get<2>();
        mo_symmetry_qt_.push_back(ha);

        //        double ea = std::get<0>(sorted_ea[p]);
        //        double eb = std::get<0>(sorted_eb[p]);
        //        int ha = std::get<1>(sorted_ea[p]);
        //        int hb = std::get<1>(sorted_eb[p]);
        //        int pa = std::get<2>(sorted_ea[p]);
        //        int pb = std::get<2>(sorted_eb[p]);
        // if (std::max(std::fabs(ea),std::fabs(eb)) < denominator_threshold_ * 1.25)

        bool excluded = false;
        outfile->Printf("\n %6d    %3d %12.6f  %1d    %3d %12.6f  %1d", p, ha, ea, p < nalpha_, hb,
                        eb, p < nbeta_);
        if (std::find(rdocc.begin(), rdocc.end(), pa) != rdocc.end()) {
            excluded = true;
            outfile->Printf(" <- restricted docc");
        }
        if (std::find(ruocc.begin(), ruocc.end(), pa) != ruocc.end()) {
            excluded = true;
            outfile->Printf(" <- restricted uocc");
        }
        if (not excluded) {
            epsilon_a_qt_.push_back(ea);
            epsilon_b_qt_.push_back(eb);
            qt_to_pitzer_.push_back(pa);
        }
    }
    outfile->Printf("\n  ----------------------------------------------------");

    for (int p = 0; p < nalpha_; ++p)
        nalphapi_ref_[sorted_ea[p].get<1>()] += 1;
    for (int p = 0; p < nbeta_; ++p)
        nbetapi_ref_[sorted_eb[p].get<1>()] += 1;
    //    for (int p = 0; p < nalpha_; ++p) nalphapi_ref_[std::get<1>(sorted_ea[p])] += 1;
    //    for (int p = 0; p < nbeta_; ++p) nbetapi_ref_[ std::get<1>(sorted_eb[p])] += 1;

    outfile->Printf("\n  Occupation numbers of the refence determinant:");
    outfile->Printf("|");
    for (int h = 0; h < nirrep_; ++h) {
        outfile->Printf(" %d", nalphapi_ref_[h]);
    }
    outfile->Printf(" > x ");
    outfile->Printf("|");
    for (int h = 0; h < nirrep_; ++h) {
        outfile->Printf(" %d", nbetapi_ref_[h]);
    }
    outfile->Printf(" >");

    double e_ahomo = sorted_ea[nalpha_ - 1].get<0>();
    double e_bhomo = sorted_eb[nbeta_ - 1].get<0>();
    double e_alumo = sorted_ea[nalpha_].get<0>();
    double e_blumo = sorted_eb[nbeta_].get<0>();

    //    double e_ahomo = std::get<0>(sorted_ea[nalpha_ - 1]);
    //    double e_bhomo = std::get<0>(sorted_eb[nbeta_ - 1]);
    //    double e_alumo = std::get<0>(sorted_ea[nalpha_]);
    //    double e_blumo = std::get<0>(sorted_eb[nbeta_]);

    outfile->Printf("\n  Energy of the alpha/beta HOMO: %12.6f %12.6f", e_ahomo, e_bhomo);
    outfile->Printf("\n  Energy of the alpha/beta LUMO: %12.6f %12.6f", e_alumo, e_blumo);
    // Determine the range of MOs to consider
    for (int h = 0; h < nirrep_; ++h) {
        for (int p = ncmopi_[h] - 1; p >= 0; --p) {
            if (e_alumo - ref_eps_a_->get(h, p) < denominator_threshold_) {
                minalphapi_[h] = p;
            }
            if (e_blumo - ref_eps_b_->get(h, p) < denominator_threshold_) {
                minbetapi_[h] = p;
            }
        }
        for (int p = 0; p < ncmopi_[h]; ++p) {
            if (ref_eps_a_->get(h, p) - e_ahomo < denominator_threshold_) {
                maxalphapi_[h] = p + 1;
            }
            if (ref_eps_b_->get(h, p) - e_bhomo < denominator_threshold_) {
                maxbetapi_[h] = p + 1;
            }
        }
    }

    outfile->Printf("\n  Orbital ranges:");
    outfile->Printf("|");
    for (int h = 0; h < nirrep_; ++h) {
        outfile->Printf(" %d/%d", minalphapi_[h], maxalphapi_[h]);
    }
    outfile->Printf(" > x ");
    outfile->Printf("|");
    for (int h = 0; h < nirrep_; ++h) {
        outfile->Printf(" %d/%d", minbetapi_[h], maxbetapi_[h]);
    }
    outfile->Printf(" >");
}
}
} // EndNamespaces
