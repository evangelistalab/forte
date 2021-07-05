/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include <map>
#include <numeric>
#include <regex>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/element_to_Z.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/petitelist.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/masses.h"

#include "boost/format.hpp"

#include "helpers/string_algorithms.h"
#include "base_classes/forte_options.h"
#include "helpers/printing.h"

#include "aosubspace.h"

std::vector<std::string> mysplit(const std::string& input, const std::string& regex) {
    // passing -1 as the submatch index parameter performs splitting
    std::regex re(regex);
    std::sregex_token_iterator first{input.begin(), input.end(), re, -1}, last;
    return {first, last};
}

using namespace psi;

namespace forte {
psi::SharedMatrix make_aosubspace_projector(psi::SharedWavefunction wfn,
                                            std::shared_ptr<ForteOptions> options,
                                            const py::dict& atom_normals) {
    psi::SharedMatrix Ps;

    py::list subspace_list = options->get_gen_list("SUBSPACE");

    int subspace_list_size = static_cast<int>(subspace_list.size());
    if (subspace_list_size <= 0)
        return Ps;

    // Run this code only if user specified a subspace
    std::vector<std::string> subspace_str;
    for (int entry = 0; entry < subspace_list_size; ++entry) {
        std::string s = py::str(subspace_list[entry]);
        to_upper_string(s); // convert to upper case
        subspace_str.push_back(s);
    }

    // Create a basis set parser object and read the minimal basis
    std::shared_ptr<psi::Molecule> molecule = wfn->molecule();
    std::shared_ptr<psi::BasisSet> min_basis = wfn->get_basisset("MINAO_BASIS");

    // Parse the atom normals used to project the p orbitals
    std::map<std::pair<int, int>, psi::Vector3> atom_to_normal;
    for (const auto& item : atom_normals) {
        auto atom = item.first.cast<std::pair<int, int>>();
        auto normal = item.second.cast<std::array<double, 3>>();
        atom_to_normal[atom] = psi::Vector3(normal);
    }

    // Create an AOSubspace object
    AOSubspace aosub(subspace_str, molecule, min_basis, atom_to_normal);

    // build the projector
    Ps = aosub.build_projector(wfn->basisset());

    // print the overlap of the projector
    psi::SharedMatrix CPsC = Ps->clone();
    CPsC->transform(wfn->Ca());
    double print_threshold = 1.0e-3;
    auto irrep_labels = molecule->irrep_labels();
    outfile->Printf("\n  ==> Orbital Overlap with AO Subspace (> %.2e) <==\n", print_threshold);
    outfile->Printf("\n    =======================");
    outfile->Printf("\n    Irrep   MO  <phi|P|phi>");
    outfile->Printf("\n    -----------------------");
    for (int h = 0, nirrep = CPsC->nirrep(); h < nirrep; h++) {
        for (int i = 0, size = CPsC->rowspi(h); i < size; i++) {
            if (CPsC->get(h, i, i) > print_threshold) {
                outfile->Printf("\n    %4s  %4d  %10.6f", irrep_labels[h].c_str(), i + 1,
                                CPsC->get(h, i, i));
            }
        }
    }
    outfile->Printf("\n    ========================");
    return Ps;
}

AOSubspace::AOSubspace(std::vector<std::string> subspace_str,
                       std::shared_ptr<psi::Molecule> molecule,
                       std::shared_ptr<psi::BasisSet> minao_basis,
                       std::map<std::pair<int, int>, psi::Vector3> atom_normals, bool debug_mode)
    : subspace_str_(subspace_str), molecule_(molecule), min_basis_(minao_basis), subspace_counter_(0),
      atom_normals_(atom_normals), debug_(debug_mode) {
    startup();
}

void AOSubspace::startup() {
    if (not min_basis_->has_puream()) {
        throw std::runtime_error("Only support spherical MINAO basis!");
    }

    l_labels_ = {"S", "P", "D", "F", "G", "H", "I", "K", "L", "M"};

    lm_labels_spherical_ = {{"S"},
                            {"PZ", "PX", "PY"},
                            {"DZ2", "DXZ", "DYZ", "DX2-Y2", "DXY"},
                            {"FZ3", "FXZ2", "FYZ2", "FZX2-ZY2", "FXYZ", "FX3-3XY2", "F3X2Y-Y3"},
                            {"G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9"},
                            {"H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "H11"}};

    for (int l = 0, lmax = static_cast<int>(lm_labels_spherical_.size()); l < lmax; ++l) {
        for (int m = 0, mmax = static_cast<int>(lm_labels_spherical_[l].size()); m < mmax; ++m) {
            labels_spherical_to_lm_[lm_labels_spherical_[l][m]] = {std::make_pair(l, m)};
        }
    }

    for (int l = 0, lmax = static_cast<int>(l_labels_.size()); l < lmax; ++l) {
        int mmax = 2 * l + 1;
        std::vector<std::pair<int, int>> lm_vec(mmax);
        for (int m = 0; m < mmax; ++m) {
            lm_vec[m] = {l, m};
        }
        labels_spherical_to_lm_[l_labels_[l]] = lm_vec;
    }

    // check the point group of the molecule. If it is not set, set it
    if (!molecule_->point_group()) {
        molecule_->set_point_group(molecule_->find_point_group());
    }

    // parse the basis set
    parse_basis_set();

    // find subspace orbitals
    parse_subspace();
}

void AOSubspace::parse_basis_set() {
    // Form a map that lists all functions on a given atom and with a given angular momentum
    std::map<std::pair<int, int>, std::vector<int>> atom_am_to_f;
    bool pure_am = min_basis_->has_puream();

    if (debug_) {
        outfile->Printf("\n  Parsing basis set\n");
        outfile->Printf("  Pure Angular Momentum: %s\n", pure_am ? "True" : "False");
    }

    int count = 0;

    std::vector<int> element_count(130);

    for (int A = 0; A < molecule_->natom(); A++) {
        int Z = static_cast<int>(round(molecule_->Z(A)));

        int n_shell = min_basis_->nshell_on_center(A);

        std::vector<int> n_count(10, 1);
        std::iota(n_count.begin(), n_count.end(), 1);

        std::vector<int> ao_list;

        if (debug_)
            outfile->Printf("\n  Atom %d (Z = %d) has %d shells\n", A, Z, n_shell);

        for (int Q = 0; Q < n_shell; Q++) {
            const GaussianShell& shell = min_basis_->shell(A, Q);
            int nfunction = shell.nfunction();
            int l = shell.am();
            if (debug_)
                outfile->Printf("    Shell %d: L = %d, N = %d (%d -> %d)\n", Q, l, nfunction, count,
                                count + nfunction);
            for (int m = 0; m < nfunction; ++m) {
                AOInfo ao(A, Z, element_count[Z], n_count[l], l, m);
                aoinfo_vec_.push_back(ao);
                ao_list.push_back(count);
                count += 1;
            }
            n_count[l] += 1; // increase the angular momentum count
        }

        atom_to_aos_[Z].push_back(ao_list);

        element_count[Z] += 1; // increase the element count
    }
}

void AOSubspace::parse_subspace() {
    print_h2("List of Subspace Orbitals Requested");
    for (size_t i = 0, size = subspace_str_.size(); i < size; ++i) {
        outfile->Printf(" %13s", subspace_str_[i].c_str());
        if ((i + 1) % 5 == 0 and i + 1 != size)
            outfile->Printf("\n  ");
    }
    outfile->Printf("\n");
    if (not atom_normals_.empty()) {
        outfile->Printf("  NOTE: Subspace orbitals may be truncated based on requested planes!");
    }

    // parse subspace orbitals
    subspace_counter_ = 0;
    bool all_found = true;
    for (const std::string& s : subspace_str_) {
        all_found &= parse_subspace_entry(s);
    }

    // print basis and mark those are selected into the subspace
    print_h2("AO Basis Set Selected By Subspace");
    outfile->Printf("\n    =======================================");
    outfile->Printf("\n      AO  Atom  Label     Type  Coefficient");
    outfile->Printf("\n    ---------------------------------------");
    for (const auto& tup : subspace_) {
        int i_min, i_sub;
        double c;
        std::tie(i_min, i_sub, c) = tup;

        const auto& ao_info = aoinfo_vec_[i_min];
        int A = ao_info.A() + 1;
        auto atom_label = atomic_labels[ao_info.Z()] + std::to_string(ao_info.element_count() + 1);
        auto ao_type = std::to_string(ao_info.n()) + lm_labels_spherical_[ao_info.l()][ao_info.m()];

        outfile->Printf("\n    %4d  %4d %6s %8s  %11.4E", i_min, A, atom_label.c_str(),
                        ao_type.c_str(), c);
    }
    outfile->Printf("\n    ---------------------------------------");
    outfile->Printf("\n    Number of subspace orbitals: %10d", subspace_counter_);
    outfile->Printf("\n    =======================================\n");

    // throw if input is wrong
    if (not all_found) {
        outfile->Printf("\n  Some subspace orbitals are not found. Please check the input.");
        outfile->Printf("\n  Orbital labels available:");
        for (const auto& am_labels : lm_labels_spherical_) {
            outfile->Printf("\n");
            for (const auto& label : am_labels) {
                outfile->Printf("  %8s", label.c_str());
            }
        }
        throw psi::PSIEXCEPTION("Some subspace orbitals are not found. Please check the input.");
    }
}

bool AOSubspace::parse_subspace_entry(const std::string& s) {
    // The regex to parse the entries
    std::regex re("([a-zA-Z]{1,2})([0-9]+)?-?([0-9]+)?\\(?((?:\\/"
                  "?[1-9]{1}[SPDFGH]{1}[a-zA-Z0-9-]*)*)\\)?");
    std::smatch match;

    Element_to_Z etoZ;

    bool found = true;

    std::regex_match(s, match, re);
    if (debug_) {
        outfile->Printf("\n  Parsing entry: '%s'", s.c_str());
        for (std::ssub_match base_sub_match : match) {
            std::string base = base_sub_match.str();
            outfile->Printf("\n  --> '%s'", base.c_str());
        }
    }
    if (match.size() == 5) {
        // Find Z
        int Z = static_cast<int>(etoZ[match[1].str()]);

        // Find the range
        int minA = 0;
        int maxA = static_cast<int>(atom_to_aos_[Z].size());
        if (not match[2].str().empty()) {
            minA = std::stoi(match[2].str()) - 1;
            maxA = match[3].str().empty() ? minA + 1 : std::stoi(match[3].str());
        }

        if (debug_) {
            outfile->Printf("\n  Element %s -> %d", match[1].str().c_str(), Z);
            outfile->Printf("\n  Range %d -> %d", minA, maxA);
        }

        // Find the subset of AOs
        if (not match[4].str().empty()) {
            // Include some of the AOs
            std::vector<std::string> vec_str = mysplit(match[4].str(), "/");
            for (std::string str : vec_str) {
                // possible format of str: "2P", "3DZ2", "4DX2-Y2", etc
                int n = std::stoi(&str[0]);
                str.erase(0, 1);
                if (labels_spherical_to_lm_.count(str) > 0) {
                    for (int A = minA; A < maxA; ++A) {
                        std::pair<int, int> atom_key{Z, A};
                        if (str == "P" and atom_normals_.find(atom_key) != atom_normals_.end()) {
                            for (int pos : atom_to_aos_[Z][A]) {
                                if ((aoinfo_vec_[pos].n() == n) and (aoinfo_vec_[pos].l() == 1)) {
                                    if (debug_)
                                        outfile->Printf("\n     + found AO @ %d, SUB @ %d", pos,
                                                        subspace_counter_);
                                    // directions: AO: pz, px, py; plane normal: x, y, z
                                    int m = (aoinfo_vec_[pos].m() + 2) % 3;
                                    double c = atom_normals_[atom_key][m];
                                    subspace_.emplace_back(pos, subspace_counter_, c);
                                }
                            }
                            subspace_counter_ += 1;
                        } else {
                            for (std::pair<int, int> lm : labels_spherical_to_lm_[str]) {
                                int l = lm.first;
                                int m = lm.second;
                                if (debug_)
                                    outfile->Printf("\n     -> %s (n = %d,l = %d, m = %d)",
                                                    str.c_str(), n, l, m);
                                for (int pos : atom_to_aos_[Z][A]) {
                                    if ((aoinfo_vec_[pos].n() == n) and
                                        (aoinfo_vec_[pos].l() == l) and
                                        (aoinfo_vec_[pos].m() == m)) {
                                        if (debug_)
                                            outfile->Printf("\n     + found AO @ %d, SUB @ %d", pos,
                                                            subspace_counter_);
                                        subspace_.emplace_back(pos, subspace_counter_++, 1.0);
                                    }
                                }
                            }
                        }
                    }
                } else {
                    outfile->Printf("\n  AO label '%s' is not valid.", str.c_str());
                    found = false;
                }
            }
        } else {
            // Include all the AOs
            for (int A = minA; A < maxA; A++) {
                std::pair<int, int> atom_key{Z, A};
                bool in_plane = atom_normals_.find(atom_key) != atom_normals_.end();

                for (int pos : atom_to_aos_[Z][A]) {
                    if (aoinfo_vec_[pos].l() == 1 and in_plane) {
                        if (debug_)
                            outfile->Printf("\n     + found AO @ %d, SUB @ %d", pos,
                                            subspace_counter_);
                        // directions: AO: pz, px, py; plane normal: x, y, z
                        int m = (aoinfo_vec_[pos].m() + 2) % 3;
                        double c = atom_normals_[atom_key][m];
                        subspace_.emplace_back(pos, subspace_counter_, c);
                        subspace_counter_ += aoinfo_vec_[pos].m() == 2 ? 1 : 0;
                    } else {
                        if (debug_)
                            outfile->Printf("\n     + found AO @ %d, SUB @ %d", pos,
                                            subspace_counter_);
                        subspace_.emplace_back(pos, subspace_counter_++, 1.0);
                    }
                }
            }
        }
    }

    return found;
}

std::shared_ptr<psi::Matrix>
AOSubspace::build_projector(const std::shared_ptr<psi::BasisSet>& large_basis) {
    // compute the overlap matrix between minimal (MINAO) orbitals
    auto ints_mm =
        std::make_unique<IntegralFactory>(min_basis_, min_basis_, min_basis_, min_basis_);
    int nbf_m = min_basis_->nbf();
    std::unique_ptr<OneBodyAOInt> ao_mm(ints_mm->ao_overlap());
    auto Smm = std::make_shared<psi::Matrix>("Smm", nbf_m, nbf_m);
    ao_mm->compute(Smm);
    if (debug_) {
        Smm->print();
    }

    // compute the overlap matrix between target subspace orbitals
    int nbf_s = subspace_counter_;
    auto Cms = std::make_shared<psi::Matrix>("Cms", nbf_m, nbf_s);
    for (const auto& tup : subspace_) {
        int m, s;
        double c;
        std::tie(m, s, c) = tup;
        Cms->set(m, s, c);
    }
    if (debug_) {
        Cms->print();
    }

    auto Sss = psi::linalg::triplet(Cms, Smm, Cms, true, false, false);
    Sss->set_name("Sss");
    if (debug_) {
        Sss->print();
    }

    // orthogonalize the subspace
    auto Xss = Sss->clone();
    Xss->set_name("Xss");
    Xss->power(-0.5);
    if (debug_) {
        Xss->print();
        auto Sss_clone = Sss->clone();
        Sss_clone->transform(Xss);
        Sss_clone->print();
    }

    // compute the overlap between the subspace and large bases
    auto ints_ml =
        std::make_unique<IntegralFactory>(min_basis_, large_basis, min_basis_, large_basis);
    int nbf_l = large_basis->nbf();
    std::unique_ptr<OneBodyAOInt> ao_ml(ints_ml->ao_overlap());
    auto Sml = std::make_shared<psi::Matrix>("Sml", nbf_m, nbf_l);
    ao_ml->compute(Sml);

    auto Ssl = psi::linalg::doublet(Cms, Sml, true, false);
    Ssl->set_name("Ssl");
    if (debug_) {
        Ssl->print();
    }

    // build AO projector
    // Pao = Ssl^T Sss^-1 Ssl = (Cms^T Sml)^T (Xss^T Xss) (Cms^T Sml)
    auto Xsl = psi::linalg::doublet(Xss, Ssl, false, false);
    Xsl->set_name("Xsl");
    if (debug_) {
        Xsl->print();
        auto Sss_clone = Sss->clone();
        Sss_clone->transform(Xsl);
        Sss_clone->print();
    }

    auto Pao = psi::linalg::doublet(Xsl, Xsl, true, false);
    Pao->set_name("PAO");
    if (debug_) {
        Pao->print();
    }

    // transform AO projector to account for symmetry
    auto ints_ll =
        std::make_shared<IntegralFactory>(large_basis, large_basis, large_basis, large_basis);
    auto plist = std::make_unique<psi::PetiteList>(large_basis, ints_ll);
    auto ao_to_so = plist->aotoso();

    auto dim = plist->SO_basisdim();
    auto Pso = std::make_shared<psi::Matrix>("PSO", dim, dim);
    Pso->apply_symmetry(Pao, ao_to_so);
    if (debug_) {
        Pso->print();
    }

    return Pso;
}
} // namespace forte
