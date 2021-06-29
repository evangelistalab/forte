/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

namespace py = pybind11;

#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/element_to_Z.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/petitelist.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/masses.h"

#include "boost/format.hpp"

#include "helpers/string_algorithms.h"
#include "base_classes/forte_options.h"
#include "helpers/printing.h"

#include "aosubspace.h"

std::vector<std::string> mysplit(const std::string& input, const std::string& regex);

std::vector<std::string> mysplit(const std::string& input, const std::string& regex) {
    // passing -1 as the submatch index parameter performs splitting
    std::regex re(regex);
    std::sregex_token_iterator first{input.begin(), input.end(), re, -1}, last;
    return {first, last};
}

using namespace psi;

namespace forte {

psi::SharedMatrix make_aosubspace_projector(psi::SharedWavefunction wfn,
                                            std::shared_ptr<ForteOptions> options) {
    psi::SharedMatrix Ps;

    // Run this code only if user specified a subspace
    py::list subspace_list = options->get_gen_list("SUBSPACE");
    py::list subspace_pi_list = options->get_gen_list("SUBSPACE_PI_PLANES");

    int subspace_list_size = subspace_list.size();
    if (subspace_list_size > 0) {
        std::vector<std::string> subspace_str;
        for (int entry = 0; entry < subspace_list_size; ++entry) {
            std::string s = py::str(subspace_list[entry]);
            // convert to upper case
            to_upper_string(s);
            subspace_str.push_back(s);
        }

        // Create a basis set parser object and read the minimal basis
        std::shared_ptr<psi::Molecule> molecule = wfn->molecule();
        std::shared_ptr<psi::BasisSet> min_basis = wfn->get_basisset("MINAO_BASIS");

        // Read subspace pi planes
        std::vector<std::vector<std::string>> pi_planes;
        int n_planes = subspace_pi_list.size();
        if (n_planes) {
            for (int entry = 0; entry < n_planes; ++entry) {
                std::vector<std::string> pi_plane;
                py::list plane = subspace_pi_list[entry];
                for (int atoms = 0, n = plane.size(); atoms < n; ++atoms) {
                    std::string s = py::str(plane[atoms]);
                    to_upper_string(s);
                    pi_plane.push_back(s);
                }
                pi_planes.push_back(pi_plane);
            }
        }

        // Create an AOSubspace object
        AOSubspace aosub(subspace_str, molecule, min_basis, pi_planes);

        // Compute the subspaces (right now this is required before any other call)
        aosub.find_subspace();

        // build the projector
        Ps = aosub.build_projector(wfn->basisset());

        //        const std::vector<int>& subspace = aosub.subspace();
        //
        //        // build the projector
        //        Ps = aosub.build_projector(subspace, molecule, min_basis, wfn->basisset());

        // print the overlap of the projector
        psi::SharedMatrix CPsC = Ps->clone();
        CPsC->transform(wfn->Ca());
        double print_threshold = 1.0e-3;
        auto irrep_labels = molecule->irrep_labels();
        outfile->Printf("\n  Orbital overlap with AO subspace (> %.2e):\n", print_threshold);
        outfile->Printf("    =======================\n");
        outfile->Printf("    Irrep   MO  <phi|P|phi>\n");
        outfile->Printf("    -----------------------\n");
        for (int h = 0, nirrep = CPsC->nirrep(); h < nirrep; h++) {
            for (int i = 0, size = CPsC->rowspi(h); i < size; i++) {
                if (CPsC->get(h, i, i) > print_threshold) {
                    outfile->Printf("    %4s  %4d  %10.6f\n", irrep_labels[h].c_str(), i + 1,
                                    CPsC->get(h, i, i));
                }
            }
        }
        outfile->Printf("    ========================\n");
    }
    return Ps;
}

AOSubspace::AOSubspace(std::shared_ptr<Molecule> molecule, std::shared_ptr<BasisSet> basis)
    : molecule_(molecule), min_basis_(basis) {
    startup();
}

AOSubspace::AOSubspace(std::vector<std::string> subspace_str,
                       std::shared_ptr<psi::Molecule> molecule,
                       std::shared_ptr<psi::BasisSet> basis)
    : subspace_str_(subspace_str), molecule_(molecule), min_basis_(basis) {
    startup();
}

AOSubspace::AOSubspace(std::vector<std::string> subspace_str,
                       std::shared_ptr<psi::Molecule> molecule,
                       std::shared_ptr<psi::BasisSet> basis,
                       std::vector<std::vector<std::string>> subspace_pi_str)
    : subspace_str_(subspace_str), molecule_(molecule), min_basis_(basis),
      subspace_pi_str_(subspace_pi_str) {
    startup();
}

void AOSubspace::find_subspace() {
    parse_basis_set();
    parse_pi_planes();
    parse_subspace();
}

void AOSubspace::startup() {
    //    outfile->Printf("  ---------------------------------------\n");
    //    outfile->Printf("    Atomic Orbital Subspace\n");
    //    outfile->Printf("    written by Francesco A. Evangelista\n");
    //    outfile->Printf("  ---------------------------------------\n");

    lm_labels_cartesian_ = {
        {"S"},
        {"PX", "PY", "PZ"},
        {"DX2", "DXY", "DXZ", "DY2", "DYZ", "DZ2"},
        {"FX3", "FX2Y", "FX2Z", "FXY2", "FXYZ", "FXZ2", "FY3", "FY2Z", "FYZ2", "FZ3"}};

    l_labels_ = {"S", "P", "D", "F", "G", "H", "I", "K", "L", "M"};

    lm_labels_sperical_ = {{"S"},
                           {"PZ", "PX", "PY"},
                           {"DZ2", "DXZ", "DYZ", "DX2-Y2", "DXY"},
                           {"FZ3", "FXZ2", "FYZ2", "FZX2-ZY2", "FXYZ", "FX3-3XY2", "F3X2Y-Y3"},
                           {"G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9"},
                           {"H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "H11"}};

    for (int l = 0; l < (int)lm_labels_sperical_.size(); ++l) {
        for (int m = 0; m < (int)lm_labels_sperical_[l].size(); ++m) {
            labels_sperical_to_lm_[lm_labels_sperical_[l][m]] = {std::make_pair(l, m)};
        }
    }
    for (int l = 0; l < (int)l_labels_.size(); ++l) {
        std::vector<std::pair<int, int>> lm_vec;
        for (int m = 0; m < 2 * l + 1; ++m) {
            lm_vec.push_back(std::make_pair(l, m));
        }
        labels_sperical_to_lm_[l_labels_[l]] = lm_vec;
    }

    // Check the point group of the molecule. If it is not set, set it
    if (!molecule_->point_group()) {
        molecule_->set_point_group(molecule_->find_point_group());
    }
}

std::shared_ptr<psi::Matrix>
AOSubspace::build_projector(std::shared_ptr<psi::BasisSet> large_basis) {
    // compute the overlap matrix between minimal (MINAO) orbitals
    auto ints_mm =
        std::make_unique<IntegralFactory>(min_basis_, min_basis_, min_basis_, min_basis_);
    int nbf_m = min_basis_->nbf();
    std::unique_ptr<OneBodyAOInt> ao_mm(ints_mm->ao_overlap());
    auto Smm = std::make_shared<psi::Matrix>("Smm", nbf_m, nbf_m);
    ao_mm->compute(Smm);

    // compute the overlap matrix between target subspace orbitals
    int nbf_s = subspace_counter_;
    auto Cms = std::make_shared<psi::Matrix>("Cms", nbf_m, nbf_s);
    for (const auto& tup : subspace_tuple_) {
        int m, s;
        double c;
        std::tie(m, s, c) = tup;
        Cms->set(m, s, c);
    }

    auto Sss = psi::linalg::triplet(Cms, Smm, Cms, true, false, false);
    Sss->set_name("Sss");

    // orthogonalize the subspace
    auto Xss = Sss->clone();
    Xss->set_name("Xss");
    Xss->power(-0.5);

    // compute the overlap between the subspace and large bases
    auto ints_ml =
        std::make_unique<IntegralFactory>(min_basis_, large_basis, min_basis_, large_basis);
    int nbf_l = large_basis->nbf();
    std::unique_ptr<OneBodyAOInt> ao_ml(ints_ml->ao_overlap());
    auto Sml = std::make_shared<psi::Matrix>("Sml", nbf_m, nbf_l);
    ao_ml->compute(Sml);

    auto Ssl = psi::linalg::doublet(Cms, Sml, true, false);
    Ssl->set_name("Ssl");

    // build AO projector
    // Pao = Ssl^T Sss^-1 Ssl = (Cms^T Sml)^T (Xss^T Xss) (Cms^T Sml)
    auto Xsl = psi::linalg::doublet(Xss, Ssl, false, false);
    Xsl->set_name("Xsl");

    auto Pao = psi::linalg::doublet(Xsl, Xsl, true, false);
    Pao->set_name("PAO");

    // transform AO projector to account for symmetry
    auto ints_ll =
        std::make_shared<IntegralFactory>(large_basis, large_basis, large_basis, large_basis);
    auto plist = std::make_unique<psi::PetiteList>(large_basis, ints_ll);
    auto ao_to_so = plist->aotoso();

    auto Pso = psi::linalg::triplet(ao_to_so, Pao, ao_to_so, true, false, false);

    // debug printing
    if (debug_) {
        Smm->print();
        Cms->print();
        Sss->print();
        Xss->print();

        auto Sss_clone = Sss->clone();
        Sss_clone->transform(Xss);
        Sss_clone->print();

        Ssl->print();
        Xsl->print();

        Sss_clone = Sss->clone();
        Sss_clone->transform(Xsl);
        Sss_clone->print();

        Pao->print();
        Pso->print();
    }

    return Pso;
}

psi::SharedMatrix AOSubspace::build_projector(const std::vector<int>& subspace,
                                              std::shared_ptr<psi::Molecule> molecule,
                                              std::shared_ptr<psi::BasisSet> min_basis,
                                              std::shared_ptr<psi::BasisSet> large_basis) {

    auto integral_mm =
        std::make_shared<IntegralFactory>(min_basis, min_basis, min_basis, min_basis);
    auto integral_ml =
        std::make_shared<IntegralFactory>(min_basis, large_basis, large_basis, large_basis);
    auto integral_ll =
        std::make_shared<IntegralFactory>(large_basis, large_basis, large_basis, large_basis);

    int nbf_s = static_cast<int>(subspace.size());
    int nbf_m = min_basis->nbf();
    int nbf_l = large_basis->nbf();

    // Check the point group of the molecule. If it is not set, set it
    if (!molecule->point_group()) {
        molecule->set_point_group(molecule->find_point_group());
    }

    // Compute the overlap integral in the minimal basis
    std::shared_ptr<OneBodyAOInt> sOBI_mm(integral_mm->ao_overlap());
    psi::SharedMatrix S_mm = std::make_shared<psi::Matrix>("S_mm", nbf_m, nbf_m);
    sOBI_mm->compute(S_mm);

#if _DEBUG_AOSUBSPACE_
    S_mm->print();
#endif

    // Extract the subspace block
    psi::SharedMatrix S_ss = std::make_shared<psi::Matrix>("S_ss", nbf_s, nbf_s);
    for (int mu = 0; mu < nbf_s; mu++) {
        for (int nu = 0; nu < nbf_s; nu++) {
            S_ss->set(mu, nu, S_mm->get(subspace[mu], subspace[nu]));
        }
    }
#if _DEBUG_AOSUBSPACE_
    S_ss->print();
#endif

    // Orthogonalize the subspace
    psi::SharedMatrix X_ss = std::make_shared<psi::Matrix>("X", nbf_s, nbf_s);
    X_ss->copy(S_ss);
    X_ss->power(-0.5);

#if _DEBUG_AOSUBSPACE_
    X_ss->print();
#endif

    S_ss->transform(X_ss);

#if _DEBUG_AOSUBSPACE_
    S_ss->print();
#endif

    psi::SharedMatrix X_mm = std::make_shared<psi::Matrix>("X_mm", nbf_m, nbf_m);
    for (int mu = 0; mu < nbf_s; mu++) {
        for (int nu = 0; nu < nbf_s; nu++) {
            X_mm->set(subspace[mu], subspace[nu], X_ss->get(mu, nu));
        }
    }
#if _DEBUG_AOSUBSPACE_
    X_mm->print();
#endif

    // Compute the overlap integral in the minimal/large basis
    std::shared_ptr<OneBodyAOInt> sOBI_ml(integral_ml->ao_overlap());
    psi::SharedMatrix S_ml = std::make_shared<psi::Matrix>("S_ml", nbf_m, nbf_l);
    sOBI_ml->compute(S_ml);
#if _DEBUG_AOSUBSPACE_
    S_ml->print();
#endif

    psi::SharedMatrix XS_ml = std::make_shared<psi::Matrix>("XS_ml", nbf_m, nbf_l);
    psi::SharedMatrix SXXS_ll = std::make_shared<psi::Matrix>("SXXS_ll", nbf_l, nbf_l);
    XS_ml->gemm(false, false, 1.0, X_mm, S_ml, 0.0);
    SXXS_ll->gemm(true, false, 1.0, XS_ml, XS_ml, 0.0);
#if _DEBUG_AOSUBSPACE_
    XS_ml->print();
    SXXS_ll->print();
#endif

    std::shared_ptr<PetiteList> plist(new PetiteList(large_basis, integral_ll));
    psi::SharedMatrix AO2SO_ = plist->aotoso();
    psi::Dimension large_basis_so_dim = plist->SO_basisdim();
    auto SXXS_ll_so =
        std::make_shared<psi::Matrix>("SXXS_ll_so", large_basis_so_dim, large_basis_so_dim);
    SXXS_ll_so->apply_symmetry(SXXS_ll, AO2SO_);
#if _DEBUG_AOSUBSPACE_
    SXXS_ll_so->print();
#endif

    return SXXS_ll_so;
}

const std::vector<int>& AOSubspace::subspace() { return subspace_; }

std::vector<std::string> AOSubspace::aolabels(std::string str_format) const {
    std::vector<std::string> aolbl;
    for (const AOInfo& aoinfo : aoinfo_vec_) {
        std::string s = boost::str(boost::format(str_format) % (aoinfo.A() + 1) %
                                   atomic_labels[aoinfo.Z()] % (aoinfo.element_count() + 1) %
                                   aoinfo.n() % lm_labels_sperical_[aoinfo.l()][aoinfo.m()]);
        aolbl.push_back(s);
    }
    return aolbl;
}

const std::vector<AOInfo>& AOSubspace::aoinfo() const { return aoinfo_vec_; }

void AOSubspace::parse_subspace() {
    print_h2("List of Subspace Orbitals Requested");
    for (size_t i = 0, size = subspace_str_.size(); i < size; ++i) {
        outfile->Printf(" %13s", subspace_str_[i].c_str());
        if ((i + 1) % 5 == 0 and i + 1 != size)
            outfile->Printf("\n  ");
    }
    outfile->Printf("\n");
    if (not subspace_pi_str_.empty()) {
        outfile->Printf("  NOTE: Subspace orbitals may be truncated based on requested planes!\n");
    }

    // parse subspace orbitals
    subspace_counter_ = 0;
    bool all_found = true;
    for (const std::string& s : subspace_str_) {
        all_found &= parse_subspace_entry(s);
    }

    // AOs that build the subspace orbitals
    std::unordered_set<int> minao_idx;
    for (const auto& tup : subspace_tuple_) {
        int i_min;
        std::tie(i_min, std::ignore, std::ignore) = tup;
        minao_idx.insert(i_min);
    }

    // print basis and mark those are selected into the subspace
    print_h2("AO Basis Set Selected By Subspace");
    outfile->Printf("\n    =======================================");
    outfile->Printf("\n      AO  Atom  Label     Type  Coefficient");
    outfile->Printf("\n    ---------------------------------------");
    for (const auto& tup : subspace_tuple_) {
        int i_min, i_sub;
        double c;
        std::tie(i_min, i_sub, c) = tup;

        const auto& ao_info = aoinfo_vec_[i_min];
        int A = ao_info.A() + 1;
        auto atom_label = atomic_labels[ao_info.Z()] + std::to_string(ao_info.element_count() + 1);
        auto ao_type = std::to_string(ao_info.n()) + lm_labels_sperical_[ao_info.l()][ao_info.m()];

        outfile->Printf("\n    %4d  %4d %6s %8s  %11.4E", i_min, A, atom_label.c_str(),
                        ao_type.c_str(), c);
    }
    outfile->Printf("\n    =======================================\n");

    //    outfile->Printf("\n  The AO basis set (The subspace contains %d AOs marked by *):\n",
    //                    minao_idx.size());
    //    outfile->Printf("    ==================================\n");
    //    outfile->Printf("       AO      Atom  Label  AO type\n");
    //    outfile->Printf("    ----------------------------------\n");
    //    std::vector<std::string> labels = aolabels("%1$4d%2$-2s%3$6d     %4$d%5$s");
    //    for (int i = 0, size = labels.size(); i < size; ++i) {
    //        outfile->Printf("    %5d %c  %s\n", i + 1, minao_idx.count(i) ? '*' : ' ',
    //                        labels[i].c_str());
    //    }
    //    outfile->Printf("    ==================================\n");

    // throw if input is wrong
    if (not all_found) {
        outfile->Printf("\n  Some subspace orbitals are not found. Please check the input.");
        outfile->Printf("\n  Orbital labels available:");
        for (const auto& am_labels : lm_labels_sperical_) {
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
        outfile->Printf("\n  Parsing entry: '%s'\n", s.c_str());
        for (std::ssub_match base_sub_match : match) {
            std::string base = base_sub_match.str();
            outfile->Printf("  --> '%s'\n", base.c_str());
        }
    }
    if (match.size() == 5) {
        // Find Z
        int Z = static_cast<int>(etoZ[match[1].str()]);

        // Find the range
        int minA = 0;
        int maxA = atom_to_aos_[Z].size();
        if (match[2].str().size() != 0) {
            minA = stoi(match[2].str()) - 1;
            if (match[3].str().size() == 0) {
                maxA = minA + 1;
            } else {
                maxA = stoi(match[3].str());
            }
        }

        if (debug_) {
            outfile->Printf("  Element %s -> %d\n", match[1].str().c_str(), Z);
            outfile->Printf("  Range %d -> %d\n", minA, maxA);
        }

        // Find the subset of AOs
        if (match[4].str().size() != 0) {
            // Include some of the AOs
            std::vector<std::string> vec_str = mysplit(match[4].str(), "/");
            for (std::string str : vec_str) {
                int n = atoi(&str[0]);
                str.erase(0, 1);
                if (labels_sperical_to_lm_.count(str) > 0) {
                    for (int A = minA; A < maxA; ++A) {
                        std::pair<int, int> atom_key{Z, A};
                        if (str == "P" and atom_to_plane_.find(atom_key) != atom_to_plane_.end()) {
                            for (int pos : atom_to_aos_[Z][A]) {
                                if ((aoinfo_vec_[pos].n() == n) and (aoinfo_vec_[pos].l() == 1)) {
                                    if (debug_)
                                        outfile->Printf("     + found AO @ %d, SUB @ %d\n", pos,
                                                        subspace_counter_);
                                    // directions: AO: pz, px, py; plane normal: x, y, z
                                    int m = (aoinfo_vec_[pos].m() + 2) % 3;
                                    double c = atom_to_plane_[atom_key]->get(m);
                                    subspace_tuple_.emplace_back(pos, subspace_counter_, c);
                                }
                            }
                            subspace_counter_ += 1;
                        } else {
                            for (std::pair<int, int> lm : labels_sperical_to_lm_[str]) {
                                int l = lm.first;
                                int m = lm.second;
                                if (debug_)
                                    outfile->Printf("     -> %s (n = %d,l = %d, m = %d)\n",
                                                    str.c_str(), n, l, m);
                                for (int pos : atom_to_aos_[Z][A]) {
                                    if ((aoinfo_vec_[pos].n() == n) and
                                        (aoinfo_vec_[pos].l() == l) and
                                        (aoinfo_vec_[pos].m() == m)) {
                                        if (debug_)
                                            outfile->Printf("     + found AO @ %d, SUB @ %d\n", pos,
                                                            subspace_counter_);
                                        subspace_tuple_.emplace_back(pos, subspace_counter_++, 1.0);
                                    }
                                }
                            }
                        }
                    }
                } else {
                    outfile->Printf("  AO label '%s' is not valid.\n", str.c_str());
                    found = false;
                }
            }
        } else {
            // Include all the AOs
            for (int A = minA; A < maxA; A++) {
                std::pair<int, int> atom_key{Z, A};
                bool in_plane = atom_to_plane_.find(atom_key) != atom_to_plane_.end();

                for (int pos : atom_to_aos_[Z][A]) {
                    if (aoinfo_vec_[pos].l() == 1 and in_plane) {
                        if (debug_)
                            outfile->Printf("     + found AO @ %d, SUB @ %d\n", pos,
                                            subspace_counter_);
                        // directions: AO: pz, px, py; plane normal: x, y, z
                        int m = (aoinfo_vec_[pos].m() + 2) % 3;
                        double c = atom_to_plane_[atom_key]->get(m);
                        subspace_tuple_.emplace_back(pos, subspace_counter_, c);
                        subspace_counter_ += aoinfo_vec_[pos].m() == 2 ? 1 : 0;
                    } else {
                        if (debug_)
                            outfile->Printf("     + found AO @ %d, SUB @ %d\n", pos,
                                            subspace_counter_);
                        subspace_tuple_.emplace_back(pos, subspace_counter_++, 1.0);
                    }
                }
            }
        }
    }

    return found;
}

void AOSubspace::parse_basis_set() {
    // Form a map that lists all functions on a given atom and with a given ang.
    // momentum
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

void AOSubspace::parse_pi_planes() {
    // print information to be parsed
    print_h2("List of Planes Requested");
    for (size_t i = 0, n_planes = subspace_pi_str_.size(); i < n_planes; ++i) {
        outfile->Printf("\n    Plane %2d", i + 1);
        for (size_t a = 0, n_entries = subspace_pi_str_[i].size(); a < n_entries; ++a) {
            if (a % 8 == 0) {
                outfile->Printf("\n    ");
            }
            outfile->Printf("%10s", subspace_pi_str_[i][a].c_str());
        }
    }

    // clear parsed plane
    atom_to_plane_.clear();

    // prepare a map from relative atomic indices to absolute atomic indices
    auto n_atoms = molecule_->natom();
    std::map<std::string, std::vector<int>> atom_to_abs_indices;
    for (int i = 0; i < n_atoms; ++i) {
        std::string atom_label = molecule_->label(i);
        to_upper_string(atom_label);
        atom_to_abs_indices[atom_label].push_back(i);
    }

    // find plane normal on each atom
    for (const std::vector<std::string>& atoms_labels : subspace_pi_str_) {
        std::vector<std::pair<int, int>> atoms;
        psi::SharedVector plane_normal;
        std::tie(atoms, plane_normal) = parse_pi_plane(atoms_labels, atom_to_abs_indices);
        for (const auto& atom : atoms) {
            if (atom_to_plane_.find(atom) != atom_to_plane_.end()) {
                atom_to_plane_[atom]->add(plane_normal);
            } else {
                atom_to_plane_[atom] = std::make_shared<psi::Vector>(*plane_normal);
            }
        }
    }

    // normalize vectors
    for (auto& key_value : atom_to_plane_) {
        const auto& atom = key_value.first;
        auto& plane = key_value.second;
        atom_to_plane_[atom]->scale(1.0 / plane->norm());
    }
}

std::tuple<std::vector<std::pair<int, int>>, psi::SharedVector>
AOSubspace::parse_pi_plane(const std::vector<std::string>& atoms_labels,
                           const std::map<std::string, std::vector<int>>& atom_to_abs_indices) {
    std::regex re(R"(([A-Za-z]{1,2})\s*(\d*)\s*-?\s*(\d*))");
    Element_to_Z etoZ;

    // grab plane atoms indices
    std::vector<int> atoms_abs_indices;
    std::vector<std::pair<int, int>> plane_atoms;
    for (const std::string& labels : atoms_labels) {
        std::smatch sm;
        std::regex_match(labels, sm, re);

        if (sm.size() != 4) {
            throw std::runtime_error("Invalid syntax for atoms forming the pi plane");
        }

        std::string atom_label = sm[1];
        int Z = static_cast<int>(etoZ[atom_label]);

        int start = 1;
        int end = static_cast<int>(atom_to_abs_indices.at(atom_label).size());
        if (sm[2].length()) {
            start = std::stoi(sm[2]);
            end = sm[3].length() ? std::stoi(sm[3]) : start;
        }

        for (int i = start - 1; i < end; ++i) {
            atoms_abs_indices.push_back(atom_to_abs_indices.at(atom_label)[i]);
            plane_atoms.push_back({Z, i});
        }
    }

    int n_atoms = atoms_abs_indices.size();
    if (n_atoms < 3) {
        throw std::runtime_error("Not enough atoms to define a plane");
    }

    // form the plane
    std::vector<double> centroid(3, 0.0);
    auto xyz0 = std::make_shared<psi::Matrix>(molecule_->name() + " sub", n_atoms, 3);
    for (int i = 0; i < n_atoms; ++i) {
        std::vector<double> xyz{molecule_->x(atoms_abs_indices[i]),
                                molecule_->y(atoms_abs_indices[i]),
                                molecule_->z(atoms_abs_indices[i])};
        for (int z = 0; z < 3; ++z) {
            centroid[z] += xyz[z];
            xyz0->set(i, z, xyz[z]);
        }
    }
    for (int z = 0; z < 3; ++z) {
        centroid[z] /= n_atoms;
    }

    // shift the plane
    for (int i = 0; i < n_atoms; ++i) {
        for (int z = 0; z < 3; ++z) {
            xyz0->set(i, z, xyz0->get(i, z) - centroid[z]);
        }
    }

    // find normal vector (smallest principal axis)
    SharedMatrix U, Vh;
    SharedVector S;
    std::tie(U, S, Vh) = xyz0->svd_temps();
    xyz0->svd(U, S, Vh);

    if (debug_) {
        outfile->Printf("\n  ==> XYZ Singular Value Results <==");
        S->print();
        Vh->print();
    }

    return {plane_atoms, Vh->get_row(0, 2)};
}
} // namespace forte
