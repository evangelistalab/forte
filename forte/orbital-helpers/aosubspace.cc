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

namespace py = pybind11;

#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/element_to_Z.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/petitelist.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/masses.h"

#include "boost/format.hpp"

#include "helpers/string_algorithms.h"
#include "base_classes/forte_options.h"

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
        auto irrep_labels = molecule->irrep_labels();

        // Create an AOSubspace object
        AOSubspace aosub(subspace_str, molecule, min_basis);

        // Compute the subspaces (right now this is required before any other call)
        aosub.find_subspace();

        const std::vector<int>& subspace = aosub.subspace();

        // build the projector
        Ps = aosub.build_projector(subspace, molecule, min_basis, wfn->basisset());

        // print the overlap of the projector
        psi::SharedMatrix CPsC = Ps->clone();
        CPsC->transform(wfn->Ca());
        double print_threshold = 1.0e-3;
        outfile->Printf("\n  Orbital overlap with AO subspace (> %.2e):\n", print_threshold);
        outfile->Printf("    =======================\n");
        outfile->Printf("    Irrep   MO  <phi|P|phi>\n");
        outfile->Printf("    -----------------------\n");
        for (int h = 0; h < CPsC->nirrep(); h++) {
            for (int i = 0; i < CPsC->rowspi(h); i++) {
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
    : molecule_(molecule), basis_(basis) {
    startup();
}

AOSubspace::AOSubspace(std::vector<std::string> subspace_str,
                       std::shared_ptr<psi::Molecule> molecule,
                       std::shared_ptr<psi::BasisSet> basis)
    : subspace_str_(subspace_str), molecule_(molecule), basis_(basis) {
    startup();
}

void AOSubspace::find_subspace() {
    parse_basis_set();
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
    outfile->Printf("\n\n  List of subspace orbitals requested:\n  ");
    for (int i = 0, size = subspace_str_.size(); i < size; ++i) {
        outfile->Printf(" %13s", subspace_str_[i].c_str());
        if ((i + 1) % 5 == 0 and i + 1 != size)
            outfile->Printf("\n  ");
    }
    outfile->Printf("\n");

    // parse subspace orbitals
    bool all_found = true;
    for (const std::string& s : subspace_str_) {
        all_found &= parse_subspace_entry(s);
    }

    // print basis and mark those are selected into the subspace
    std::unordered_set<int> subspace_idx(subspace_.begin(), subspace_.end());

    outfile->Printf("\n  The AO basis set (The subspace contains %d AOs marked by *):\n",
                    subspace_.size());
    outfile->Printf("    ==================================\n");
    outfile->Printf("       AO      Atom  Label  AO type\n");
    outfile->Printf("    ----------------------------------\n");
    std::vector<std::string> labels = aolabels("%1$4d%2$-2s%3$6d     %4$d%5$s");
    for (int i = 0, size = labels.size(); i < size; ++i) {
        outfile->Printf("    %5d %c  %s\n", i + 1, subspace_idx.count(i) ? '*' : ' ',
                        labels[i].c_str());
    }
    outfile->Printf("    ==================================\n");

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
                    for (std::pair<int, int> lm : labels_sperical_to_lm_[str]) {
                        int l = lm.first;
                        int m = lm.second;
                        if (debug_)
                            outfile->Printf("     -> %s (n = %d,l = %d, m = %d)\n", str.c_str(), n,
                                            l, m);
                        for (int A = minA; A < maxA; A++) {
                            for (int pos : atom_to_aos_[Z][A]) {
                                if ((aoinfo_vec_[pos].n() == n) and (aoinfo_vec_[pos].l() == l) and
                                    (aoinfo_vec_[pos].m() == m)) {
                                    if (debug_)
                                        outfile->Printf("     + found at position %d\n", pos);
                                    subspace_.push_back(pos);
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
                for (int pos : atom_to_aos_[Z][A]) {
                    if (debug_)
                        outfile->Printf("     + found at position %d\n", pos);
                    subspace_.push_back(pos);
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
    bool pure_am = basis_->has_puream();

    if (debug_) {
        outfile->Printf("\n  Parsing basis set\n");
        outfile->Printf("  Pure Angular Momentum: %s\n", pure_am ? "True" : "False");
    }

    int count = 0;

    std::vector<int> element_count(130);

    for (int A = 0; A < molecule_->natom(); A++) {
        int Z = static_cast<int>(round(molecule_->Z(A)));

        int n_shell = basis_->nshell_on_center(A);

        std::vector<int> n_count(10, 1);
        std::iota(n_count.begin(), n_count.end(), 1);

        std::vector<int> ao_list;

        if (debug_)
            outfile->Printf("\n  Atom %d (Z = %d) has %d shells\n", A, Z, n_shell);

        for (int Q = 0; Q < n_shell; Q++) {
            const GaussianShell& shell = basis_->shell(A, Q);
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
} // namespace forte
