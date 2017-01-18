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
#include <numeric>
#include <vector>
#include <regex>

#include "psi4/masses.h"
//#include "psi4/libmints/Z_to_element.h"
#include "psi4/libmints/element_to_Z.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/petitelist.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libmints/basisset_parser.h"

#include "boost/format.hpp"

#include "aosubspace.h"

using namespace psi;

std::vector<std::string> mysplit(const std::string& input, const std::string& regex);

std::vector<std::string> mysplit(const std::string& input, const std::string& regex) {
    // passing -1 as the submatch index parameter performs splitting
    std::regex re(regex);
    std::sregex_token_iterator
        first{input.begin(), input.end(), re, -1},
        last;
    return {first, last};
}

namespace psi{ namespace forte{

SharedMatrix create_aosubspace_projector(SharedWavefunction wfn, Options& options)
{
    SharedMatrix Ps;

    /* PORTTODO: re-enable this block
    // Run this code only if user specified a subspace
    if (options["SUBSPACE"].size() > 0){
        std::vector<std::string> subspace_str;
        for (int entry = 0; entry < (int)options["SUBSPACE"].size(); ++entry){
            std::string s = options["SUBSPACE"][entry].to_string();
            subspace_str.push_back(s);
        }

        // Create a basis set parser object and read the minimal basis
        std::shared_ptr<Molecule> molecule = wfn->molecule();
        std::shared_ptr<BasisSetParser> parser(new Gaussian94BasisSetParser());
        std::shared_ptr<BasisSet> min_basis = BasisSet::pyconstruct_orbital(molecule,"BASIS",options.get_str("MIN_BASIS"));

        // Create an AOSubspace object
        AOSubspace aosub(subspace_str,molecule,min_basis);

        // Compute the subspaces (right now this is required before any other call)
        aosub.find_subspace();

        // Show minimal basis using custom formatting
        outfile->Printf("\n  Minimal basis:\n");
        outfile->Printf("    ==================================\n");
        outfile->Printf("       AO    Atom    Label  AO type   \n");
        outfile->Printf("    ----------------------------------\n");
        {
            std::vector<std::string> aolabels = aosub.aolabels("%1$4d       %2$-2s %3$-4d  %4$d%5$s");

            int nbf = 0;
            for (const auto& s : aolabels){
                outfile->Printf("    %5d  %s\n",nbf + 1,s.c_str());
                nbf++;
            }
        }
        outfile->Printf("    ==================================\n");

        const std::vector<int>& subspace = aosub.subspace();

        Ps =  aosub.build_projector(subspace,molecule,min_basis,wfn->basisset());
    }
    */
    return Ps;
}

AOSubspace::AOSubspace(std::shared_ptr<Molecule> molecule,std::shared_ptr<BasisSet> basis)
{
    startup();
}

AOSubspace::AOSubspace(std::vector<std::string> subspace_str,std::shared_ptr<Molecule> molecule, std::shared_ptr<BasisSet> basis)
    : subspace_str_(subspace_str), molecule_(molecule), basis_(basis)
{
    startup();
}

void AOSubspace::find_subspace()
{
    parse_basis_set();
    parse_subspace();
}

void AOSubspace::startup()
{
//    outfile->Printf("  ---------------------------------------\n");
//    outfile->Printf("    Atomic Orbital Subspace\n");
//    outfile->Printf("    written by Francesco A. Evangelista\n");
//    outfile->Printf("  ---------------------------------------\n");

    lm_labels_cartesian_ = {{"S"},
                            {"PX","PY","PZ"},
                            {"DX2","DXY","DXZ","DY2","DYZ","DZ2"},
                            {"FX3","FX2Y","FX2Z","FXY2","FXYZ","FXZ2","FY3","FY2Z","FYZ2","FZ3"}};

    l_labels_ = {"S","P","D","F","G","H","I","K","L","M"};

    lm_labels_sperical_ = {{"S"},
                           {"PZ","PX","PY"},
                           {"DZ2","DXZ","DYZ","DX2Y2","DXY"},
                           {"FZ3","FXZ2","FYZ2","FZX2-ZY2","FXYZ","FX3-3XY2","F3X2Y-Y3"},
                           {"G1","G2","G3","G4","G5","G6","G7","G8","G9"},
                           {"H1","H2","H3","H4","H5","H6","H7","H8","H9","H10","H11"}};

    for (int l = 0; l < (int)lm_labels_sperical_.size(); ++l){
        for (int m = 0; m < (int)lm_labels_sperical_[l].size(); ++m){
            labels_sperical_to_lm_[lm_labels_sperical_[l][m]] = {std::make_pair(l,m)};
        }
    }
    for (int l = 0; l < (int)l_labels_.size(); ++l){
        std::vector<std::pair<int,int>> lm_vec;
        for (int m = 0; m < 2 * l + 1; ++m){
            lm_vec.push_back(std::make_pair(l,m));
        }
        labels_sperical_to_lm_[l_labels_[l]] = lm_vec;
    }
}

SharedMatrix AOSubspace::build_projector(const std::vector<int>& subspace,
                                         std::shared_ptr<Molecule> molecule,
                                         std::shared_ptr<BasisSet> min_basis,
                                         std::shared_ptr<BasisSet> large_basis)
{
    bool debug = false;

    std::shared_ptr<IntegralFactory> integral_mm(
                new IntegralFactory(min_basis, min_basis, min_basis, min_basis));
    std::shared_ptr<IntegralFactory> integral_ml(
                new IntegralFactory(min_basis, large_basis, large_basis, large_basis));
    std::shared_ptr<IntegralFactory> integral_ll(
                new IntegralFactory(large_basis, large_basis, large_basis, large_basis));


    int nbf_s = static_cast<int>(subspace.size());
    int nbf_m = min_basis->nbf();
    int nbf_l = large_basis->nbf();

    // Check the point group of the molecule. If it is not set, set it
    if (!molecule->point_group()) {
        molecule->set_point_group(molecule->find_point_group());
    }

    // Compute the overlap integral in the minimal basis
    std::shared_ptr<OneBodyAOInt> sOBI_mm(integral_mm->ao_overlap());
    SharedMatrix S_mm = SharedMatrix(new Matrix("S_mm",nbf_m,nbf_m));
    sOBI_mm->compute(S_mm);

#if _DEBUG_AOSUBSPACE_
    S_mm->print();
#endif

    // Extract the subspace block
    SharedMatrix S_ss = SharedMatrix(new Matrix("S_ss",nbf_s,nbf_s));
    for (int mu = 0; mu < nbf_s; mu++){
        for (int nu = 0; nu < nbf_s; nu++){
            S_ss->set(mu,nu,S_mm->get(subspace[mu],subspace[nu]));
        }
    }
#if _DEBUG_AOSUBSPACE_
    S_ss->print();
#endif

    // Orthogonalize the subspace
    SharedMatrix X_ss = SharedMatrix(new Matrix("X",nbf_s,nbf_s));
    X_ss->copy(S_ss);
    X_ss->power(-0.5);

#if _DEBUG_AOSUBSPACE_
    X_ss->print();
#endif

    S_ss->transform(X_ss);

#if _DEBUG_AOSUBSPACE_
    S_ss->print();
#endif

    SharedMatrix X_mm = SharedMatrix(new Matrix("X_mm",nbf_m,nbf_m));
    for (int mu = 0; mu < nbf_s; mu++){
        for (int nu = 0; nu < nbf_s; nu++){
            X_mm->set(subspace[mu],subspace[nu],X_ss->get(mu,nu));
        }
    }
#if _DEBUG_AOSUBSPACE_
    X_mm->print();
#endif

    // Compute the overlap integral in the minimal/large basis
    std::shared_ptr<OneBodyAOInt> sOBI_ml(integral_ml->ao_overlap());
    SharedMatrix S_ml = SharedMatrix(new Matrix("S_ml",nbf_m,nbf_l));
    sOBI_ml->compute(S_ml);
#if _DEBUG_AOSUBSPACE_
    S_ml->print();
#endif

    SharedMatrix XS_ml = SharedMatrix(new Matrix("XS_ml",nbf_m,nbf_l));
    SharedMatrix SXXS_ll = SharedMatrix(new Matrix("SXXS_ll",nbf_l,nbf_l));
    XS_ml->gemm(false,false,1.0,X_mm,S_ml,0.0);
    SXXS_ll->gemm(true,false,1.0,XS_ml,XS_ml,0.0);
#if _DEBUG_AOSUBSPACE_
    XS_ml->print();
    SXXS_ll->print();
#endif

    std::shared_ptr<PetiteList> plist(new PetiteList(large_basis, integral_ll));
    SharedMatrix AO2SO_ = plist->aotoso();
    Dimension large_basis_so_dim = plist->SO_basisdim();
    SharedMatrix SXXS_ll_so(new Matrix("SXXS_ll_so",large_basis_so_dim,large_basis_so_dim));
    SXXS_ll_so->apply_symmetry(SXXS_ll,AO2SO_);
#if _DEBUG_AOSUBSPACE_
    SXXS_ll_so->print();
#endif

    return SXXS_ll_so;
}


const std::vector<int>& AOSubspace::subspace()
{
    return subspace_;
}

std::vector<std::string> AOSubspace::aolabels(std::string str_format) const
{
    std::vector<std::string> aolbl;
    for (const AOInfo& aoinfo : aoinfo_vec_){
        std::string s = boost::str( boost::format(str_format)
                                    % (aoinfo.A() + 1)
                                    % atomic_labels[aoinfo.Z()]
                                    % (aoinfo.element_count() + 1)
                                    % aoinfo.n()
                                    % lm_labels_sperical_[aoinfo.l()][aoinfo.m()]);
        aolbl.push_back(s);
    }
    return aolbl;
}

const std::vector<AOInfo>& AOSubspace::aoinfo() const
{
    return aoinfo_vec_;
}

void AOSubspace::parse_subspace()
{
    outfile->Printf("\n\n  List of subspaces:");
    for (const std::string& s : subspace_str_){
        outfile->Printf(" %s",s.c_str());
    }
    outfile->Printf("\n");

    for (const std::string& s : subspace_str_){
        parse_subspace_entry(s);
    }

    outfile->Printf("\n  Subspace contains AOs:\n");
    for (size_t i = 0; i < subspace_.size(); ++i){
        outfile->Printf("  %6d",subspace_[i] + 1);
        if ((i + 1) % 8 == 0)
            outfile->Printf("\n");
    }
    outfile->Printf("\n");
}

void AOSubspace::parse_subspace_entry(const std::string& s)
{
    // The regex to parse the entries
    std::regex re("([a-zA-Z]{1,2})([1-9]+)?-?([1-9]+)?\\(?((?:\\/?[1-9]{1}[SPDF]{1}[a-zA-Z]*)*)\\)?");
    std::smatch match;

    Element_to_Z etoZ;

    std::regex_match(s,match,re);
    if (debug_){
        outfile->Printf("\n  Parsing entry: '%s'\n",s.c_str());
        for (std::ssub_match base_sub_match : match){
            std::string base = base_sub_match.str();
            outfile->Printf("  --> '%s'\n",base.c_str());
        }
    }
    if (match.size() == 5){
        // Find Z
        int Z = static_cast<int>(etoZ[match[1].str()]);

        // Find the range
        int minA = 0;
        int maxA = atom_to_aos_[Z].size();
        if (match[2].str().size() != 0){
            minA = stoi(match[2].str()) - 1;
            if (match[3].str().size() == 0){
                maxA = minA + 1;
            }else{
                maxA = stoi(match[3].str());
            }
        }

        if (debug_){
            outfile->Printf("  Element %s -> %d\n",match[1].str().c_str(),Z);
            outfile->Printf("  Range %d -> %d\n",minA,maxA);
        }

        // Find the subset of AOs
        if (match[4].str().size() != 0){
            // Include some of the AOs
            std::vector<std::string> vec_str = mysplit(match[4].str(),"/");
            for (std::string str : vec_str){
                int n = atoi(&str[0]);
                str.erase(0,1);
                if (labels_sperical_to_lm_.count(str) > 0){
                    for (std::pair<int,int> lm : labels_sperical_to_lm_[str]){
                        int l = lm.first;
                        int m = lm.second;
                        if (debug_) outfile->Printf("     -> %s (n = %d,l = %d, m = %d)\n",str.c_str(),n,l,m);
                        for (int A = minA; A < maxA; A++){
                            for (int pos : atom_to_aos_[Z][A]){
                                if ((aoinfo_vec_[pos].n() == n) and (aoinfo_vec_[pos].l() == l) and (aoinfo_vec_[pos].m() == m)){
                                    if (debug_) outfile->Printf("     + found at position %d\n",pos);
                                    subspace_.push_back(pos);
                                }
                            }
                        }

                    }
                }else{
                    outfile->Printf("  AO label '%s' is not valid.\n",str.c_str());
                }
            }
        }else{
            // Include all the AOs
            for (int A = minA; A < maxA; A++){
                for (int pos : atom_to_aos_[Z][A]){
                    if (debug_) outfile->Printf("     + found at position %d\n",pos);
                    subspace_.push_back(pos);
                }
            }
        }
    }
}

void AOSubspace::parse_basis_set()
{
    // Form a map that lists all functions on a given atom and with a given ang. momentum
    std::map<std::pair<int,int>,std::vector<int>> atom_am_to_f;
    bool pure_am = basis_->has_puream();

    if (debug_){
        outfile->Printf("\n  Parsing basis set\n");
        outfile->Printf("  Pure Angular Momentum: %s\n",pure_am ? "True" : "False");
    }

    int count = 0;

    std::vector<int> element_count(130);

    for (int A = 0; A < molecule_->natom(); A++) {
        int Z = static_cast<int>(round(molecule_->Z(A)));

        int n_shell = basis_->nshell_on_center(A);

        std::vector<int> n_count(10,1);
        std::iota(n_count.begin(),n_count.end(),1);

        std::vector<int> ao_list;

        if (debug_) outfile->Printf("\n  Atom %d (Z = %d) has %d shells\n",A,Z,n_shell);

        for (int Q = 0; Q < n_shell; Q++){
            const GaussianShell& shell = basis_->shell(A,Q);
            int nfunction = shell.nfunction();
            int l = shell.am();
            if (debug_) outfile->Printf("    Shell %d: L = %d, N = %d (%d -> %d)\n",Q,l,nfunction,count,count + nfunction);
            for (int m = 0; m < nfunction; ++m){
                AOInfo ao(A,Z,element_count[Z],n_count[l],l,m);
                aoinfo_vec_.push_back(ao);
                ao_list.push_back(count);
                count += 1;
            }
            n_count[l] += 1;  // increase the angular momentum count
        }

        atom_to_aos_[Z].push_back(ao_list);

        element_count[Z] += 1; // increase the element count
    }
}

}}
