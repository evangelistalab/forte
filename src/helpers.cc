/*
 *@BEGIN LICENSE
 *
 * Libadaptive: an ab initio quantum chemistry software package
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 *@END LICENSE
 */

#include <numeric>

#include "psi4-dec.h"

#include <libmints/molecule.h>
#include <libmints/pointgrp.h>
#include "libmints/wavefunction.h"

#include "helpers.h"

namespace psi{ namespace forte{


MOSpaceInfo::MOSpaceInfo()
{
    // Now we want the reference (SCF) wavefunction
    boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();
    nirrep_ = wfn->nirrep();
    nmopi_ = wfn->nmopi();

    // Add the elementary spaces to the list of composite spaces
    for (const std::string& es : elementary_spaces_){
        composite_spaces_[es] = {es};
    }
}

MOSpaceInfo::~MOSpaceInfo()
{
}

size_t MOSpaceInfo::size(const std::string& space)
{
    size_t s = 0;
    if (composite_spaces_.count(space) == 0){
        std::string msg = "\n  MOSpaceInfo::size - composite space " + space + " is not defined.";
        throw PSIEXCEPTION(msg.c_str());
    }else{
        for (const auto& el_space : composite_spaces_[space]){
            if (mo_spaces_.count(el_space))
                s += mo_spaces_[el_space].first.sum();
        }
    }
    return s;
}

Dimension MOSpaceInfo::get_dimension(const std::string& space)
{
    Dimension result(nirrep_);
    if (composite_spaces_.count(space) == 0){
        std::string msg = "\n  MOSpaceInfo::size - composite space " + space + " is not defined.";
        throw PSIEXCEPTION(msg.c_str());
    }else{
        for (const auto& el_space : composite_spaces_[space]){
            if (mo_spaces_.count(el_space))
                result += mo_spaces_[el_space].first;
        }
    }
    return result;
}

std::vector<int> MOSpaceInfo::symmetry(const std::string& space)
{
    Dimension dims = get_dimension(space);
    std::vector<int> result;
    for (int h = 0; h < dims.n(); ++h){
        for (int i = 0; i < dims[h]; ++i){
            result.push_back(h);
        }
    }
    return result;
}

std::vector<size_t> MOSpaceInfo::get_absolute_mo(const std::string& space)
{
    std::vector<size_t> result;
    if (composite_spaces_.count(space) == 0){
        std::string msg = "\n  MOSpaceInfo::size - composite space " + space + " is not defined.";
        throw PSIEXCEPTION(msg.c_str());
    }else{
        for (const auto& el_space : composite_spaces_[space]){
            if (mo_spaces_.count(el_space)){
                auto& vec_mo_info = mo_spaces_[el_space].second;
                for (auto& mo_info : vec_mo_info){
                    result.push_back(std::get<0>(mo_info)); // <- grab the absolute index
                }
            }
        }
    }
    return result;
}

std::vector<size_t> MOSpaceInfo::get_corr_abs_mo(const std::string& space)
{
    std::vector<size_t> result;
    if (composite_spaces_.count(space) == 0){
        std::string msg = "\n  MOSpaceInfo::size - composite space " + space + " is not defined.";
        throw PSIEXCEPTION(msg.c_str());
    }else{
        for (const auto& el_space : composite_spaces_[space]){
            if (mo_spaces_.count(el_space)){
                auto& vec_mo_info = mo_spaces_[el_space].second;
                for (auto& mo_info : vec_mo_info){
                    result.push_back(mo_to_cmo_[std::get<0>(mo_info)]); // <- grab the absolute index and convert to correlated MOs
                }
            }
        }
    }
    return result;
}

std::vector<std::pair<size_t,size_t>> MOSpaceInfo::get_relative_mo(const std::string& space)
{
    std::vector<std::pair<size_t,size_t>> result;
    if (composite_spaces_.count(space) == 0){
        std::string msg = "\n  MOSpaceInfo::size - composite space " + space + " is not defined.";
        throw PSIEXCEPTION(msg.c_str());
    }else{
        for (const auto& el_space : composite_spaces_[space]){
            if (mo_spaces_.count(el_space)){
                auto& vec_mo_info = mo_spaces_[el_space].second;
                for (auto& mo_info : vec_mo_info){
                    result.push_back(std::make_pair(std::get<1>(mo_info),std::get<2>(mo_info))); // <- grab the irrep and relative index
                }
            }
        }
    }
    return result;
}

void MOSpaceInfo::read_options(Options& options)
{
    outfile->Printf("\n\n  ==> MO Space Information <==\n");

    // Read the elementary spaces
    for (std::string& space : elementary_spaces_){
        std::pair<SpaceInfo,bool> result = read_mo_space(space,options);
        if (result.second){
            mo_spaces_[space] = result.first;
        }
    }

    // Handle frozen core


    // Count the assigned orbitals
    Dimension unassigned = nmopi_;
    for (auto& str_si : mo_spaces_){
        unassigned -= str_si.second.first;
    }

    for (size_t h = 0; h < nirrep_; ++h){
        if (unassigned[h] < 0){
            outfile->Printf("\n  There is an error in the definition of the orbital spaces.  Total unassigned MOs for irrep %d is %d.",
                            h,unassigned[h]);
        }
    }

    // Adjust size of undefined spaces
    for (std::string space : elementary_spaces_priority_){
        // Assign MOs to the undefined space with the highest priority
        if (not mo_spaces_.count(space)){
            std::vector<MOInfo> vec_mo_info;
            mo_spaces_[space] = std::make_pair(unassigned,vec_mo_info);
            for (size_t h = 0; h < nirrep_; ++h){ unassigned[h] = 0; }
        }
    }
    if (unassigned.sum() != 0){
        outfile->Printf("\n  There is an error in the definition of the orbital spaces.  There are %d unassigned MOs.",unassigned.sum());
        exit(1);
    }


    // Compute orbital mappings
    for (size_t h = 0, p_abs = 0; h < nirrep_; ++h){
        size_t p_rel = 0;
        for (std::string space : elementary_spaces_){
            size_t n = mo_spaces_[space].first[h];
            for (size_t q = 0; q < n; ++q){
                mo_spaces_[space].second.push_back(std::make_tuple(p_abs,h,p_rel));
                p_abs += 1;
                p_rel += 1;
            }
        }
    }


    // Compute the MO to correlated MO mapping
    std::vector<size_t> vec(nmopi_.sum());
    std::iota(vec.begin(),vec.end(),0);

    // Remove the frozen core/virtuals
    for (MOInfo& mo_info : mo_spaces_["FROZEN_DOCC"].second){
        outfile->Printf("\n Removing orbital %d",std::get<0>(mo_info));
        vec.erase(std::remove(vec.begin(), vec.end(),std::get<0>(mo_info)), vec.end());
    }
    for (MOInfo& mo_info : mo_spaces_["FROZEN_UOCC"].second){
        vec.erase(std::remove(vec.begin(), vec.end(),std::get<0>(mo_info)), vec.end());
    }

    mo_to_cmo_.assign(nmopi_.sum(),1000000000);
    for (size_t n = 0; n < vec.size(); ++n){
        mo_to_cmo_[vec[n]] = n;
    }

    // Define composite spaces

    // Print the space information
    size_t label_size = 1;
    for (std::string space : elementary_spaces_){
        label_size = std::max(space.size(),label_size);
    }

    int banner_width = label_size + 4 + 6 * (nirrep_ + 1);
    CharacterTable ct = Process::environment.molecule()->point_group()->char_table();
    outfile->Printf("\n  %s",std::string(banner_width,'-').c_str());
    outfile->Printf("\n    %s",std::string(label_size,' ').c_str());
    for (size_t h = 0; h < nirrep_; ++h) outfile->Printf(" %5s",ct.gamma(h).symbol());
    outfile->Printf("   Sum");
    outfile->Printf("\n  %s",std::string(banner_width,'-').c_str());

    for (std::string space : elementary_spaces_){
        Dimension& dim = mo_spaces_[space].first;
        outfile->Printf("\n    %-*s",label_size,space.c_str());
        for (size_t h = 0; h < nirrep_; ++h){
            outfile->Printf("%6d",dim[h]);
        }
        outfile->Printf("%6d",dim.sum());
    }
    outfile->Printf("\n    %-*s",label_size,"Total");
    for (size_t h = 0; h < nirrep_; ++h){
        outfile->Printf("%6d",nmopi_[h]);
    }
    outfile->Printf("%6d",nmopi_.sum());
    outfile->Printf("\n  %s",std::string(banner_width,'-').c_str());
}

std::pair<SpaceInfo,bool> MOSpaceInfo::read_mo_space(const std::string& space,Options& options)
{
    bool read = false;
    Dimension space_dim(nirrep_);
    std::vector<MOInfo> vec_mo_info;
    if ((options[space].has_changed()) && (options[space].size() == nirrep_)){
        for (size_t h = 0; h < nirrep_; ++h){
            space_dim[h] = options[space][h].to_integer();
        }
        read = true;
        outfile->Printf("\n  Read options for space %s",space.c_str());
    }else{
//        outfile->Printf("\n  The size of space \"%s\" (%d) does not match the number of irreducible representations (%zu).",
//                        space.c_str(),options[space].size(),nirrep_);
    }
    SpaceInfo space_info(space_dim,vec_mo_info);
    return std::make_pair(space_info,read);
}


void print_method_banner(const std::vector<std::string>& text, const std::string &separator)
{
    size_t max_width = 80;

    size_t width = 0;
    for (auto& line : text){
        width = std::max(width,line.size());
    }

    std::string tab((max_width - width - 4)/2,' ');
    std::string header(width + 4,char(separator[0]));

    *outfile << "\n\n" << tab << header << std::endl;
    for (auto& line : text){
        size_t padding = 2 + (width - line.size()) / 2;
        *outfile << tab << std::string(padding,' ') << line << std::endl;
    }
    *outfile << tab << header << std::endl;

    outfile->Flush();
}


void print_h2(const std::string& text, const std::string& left_separator, const std::string& right_separator)
{
    outfile->Printf("\n\n  %s %s %s\n",left_separator.c_str(),
                    text.c_str(),right_separator.c_str());
}

std::string to_string(const std::vector<std::string> &vec_str, const std::string &sep)
{
    if (vec_str.size() == 0)
        return std::string();

    std::string ss;

    std::for_each(vec_str.begin(), vec_str.end() - 1,[&](const std::string& s){ss += s + sep;});
    ss += vec_str.back();

    return ss;
}

Matrix tensor_to_matrix(ambit::Tensor t,Dimension dims)
{
    // Copy the tensor to a plain matrix
    size_t size = dims.sum();
    Matrix M("M",size,size);
    t.iterate([&](const std::vector<size_t>& i,double& value){
        M.set(i[0],i[1],value);
    });

    Matrix M_sym("M",dims,dims);
    size_t offset = 0;
    for (size_t h = 0; h < static_cast<size_t>(dims.n()); ++h){
        for (size_t p = 0; p < static_cast<size_t>(dims[h]); ++p){
            for (size_t q = 0; q < static_cast<size_t>(dims[h]); ++q){
                double value = M.get(p + offset,q + offset);
                M_sym.set(h,p,q,value);
            }
        }
        offset += dims[h];
    }
    return M_sym;
}

}} // End Namespaces
