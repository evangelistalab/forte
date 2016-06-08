#include <map>
#include <algorithm>
#include <libmints/pointgrp.h>

#include "active_dsrgpt2.h"

namespace psi{ namespace forte{

ACTIVE_DSRGPT2::ACTIVE_DSRGPT2(SharedWavefunction ref_wfn, Options &options,
                               std::shared_ptr<ForteIntegrals> ints,
                               std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(options), ints_(ints), mo_space_info_(mo_space_info), total_nroots_(0)
{
    // Copy the wavefunction information
    shallow_copy(ref_wfn);
    reference_wavefunction_ = ref_wfn;

    print_method_banner({"ACTIVE-DSRGPT2", "Chenyang Li"});
    outfile->Printf("\n    The orbitals are fixed throughout the process.");
    outfile->Printf("\n    If different orbitals (or reference) are desired for different roots,");
    outfile->Printf("\n    you need to run those separately using the regular DSRG-MRPT2 (or DF/CD) code.\n");
    startup();
}

ACTIVE_DSRGPT2::~ACTIVE_DSRGPT2(){}

void ACTIVE_DSRGPT2::startup(){
    if(options_["NROOTPI"].size() == 0){
        throw PSIEXCEPTION("Please specify NROOTPI for ACTIVE-DSRGPT2 jobs.");
    }else{
        int nirrep = this->nirrep();
        ref_energies_ = vector<vector<double>> (nirrep,vector<double>());
        pt2_energies_ = vector<vector<double>> (nirrep,vector<double>());
        dominant_dets_ = vector<vector<STLBitsetDeterminant>> (nirrep,vector<STLBitsetDeterminant>());
        CharacterTable ct = Process::environment.molecule()->point_group()->char_table();
        if(options_.get_str("ACTIVE_SPACE_TYPE") == "CIS"){
            t1_percentage_ = vector<vector<pair<int,double>>> (nirrep,vector<pair<int,double>>());
        }
        for(int h = 0; h < nirrep; ++h){
            nrootpi_.push_back(options_["NROOTPI"][h].to_integer());
            irrep_symbol_.push_back(std::string(ct.gamma(h).symbol()));
        }

        // print request
        int total_width = 4 + 6 + 6 * nirrep;
        outfile->Printf("\n      %s",std::string(6,' ').c_str());
        for(int h = 0; h < nirrep; ++h) outfile->Printf(" %5s",irrep_symbol_[h].c_str());
        outfile->Printf("\n    %s",std::string(total_width,'-').c_str());
        outfile->Printf("\n      NROOTS");
        for(int h = 0; h < nirrep; ++h){
            outfile->Printf("%6d",nrootpi_[h]);
            total_nroots_ += nrootpi_[h];
        }
        outfile->Printf("\n    %s",std::string(total_width,'-').c_str());
    }
}

double ACTIVE_DSRGPT2::compute_energy(){
    if(total_nroots_ == 0){
        outfile->Printf("\n  NROOTPI is zero. Did nothing.");
    }else{
        FCI_MO fci_mo(reference_wavefunction_,options_,ints_,mo_space_info_);
        int nirrep = nrootpi_.size();
        for(int h = 0; h < nirrep; ++h){
            if(nrootpi_[h] == 0) continue;
            else{
                fci_mo.set_root_sym(h);
                int cisd_nroot = nrootpi_[h] < 5 ? 2 * nrootpi_[h] : nrootpi_[h] + 5;
                Matrix overlap ("overlap", cisd_nroot, nrootpi_[h]); // CISD by CIS
                for(int i = 0; i < nrootpi_[h]; ++i){
                    // CI routine
                    outfile->Printf("\n\n  %s", std::string(35,'=').c_str());
                    outfile->Printf("\n    Current Job: %3s state, root %2d", irrep_symbol_[h].c_str(), i);
                    outfile->Printf("\n  %s\n", std::string(35,'=').c_str());
                    fci_mo.set_nroots(i+1);
                    fci_mo.set_root(i);
                    ref_energies_[h].push_back(fci_mo.compute_energy());
                    Reference reference = fci_mo.reference();
                    dominant_dets_[h].push_back(fci_mo.dominant_det());

                    // PT2 routine
                    double pt2 = 0.0;
                    if(options_.get_str("INT_TYPE") == "CONVENTIONAL"){
                        auto dsrg = std::make_shared<DSRG_MRPT2>(reference,reference_wavefunction_,options_,ints_,mo_space_info_);
                        dsrg->ignore_semicanonical(true);
                        pt2 = dsrg->compute_energy();
                    }else{
                        auto dsrg = std::make_shared<THREE_DSRG_MRPT2>(reference,reference_wavefunction_,options_,ints_,mo_space_info_);
                        pt2 = dsrg->compute_energy();
                    }
                    pt2_energies_[h].push_back(pt2);

                    // compute singles percentages
                    if(options_.get_str("ACTIVE_SPACE_TYPE") == "CIS"){
                        std::vector<STLBitsetDeterminant> vecdet_cis = fci_mo.p_space();
                        SharedVector eigen_cis = fci_mo.eigen().back().first;

                        // set quiet mode for CISD, use CIS semi-canonicalized orbitals
                        fci_mo.set_active_space_type("CISD");
                        fci_mo.set_semicanonical(false);
                        fci_mo.set_quite_mode(true);

                        // loop over CISD states
                        print_h2("Compute Singles Percentage");
                        outfile->Printf("    Will compute %d roots for CISD in Irrep %s.\n", cisd_nroot, irrep_symbol_[h].c_str());
                        for(int j = 0; j < cisd_nroot; ++j){
//                            outfile->Printf("\n\n  %s", std::string(42,'=').c_str());
//                            outfile->Printf("\n    Current Job (CISD): %3s state, root %2d", irrep_symbol_[h].c_str(), j);
//                            outfile->Printf("\n  %s\n", std::string(42,'=').c_str());
                            fci_mo.set_nroots(j+1);
                            fci_mo.set_root(j);
                            fci_mo.compute_energy();

                            std::vector<STLBitsetDeterminant> vecdet_cisd = fci_mo.p_space();
                            SharedVector eigen_cisd = fci_mo.eigen().back().first;

                            for(int s = 0; s < vecdet_cis.size(); ++s){
                                for(int sd = 0; sd < vecdet_cisd.size(); ++sd){
                                    if(vecdet_cis[s] == vecdet_cisd[sd]){
                                        double value = eigen_cisd->get(sd) * eigen_cis->get(s);
                                        overlap.add(j,i,value);
                                    }
                                }
                            }
                        }

                        // set back to CIS computation
                        fci_mo.set_active_space_type("CIS");
                        fci_mo.set_semicanonical(true);
                        fci_mo.set_quite_mode(false);
                    }
                }

                if(options_.get_str("ACTIVE_SPACE_TYPE") == "CIS"){
                    outfile->Printf("\n  ==> Overlap Metric of <CISD|CIS> for Irrep %s <==\n\n", irrep_symbol_[h].c_str());
                    overlap.print();
                    for(int i = 0; i < overlap.ncol(); ++i){
                        double max = std::fabs(overlap.get(i,i));
                        int root = i;
                        for(int j = 0; j < overlap.nrow(); ++j){
                            double value = std::fabs(overlap.get(j,i));
                            if(max < value){
                                max = value;
                                root = j;
                            }
                        }
                        t1_percentage_[h].push_back(make_pair(root,100.0*max));
                    }
                }
            }
        }
        print_summary();
    }
    return 0.0;
}

void ACTIVE_DSRGPT2::print_summary(){
    print_h2("ACTIVE-DSRGPT2 Summary");

    std::string ref_type = options_.get_str("ACTIVE_SPACE_TYPE");
    if(ref_type == "COMPLETE") ref_type = std::string("CAS");

    int nirrep = nrootpi_.size();
    if(ref_type != "CIS"){
        int total_width = 4 + 6 + 18 + 18 + 3 * 2;
        outfile->Printf("\n    %4s  %6s  %11s%7s  %11s", "Sym.", "ROOT", ref_type.c_str(), std::string(7,' ').c_str(), "PT2");
        outfile->Printf("\n    %s", std::string(total_width,'-').c_str());
        for(int h = 0; h < nirrep; ++h){
            if(nrootpi_[h] != 0){
                for(int i = nrootpi_[h]; i > 0; --i){
                    std::string sym(4,' ');
                    if(i == 1) sym = irrep_symbol_[h];
                    outfile->Printf("\n    %4s  %6d  %18.10f  %18.10f", sym.c_str(), i-1, ref_energies_[h][i-1], pt2_energies_[h][i-1]);
                }
                outfile->Printf("\n    %s", std::string(total_width,'-').c_str());
            }
        }
    }else{
        int total_width = 4 + 6 + 18 + 18 + 11 + 4 * 2;
        outfile->Printf("\n    %4s  %6s  %11s%7s  %11s%7s  %11s", "Sym.", "ROOT", ref_type.c_str(), std::string(7,' ').c_str(),
                        "PT2", std::string(7,' ').c_str(), "% in CISD ");
        outfile->Printf("\n    %s", std::string(total_width,'-').c_str());
        for(int h = 0; h < nirrep; ++h){
            if(nrootpi_[h] != 0){
                for(int i = nrootpi_[h]; i > 0; --i){
                    std::string sym(4,' ');
                    if(i == 1) sym = irrep_symbol_[h];
                    outfile->Printf("\n    %4s  %6d  %18.10f  %18.10f  %5.1f (%3d)", sym.c_str(), i-1, ref_energies_[h][i-1],
                            pt2_energies_[h][i-1], t1_percentage_[h][i-1].second, t1_percentage_[h][i-1].first);
                }
                outfile->Printf("\n    %s", std::string(total_width,'-').c_str());
            }
        }
    }

    if(nrootpi_[0] > 0 && total_nroots_ > 1){
        print_h2("Relative Energy WRT Totally Symmetric Ground State (eV)");

        double ev = 27.211399;
        if(ref_type == "CAS"){
            int width = 4 + 6 + 8 + 8 + 3 * 2;
            outfile->Printf("\n    %4s  %6s  %6s%2s  %6s", "Sym.", "ROOT", ref_type.c_str(), std::string(2,' ').c_str(), "PT2");
            outfile->Printf("\n    %s", std::string(width,'-').c_str());
            for(int h = 0; h < nirrep; ++h){
                if(nrootpi_[h] != 0){
                    for(int i = nrootpi_[h]; i > 0; --i){
                        std::string sym(4,' ');
                        if(h == 0 && i == 1) continue;
                        if(h == 0 && i == 2) sym = irrep_symbol_[h];
                        if(i == 1) sym = irrep_symbol_[h];

                        double Eci = (ref_energies_[h][i-1] - ref_energies_[0][0]) * ev;
                        double Ept = (pt2_energies_[h][i-1] - pt2_energies_[0][0]) * ev;
                        outfile->Printf("\n    %4s  %6d  %8.3f  %8.3f", sym.c_str(), i-1, Eci, Ept);
                    }
                    if(h != 0 || nrootpi_[0] != 1)
                        outfile->Printf("\n    %s", std::string(width,'-').c_str());
                }
            }
        }else{
            int width = 4 + 6 + 8 + 8 + 25 + 4 * 2;
            outfile->Printf("\n    %4s  %6s  %6s%2s  %6s%2s  %25s", "Sym.", "ROOT", ref_type.c_str(),
                            "  ", "PT2", "  ", "Excitation Type");
            outfile->Printf("\n    %s", std::string(width,'-').c_str());
            for(int h = 0; h < nirrep; ++h){
                if(nrootpi_[h] != 0){
                    for(int i = nrootpi_[h]; i > 0; --i){
                        std::string sym(4,' ');
                        if(h == 0 && i == 1) continue;
                        if(h == 0 && i == 2) sym = irrep_symbol_[h];
                        if(i == 1) sym = irrep_symbol_[h];

                        double Eci = (ref_energies_[h][i-1] - ref_energies_[0][0]) * ev;
                        double Ept = (pt2_energies_[h][i-1] - pt2_energies_[0][0]) * ev;
                        std::string ex_type = compute_ex_type(dominant_dets_[h][i-1], dominant_dets_[0][0]);
                        outfile->Printf("\n    %4s  %6d  %8.3f  %8.3f  %25s", sym.c_str(), i-1, Eci, Ept, ex_type.c_str());
                    }
                    if(h != 0 || nrootpi_[0] != 1)
                        outfile->Printf("\n    %s", std::string(width,'-').c_str());
                }
            }
            outfile->Printf("\n    Excitation type: orbitals are zero-based (active only).");
            outfile->Printf("\n    (S) for singles; (D) for doubles.");
        }
    }
}

std::string ACTIVE_DSRGPT2::compute_ex_type(const STLBitsetDeterminant& det, const STLBitsetDeterminant& ref_det){
    Dimension active = mo_space_info_->get_dimension("ACTIVE");
    int nirrep = this->nirrep();
    std::vector<std::string> sym_active;
    for(int h = 0; h < nirrep; ++h){
        for(int i = 0; i < active[h]; ++i){
            sym_active.push_back(std::to_string(i) + irrep_symbol_[h]);
        }
    }

    // compare alpha occ
    std::vector<int> occA_ref (ref_det.get_alfa_occ());
    std::vector<int> occA_det (det.get_alfa_occ());
    std::vector<int> commonA;
    std::set_intersection(occA_ref.begin(), occA_ref.end(), occA_det.begin(), occA_det.end(), back_inserter(commonA));
    occA_ref.erase(std::set_difference(occA_ref.begin(), occA_ref.end(), commonA.begin(), commonA.end(),
                                       occA_ref.begin()), occA_ref.end());
    occA_det.erase(std::set_difference(occA_det.begin(), occA_det.end(), commonA.begin(), commonA.end(),
                                       occA_det.begin()), occA_det.end());

    // compare beta occ
    std::vector<int> occB_ref (ref_det.get_beta_occ());
    std::vector<int> occB_det (det.get_beta_occ());
    std::vector<int> commonB;
    std::set_intersection(occB_ref.begin(), occB_ref.end(), occB_det.begin(), occB_det.end(), back_inserter(commonB));
    occB_ref.erase(std::set_difference(occB_ref.begin(), occB_ref.end(), commonB.begin(), commonB.end(),
                                       occB_ref.begin()), occB_ref.end());
    occB_det.erase(std::set_difference(occB_det.begin(), occB_det.end(), commonB.begin(), commonB.end(),
                                       occB_det.begin()), occB_det.end());

    // output string
    std::string output;
    size_t A = occA_ref.size();
    size_t B = occB_ref.size();

    // CIS
    if(A + B == 1){
        if (A == 1 && B == 0){
            int idx_ref = occA_ref[0];
            int idx_det = occA_det[0];
            output = sym_active[idx_ref] + " -> " + sym_active[idx_det] + " (S)";
        }
        else if (A == 0 && B ==1){
            int idx_ref = occB_ref[0];
            int idx_det = occB_det[0];
            output = sym_active[idx_ref] + " -> " + sym_active[idx_det] + " (S)";
        }
    }else{
        // CISD
        if (A == 2){
            int i_ref = occA_ref[0], j_ref = occA_ref[1];
            int i_det = occA_det[0], j_det = occA_det[1];
            output = sym_active[i_ref] + "," + sym_active[j_ref] + " -> "
                    + sym_active[i_det] + "," + sym_active[j_det];
        }
        else if (B == 2){
            int i_ref = occB_ref[0], j_ref = occB_ref[1];
            int i_det = occB_det[0], j_det = occB_det[1];
            output = sym_active[i_ref] + "," + sym_active[j_ref] + " -> "
                    + sym_active[i_det] + "," + sym_active[j_det];
        }
        else if (A == 1 && B == 1){
            int i_ref = occA_ref[0], j_ref = occB_ref[0];
            int i_det = occA_det[0], j_det = occB_det[0];
            if(i_ref == j_ref && i_det == j_det){
                output = sym_active[i_ref] + " -> " + sym_active[i_det] + " (D)";
            }else{
                output = sym_active[i_ref] + "," + sym_active[j_ref] + " -> "
                        + sym_active[i_det] + "," + sym_active[j_det];
            }
        }
    }

    return output;
}

}}
