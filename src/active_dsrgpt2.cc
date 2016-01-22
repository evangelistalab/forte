#include <libmints/pointgrp.h>

#include "active_dsrgpt2.h"

namespace psi{ namespace forte{
ACTIVE_DSRGPT2::ACTIVE_DSRGPT2(boost::shared_ptr<Wavefunction> wfn, Options &options,
                               std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info)
    : wfn_(wfn), options_(options), ints_(ints), mo_space_info_(mo_space_info), total_nroots_(0)
{
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
        int nirrep = wfn_->nirrep();
        ref_energies_ = vector<vector<double>> (nirrep,vector<double>());
        pt2_energies_ = vector<vector<double>> (nirrep,vector<double>());
        vecdet_cis_ = vector<vector<STLBitsetDeterminant>> (nirrep,vector<STLBitsetDeterminant>());
        eigen_cis_ = vector<vector<SharedVector>> (nirrep,vector<SharedVector>());
        t1_percentage_ = vector<vector<pair<int,double>>> (nirrep,vector<pair<int,double>>());
        CharacterTable ct = Process::environment.molecule()->point_group()->char_table();

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

void ACTIVE_DSRGPT2::compute_energy(){
    if(total_nroots_ == 0){
        outfile->Printf("\n  NROOTPI is zero. Did nothing.");
    }else{
        FCI_MO fci_mo(wfn_,options_,ints_,mo_space_info_);
        int nirrep = nrootpi_.size();
        for(int h = 0; h < nirrep; ++h){
            if(nrootpi_[h] == 0) continue;
            else{
                fci_mo.set_root_sym(h);
                for(int i = 0; i < nrootpi_[h]; ++i){
                    // CI routine
                    outfile->Printf("\n\n  %s", std::string(35,'=').c_str());
                    outfile->Printf("\n    Current Job: %3s state, root %2d", irrep_symbol_[h].c_str(), i);
                    outfile->Printf("\n  %s\n", std::string(35,'=').c_str());
                    fci_mo.set_nroots(i+1);
                    fci_mo.set_root(i);
                    ref_energies_[h].push_back(fci_mo.compute_energy());
                    Reference reference = fci_mo.reference();

                    // save the CIS stuff
                    eigen_cis_[h].push_back(fci_mo.eigen().back().first);
                    if(i == nrootpi_[h] - 1){
                        if(options_.get_str("ACTIVE_SPACE_TYPE") == "CIS"){
                            vecdet_cis_[h] = fci_mo.p_space();
                        }
                    }

                    // PT2 routine
                    double pt2 = 0.0;
                    if(options_.get_str("INT_TYPE") == "CONVENTIONAL"){
                        auto dsrg = std::make_shared<DSRG_MRPT2>(reference,wfn_,options_,ints_,mo_space_info_);
                        pt2 = dsrg->compute_energy();
                    }else{
                        auto dsrg = std::make_shared<THREE_DSRG_MRPT2>(reference,wfn_,options_,ints_,mo_space_info_);
                        pt2 = dsrg->compute_energy();
                    }
                    pt2_energies_[h].push_back(pt2);
                }
            }
        }
        compute_t1_percentage();
        print_summary();
    }
}

void ACTIVE_DSRGPT2::compute_t1_percentage(){
    print_h2("Compute the singles percentage");

    // set to CISD first
    FCI_MO fci_mo(wfn_,options_,ints_,mo_space_info_);
    fci_mo.set_active_space_type("CISD");
    fci_mo.set_quite_mode(true);
    int nirrep = nrootpi_.size();
    for(int h = 0; h < nirrep; ++h){
        if(nrootpi_[h] == 0) continue;
        else{
            fci_mo.set_root_sym(h);
            Matrix overlap ("overlap", 2 * nrootpi_[h], nrootpi_[h]);
            for(int i = 0; i < 2 * nrootpi_[h]; ++i){
                // CI routine
                outfile->Printf("\n\n  %s", std::string(35,'=').c_str());
                outfile->Printf("\n    Current Job: %3s state, root %2d", irrep_symbol_[h].c_str(), i);
                outfile->Printf("\n  %s\n", std::string(35,'=').c_str());
                fci_mo.set_nroots(i+1);
                fci_mo.set_root(i);
                fci_mo.compute_energy();

                std::vector<STLBitsetDeterminant> vecdet_cisd = fci_mo.p_space();
                SharedVector eigen_cisd = fci_mo.eigen().back().first;

                // loop over CIS states
                for(int j = 0; j < eigen_cis_[h].size(); ++j){
                    for(int s = 0; s < vecdet_cis_[h].size(); ++s){
                        for(int sd = 0; sd < vecdet_cisd.size(); ++sd){
                            if(vecdet_cis_[h][s] == vecdet_cisd[sd]){
                                double value = std::fabs(eigen_cisd->get(sd)) * std::fabs(eigen_cis_[h][j]->get(s));
                                overlap.add(i,j,value);
                            }
                        }
                    }
                }

            }
            overlap.print();
            for(int i = 0; i < overlap.ncol(); ++i){
                double max = overlap.get(i,i);
                int root = i;
                for(int j = 0; j < overlap.nrow(); ++j){
                    double value = overlap.get(j,i);
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
        outfile->Printf("\n    %4s  %6s  %11s%7s  %11s%7s  %8s", "Sym.", "ROOT", ref_type.c_str(), std::string(7,' ').c_str(),
                        "PT2", std::string(7,' ').c_str(), "T1% (#CISD)");
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
        print_h2("Relative Energy WRT closed-shell Ground State (eV)");
        int width = 4 + 6 + 10 + 10 + 3 * 2;
        outfile->Printf("\n    %4s  %6s  %6s%2s  %6s", "Sym.", "ROOT", ref_type.c_str(), std::string(2,' ').c_str(), "PT2");
        outfile->Printf("\n    %s", std::string(width,'-').c_str());
        for(int h = 0; h < nirrep; ++h){
            if(nrootpi_[h] != 0){
                for(int i = nrootpi_[h]; i > 0; --i){
                    std::string sym(4,' ');
                    if(h == 0 && i == 1) continue;
                    if(h == 0 && i == 2) sym = irrep_symbol_[h];
                    if(i == 1) sym = irrep_symbol_[h];

                    double ev = 27.211399;
                    double Eci = (ref_energies_[h][i-1] - ref_energies_[0][0]) * ev;
                    double Ept = (pt2_energies_[h][i-1] - pt2_energies_[0][0]) * ev;
                    outfile->Printf("\n    %4s  %6d  %8.3f  %8.3f", sym.c_str(), i-1, Eci, Ept);
                }
                if(h == 0 && nrootpi_[0] == 1){}
                else
                    outfile->Printf("\n    %s", std::string(width,'-').c_str());
            }
        }
    }
}

}}
