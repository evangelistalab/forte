#include "ambit/blocked_tensor.h"

#include <libpsio/psio.h>
#include <libpsio/psio.hpp>

#include <libmints/matrix.h>
#include <libmints/vector.h>

#include "helpers.h"
#include "mp2_nos.h"

namespace psi{ namespace forte{

using namespace ambit;

MP2_NOS::MP2_NOS(boost::shared_ptr<Wavefunction> wfn, Options &options, std::shared_ptr<ForteIntegrals>  ints, std::shared_ptr<MOSpaceInfo> mo_space_info)
{
    print_method_banner({"Second-Order Moller-Plesset Natural Orbitals",
                         "written by Francesco A. Evangelista"});

    BlockedTensor::set_expert_mode(true);
    BlockedTensor::reset_mo_spaces();

    /// List of alpha occupied MOs
    std::vector<size_t> a_occ_mos;
    /// List of beta occupied MOs
    std::vector<size_t> b_occ_mos;
    /// List of alpha virtual MOs
    std::vector<size_t> a_vir_mos;
    /// List of beta virtual MOs
    std::vector<size_t> b_vir_mos;

    /// Map from all the MOs to the alpha occupied
    std::map<size_t,size_t> mos_to_aocc;
    /// Map from all the MOs to the beta occupied
    std::map<size_t,size_t> mos_to_bocc;
    /// Map from all the MOs to the alpha virtual
    std::map<size_t,size_t> mos_to_avir;
    /// Map from all the MOs to the beta virtual
    std::map<size_t,size_t> mos_to_bvir;

    Dimension ncmopi_ = mo_space_info->get_dimension("CORRELATED");
    Dimension frzcpi =  mo_space_info->get_dimension("FROZEN_DOCC");
    Dimension frzvpi =  mo_space_info->get_dimension("FROZEN_UOCC");

    Dimension nmopi =  wfn->nmopi();
    Dimension doccpi = wfn->doccpi();
    Dimension soccpi = wfn->soccpi();

    Dimension corr_docc(doccpi);
    corr_docc -= frzcpi;

    Dimension aoccpi = corr_docc + wfn->soccpi();
    Dimension boccpi = corr_docc;
    Dimension avirpi = ncmopi_ - aoccpi;
    Dimension bvirpi = ncmopi_ - boccpi;

    int nirrep = wfn->nirrep();

    for (int h = 0, p = 0; h < nirrep; ++h){
        for (int i = 0; i < corr_docc[h]; ++i,++p){
            a_occ_mos.push_back(p);
            b_occ_mos.push_back(p);
        }
        for (int i = 0; i < soccpi[h]; ++i,++p){
            a_occ_mos.push_back(p);
            b_vir_mos.push_back(p);
        }
        for (int a = 0; a < ncmopi_[h] - corr_docc[h] - soccpi[h]; ++a,++p){
            a_vir_mos.push_back(p);
            b_vir_mos.push_back(p);
        }
    }
    //f r (size_t p = 0; p < a_occ_mos.size(); ++p) mos_to_aocc[a_occ_mos[p]] = p;
    //for (size_t p = 0; p < b_occ_mos.size(); ++p) mos_to_bocc[b_occ_mos[p]] = p;
    //for (size_t p = 0; p < a_vir_mos.size(); ++p) mos_to_avir[a_vir_mos[p]] = p;
    //for (size_t p = 0; p < b_vir_mos.size(); ++p) mos_to_bvir[b_vir_mos[p]] = p;


    BlockedTensor::add_mo_space("o","ijklmn",a_occ_mos,AlphaSpin);
    BlockedTensor::add_mo_space("O","IJKLMN",b_occ_mos,BetaSpin);
    BlockedTensor::add_mo_space("v","abcdef",a_vir_mos,AlphaSpin);
    BlockedTensor::add_mo_space("V","ABCDEF",b_vir_mos,BetaSpin);
    BlockedTensor::add_composite_mo_space("i","pqrstuvwxyz",{"o","v"});
    BlockedTensor::add_composite_mo_space("I","PQRSTUVWXYZ",{"O","V"});

    BlockedTensor G1 = BlockedTensor::build(CoreTensor,"G1",spin_cases({"oo"}));
    BlockedTensor D1 = BlockedTensor::build(CoreTensor,"D1",spin_cases({"oo","vv"}));
    BlockedTensor H = BlockedTensor::build(CoreTensor,"H",spin_cases({"ii"}));
    BlockedTensor F = BlockedTensor::build(CoreTensor,"F",spin_cases({"ii"}));
    BlockedTensor V = BlockedTensor::build(CoreTensor,"V",spin_cases({"iiii"}));
    BlockedTensor T2 = BlockedTensor::build(CoreTensor,"T2",spin_cases({"oovv"}));
    BlockedTensor InvD2 = BlockedTensor::build(CoreTensor,"Inverse D2",spin_cases({"oovv"}));

    // Fill in the one-electron operator (H)
    H.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0] == AlphaSpin)
            value = ints->oei_a(i[0],i[1]);
        else
            value = ints->oei_b(i[0],i[1]);
    });

    // Fill in the two-electron operator (V)
    V.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)) value = ints->aptei_aa(i[0],i[1],i[2],i[3]);
        if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin) ) value = ints->aptei_ab(i[0],i[1],i[2],i[3]);
        if ((spin[0] == BetaSpin)  and (spin[1] == BetaSpin) ) value = ints->aptei_bb(i[0],i[1],i[2],i[3]);
    });

    H.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0] == AlphaSpin)
            value = ints->oei_a(i[0],i[1]);
        else
            value = ints->oei_b(i[0],i[1]);
    });

    G1.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        value = i[0] == i[1] ? 1.0 : 0.0;});

    D1.block("oo").iterate([&](const std::vector<size_t>& i,double& value){
        value = i[0] == i[1] ? 1.0 : 0.0;});

    D1.block("OO").iterate([&](const std::vector<size_t>& i,double& value){
        value = i[0] == i[1] ? 1.0 : 0.0;});

    // Form the Fock matrix
    F["ij"]  = H["ij"];
    F["ab"]  = H["ab"];
    F["pq"] += V["prqs"] * G1["sr"];
    F["pq"] += V["pRqS"] * G1["SR"];

    F["IJ"] += H["IJ"];
    F["AB"] += H["AB"];
    F["PQ"] += V["rPsQ"] * G1["sr"];
    F["PQ"] += V["PRQS"] * G1["SR"];

    size_t ncmo_ = mo_space_info->size("CORRELATED");
    std::vector<double> Fa(ncmo_);
    std::vector<double> Fb(ncmo_);

    F.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0] == AlphaSpin and (i[0] == i[1])){
            Fa[i[0]] = value;
        }
        if (spin[0] == BetaSpin and (i[0] == i[1])){
            Fb[i[0]] = value;
        }
    });

    InvD2.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)){
            value = 1.0 / (Fa[i[0]] + Fa[i[1]] - Fa[i[2]] - Fa[i[3]]);
        }else if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin) ){
            value = 1.0 / (Fa[i[0]] + Fb[i[1]] - Fa[i[2]] - Fb[i[3]]);
        }else if ((spin[0] == BetaSpin)  and (spin[1] == BetaSpin) ){
            value = 1.0 / (Fb[i[0]] + Fb[i[1]] - Fb[i[2]] - Fb[i[3]]);
        }
    });

    T2["ijab"] = V["ijab"] * InvD2["ijab"];
    T2["iJaB"] = V["iJaB"] * InvD2["iJaB"];
    T2["IJAB"] = V["IJAB"] * InvD2["IJAB"];

    double Eaa = 0.25 * T2["ijab"] * V["ijab"];
    double Eab = T2["iJaB"] * V["iJaB"];
    double Ebb = 0.25 * T2["IJAB"] * V["IJAB"];

    double mp2_correlation_energy = Eaa + Eab + Ebb;
    double ref_energy = wfn->reference_energy();
    outfile->Printf("\n\n    SCF energy                            = %20.15f",ref_energy);
    outfile->Printf("\n    MP2 correlation energy                = %20.15f",mp2_correlation_energy);
    outfile->Printf("\n  * MP2 total energy                      = %20.15f\n\n",ref_energy + mp2_correlation_energy);

    D1["ab"] += 0.5 * T2["ijbc"] * T2["ijac"];
    D1["ab"] += 1.0 * T2["iJbC"] * T2["iJaC"];

    D1["AB"] += 0.5 * T2["IJCB"] * T2["IJCA"];
    D1["AB"] += 1.0 * T2["iJcB"] * T2["iJcA"];

    D1["ij"] -= 0.5 * T2["ikab"] * T2["jkab"];
    D1["ij"] -= 1.0 * T2["iKaB"] * T2["jKaB"];

    D1["IJ"] -= 0.5 * T2["IKAB"] * T2["JKAB"];
    D1["IJ"] -= 1.0 * T2["kIaB"] * T2["kJaB"];

    // Copy the density matrix to matrix objects
    Matrix D1oo = tensor_to_matrix(D1.block("oo"),aoccpi);
    Matrix D1OO = tensor_to_matrix(D1.block("OO"),boccpi);
    Matrix D1vv = tensor_to_matrix(D1.block("vv"),avirpi);
    Matrix D1VV = tensor_to_matrix(D1.block("VV"),bvirpi);

    Matrix D1oo_evecs("D1oo_evecs",aoccpi,aoccpi);
    Matrix D1OO_evecs("D1OO_evecs",boccpi,boccpi);
    Matrix D1vv_evecs("D1vv_evecs",avirpi,avirpi);
    Matrix D1VV_evecs("D1VV_evecs",bvirpi,bvirpi);

    Vector D1oo_evals("D1oo_evals",aoccpi);
    Vector D1OO_evals("D1OO_evals",boccpi);
    Vector D1vv_evals("D1vv_evals",avirpi);
    Vector D1VV_evals("D1VV_evals",bvirpi);

    D1oo.diagonalize(D1oo_evecs,D1oo_evals);
    D1vv.diagonalize(D1vv_evecs,D1vv_evals);
    D1OO.diagonalize(D1OO_evecs,D1OO_evals);
    D1VV.diagonalize(D1VV_evecs,D1VV_evals);

    //Print natural orbitals
    if(options.get_bool("NAT_ORBS_PRINT"))

    {
        D1oo_evals.print();
        D1vv_evals.print();
        D1OO_evals.print();
        D1VV_evals.print();
    }
    //This will suggested a restricted_docc and a active
    //Does not take in account frozen_docc
    if(options.get_bool("NAT_ACT"))
    {
        std::vector<size_t> restricted_docc(nirrep);
        std::vector<size_t> active(nirrep);
        double occupied = options.get_double("OCC_NATURAL");
        double virtual_orb = options.get_double("VIRT_NATURAL");
        outfile->Printf("\n Suggested Active Space \n");
        outfile->Printf("\n Occupied orbitals with an occupation less than %6.4f are active", occupied);
        outfile->Printf("\n Virtual orbitals with an occupation greater than %6.4f are active", virtual_orb);
        outfile->Printf("\n Remember, these are suggestions  :-)!\n");
        for(size_t h = 0; h < nirrep; ++h){
            size_t restricted_docc_number = 0;
            size_t active_number          = 0;
            for(size_t i = 0; i < aoccpi[h]; ++i) {
                if(D1oo_evals.get(h,i) < occupied)
                {
                    active_number++;
                    outfile->Printf("\n In %u, orbital occupation %u = %8.6f Active occupied", h,i, D1oo_evals.get(h,i));
                    active[h] = active_number;
                }
                else if(D1oo_evals.get(h,i) >= occupied)
                {
                    restricted_docc_number++;
                    outfile->Printf("\n In %u, orbital occupation %u = %8.6f  RDOCC", h, i, D1oo_evals.get(h,i));
                    restricted_docc[h] = restricted_docc_number;
                }
            }
            for(size_t a = 0; a < avirpi[h]; ++a){
                if(D1vv_evals.get(h,a) > virtual_orb)
                {
                    active_number++;
                    active[h] =active_number; 
                    outfile->Printf("\n In %u, orbital occupation %u = %8.6f Active virtual", h,a, D1vv_evals.get(h,a));
                }
            }

        }
        outfile->Printf("\n By occupation analysis, your restricted docc should be\n");
        outfile->Printf("\n Restricted_docc = [");
        for(auto &rocc : restricted_docc){
            outfile->Printf("%u, ", rocc);
        }
        outfile->Printf("]\n");
        outfile->Printf("\n By occupation analysis, active space should be \n");
        outfile->Printf("\n Active = [");
        for(auto &ract : active){
            outfile->Printf("%u, ", ract);
        }
        outfile->Printf("]\n");
    }

    Matrix Ua("Ua",nmopi,nmopi);
    // Patch together the transformation matrices
    for (size_t h = 0; h < nirrep; ++h){
        size_t irrep_offset = 0;

        // Frozen core orbitals are unchanged
        for (size_t p = 0; p < frzcpi[h]; ++p){
            Ua.set(h,p,p,1.0);
        }
        irrep_offset += frzcpi[h];

        // Occupied alpha
        for (size_t p = 0; p < aoccpi[h]; ++p){
            for (size_t q = 0; q < aoccpi[h]; ++q){
                double value = D1oo_evecs.get(h,p,q);
                Ua.set(h,p + irrep_offset,q + irrep_offset,value);
            }
        }
        irrep_offset += aoccpi[h];

        // Virtual alpha
        for (size_t p = 0; p < avirpi[h]; ++p){
            for (size_t q = 0; q < avirpi[h]; ++q){
                double value = D1vv_evecs.get(h,p,q);
                Ua.set(h,p + irrep_offset,q + irrep_offset,value);
            }
        }
        irrep_offset += avirpi[h];

        // Frozen virtual orbitals are unchanged
        for (size_t p = 0; p < frzvpi[h]; ++p){
            Ua.set(h,p + irrep_offset,p + irrep_offset,1.0);
        }
    }

    Matrix Ub("Ub",nmopi,nmopi);
    // Patch together the transformation matrices
    for (size_t h = 0; h < nirrep; ++h){
        size_t irrep_offset = 0;

        // Frozen core orbitals are unchanged
        for (size_t p = 0; p < frzcpi[h]; ++p){
            Ub.set(h,p,p,1.0);
        }
        irrep_offset += frzcpi[h];

        // Occupied alpha
        for (size_t p = 0; p < boccpi[h]; ++p){
            for (size_t q = 0; q < boccpi[h]; ++q){
                double value = D1OO_evecs.get(h,p,q);
                Ub.set(h,p + irrep_offset,q + irrep_offset,value);
            }
        }
        irrep_offset += boccpi[h];

        // Virtual alpha
        for (size_t p = 0; p < bvirpi[h]; ++p){
            for (size_t q = 0; q < bvirpi[h]; ++q){
                double value = D1VV_evecs.get(h,p,q);
                Ub.set(h,p + irrep_offset,q + irrep_offset,value);
            }
        }
        irrep_offset += bvirpi[h];

        // Frozen virtual orbitals are unchanged
        for (size_t p = 0; p < frzvpi[h]; ++p){
            Ub.set(h,p + irrep_offset,p + irrep_offset,1.0);
        }
    }

    // Modify the orbital coefficients
    SharedMatrix Ca = wfn->Ca();
    SharedMatrix Cb = wfn->Cb();
    SharedMatrix Ca_new(Ca->clone());
    SharedMatrix Cb_new(Cb->clone());
    Ca_new->gemm(false,false,1.0,Ca,Ua,0.0);
    Cb_new->gemm(false,false,1.0,Cb,Ub,0.0);
    Ca->copy(Ca_new);
    Cb->copy(Cb_new);

    // Retransform the integrals in the new basis
    ints->retransform_integrals();

    BlockedTensor::set_expert_mode(false);
    // Erase all mo_space information
    BlockedTensor::reset_mo_spaces();
}

SemiCanonical::SemiCanonical(boost::shared_ptr<Wavefunction> wfn,
                             Options &options, std::shared_ptr<ForteIntegrals> ints,
                             std::shared_ptr<MOSpaceInfo> mo_space_info, Reference &reference)
{
    print_method_banner({"Semi-Canonical Orbitals","Francesco A. Evangelista"});
    Timer SemiCanonicalize;

    // 1. Build the Fock matrix
    int nirrep = wfn->nirrep();
    size_t ncmo = mo_space_info->size("CORRELATED");
    Dimension nmopi = wfn->nmopi();
    Dimension ncmopi = mo_space_info->get_dimension("CORRELATED");
    Dimension fdocc = mo_space_info->get_dimension("FROZEN_DOCC");
    Dimension rdocc = mo_space_info->get_dimension("RESTRICTED_DOCC");
    Dimension actv = mo_space_info->get_dimension("ACTIVE");
    Dimension ruocc = mo_space_info->get_dimension("RESTRICTED_UOCC");

    SharedMatrix Da(new Matrix("Da", ncmo, ncmo));
    SharedMatrix Db(new Matrix("Db", ncmo, ncmo));

    Matrix L1a = tensor_to_matrix(reference.L1a(),actv);
    Matrix L1b = tensor_to_matrix(reference.L1b(),actv);

    for (int h = 0, offset = 0; h < nirrep; ++h){
        // core block (diagonal)
        for (int i = 0; i < rdocc[h]; ++i){
            Da->set(offset + i, offset + i,1.0);
            Db->set(offset + i, offset + i,1.0);
        }

        offset += rdocc[h];

        // active block
        for (int u = 0; u < actv[h]; ++u){
            for (int v = 0; v < actv[h]; ++v){
                Da->set(offset + u, offset + v,L1a.get(h,u,v));
                Db->set(offset + u, offset + v,L1b.get(h,u,v));
            }
        }

        offset += ncmopi[h] - rdocc[h];
    }

    Timer FockTime;
    ints->make_fock_matrix(Da,Db);
    outfile->Printf("\n Took %8.6f s to build fock matrix", FockTime.get());

    // 2. Diagonalize the diagonal blocks of the Fock matrix
    SharedMatrix Fc_a(new Matrix("Fock core alpha",rdocc,rdocc));
    SharedMatrix Fc_b(new Matrix("Fock core beta",rdocc,rdocc));
    SharedMatrix Fa_a(new Matrix("Fock active alpha",actv,actv));
    SharedMatrix Fa_b(new Matrix("Fock active beta",actv,actv));
    SharedMatrix Fv_a(new Matrix("Fock virtual alpha",ruocc,ruocc));
    SharedMatrix Fv_b(new Matrix("Fock virtual beta",ruocc,ruocc));

    for (int h = 0, offset = 0; h < nirrep; ++h){
        // core block
        for (int i = 0; i < rdocc[h]; ++i){
            for (int j = 0; j < rdocc[h]; ++j){
                Fc_a->set(h,i,j,ints->fock_a(offset + i,offset + j));
                Fc_b->set(h,i,j,ints->fock_b(offset + i,offset + j));
            }
        }
        offset += rdocc[h];

        // active block
        for (int u = 0; u < actv[h]; ++u){
            for (int v = 0; v < actv[h]; ++v){
                Fa_a->set(h,u,v,ints->fock_a(offset + u,offset + v));
                Fa_b->set(h,u,v,ints->fock_b(offset + u,offset + v));

            }
        }
        offset += actv[h];

        // virtual block
        for (int a = 0; a < ruocc[h]; ++a){
            for (int b = 0; b < ruocc[h]; ++b){
                Fv_a->set(h,a,b,ints->fock_a(offset + a,offset + b));
                Fv_b->set(h,a,b,ints->fock_b(offset + a,offset + b));
            }
        }
        offset += ruocc[h];
    }

    // Diagonalize each block of the Fock matrix
    std::vector<SharedMatrix> evecs;
    std::vector<SharedVector> evals;
    for (auto F : {Fc_a,Fc_b,Fa_a,Fa_b,Fv_a,Fv_b}){
        SharedMatrix U(new Matrix("U",F->rowspi(),F->colspi()));
        SharedVector lambda(new Vector("lambda",F->rowspi()));
        F->diagonalize(U,lambda);
        evecs.push_back(U);
        evals.push_back(lambda);
    }
//    Fv_a->print();
//    SharedMatrix Uv = evecs[4];
//    Fv_a->transform(Uv);
//    Fv_a->print();

    // 3. Build the unitary matrices
    Matrix Ua("Ua",nmopi,nmopi);
    Matrix Ub("Ub",nmopi,nmopi);
    for (int h = 0; h < nirrep; ++h){
        size_t offset = 0;

        // Set the matrices to the identity,
        // this takes care of the frozen core and virtual spaces
        for (int p = 0; p < nmopi[h]; ++p){
            Ua.set(h,p,p,1.0);
            Ub.set(h,p,p,1.0);
        }

        offset += fdocc[h];

        // core block
        for (int i = 0; i < rdocc[h]; ++i){
            for (int j = 0; j < rdocc[h]; ++j){
                Ua.set(h,offset + i, offset + j,evecs[0]->get(h,i,j));
                Ub.set(h,offset + i, offset + j,evecs[1]->get(h,i,j));
            }
        }
        offset += rdocc[h];

        // active block
        for (int u = 0; u < actv[h]; ++u){
            for (int v = 0; v < actv[h]; ++v){
                Ua.set(h,offset + u, offset + v,evecs[2]->get(h,u,v));
                Ub.set(h,offset + u, offset + v,evecs[3]->get(h,u,v));
            }
        }
        offset += actv[h];

        // virtual block
        for (int a = 0; a < ruocc[h]; ++a){
            for (int b = 0; b < ruocc[h]; ++b){
                Ua.set(h,offset + a, offset + b,evecs[4]->get(h,a,b));
                Ub.set(h,offset + a, offset + b,evecs[5]->get(h,a,b));
            }
        }
    }

    // 4. Transform the MO coefficients
    SharedMatrix Ca = wfn->Ca();
    SharedMatrix Cb = wfn->Cb();
    SharedMatrix Ca_new(Ca->clone());
    SharedMatrix Cb_new(Cb->clone());
    Ca_new->gemm(false,false,1.0,Ca,Ua,0.0);
    Cb_new->gemm(false,false,1.0,Cb,Ub,0.0);
    Ca->copy(Ca_new);
    Cb->copy(Cb_new);

    // 5. Retransform the integrals in the new basis
    print_h2("Integral transformation");
    ints->retransform_integrals();
    outfile->Printf("\n SemiCanonicalize takes %8.6f s.", SemiCanonicalize.get());
}

}} // End Namespaces
