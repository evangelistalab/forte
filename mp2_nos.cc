#include "mp2_nos.h"

#include "ambit/blocked_tensor.h"

#include <libpsio/psio.h>
#include <libpsio/psio.hpp>

#include <libmints/matrix.h>
#include <libmints/vector.h>

namespace psi{ namespace libadaptive{

using namespace ambit;
Matrix tensor_to_matrix(ambit::Tensor t,Dimension dims);

MP2_NOS::MP2_NOS(boost::shared_ptr<Wavefunction> wfn, Options &options, ExplorerIntegrals* ints)
{
    outfile->Printf("\n\n      ------------------------------------------------");
    outfile->Printf("\n          Second-Order Moller-Plesset Natural Orbitals");
    outfile->Printf("\n               written by Francesco A. Evangelista");
    outfile->Printf("\n      ------------------------------------------------\n");
    outfile->Flush();

    BlockedTensor::set_expert_mode(true);

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

    Dimension ncmopi_ = ints->ncmopi();
    Dimension frzcpi = ints->frzcpi();
    Dimension frzvpi = ints->frzvpi();

    Dimension nmopi = wfn->nmopi();
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

    for (size_t p = 0; p < a_occ_mos.size(); ++p) mos_to_aocc[a_occ_mos[p]] = p;
    for (size_t p = 0; p < b_occ_mos.size(); ++p) mos_to_bocc[b_occ_mos[p]] = p;
    for (size_t p = 0; p < a_vir_mos.size(); ++p) mos_to_avir[a_vir_mos[p]] = p;
    for (size_t p = 0; p < b_vir_mos.size(); ++p) mos_to_bvir[b_vir_mos[p]] = p;

    size_t naocc = a_occ_mos.size();
    size_t nbocc = b_occ_mos.size();
    size_t navir = a_vir_mos.size();
    size_t nbvir = b_vir_mos.size();

    BlockedTensor::add_mo_space("o","ijklmn",a_occ_mos,AlphaSpin);
    BlockedTensor::add_mo_space("O","IJKLMN",b_occ_mos,BetaSpin);
    BlockedTensor::add_mo_space("v","abcdef",a_vir_mos,AlphaSpin);
    BlockedTensor::add_mo_space("V","ABCDEF",b_vir_mos,BetaSpin);
    BlockedTensor::add_composite_mo_space("i","pqrstuvwxyz",{"o","v"});
    BlockedTensor::add_composite_mo_space("I","PQRSTUVWXYZ",{"O","V"});

    BlockedTensor G1 = BlockedTensor::build(kCore,"G1",spin_cases({"oo"}));
    BlockedTensor D1 = BlockedTensor::build(kCore,"D1",spin_cases({"oo","vv"}));
    BlockedTensor H = BlockedTensor::build(kCore,"H",spin_cases({"ii"}));
    BlockedTensor F = BlockedTensor::build(kCore,"F",spin_cases({"ii"}));
    BlockedTensor V = BlockedTensor::build(kCore,"V",spin_cases({"iiii"}));
    BlockedTensor T2 = BlockedTensor::build(kCore,"T2",spin_cases({"oovv"}));
    BlockedTensor InvD2 = BlockedTensor::build(kCore,"Inverse D2",spin_cases({"oovv"}));

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

    size_t ncmo_ = ints->ncmo();
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
    Cb_new->gemm(false,false,1.0,Cb,Ua,0.0);
    Ca->copy(Ca_new);
    Cb->copy(Cb_new);

    // Retransform the integrals in the new basis
    ints->retransform_integrals();

    BlockedTensor::set_expert_mode(false);
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
    for (size_t h = 0; h < dims.n(); ++h){
        for (size_t p = 0; p < dims[h]; ++p){
            for (size_t q = 0; q < dims[h]; ++q){
                double value = M.get(p + offset,q + offset);
                M_sym.set(h,p,q,value);
            }
        }
        offset += dims[h];
    }
    return M_sym;
}

}} // End Namespaces
