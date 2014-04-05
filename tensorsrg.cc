//#ifdef _HAS_LIBBTL_
#include "tensor_basic.h"
#include "tensor_labeled.h"
#include "tensor_product.h"
#include "tensorsrg.h"


#include <cmath>

#include <boost/numeric/odeint.hpp>

#include <libpsio/psio.hpp>
#include <libmints/wavefunction.h>
#include <libmints/molecule.h>
#include <libmints/vector.h>

#include "mosrg.h"

using namespace std;
using namespace psi;

namespace psi{ namespace libadaptive{

TensorSRG::TensorSRG(boost::shared_ptr<Wavefunction> wfn, Options &options, ExplorerIntegrals* ints)
    : MethodBase(wfn,options,ints)
{
    startup();
}

TensorSRG::~TensorSRG()
{
    cleanup();
}

void TensorSRG::startup()
{
    fprintf(outfile,"\n\n      --------------------------------------");
    fprintf(outfile,"\n          Similarity Renormalization Group");
    fprintf(outfile,"\n                tensor-based code");
    fprintf(outfile,"\n");
    fprintf(outfile,"\n                Version 0.1.0");
    fprintf(outfile,"\n");
    fprintf(outfile,"\n       written by Francesco A. Evangelista");
    fprintf(outfile,"\n      --------------------------------------\n");

    fprintf(outfile,"\n      Debug level = %d\n",debug_);
    fflush(outfile);

    BlockedTensor::print_mo_spaces();

    S1.resize_spin_components("S1","ov");
    S2.resize_spin_components("S2","oovv");
    Hbar1.resize_spin_components("Hbar1","ii");
    Hbar2.resize_spin_components("Hbar2","iiii");
    O1.resize_spin_components("O1","ii");
    O2.resize_spin_components("O2","iiii");
    C1.resize_spin_components("C1","ii");
    C2.resize_spin_components("C2","iiii");
    I_ioiv.resize_spin_components("C2","ioiv");


//    S2.aa.resize("T2aa","oovv");
//    S2.ab.resize("T2ab","oOvV");
//    S2.bb.resize("T2bb","OOVV");
}

void TensorSRG::cleanup()
{
    print_timings();
}

double TensorSRG::compute_mp2_guess()
{
//    S2.aa["ijab"] = V.aa["ijab"] / D2.aa["ijab"];
//    S2.ab["iJaB"] = V.ab["iJaB"] / D2.ab["iJaB"];
//    S2.bb["IJAB"] = V.bb["IJAB"] / D2.bb["IJAB"];

    S2["ijab"] = V["ijab"] / D2["ijab"];
    S2["iJaB"] = V["iJaB"] / D2["iJaB"];
    S2["IJAB"] = V["IJAB"] / D2["IJAB"];

//    double Eaa = 0.25 * BlockedTensor::dot(S2.aa["ijab"],V.aa["ijab"]);
//    double Eab = BlockedTensor::dot(S2.ab["iJaB"],V.ab["iJaB"]);
//    double Ebb = 0.25 * BlockedTensor::dot(S2.bb["IJAB"],V.bb["IJAB"]);


    double Eaa = 0.25 * BlockedTensor::dot(S2["ijab"],V["ijab"]);
    double Eab = BlockedTensor::dot(S2["iJaB"],V["iJaB"]);
    double Ebb = 0.25 * BlockedTensor::dot(S2["IJAB"],V["IJAB"]);

    double mp2_correlation_energy = Eaa + Eab + Ebb;
    double ref_energy = reference_energy();
    fprintf(outfile,"\n\n    SCF energy                            = %20.15f",ref_energy);
    fprintf(outfile,"\n    MP2 correlation energy                = %20.15f",mp2_correlation_energy);
    fprintf(outfile,"\n  * MP2 total energy                      = %20.15f\n\n",ref_energy + mp2_correlation_energy);
    return ref_energy + mp2_correlation_energy;
}

double TensorSRG::compute_energy()
{
    if(options_.get_str("SRG_MODE") == "SRG"){
//        compute_similarity_renormalization_group();
    }else if(options_.get_str("SRG_MODE") == "CT"){
        return compute_ct_energy();
    }else if(options_.get_str("SRG_MODE") == "DSRG"){
//        compute_driven_srg_energy();
    }
    print_timings();
    return 0.0;
}

double TensorSRG::compute_ct_energy()
{
    compute_mp2_guess();
    fprintf(outfile,"\n\n  Computing the single-reference Canonical Transformation energy");

    // Start the CCSD cycle
    double old_energy = 0.0;
    bool   converged  = false;
    int    cycle      = 0;

    int max_diis_vectors = options_.get_int("DIIS_MAX_VECS");
    DIISManager diis_manager(max_diis_vectors, "L-CTSD DIIS vector", DIISManager::OldestAdded, DIISManager::InCore);
    size_t nmo2 = nmo_ * nmo_;
    size_t nmo4 = nmo_ * nmo_ * nmo_ * nmo_;

    Vector diis_error("De",3 * nmo4 + 2 * nmo2);
    Vector diis_var("Dv",3 * nmo4 + 2 * nmo2);
    diis_manager.set_error_vector_size(1,DIISEntry::Vector,&diis_error);
    diis_manager.set_vector_size(1,DIISEntry::Vector,&diis_var);
    fprintf(outfile,"\n\n  norm(S2_) = %15e",S2.norm(2.0));
    compute_hbar();
    while(!converged){
        fprintf(outfile,"\n  Updating the S amplitudes...");
        fflush(outfile);
        update_S1();
        update_S2();
        fprintf(outfile," done.");

//        diis_manager.add_entry(2,&diis_error,&diis_var);
//        if (cycle > max_diis_vectors){
//            if (cycle % max_diis_vectors == 2){
//                fprintf(outfile,"\n\n  Performing DIIS extrapolation\n");
//                diis_manager.extrapolate(1,&diis_var);
//            }
//        }

        fprintf(outfile,"\n  Compute recursive single commutator...");
        fflush(outfile);
        double energy = compute_hbar();
        fprintf(outfile," done.");
        fflush(outfile);

        fprintf(outfile,"\n  --------------------------------------------");
        fprintf(outfile,"\n  nExc           |S|                  |R|");
        fprintf(outfile,"\n  --------------------------------------------");
        fprintf(outfile,"\n    1     %15e      %15e",S1.norm(),0.0);
        fprintf(outfile,"\n    2     %15e      %15e",S2.norm(),0.0);
        fprintf(outfile,"\n  --------------------------------------------");
        double delta_energy = energy-old_energy;
        old_energy = energy;
        fprintf(outfile,"\n  @CC %4d %25.15f %25.15f",cycle,energy,delta_energy);

        if(fabs(delta_energy) < options_.get_double("E_CONVERGENCE")){
            converged = true;
        }

        if(cycle > options_.get_int("MAXITER")){
            fprintf(outfile,"\n\n\tThe calculation did not converge in %d cycles\n\tQuitting PSIMRCC\n",options_.get_int("MAXITER"));
            fflush(outfile);
//            exit(1);
            converged = true;
//            break;
        }
        fflush(outfile);
        cycle++;

        fprintf(outfile,"\n  NEXT CYCLE");
        fflush(outfile);
    }
    fprintf(outfile,"\n\n      * CCSD-BCH total energy      = %25.15f",old_energy);
    // Set some environment variables
    Process::environment.globals["CURRENT ENERGY"] = old_energy;
    Process::environment.globals["CTSD ENERGY"] = old_energy;
    Process::environment.globals["LCTSD ENERGY"] = old_energy;
    return old_energy;
}

double TensorSRG::compute_hbar()
{
    fprintf(outfile,"\n\n  Computing the similarity-transformed Hamiltonian");
    fprintf(outfile,"\n  -----------------------------------------------------------------");
    fprintf(outfile,"\n  nComm           C0                 |C1|                  |C2|" );
    fprintf(outfile,"\n  -----------------------------------------------------------------");

    // Initialize Hbar and O with the normal ordered Hamiltonian
    Hbar0 = reference_energy();

    Hbar1["pq"] = F["pq"];
    Hbar1["PQ"] = F["PQ"];
    Hbar2["pqrs"] = V["pqrs"];
    Hbar2["pQrS"] = V["pQrS"];
    Hbar2["PQRS"] = V["PQRS"];

    O1["pq"] = F["pq"];
    O1["PQ"] = F["PQ"];
    O2["pqrs"] = V["pqrs"];
    O2["pQrS"] = V["pQrS"];
    O2["PQRS"] = V["PQRS"];

//    double normHbar1 = Hbar1a.norm() + Hbar1b.norm();
    fprintf(outfile,"\n  %2d %20.12f %20e %20e",0,Hbar0,Hbar1.norm(),Hbar2.norm());

    int maxn = options_.get_int("SRG_RSC_NCOMM");
    double ct_threshold = options_.get_double("SRG_RSC_THRESHOLD");
    for (int n = 1; n <= maxn; ++n) {
        double factor = 1.0 / static_cast<double>(n);

        double C0 = 0;
        C1.zero();
        C2.zero();

        if (options_.get_str("SRG_COMM") == "STANDARD"){
            commutator_A_B_C(factor,O1,O2,S1,S2,C0,C1,C2);
        }else if (options_.get_str("SRG_COMM") == "FO"){
            commutator_A_B_C_fourth_order(factor,O1,O2,S1,S2,C0,C1,C2);
        }

        // Hbar += C
        Hbar0 += C0;
        Hbar1["pq"] += C1["pq"];
        Hbar1["PQ"] += C1["PQ"];
        Hbar2["pqrs"] += C2["pqrs"];
        Hbar2["pQrS"] += C2["pQrS"];
        Hbar2["PQRS"] += C2["PQRS"];

        // O = C
        O1["pq"] = C1["pq"];
        O1["PQ"] = C1["PQ"];
        O2["pqrs"] = C2["pqrs"];
        O2["pQrS"] = C2["pQrS"];
        O2["PQRS"] = C2["PQRS"];

        // Check |C|
        double norm_C1 = C1.norm();
        double norm_C2 = C2.norm();
        double norm_Hb1 = Hbar1.norm();
        double norm_Hb2 = Hbar2.norm();
        fprintf(outfile,"\n  %2d %20.12f %20e %20e %20e %20e",n,C0,norm_C1,norm_C2,norm_Hb1,norm_Hb2);
        fflush(outfile);
        if (std::sqrt(norm_C2 * norm_C2 + norm_C1 * norm_C1) < ct_threshold){
            break;
        }
    }
    fprintf(outfile,"\n  -----------------------------------------------------------------");
    fflush(outfile);
    return Hbar0;
}


void TensorSRG::update_S1()
{
    S1["ia"] += Hbar1["ia"] / D1["ia"];
    S1["IA"] += Hbar1["IA"] / D1["IA"];
}

void TensorSRG::update_S2()
{
    S2["ijab"] += Hbar2["ijab"] / D2["ijab"];
    S2["iJaB"] += Hbar2["iJaB"] / D2["iJaB"];
    S2["IJAB"] += Hbar2["IJAB"] / D2["IJAB"];
}

}} // EndNamespaces

//// Fill in the one-electron operator (H)
//Ha.fill_one_electron([&](size_t p,size_t q){return ints_->oei_a(p,q);});
//Hb.fill_one_electron([&](size_t p,size_t q){return ints_->oei_b(p,q);});
//// Fill in the two-electron operator (V)
//Vaa.fill_two_electron([&](size_t p,size_t q,size_t r,size_t s){return ints_->aptei_aa(p,q,r,s);});
//Vab.fill_two_electron([&](size_t p,size_t q,size_t r,size_t s){return ints_->aptei_ab(p,q,r,s);});
//Vbb.fill_two_electron([&](size_t p,size_t q,size_t r,size_t s){return ints_->aptei_bb(p,q,r,s);});

//G1a.fill_one_electron([&](size_t p,size_t q){return (p == q ? 1.0 : 0.0);});
//G1b.fill_one_electron([&](size_t p,size_t q){return (p == q ? 1.0 : 0.0);});

//// Form the Fock matrix
//Fa["pq"]  = Ha["pq"];
//Fa["pq"] += Vaa["prqs"] * G1a["sr"];
//Fa["pq"] += Vab["pRqS"] * G1b["SR"];

//Fb["PQ"]  = Hb["PQ"];
//Fb["PQ"] += Vab["rPsQ"] * G1a["sr"];
//Fb["PQ"] += Vbb["PRQS"] * G1b["SR"];

//Tensor& Fa_oo = *Fa.block("oo");
//Tensor& Fa_vv = *Fa.block("vv");
//Tensor& Fb_OO = *Fb.block("OO");
//Tensor& Fb_VV = *Fb.block("VV");

//D2aa.fill_two_electron(
//            [&](size_t p,size_t q,size_t r,size_t s){
//    size_t pp = mos_to_aocc[p];
//    size_t qq = mos_to_aocc[q];
//    size_t rr = mos_to_avir[r];
//    size_t ss = mos_to_avir[s];
//    return Fa_oo(pp,pp) + Fa_oo(qq,qq) - Fa_vv(rr,rr) - Fa_vv(ss,ss);});
//D2ab.fill_two_electron(
//            [&](size_t p,size_t q,size_t r,size_t s){
//    size_t pp = mos_to_aocc[p];
//    size_t qq = mos_to_bocc[q];
//    size_t rr = mos_to_avir[r];
//    size_t ss = mos_to_bvir[s];
//    return Fa_oo(pp,pp) + Fb_OO(qq,qq) - Fa_vv(rr,rr) - Fb_VV(ss,ss);});
//D2bb.fill_two_electron(
//            [&](size_t p,size_t q,size_t r,size_t s){
//    size_t pp = mos_to_bocc[p];
//    size_t qq = mos_to_bocc[q];
//    size_t rr = mos_to_bvir[r];
//    size_t ss = mos_to_bvir[s];
//    return Fb_OO(pp,pp) + Fb_OO(qq,qq) - Fb_VV(rr,rr) - Fb_VV(ss,ss);});




//std::vector<size_t> a_occ_mos;
//std::vector<size_t> b_occ_mos;
//std::vector<size_t> a_vir_mos;
//std::vector<size_t> b_vir_mos;

//std::map<size_t,size_t> mos_to_aocc;
//std::map<size_t,size_t> mos_to_bocc;
//std::map<size_t,size_t> mos_to_avir;
//std::map<size_t,size_t> mos_to_bvir;

//for (int h = 0, p = 0; h < nirrep_; ++h){
//    for (int i = 0; i < doccpi_[h]; ++i,++p){
//        a_occ_mos.push_back(p);
//        b_occ_mos.push_back(p);
//        mos_to_aocc[p] = a_occ_mos.size()-1;
//        mos_to_bocc[p] = a_occ_mos.size()-1;
//    }
//    for (int i = 0; i < soccpi_[h]; ++i,++p){
//        a_occ_mos.push_back(p);
//        b_vir_mos.push_back(p);
//    }
//    for (int a = 0; a < nmopi_[h] - doccpi_[h] - soccpi_[h]; ++a,++p){
//        a_vir_mos.push_back(p);
//        b_vir_mos.push_back(p);
//    }
//}

//for (size_t p = 0; p < a_occ_mos.size(); ++p) mos_to_aocc[a_occ_mos[p]] = p;
//for (size_t p = 0; p < b_occ_mos.size(); ++p) mos_to_bocc[b_occ_mos[p]] = p;
//for (size_t p = 0; p < a_vir_mos.size(); ++p) mos_to_avir[a_vir_mos[p]] = p;
//for (size_t p = 0; p < b_vir_mos.size(); ++p) mos_to_bvir[b_vir_mos[p]] = p;

//size_t naocc = a_occ_mos.size();
//size_t nbocc = b_occ_mos.size();
//size_t navir = a_vir_mos.size();
//size_t nbvir = b_vir_mos.size();

//BlockedTensor::add_primitive_mo_space("o","ijkl",a_occ_mos);
//BlockedTensor::add_primitive_mo_space("O","IJKL",b_occ_mos);
//BlockedTensor::add_primitive_mo_space("v","abcd",a_vir_mos);
//BlockedTensor::add_primitive_mo_space("V","ABCD",b_vir_mos);
//BlockedTensor::add_composite_mo_space("i","pqrstu",{"o","v"});
//BlockedTensor::add_composite_mo_space("I","PQRSTU",{"O","V"});
//BlockedTensor::print_mo_spaces();















