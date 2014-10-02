#include <cmath>

#include <boost/numeric/odeint.hpp>

#include <libmints/molecule.h>
#include <libmints/wavefunction.h>

#include "tensorsrg.h"

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
    outfile->Printf("\n\n      --------------------------------------");
    outfile->Printf("\n          Similarity Renormalization Group");
    outfile->Printf("\n                tensor-based code");
    outfile->Printf("\n");
    outfile->Printf("\n                Version 0.2.2");
    outfile->Printf("\n");
    outfile->Printf("\n       written by Francesco A. Evangelista");
    outfile->Printf("\n      --------------------------------------\n");
    outfile->Printf("\n      Debug level = %d",debug_);
    outfile->Printf("\n      Print level = %d\n",print_);
    fflush(outfile);

    BlockedTensor::print_mo_spaces();

    outfile->Printf("\n      Energy convergence = %e\n",options_.get_double("E_CONVERGENCE"));

    S1.resize_spin_components("S1","ov");
    S2.resize_spin_components("S2","oovv");
    DS1.resize_spin_components("S1","ov");
    DS2.resize_spin_components("S2","oovv");
    R1.resize_spin_components("R1","ov");
    R2.resize_spin_components("R2","oovv");
    Hbar1.resize_spin_components("Hbar1","ii");
    Hbar2.resize_spin_components("Hbar2","iiii");
    eta1.resize_spin_components("eta1","ii");
    eta2.resize_spin_components("eta2","iiii");
    g1.resize_spin_components("g1","ii");
    g2.resize_spin_components("g2","iiii");
    O1.resize_spin_components("O1","ii");
    O2.resize_spin_components("O2","iiii");
    C1.resize_spin_components("C1","ii");
    C2.resize_spin_components("C2","iiii");
    I_ioiv.resize_spin_components("I_ioiv","ioiv");

    dsrg_power_ = options_.get_double("DSRG_POWER");
}

void TensorSRG::cleanup()
{
    print_timings();
}

double TensorSRG::compute_mp2_guess()
{
    S2["ijab"] = V["ijab"] / D2["ijab"];
    S2["iJaB"] = V["iJaB"] / D2["iJaB"];
    S2["IJAB"] = V["IJAB"] / D2["IJAB"];

    double Eaa = 0.25 * BlockedTensor::dot(S2["ijab"],V["ijab"]);
    double Eab = BlockedTensor::dot(S2["iJaB"],V["iJaB"]);
    double Ebb = 0.25 * BlockedTensor::dot(S2["IJAB"],V["IJAB"]);

    double mp2_correlation_energy = Eaa + Eab + Ebb;
    double ref_energy = reference_energy();
    outfile->Printf("\n\n    SCF energy                            = %20.15f",ref_energy);
    outfile->Printf("\n    SRG-PT2 correlation energy            = %20.15f",mp2_correlation_energy);
    outfile->Printf("\n  * SRG-PT2 total energy                  = %20.15f\n",ref_energy + mp2_correlation_energy);

//    outfile->Printf("\n\n    SCF energy                            = %20.15f",E0_);
//    outfile->Printf("\n\n    SCF energy                            = %20.15f",E0_);
//    outfile->Printf("\n    MP2 correlation energy                = %20.15f",mp2_correlation_energy);
//    outfile->Printf("\n  * MP2 total energy                      = %20.15f\n",E0_ + mp2_correlation_energy);
    return E0_ + mp2_correlation_energy;
}

double TensorSRG::compute_mp2_guess_driven_srg()
{
    double srg_s = options_.get_double("DSRG_S");

    Tensor& Fa_oo = *F.block("oo");
    Tensor& Fa_vv = *F.block("vv");
    Tensor& Fb_OO = *F.block("OO");
    Tensor& Fb_VV = *F.block("VV");
    Tensor& V_oovv = *V.block("oovv");
    Tensor& V_oOvV = *V.block("oOvV");
    Tensor& V_OOVV = *V.block("OOVV");

    S2.zero();
    S2.fill_two_electron_spin([&](size_t p,MOSetSpinType sp,
                              size_t q,MOSetSpinType sq,
                              size_t r,MOSetSpinType sr,
                              size_t s,MOSetSpinType ss){
        if ((sp == Alpha) and (sq == Alpha)){
            size_t pp = mos_to_aocc[p];
            size_t qq = mos_to_aocc[q];
            size_t rr = mos_to_avir[r];
            size_t ss = mos_to_avir[s];
            double denominator = Fa_oo(pp,pp) + Fa_oo(qq,qq) - Fa_vv(rr,rr) - Fa_vv(ss,ss);
            double exp_factor = one_minus_exp_div_x(srg_s,denominator,dsrg_power_);
            return V_oovv(pp,qq,rr,ss) * exp_factor;
        }else if ((sp == Alpha) and (sq == Beta) ){
            size_t pp = mos_to_aocc[p];
            size_t qq = mos_to_bocc[q];
            size_t rr = mos_to_avir[r];
            size_t ss = mos_to_bvir[s];
            double denominator = Fa_oo(pp,pp) + Fb_OO(qq,qq) - Fa_vv(rr,rr) - Fb_VV(ss,ss);
            double exp_factor = one_minus_exp_div_x(srg_s,denominator,dsrg_power_);
            return V_oOvV(pp,qq,rr,ss) * exp_factor;
        }else if ((sp == Beta)  and (sq == Beta) ){
            size_t pp = mos_to_bocc[p];
            size_t qq = mos_to_bocc[q];
            size_t rr = mos_to_bvir[r];
            size_t ss = mos_to_bvir[s];
            double denominator = Fb_OO(pp,pp) + Fb_OO(qq,qq) - Fb_VV(rr,rr) - Fb_VV(ss,ss);
            double exp_factor = one_minus_exp_div_x(srg_s,denominator,dsrg_power_);
            return V_OOVV(pp,qq,rr,ss) * exp_factor;
        }
        return 0.0;
    });

    double Eaa = 0.25 * BlockedTensor::dot(S2["ijab"],V["ijab"]);
    double Eab = BlockedTensor::dot(S2["iJaB"],V["iJaB"]);
    double Ebb = 0.25 * BlockedTensor::dot(S2["IJAB"],V["IJAB"]);

    double mp2_correlation_energy = Eaa + Eab + Ebb;
    double ref_energy = reference_energy();
    outfile->Printf("\n\n    SCF energy                            = %20.15f",ref_energy);
    outfile->Printf("\n    SRG-PT2 correlation energy            = %20.15f",mp2_correlation_energy);
    outfile->Printf("\n  * SRG-PT2 total energy                  = %20.15f\n",ref_energy + mp2_correlation_energy);
    return ref_energy + mp2_correlation_energy;
}

double one_minus_exp_div_x(double s,double x,double power)
{
    return ((1.0 - std::exp(-s * std::pow(x,power))) / x);
}

double TensorSRG::compute_energy()
{
    if(options_.get_str("SRG_MODE") == "SRG"){
        compute_srg_energy();
    }else if(options_.get_str("SRG_MODE") == "CT"){
        return compute_ct_energy();
    }else if(options_.get_str("SRG_MODE") == "DSRG"){
//        compute_driven_srg_energy();
    }
    return 0.0;
}

void TensorSRG::transfer_integrals()
{
    // Scalar term
    double scalar0 = E0_ + Hbar0;
    double scalar1 = 0.0;
    double scalar2 = 0.0;
    {
        Tensor& Hbar_oo = *Hbar1.block("oo");
        Tensor::iterator it = Hbar_oo.begin();
        Tensor::iterator endit = Hbar_oo.end();
        for (; it != endit; ++it){
            std::vector<size_t>& i = it.address();
            if (i[0] == i[1]){
                scalar1 -= *it;
            }
        }
    }
    {
        Tensor& Hbar_OO = *Hbar1.block("OO");
        Tensor::iterator it = Hbar_OO.begin();
        Tensor::iterator endit = Hbar_OO.end();
        for (; it != endit; ++it){
            std::vector<size_t>& i = it.address();
            if (i[0] == i[1]){
                scalar1 -= *it;
            }
        }
    }
    {
        Tensor& Hbar_oooo = *Hbar2.block("oooo");
        Tensor::iterator it = Hbar_oooo.begin();
        Tensor::iterator endit = Hbar_oooo.end();
        for (; it != endit; ++it){
            std::vector<size_t>& i = it.address();
            if ((i[0] == i[2]) and (i[1] == i[3])){
                scalar2 += 0.5 * (*it);
            }
        }
    }
    {
        Tensor& Hbar_oOoO = *Hbar2.block("oOoO");
        Tensor::iterator it = Hbar_oOoO.begin();
        Tensor::iterator endit = Hbar_oOoO.end();
        for (; it != endit; ++it){
            std::vector<size_t>& i = it.address();
            if ((i[0] == i[2]) and (i[1] == i[3])){
                scalar2 += (*it);
            }
        }
    }
    {
        Tensor& Hbar_OOOO = *Hbar2.block("OOOO");
        Tensor::iterator it = Hbar_OOOO.begin();
        Tensor::iterator endit = Hbar_OOOO.end();
        for (; it != endit; ++it){
            std::vector<size_t>& i = it.address();
            if ((i[0] == i[2]) and (i[1] == i[3])){
                scalar2 += 0.5 * (*it);
            }
        }
    }
    double scalar = scalar0 + scalar1 + scalar2 - molecule()->nuclear_repulsion_energy();
    outfile->Printf("\n  The Hamiltonian electronic scalar term (normal ordered wrt the true vacuum");
    outfile->Printf("\n  E0 = %20.12f",scalar);

    O1["pq"] = Hbar1["pq"];
    O1["PQ"] = Hbar1["PQ"];
    O1["pq"] -= Hbar2["prqs"] * G1["rs"];
    O1["pq"] -= Hbar2["pRqS"] * G1["RS"];
    O1["PQ"] -= Hbar2["rPsQ"] * G1["rs"];
    O1["PQ"] -= Hbar2["PRQS"] * G1["RS"];

    outfile->Printf("\n  Updating all the integrals");
    ints_->set_scalar(scalar);
    O1.iterate_over_elements([&](std::vector<size_t>& m,std::vector<MOSetSpinType>& spin,double& value)
    {
        if ((spin[0] == Alpha) and (spin[1] == Alpha)){
            ints_->set_oei(m[0],m[1],value,true);
        }
        if ((spin[0] == Beta) and (spin[1] == Beta)){
            ints_->set_oei(m[0],m[1],value,false);
        }
    });
    Hbar2.iterate_over_elements([&](std::vector<size_t>& m,std::vector<MOSetSpinType>& spin,double& value){
        if ((spin[0] == Alpha) and (spin[1] == Alpha) and (spin[2] == Alpha) and (spin[3] == Alpha)){
            ints_->set_tei(m[0],m[1],m[2],m[3],value,true,true);
        }
        if ((spin[0] == Alpha) and (spin[1] == Beta) and (spin[2] == Alpha) and (spin[3] == Beta)){
            ints_->set_tei(m[0],m[1],m[2],m[3],value,true,false);
        }
        if ((spin[0] == Beta) and (spin[1] == Beta) and (spin[2] == Beta) and (spin[3] == Beta)){
            ints_->set_tei(m[0],m[1],m[2],m[3],value,false,false);
        }
    });

    // As a test compute the current CT energy
    double Esth = scalar;
    {
        Tensor& Hbar_oo = *O1.block("oo");
        Tensor::iterator it = Hbar_oo.begin();
        Tensor::iterator endit = Hbar_oo.end();
        for (; it != endit; ++it){
            std::vector<size_t>& i = it.address();
            if (i[0] == i[1]){
                Esth += *it;
            }
        }
    }
    {
        Tensor& Hbar_OO = *O1.block("OO");
        Tensor::iterator it = Hbar_OO.begin();
        Tensor::iterator endit = Hbar_OO.end();
        for (; it != endit; ++it){
            std::vector<size_t>& i = it.address();
            if (i[0] == i[1]){
                Esth += *it;
            }
        }
    }
    {
        Tensor& Hbar_oooo = *Hbar2.block("oooo");
        Tensor::iterator it = Hbar_oooo.begin();
        Tensor::iterator endit = Hbar_oooo.end();
        for (; it != endit; ++it){
            std::vector<size_t>& i = it.address();
            if ((i[0] == i[2]) and (i[1] == i[3])){
                Esth += 0.5 * (*it);
            }
        }
    }
    {
        Tensor& Hbar_oOoO = *Hbar2.block("oOoO");
        Tensor::iterator it = Hbar_oOoO.begin();
        Tensor::iterator endit = Hbar_oOoO.end();
        for (; it != endit; ++it){
            std::vector<size_t>& i = it.address();
            if ((i[0] == i[2]) and (i[1] == i[3])){
                Esth += (*it);
            }
        }
    }
    {
        Tensor& Hbar_OOOO = *Hbar2.block("OOOO");
        Tensor::iterator it = Hbar_OOOO.begin();
        Tensor::iterator endit = Hbar_OOOO.end();
        for (; it != endit; ++it){
            std::vector<size_t>& i = it.address();
            if ((i[0] == i[2]) and (i[1] == i[3])){
                Esth += 0.5 * (*it);
            }
        }
    }

    outfile->Printf("\n  <H> = %24.12f",Esth + molecule()->nuclear_repulsion_energy());

    ints_->update_integrals();
    fflush(outfile);
}

}} // EndNamespaces
