#include <cmath>

#include <boost/numeric/odeint.hpp>

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
    fprintf(outfile,"\n\n      --------------------------------------");
    fprintf(outfile,"\n          Similarity Renormalization Group");
    fprintf(outfile,"\n                tensor-based code");
    fprintf(outfile,"\n");
    fprintf(outfile,"\n                Version 0.2.0");
    fprintf(outfile,"\n");
    fprintf(outfile,"\n       written by Francesco A. Evangelista");
    fprintf(outfile,"\n      --------------------------------------\n");
    fprintf(outfile,"\n      Debug level = %d",debug_);
    fprintf(outfile,"\n      Print level = %d\n",print_);
    fflush(outfile);

    BlockedTensor::print_mo_spaces();

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
    fprintf(outfile,"\n\n    SCF energy                            = %20.15f",ref_energy);
    fprintf(outfile,"\n    SRG-PT2 correlation energy            = %20.15f",mp2_correlation_energy);
    fprintf(outfile,"\n  * SRG-PT2 total energy                  = %20.15f\n",ref_energy + mp2_correlation_energy);

//    fprintf(outfile,"\n\n    SCF energy                            = %20.15f",E0_);
//    fprintf(outfile,"\n\n    SCF energy                            = %20.15f",E0_);
//    fprintf(outfile,"\n    MP2 correlation energy                = %20.15f",mp2_correlation_energy);
//    fprintf(outfile,"\n  * MP2 total energy                      = %20.15f\n",E0_ + mp2_correlation_energy);
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
    fprintf(outfile,"\n\n    SCF energy                            = %20.15f",ref_energy);
    fprintf(outfile,"\n    SRG-PT2 correlation energy            = %20.15f",mp2_correlation_energy);
    fprintf(outfile,"\n  * SRG-PT2 total energy                  = %20.15f\n",ref_energy + mp2_correlation_energy);
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

}} // EndNamespaces
