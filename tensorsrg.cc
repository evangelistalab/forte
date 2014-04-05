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
    fprintf(outfile,"\n                Version 0.1.0");
    fprintf(outfile,"\n");
    fprintf(outfile,"\n       written by Francesco A. Evangelista");
    fprintf(outfile,"\n      --------------------------------------\n");

    fprintf(outfile,"\n      Debug level = %d\n",debug_);
    fprintf(outfile,"\n      Print level = %d\n",print_);
    fflush(outfile);

    BlockedTensor::print_mo_spaces();

    S1.resize_spin_components("S1","ov");
    S2.resize_spin_components("S2","oovv");
    R1.resize_spin_components("R1","ov");
    R2.resize_spin_components("R2","oovv");
    Hbar1.resize_spin_components("Hbar1","ii");
    Hbar2.resize_spin_components("Hbar2","iiii");
    O1.resize_spin_components("O1","ii");
    O2.resize_spin_components("O2","iiii");
    C1.resize_spin_components("C1","ii");
    C2.resize_spin_components("C2","iiii");
    I_ioiv.resize_spin_components("C2","ioiv");
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
    fprintf(outfile,"\n    MP2 correlation energy                = %20.15f",mp2_correlation_energy);
    fprintf(outfile,"\n  * MP2 total energy                      = %20.15f\n",ref_energy + mp2_correlation_energy);
    return ref_energy + mp2_correlation_energy;
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
    print_timings();
    return 0.0;
}

}} // EndNamespaces
