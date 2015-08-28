#include <algorithm>
#include <vector>
#include <map>
#include <cctype>
#include <boost/format.hpp>
#include <boost/timer.hpp>

#include <libdiis/diismanager.h>

#include "helpers.h"
#include "mrdsrg.h"

namespace psi{ namespace libadaptive{

void MRDSRG::compute_hbar(){
    if (print_ > 2){
        outfile->Printf("\n\n  ==> Computing the DSRG Transformed Hamiltonian <==\n");
    }

    // copy bare Hamiltonian to Hbar
    Hbar0 = 0.0;
    Hbar1["pq"] = F["pq"];
    Hbar1["PQ"] = F["PQ"];
    Hbar2["pqrs"] = V["pqrs"];
    Hbar2["pQrS"] = V["pQrS"];
    Hbar2["PQRS"] = V["PQRS"];

    // temporary Hamiltonian used in every iteration
    O1["pq"] = F["pq"];
    O1["PQ"] = F["PQ"];
    O2["pqrs"] = V["pqrs"];
    O2["pQrS"] = V["pQrS"];
    O2["PQRS"] = V["PQRS"];

    // iterator variables
    bool converged = false;
    int maxn = options_.get_int("SRG_RSC_NCOMM");
    double ct_threshold = options_.get_double("SRG_RSC_THRESHOLD");

    // compute Hbar recursively
    for(int n = 1; n <= maxn; ++n){
        // prefactor before n-nested commutator
        double factor = 1.0 / n;

        // Compute the commutator C = 1/n [O, T]
        double C0 = 0.0;
        C1.zero();
        C2.zero();

        // printing level
        if(print_ > 2){
            std::string dash(38, '-');
            outfile->Printf("\n    %s", dash.c_str());
        }

        // zero-body
        H1_T1_C0(O1,T1,factor,C0);
        H1_T2_C0(O1,T2,factor,C0);
        H2_T1_C0(O2,T1,factor,C0);
        H2_T2_C0(O2,T2,factor,C0);
        // one-body
        H1_T1_C1(O1,T1,factor,C1);
        H1_T2_C1(O1,T2,factor,C1);
        H2_T1_C1(O2,T1,factor,C1);
        H2_T2_C1(O2,T2,factor,C1);
        // two-body
        H1_T2_C2(O1,T2,factor,C2);
        H2_T1_C2(O2,T1,factor,C2);
        H2_T2_C2(O2,T2,factor,C2);

        // printing level
        if(print_ > 2){
            std::string dash(38, '-');
            outfile->Printf("\n    %s\n", dash.c_str());
        }

        // [H, A] = [H, T] + [H, T]^dagger
        C0 *= 2.0;
        O1["pq"]  = C1["pq"];
        O1["PQ"]  = C1["PQ"];
        C1["pq"] += O1["qp"];
        C1["PQ"] += O1["QP"];
        O2["pqrs"]  = C2["pqrs"];
        O2["pQrS"]  = C2["pQrS"];
        O2["PQRS"]  = C2["PQRS"];
        C2["pqrs"] += O2["rspq"];
        C2["pQrS"] += O2["rSpQ"];
        C2["PQRS"] += O2["RSPQ"];

        // Hbar += C
        Hbar0 += C0;
        Hbar1["pq"] += C1["pq"];
        Hbar1["PQ"] += C1["PQ"];
        Hbar2["pqrs"] += C2["pqrs"];
        Hbar2["pQrS"] += C2["pQrS"];
        Hbar2["PQRS"] += C2["PQRS"];

        // copy C to O for next level commutator
        O1["pq"] = C1["pq"];
        O1["PQ"] = C1["PQ"];
        O2["pqrs"] = C2["pqrs"];
        O2["pQrS"] = C2["pQrS"];
        O2["PQRS"] = C2["PQRS"];

        // test convergence of C
        double norm_C1 = C1.norm();
        double norm_C2 = C2.norm();
        if (std::sqrt(norm_C2 * norm_C2 + norm_C1 * norm_C1) < ct_threshold){
            converged = true;
            break;
        }
    }
    if(!converged){
        outfile->Printf("\n    Warning! Hbar is not converged in %3d-nested commutators!", maxn);
        outfile->Printf("\n    Please increase SRG_RSC_NCOMM.");
        outfile->Flush();
    }
}

double MRDSRG::compute_energy_ldsrg2(){

    // print title
    outfile->Printf("\n\n  ==> Computing MR-LDSRG(2) Energy <==\n");
    outfile->Printf("\n    Reference:");
    outfile->Printf("\n      J. Chem. Phys. (in preparation)\n");
    std::string indent(4, ' ');
    std::string dash(99, '-');
    std::string title;
    title += indent + str(boost::format("%5c  %=27s  %=21s  %=21s  %=17s\n")
                          % ' ' % "Energy (a.u.)" % "Non-Diagonal Norm" % "Amplitude RMS" % "Timings (s)");
    title += indent + std::string (7, ' ') + std::string (27, '-') + "  " + std::string (21, '-')
            + "  " + std::string (21, '-') + "  " + std::string (17, '-') + "\n";
    title += indent + str(boost::format("%5s  %=16s %=10s  %=10s %=10s  %=10s %=10s  %=8s %=8s\n")
                          % "Iter." % "Corr." % "Delta" % "Hbar1" % "Hbar2" % "T1" % "T2" %"Hbar" % "Amp.");
    title += indent + dash;
    outfile->Printf("\n%s", title.c_str());

    // figure out off-diagonal block labels for Hbar1
    std::vector<std::string> blocks1;
    blocks1.push_back(acore_label + aactv_label);
    blocks1.push_back(acore_label + avirt_label);
    blocks1.push_back(aactv_label + avirt_label);
    blocks1.push_back(bcore_label + bactv_label);
    blocks1.push_back(bcore_label + bvirt_label);
    blocks1.push_back(bactv_label + bvirt_label);

    // figure out off-diagonal block labels for Hbar2
    std::vector<std::string> blocks2;
    std::vector<std::string> hole, particle;
    hole.push_back(acore_label); hole.push_back(aactv_label);
    particle.push_back(aactv_label); particle.push_back(avirt_label);
    for(auto& idx0: hole){
        for(auto& idx1: hole){
            for(auto& idx2: particle){
                for(auto& idx3: particle){
                    if(idx0 == aactv_label && idx1 == aactv_label &&
                            idx2 == aactv_label && idx3 == aactv_label){
                        continue;
                    }
                    std::string index;
                    index = idx0 + idx1 + idx2 + idx3;
                    blocks2.push_back(index);
                    index[1] = toupper(index[1]);
                    index[3] = toupper(index[3]);
                    blocks2.push_back(index);
                    index[0] = toupper(index[0]);
                    index[2] = toupper(index[2]);
                    blocks2.push_back(index);
                }
            }
        }
    }

    // iteration variables
    int cycle = 0, maxiter = options_.get_int("MAXITER");
    double e_conv = options_.get_double("E_CONVERGENCE");
    double r_conv = options_.get_double("R_CONVERGENCE");
    bool converged = false;
    double Ecorr = 0.0;
    Hbar1 = BTF->build(tensor_type_,"Hbar1",spin_cases({"gg"}));
    Hbar2 = BTF->build(tensor_type_,"Hbar2",spin_cases({"gggg"}));
    O1 = BTF->build(tensor_type_,"O1",spin_cases({"gg"}));
    O2 = BTF->build(tensor_type_,"O2",spin_cases({"gggg"}));
    C1 = BTF->build(tensor_type_,"C1",spin_cases({"gg"}));
    C2 = BTF->build(tensor_type_,"C2",spin_cases({"gggg"}));
    DT1 = BTF->build(tensor_type_,"DT1",spin_cases({"hp"}));
    DT2 = BTF->build(tensor_type_,"DT2",spin_cases({"hhpp"}));

    // setup DIIS
    std::shared_ptr<DIISManager> diis_manager;
    int max_diis_vectors = options_.get_int("DIIS_MAX_VECS");
    if (max_diis_vectors > 0){
        diis_manager = std::shared_ptr<DIISManager>(new DIISManager(max_diis_vectors, "LDSRG2 DIIS vector", DIISManager::OldestAdded, DIISManager::InCore));
        diis_manager->set_error_vector_size(51,
                                           DIISEntry::Pointer,DT1.block("cv").numel(),
                                           DIISEntry::Pointer,DT1.block("CV").numel(),
                                           DIISEntry::Pointer,DT1.block("ca").numel(),
                                           DIISEntry::Pointer,DT1.block("CA").numel(),
                                           DIISEntry::Pointer,DT1.block("av").numel(),
                                           DIISEntry::Pointer,DT1.block("AV").numel(),
                                           DIISEntry::Pointer,DT2.block("ccaa").numel(),
                                           DIISEntry::Pointer,DT2.block("ccav").numel(),
                                           DIISEntry::Pointer,DT2.block("ccva").numel(),
                                           DIISEntry::Pointer,DT2.block("ccvv").numel(),
                                           DIISEntry::Pointer,DT2.block("caaa").numel(),
                                           DIISEntry::Pointer,DT2.block("caav").numel(),
                                           DIISEntry::Pointer,DT2.block("cava").numel(),
                                           DIISEntry::Pointer,DT2.block("cavv").numel(),
                                           DIISEntry::Pointer,DT2.block("acaa").numel(),
                                           DIISEntry::Pointer,DT2.block("acav").numel(),
                                           DIISEntry::Pointer,DT2.block("acva").numel(),
                                           DIISEntry::Pointer,DT2.block("acvv").numel(),
                                           DIISEntry::Pointer,DT2.block("aaav").numel(),
                                           DIISEntry::Pointer,DT2.block("aava").numel(),
                                           DIISEntry::Pointer,DT2.block("aavv").numel(),
                                           DIISEntry::Pointer,DT2.block("cCaA").numel(),
                                           DIISEntry::Pointer,DT2.block("cCaV").numel(),
                                           DIISEntry::Pointer,DT2.block("cCvA").numel(),
                                           DIISEntry::Pointer,DT2.block("cCvV").numel(),
                                           DIISEntry::Pointer,DT2.block("cAaA").numel(),
                                           DIISEntry::Pointer,DT2.block("cAaV").numel(),
                                           DIISEntry::Pointer,DT2.block("cAvA").numel(),
                                           DIISEntry::Pointer,DT2.block("cAvV").numel(),
                                           DIISEntry::Pointer,DT2.block("aCaA").numel(),
                                           DIISEntry::Pointer,DT2.block("aCaV").numel(),
                                           DIISEntry::Pointer,DT2.block("aCvA").numel(),
                                           DIISEntry::Pointer,DT2.block("aCvV").numel(),
                                           DIISEntry::Pointer,DT2.block("aAaV").numel(),
                                           DIISEntry::Pointer,DT2.block("aAvA").numel(),
                                           DIISEntry::Pointer,DT2.block("aAvV").numel(),
                                           DIISEntry::Pointer,DT2.block("CCAA").numel(),
                                           DIISEntry::Pointer,DT2.block("CCAV").numel(),
                                           DIISEntry::Pointer,DT2.block("CCVA").numel(),
                                           DIISEntry::Pointer,DT2.block("CCVV").numel(),
                                           DIISEntry::Pointer,DT2.block("CAAA").numel(),
                                           DIISEntry::Pointer,DT2.block("CAAV").numel(),
                                           DIISEntry::Pointer,DT2.block("CAVA").numel(),
                                           DIISEntry::Pointer,DT2.block("CAVV").numel(),
                                           DIISEntry::Pointer,DT2.block("ACAA").numel(),
                                           DIISEntry::Pointer,DT2.block("ACAV").numel(),
                                           DIISEntry::Pointer,DT2.block("ACVA").numel(),
                                           DIISEntry::Pointer,DT2.block("ACVV").numel(),
                                           DIISEntry::Pointer,DT2.block("AAAV").numel(),
                                           DIISEntry::Pointer,DT2.block("AAVA").numel(),
                                           DIISEntry::Pointer,DT2.block("AAVV").numel());
        diis_manager->set_vector_size(51,
                                     DIISEntry::Pointer,T1.block("cv").numel(),
                                     DIISEntry::Pointer,T1.block("CV").numel(),
                                     DIISEntry::Pointer,T1.block("ca").numel(),
                                     DIISEntry::Pointer,T1.block("CA").numel(),
                                     DIISEntry::Pointer,T1.block("av").numel(),
                                     DIISEntry::Pointer,T1.block("AV").numel(),
                                     DIISEntry::Pointer,T2.block("ccaa").numel(),
                                     DIISEntry::Pointer,T2.block("ccav").numel(),
                                     DIISEntry::Pointer,T2.block("ccva").numel(),
                                     DIISEntry::Pointer,T2.block("ccvv").numel(),
                                     DIISEntry::Pointer,T2.block("caaa").numel(),
                                     DIISEntry::Pointer,T2.block("caav").numel(),
                                     DIISEntry::Pointer,T2.block("cava").numel(),
                                     DIISEntry::Pointer,T2.block("cavv").numel(),
                                     DIISEntry::Pointer,T2.block("acaa").numel(),
                                     DIISEntry::Pointer,T2.block("acav").numel(),
                                     DIISEntry::Pointer,T2.block("acva").numel(),
                                     DIISEntry::Pointer,T2.block("acvv").numel(),
                                     DIISEntry::Pointer,T2.block("aaav").numel(),
                                     DIISEntry::Pointer,T2.block("aava").numel(),
                                     DIISEntry::Pointer,T2.block("aavv").numel(),
                                     DIISEntry::Pointer,T2.block("cCaA").numel(),
                                     DIISEntry::Pointer,T2.block("cCaV").numel(),
                                     DIISEntry::Pointer,T2.block("cCvA").numel(),
                                     DIISEntry::Pointer,T2.block("cCvV").numel(),
                                     DIISEntry::Pointer,T2.block("cAaA").numel(),
                                     DIISEntry::Pointer,T2.block("cAaV").numel(),
                                     DIISEntry::Pointer,T2.block("cAvA").numel(),
                                     DIISEntry::Pointer,T2.block("cAvV").numel(),
                                     DIISEntry::Pointer,T2.block("aCaA").numel(),
                                     DIISEntry::Pointer,T2.block("aCaV").numel(),
                                     DIISEntry::Pointer,T2.block("aCvA").numel(),
                                     DIISEntry::Pointer,T2.block("aCvV").numel(),
                                     DIISEntry::Pointer,T2.block("aAaV").numel(),
                                     DIISEntry::Pointer,T2.block("aAvA").numel(),
                                     DIISEntry::Pointer,T2.block("aAvV").numel(),
                                     DIISEntry::Pointer,T2.block("CCAA").numel(),
                                     DIISEntry::Pointer,T2.block("CCAV").numel(),
                                     DIISEntry::Pointer,T2.block("CCVA").numel(),
                                     DIISEntry::Pointer,T2.block("CCVV").numel(),
                                     DIISEntry::Pointer,T2.block("CAAA").numel(),
                                     DIISEntry::Pointer,T2.block("CAAV").numel(),
                                     DIISEntry::Pointer,T2.block("CAVA").numel(),
                                     DIISEntry::Pointer,T2.block("CAVV").numel(),
                                     DIISEntry::Pointer,T2.block("ACAA").numel(),
                                     DIISEntry::Pointer,T2.block("ACAV").numel(),
                                     DIISEntry::Pointer,T2.block("ACVA").numel(),
                                     DIISEntry::Pointer,T2.block("ACVV").numel(),
                                     DIISEntry::Pointer,T2.block("AAAV").numel(),
                                     DIISEntry::Pointer,T2.block("AAVA").numel(),
                                     DIISEntry::Pointer,T2.block("AAVV").numel());
    }

    // start iteration
    do{
        // compute Hbar
        boost::timer t_hbar;
        compute_hbar();
        double Edelta = Hbar0 - Ecorr;
        Ecorr = Hbar0;
        double time_hbar = t_hbar.elapsed();

        // compute norms of off-diagonal Hbar
        double Hbar1od = Hbar1od_norm(blocks1);
        double Hbar2od = Hbar2od_norm(blocks2);

        // update amplitudes
        boost::timer t_amp;
        update_t2();
        update_t1();
        double time_amp = t_amp.elapsed();

        // DIIS amplitudes
        if(diis_manager){
            diis_manager->add_entry(102,
                    &(DT1.block("cv").data()[0]),
                    &(DT1.block("CV").data()[0]),
                    &(DT1.block("ca").data()[0]),
                    &(DT1.block("CA").data()[0]),
                    &(DT1.block("av").data()[0]),
                    &(DT1.block("AV").data()[0]),
                    &(DT2.block("ccaa").data()[0]),
                    &(DT2.block("ccav").data()[0]),
                    &(DT2.block("ccva").data()[0]),
                    &(DT2.block("ccvv").data()[0]),
                    &(DT2.block("caaa").data()[0]),
                    &(DT2.block("caav").data()[0]),
                    &(DT2.block("cava").data()[0]),
                    &(DT2.block("cavv").data()[0]),
                    &(DT2.block("acaa").data()[0]),
                    &(DT2.block("acav").data()[0]),
                    &(DT2.block("acva").data()[0]),
                    &(DT2.block("acvv").data()[0]),
                    &(DT2.block("aaav").data()[0]),
                    &(DT2.block("aava").data()[0]),
                    &(DT2.block("aavv").data()[0]),
                    &(DT2.block("cCaA").data()[0]),
                    &(DT2.block("cCaV").data()[0]),
                    &(DT2.block("cCvA").data()[0]),
                    &(DT2.block("cCvV").data()[0]),
                    &(DT2.block("cAaA").data()[0]),
                    &(DT2.block("cAaV").data()[0]),
                    &(DT2.block("cAvA").data()[0]),
                    &(DT2.block("cAvV").data()[0]),
                    &(DT2.block("aCaA").data()[0]),
                    &(DT2.block("aCaV").data()[0]),
                    &(DT2.block("aCvA").data()[0]),
                    &(DT2.block("aCvV").data()[0]),
                    &(DT2.block("aAaV").data()[0]),
                    &(DT2.block("aAvA").data()[0]),
                    &(DT2.block("aAvV").data()[0]),
                    &(DT2.block("CCAA").data()[0]),
                    &(DT2.block("CCAV").data()[0]),
                    &(DT2.block("CCVA").data()[0]),
                    &(DT2.block("CCVV").data()[0]),
                    &(DT2.block("CAAA").data()[0]),
                    &(DT2.block("CAAV").data()[0]),
                    &(DT2.block("CAVA").data()[0]),
                    &(DT2.block("CAVV").data()[0]),
                    &(DT2.block("ACAA").data()[0]),
                    &(DT2.block("ACAV").data()[0]),
                    &(DT2.block("ACVA").data()[0]),
                    &(DT2.block("ACVV").data()[0]),
                    &(DT2.block("AAAV").data()[0]),
                    &(DT2.block("AAVA").data()[0]),
                    &(DT2.block("AAVV").data()[0]),
                    &(T1.block("cv").data()[0]),
                    &(T1.block("CV").data()[0]),
                    &(T1.block("ca").data()[0]),
                    &(T1.block("CA").data()[0]),
                    &(T1.block("av").data()[0]),
                    &(T1.block("AV").data()[0]),
                    &(T2.block("ccaa").data()[0]),
                    &(T2.block("ccav").data()[0]),
                    &(T2.block("ccva").data()[0]),
                    &(T2.block("ccvv").data()[0]),
                    &(T2.block("caaa").data()[0]),
                    &(T2.block("caav").data()[0]),
                    &(T2.block("cava").data()[0]),
                    &(T2.block("cavv").data()[0]),
                    &(T2.block("acaa").data()[0]),
                    &(T2.block("acav").data()[0]),
                    &(T2.block("acva").data()[0]),
                    &(T2.block("acvv").data()[0]),
                    &(T2.block("aaav").data()[0]),
                    &(T2.block("aava").data()[0]),
                    &(T2.block("aavv").data()[0]),
                    &(T2.block("cCaA").data()[0]),
                    &(T2.block("cCaV").data()[0]),
                    &(T2.block("cCvA").data()[0]),
                    &(T2.block("cCvV").data()[0]),
                    &(T2.block("cAaA").data()[0]),
                    &(T2.block("cAaV").data()[0]),
                    &(T2.block("cAvA").data()[0]),
                    &(T2.block("cAvV").data()[0]),
                    &(T2.block("aCaA").data()[0]),
                    &(T2.block("aCaV").data()[0]),
                    &(T2.block("aCvA").data()[0]),
                    &(T2.block("aCvV").data()[0]),
                    &(T2.block("aAaV").data()[0]),
                    &(T2.block("aAvA").data()[0]),
                    &(T2.block("aAvV").data()[0]),
                    &(T2.block("CCAA").data()[0]),
                    &(T2.block("CCAV").data()[0]),
                    &(T2.block("CCVA").data()[0]),
                    &(T2.block("CCVV").data()[0]),
                    &(T2.block("CAAA").data()[0]),
                    &(T2.block("CAAV").data()[0]),
                    &(T2.block("CAVA").data()[0]),
                    &(T2.block("CAVV").data()[0]),
                    &(T2.block("ACAA").data()[0]),
                    &(T2.block("ACAV").data()[0]),
                    &(T2.block("ACVA").data()[0]),
                    &(T2.block("ACVV").data()[0]),
                    &(T2.block("AAAV").data()[0]),
                    &(T2.block("AAVA").data()[0]),
                    &(T2.block("AAVV").data()[0]));
            if (cycle > max_diis_vectors){
                if (cycle % max_diis_vectors == 2){
                    outfile->Printf(" -> DIIS");
                    outfile->Flush();
                    diis_manager->extrapolate(51,
                            &(T1.block("cv").data()[0]),
                            &(T1.block("CV").data()[0]),
                            &(T1.block("ca").data()[0]),
                            &(T1.block("CA").data()[0]),
                            &(T1.block("av").data()[0]),
                            &(T1.block("AV").data()[0]),
                            &(T2.block("ccaa").data()[0]),
                            &(T2.block("ccav").data()[0]),
                            &(T2.block("ccva").data()[0]),
                            &(T2.block("ccvv").data()[0]),
                            &(T2.block("caaa").data()[0]),
                            &(T2.block("caav").data()[0]),
                            &(T2.block("cava").data()[0]),
                            &(T2.block("cavv").data()[0]),
                            &(T2.block("acaa").data()[0]),
                            &(T2.block("acav").data()[0]),
                            &(T2.block("acva").data()[0]),
                            &(T2.block("acvv").data()[0]),
                            &(T2.block("aaav").data()[0]),
                            &(T2.block("aava").data()[0]),
                            &(T2.block("aavv").data()[0]),
                            &(T2.block("cCaA").data()[0]),
                            &(T2.block("cCaV").data()[0]),
                            &(T2.block("cCvA").data()[0]),
                            &(T2.block("cCvV").data()[0]),
                            &(T2.block("cAaA").data()[0]),
                            &(T2.block("cAaV").data()[0]),
                            &(T2.block("cAvA").data()[0]),
                            &(T2.block("cAvV").data()[0]),
                            &(T2.block("aCaA").data()[0]),
                            &(T2.block("aCaV").data()[0]),
                            &(T2.block("aCvA").data()[0]),
                            &(T2.block("aCvV").data()[0]),
                            &(T2.block("aAaV").data()[0]),
                            &(T2.block("aAvA").data()[0]),
                            &(T2.block("aAvV").data()[0]),
                            &(T2.block("CCAA").data()[0]),
                            &(T2.block("CCAV").data()[0]),
                            &(T2.block("CCVA").data()[0]),
                            &(T2.block("CCVV").data()[0]),
                            &(T2.block("CAAA").data()[0]),
                            &(T2.block("CAAV").data()[0]),
                            &(T2.block("CAVA").data()[0]),
                            &(T2.block("CAVV").data()[0]),
                            &(T2.block("ACAA").data()[0]),
                            &(T2.block("ACAV").data()[0]),
                            &(T2.block("ACVA").data()[0]),
                            &(T2.block("ACVV").data()[0]),
                            &(T2.block("AAAV").data()[0]),
                            &(T2.block("AAVA").data()[0]),
                            &(T2.block("AAVV").data()[0]));
                }
            }
        }

        // printing
        outfile->Printf("\n    %5d  %16.12f %10.3e  %10.3e %10.3e  %10.3e %10.3e  %8.3f %8.3f",
                        cycle, Ecorr, Edelta, Hbar1od, Hbar2od, T1rms, T2rms, time_hbar, time_amp);

        // test convergence
        double rms = T1rms > T2rms ? T1rms : T2rms;
        if(fabs(Edelta) < e_conv && rms < r_conv){
            converged = true;
        }
        if(cycle > maxiter){
            outfile->Printf("\n\n    The computation does not converge in %d iterations!\tQuitting.\n", maxiter);
            converged = true;
        }
        outfile->Flush();
        ++cycle;
    } while (!converged);

    // print summary
    outfile->Printf("\n    %s", dash.c_str());
    outfile->Printf("\n\n  ==> MR-LDSRG(2) Energy Summary <==\n");
    std::vector<std::pair<std::string,double>> energy;
    energy.push_back({"E0 (reference)", Eref});
    energy.push_back({"MR-LDSRG(2) correlation energy", Ecorr});
    energy.push_back({"MR-LDSRG(2) total energy", Eref + Ecorr});
    for (auto& str_dim: energy){
        outfile->Printf("\n    %-30s = %23.15f", str_dim.first.c_str(), str_dim.second);
    }

    // analyze converged amplitudes
    analyze_amplitudes("Final ",T1,T2);

    return Ecorr;
}

double MRDSRG::Hbar1od_norm(const std::vector<std::string> &blocks){
    double norm = 0.0;

    for(auto& block: blocks){
        double norm_block = Hbar1.block(block).norm();
        norm += 2.0 * norm_block * norm_block;
    }
    norm = std::sqrt(norm);

    return norm;
}

double MRDSRG::Hbar2od_norm(const std::vector<std::string> &blocks){
    double norm = 0.0;

    for(auto& block: blocks){
        double norm_block = Hbar2.block(block).norm();
        norm += 2.0 * norm_block * norm_block;
    }
    norm = std::sqrt(norm);

    return norm;
}

}}
