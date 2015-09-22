#include <algorithm>
#include <vector>
#include <map>
#include <cctype>
#include <boost/format.hpp>
#include <boost/timer.hpp>

#include <libdiis/diismanager.h>

#include "helpers.h"
#include "mrdsrg.h"

namespace psi{ namespace forte{

void MRDSRG::compute_hbar(){
    if (print_ > 2){
        outfile->Printf("\n\n  ==> Computing the DSRG Transformed Hamiltonian <==\n");
    }

    // copy bare Hamiltonian to Hbar
    Hbar0_ = 0.0;
    Hbar1_["pq"] = F_["pq"];
    Hbar1_["PQ"] = F_["PQ"];
    Hbar2_["pqrs"] = V_["pqrs"];
    Hbar2_["pQrS"] = V_["pQrS"];
    Hbar2_["PQRS"] = V_["PQRS"];

    // temporary Hamiltonian used in every iteration
    O1_["pq"] = F_["pq"];
    O1_["PQ"] = F_["PQ"];
    O2_["pqrs"] = V_["pqrs"];
    O2_["pQrS"] = V_["pQrS"];
    O2_["PQRS"] = V_["PQRS"];

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
        C1_.zero();
        C2_.zero();

        // printing level
        if(print_ > 2){
            std::string dash(38, '-');
            outfile->Printf("\n    %s", dash.c_str());
        }

        // zero-body
        H1_T1_C0(O1_,T1_,factor,C0);
        H1_T2_C0(O1_,T2_,factor,C0);
        H2_T1_C0(O2_,T1_,factor,C0);
        H2_T2_C0(O2_,T2_,factor,C0);
        // one-body
        H1_T1_C1(O1_,T1_,factor,C1_);
        H1_T2_C1(O1_,T2_,factor,C1_);
        H2_T1_C1(O2_,T1_,factor,C1_);
        H2_T2_C1(O2_,T2_,factor,C1_);
        // two-body
        H1_T2_C2(O1_,T2_,factor,C2_);
        H2_T1_C2(O2_,T1_,factor,C2_);
        H2_T2_C2(O2_,T2_,factor,C2_);

        // printing level
        if(print_ > 2){
            std::string dash(38, '-');
            outfile->Printf("\n    %s\n", dash.c_str());
        }

        // [H, A] = [H, T] + [H, T]^dagger
        C0 *= 2.0;
        O1_["pq"]  = C1_["pq"];
        O1_["PQ"]  = C1_["PQ"];
        C1_["pq"] += O1_["qp"];
        C1_["PQ"] += O1_["QP"];
        O2_["pqrs"]  = C2_["pqrs"];
        O2_["pQrS"]  = C2_["pQrS"];
        O2_["PQRS"]  = C2_["PQRS"];
        C2_["pqrs"] += O2_["rspq"];
        C2_["pQrS"] += O2_["rSpQ"];
        C2_["PQRS"] += O2_["RSPQ"];

        // Hbar += C
        Hbar0_ += C0;
        Hbar1_["pq"] += C1_["pq"];
        Hbar1_["PQ"] += C1_["PQ"];
        Hbar2_["pqrs"] += C2_["pqrs"];
        Hbar2_["pQrS"] += C2_["pQrS"];
        Hbar2_["PQRS"] += C2_["PQRS"];

        // copy C to O for next level commutator
        O1_["pq"] = C1_["pq"];
        O1_["PQ"] = C1_["PQ"];
        O2_["pqrs"] = C2_["pqrs"];
        O2_["pQrS"] = C2_["pQrS"];
        O2_["PQRS"] = C2_["PQRS"];

        // test convergence of C
        double norm_C1 = C1_.norm();
        double norm_C2 = C2_.norm();
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
    blocks1.push_back(acore_label_ + aactv_label_);
    blocks1.push_back(acore_label_ + avirt_label_);
    blocks1.push_back(aactv_label_ + avirt_label_);
    blocks1.push_back(bcore_label_ + bactv_label_);
    blocks1.push_back(bcore_label_ + bvirt_label_);
    blocks1.push_back(bactv_label_ + bvirt_label_);

    // figure out off-diagonal block labels for Hbar2
    std::vector<std::string> blocks2;
    std::vector<std::string> hole, particle;
    hole.push_back(acore_label_); hole.push_back(aactv_label_);
    particle.push_back(aactv_label_); particle.push_back(avirt_label_);
    for(auto& idx0: hole){
        for(auto& idx1: hole){
            for(auto& idx2: particle){
                for(auto& idx3: particle){
                    if(idx0 == aactv_label_ && idx1 == aactv_label_ &&
                            idx2 == aactv_label_ && idx3 == aactv_label_){
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
    Hbar1_ = BTF_->build(tensor_type_,"Hbar1",spin_cases({"gg"}));
    Hbar2_ = BTF_->build(tensor_type_,"Hbar2",spin_cases({"gggg"}));
    O1_ = BTF_->build(tensor_type_,"O1",spin_cases({"gg"}));
    O2_ = BTF_->build(tensor_type_,"O2",spin_cases({"gggg"}));
    C1_ = BTF_->build(tensor_type_,"C1",spin_cases({"gg"}));
    C2_ = BTF_->build(tensor_type_,"C2",spin_cases({"gggg"}));
    DT1_ = BTF_->build(tensor_type_,"DT1",spin_cases({"hp"}));
    DT2_ = BTF_->build(tensor_type_,"DT2",spin_cases({"hhpp"}));

    // setup DIIS
    std::shared_ptr<DIISManager> diis_manager;
    int max_diis_vectors = options_.get_int("DIIS_MAX_VECS");
    int min_diis_vectors = options_.get_int("DIIS_MIN_VECS");
    if (max_diis_vectors > 0){
        diis_manager = std::shared_ptr<DIISManager>(new DIISManager(max_diis_vectors, "LDSRG2 DIIS vector", DIISManager::OldestAdded, DIISManager::InCore));
        diis_manager->set_error_vector_size(51,
                                           DIISEntry::Pointer,DT1_.block("cv").numel(),
                                           DIISEntry::Pointer,DT1_.block("CV").numel(),
                                           DIISEntry::Pointer,DT1_.block("ca").numel(),
                                           DIISEntry::Pointer,DT1_.block("CA").numel(),
                                           DIISEntry::Pointer,DT1_.block("av").numel(),
                                           DIISEntry::Pointer,DT1_.block("AV").numel(),
                                           DIISEntry::Pointer,DT2_.block("ccaa").numel(),
                                           DIISEntry::Pointer,DT2_.block("ccav").numel(),
                                           DIISEntry::Pointer,DT2_.block("ccva").numel(),
                                           DIISEntry::Pointer,DT2_.block("ccvv").numel(),
                                           DIISEntry::Pointer,DT2_.block("caaa").numel(),
                                           DIISEntry::Pointer,DT2_.block("caav").numel(),
                                           DIISEntry::Pointer,DT2_.block("cava").numel(),
                                           DIISEntry::Pointer,DT2_.block("cavv").numel(),
                                           DIISEntry::Pointer,DT2_.block("acaa").numel(),
                                           DIISEntry::Pointer,DT2_.block("acav").numel(),
                                           DIISEntry::Pointer,DT2_.block("acva").numel(),
                                           DIISEntry::Pointer,DT2_.block("acvv").numel(),
                                           DIISEntry::Pointer,DT2_.block("aaav").numel(),
                                           DIISEntry::Pointer,DT2_.block("aava").numel(),
                                           DIISEntry::Pointer,DT2_.block("aavv").numel(),
                                           DIISEntry::Pointer,DT2_.block("cCaA").numel(),
                                           DIISEntry::Pointer,DT2_.block("cCaV").numel(),
                                           DIISEntry::Pointer,DT2_.block("cCvA").numel(),
                                           DIISEntry::Pointer,DT2_.block("cCvV").numel(),
                                           DIISEntry::Pointer,DT2_.block("cAaA").numel(),
                                           DIISEntry::Pointer,DT2_.block("cAaV").numel(),
                                           DIISEntry::Pointer,DT2_.block("cAvA").numel(),
                                           DIISEntry::Pointer,DT2_.block("cAvV").numel(),
                                           DIISEntry::Pointer,DT2_.block("aCaA").numel(),
                                           DIISEntry::Pointer,DT2_.block("aCaV").numel(),
                                           DIISEntry::Pointer,DT2_.block("aCvA").numel(),
                                           DIISEntry::Pointer,DT2_.block("aCvV").numel(),
                                           DIISEntry::Pointer,DT2_.block("aAaV").numel(),
                                           DIISEntry::Pointer,DT2_.block("aAvA").numel(),
                                           DIISEntry::Pointer,DT2_.block("aAvV").numel(),
                                           DIISEntry::Pointer,DT2_.block("CCAA").numel(),
                                           DIISEntry::Pointer,DT2_.block("CCAV").numel(),
                                           DIISEntry::Pointer,DT2_.block("CCVA").numel(),
                                           DIISEntry::Pointer,DT2_.block("CCVV").numel(),
                                           DIISEntry::Pointer,DT2_.block("CAAA").numel(),
                                           DIISEntry::Pointer,DT2_.block("CAAV").numel(),
                                           DIISEntry::Pointer,DT2_.block("CAVA").numel(),
                                           DIISEntry::Pointer,DT2_.block("CAVV").numel(),
                                           DIISEntry::Pointer,DT2_.block("ACAA").numel(),
                                           DIISEntry::Pointer,DT2_.block("ACAV").numel(),
                                           DIISEntry::Pointer,DT2_.block("ACVA").numel(),
                                           DIISEntry::Pointer,DT2_.block("ACVV").numel(),
                                           DIISEntry::Pointer,DT2_.block("AAAV").numel(),
                                           DIISEntry::Pointer,DT2_.block("AAVA").numel(),
                                           DIISEntry::Pointer,DT2_.block("AAVV").numel());
        diis_manager->set_vector_size(51,
                                     DIISEntry::Pointer,T1_.block("cv").numel(),
                                     DIISEntry::Pointer,T1_.block("CV").numel(),
                                     DIISEntry::Pointer,T1_.block("ca").numel(),
                                     DIISEntry::Pointer,T1_.block("CA").numel(),
                                     DIISEntry::Pointer,T1_.block("av").numel(),
                                     DIISEntry::Pointer,T1_.block("AV").numel(),
                                     DIISEntry::Pointer,T2_.block("ccaa").numel(),
                                     DIISEntry::Pointer,T2_.block("ccav").numel(),
                                     DIISEntry::Pointer,T2_.block("ccva").numel(),
                                     DIISEntry::Pointer,T2_.block("ccvv").numel(),
                                     DIISEntry::Pointer,T2_.block("caaa").numel(),
                                     DIISEntry::Pointer,T2_.block("caav").numel(),
                                     DIISEntry::Pointer,T2_.block("cava").numel(),
                                     DIISEntry::Pointer,T2_.block("cavv").numel(),
                                     DIISEntry::Pointer,T2_.block("acaa").numel(),
                                     DIISEntry::Pointer,T2_.block("acav").numel(),
                                     DIISEntry::Pointer,T2_.block("acva").numel(),
                                     DIISEntry::Pointer,T2_.block("acvv").numel(),
                                     DIISEntry::Pointer,T2_.block("aaav").numel(),
                                     DIISEntry::Pointer,T2_.block("aava").numel(),
                                     DIISEntry::Pointer,T2_.block("aavv").numel(),
                                     DIISEntry::Pointer,T2_.block("cCaA").numel(),
                                     DIISEntry::Pointer,T2_.block("cCaV").numel(),
                                     DIISEntry::Pointer,T2_.block("cCvA").numel(),
                                     DIISEntry::Pointer,T2_.block("cCvV").numel(),
                                     DIISEntry::Pointer,T2_.block("cAaA").numel(),
                                     DIISEntry::Pointer,T2_.block("cAaV").numel(),
                                     DIISEntry::Pointer,T2_.block("cAvA").numel(),
                                     DIISEntry::Pointer,T2_.block("cAvV").numel(),
                                     DIISEntry::Pointer,T2_.block("aCaA").numel(),
                                     DIISEntry::Pointer,T2_.block("aCaV").numel(),
                                     DIISEntry::Pointer,T2_.block("aCvA").numel(),
                                     DIISEntry::Pointer,T2_.block("aCvV").numel(),
                                     DIISEntry::Pointer,T2_.block("aAaV").numel(),
                                     DIISEntry::Pointer,T2_.block("aAvA").numel(),
                                     DIISEntry::Pointer,T2_.block("aAvV").numel(),
                                     DIISEntry::Pointer,T2_.block("CCAA").numel(),
                                     DIISEntry::Pointer,T2_.block("CCAV").numel(),
                                     DIISEntry::Pointer,T2_.block("CCVA").numel(),
                                     DIISEntry::Pointer,T2_.block("CCVV").numel(),
                                     DIISEntry::Pointer,T2_.block("CAAA").numel(),
                                     DIISEntry::Pointer,T2_.block("CAAV").numel(),
                                     DIISEntry::Pointer,T2_.block("CAVA").numel(),
                                     DIISEntry::Pointer,T2_.block("CAVV").numel(),
                                     DIISEntry::Pointer,T2_.block("ACAA").numel(),
                                     DIISEntry::Pointer,T2_.block("ACAV").numel(),
                                     DIISEntry::Pointer,T2_.block("ACVA").numel(),
                                     DIISEntry::Pointer,T2_.block("ACVV").numel(),
                                     DIISEntry::Pointer,T2_.block("AAAV").numel(),
                                     DIISEntry::Pointer,T2_.block("AAVA").numel(),
                                     DIISEntry::Pointer,T2_.block("AAVV").numel());
    }

    // start iteration
    do{
        // compute Hbar
        boost::timer t_hbar;
        compute_hbar();
        double Edelta = Hbar0_ - Ecorr;
        Ecorr = Hbar0_;
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
            if(cycle >= min_diis_vectors){
                diis_manager->add_entry(102,
                    &(DT1_.block("cv").data()[0]),
                    &(DT1_.block("CV").data()[0]),
                    &(DT1_.block("ca").data()[0]),
                    &(DT1_.block("CA").data()[0]),
                    &(DT1_.block("av").data()[0]),
                    &(DT1_.block("AV").data()[0]),
                    &(DT2_.block("ccaa").data()[0]),
                    &(DT2_.block("ccav").data()[0]),
                    &(DT2_.block("ccva").data()[0]),
                    &(DT2_.block("ccvv").data()[0]),
                    &(DT2_.block("caaa").data()[0]),
                    &(DT2_.block("caav").data()[0]),
                    &(DT2_.block("cava").data()[0]),
                    &(DT2_.block("cavv").data()[0]),
                    &(DT2_.block("acaa").data()[0]),
                    &(DT2_.block("acav").data()[0]),
                    &(DT2_.block("acva").data()[0]),
                    &(DT2_.block("acvv").data()[0]),
                    &(DT2_.block("aaav").data()[0]),
                    &(DT2_.block("aava").data()[0]),
                    &(DT2_.block("aavv").data()[0]),
                    &(DT2_.block("cCaA").data()[0]),
                    &(DT2_.block("cCaV").data()[0]),
                    &(DT2_.block("cCvA").data()[0]),
                    &(DT2_.block("cCvV").data()[0]),
                    &(DT2_.block("cAaA").data()[0]),
                    &(DT2_.block("cAaV").data()[0]),
                    &(DT2_.block("cAvA").data()[0]),
                    &(DT2_.block("cAvV").data()[0]),
                    &(DT2_.block("aCaA").data()[0]),
                    &(DT2_.block("aCaV").data()[0]),
                    &(DT2_.block("aCvA").data()[0]),
                    &(DT2_.block("aCvV").data()[0]),
                    &(DT2_.block("aAaV").data()[0]),
                    &(DT2_.block("aAvA").data()[0]),
                    &(DT2_.block("aAvV").data()[0]),
                    &(DT2_.block("CCAA").data()[0]),
                    &(DT2_.block("CCAV").data()[0]),
                    &(DT2_.block("CCVA").data()[0]),
                    &(DT2_.block("CCVV").data()[0]),
                    &(DT2_.block("CAAA").data()[0]),
                    &(DT2_.block("CAAV").data()[0]),
                    &(DT2_.block("CAVA").data()[0]),
                    &(DT2_.block("CAVV").data()[0]),
                    &(DT2_.block("ACAA").data()[0]),
                    &(DT2_.block("ACAV").data()[0]),
                    &(DT2_.block("ACVA").data()[0]),
                    &(DT2_.block("ACVV").data()[0]),
                    &(DT2_.block("AAAV").data()[0]),
                    &(DT2_.block("AAVA").data()[0]),
                    &(DT2_.block("AAVV").data()[0]),
                    &(T1_.block("cv").data()[0]),
                    &(T1_.block("CV").data()[0]),
                    &(T1_.block("ca").data()[0]),
                    &(T1_.block("CA").data()[0]),
                    &(T1_.block("av").data()[0]),
                    &(T1_.block("AV").data()[0]),
                    &(T2_.block("ccaa").data()[0]),
                    &(T2_.block("ccav").data()[0]),
                    &(T2_.block("ccva").data()[0]),
                    &(T2_.block("ccvv").data()[0]),
                    &(T2_.block("caaa").data()[0]),
                    &(T2_.block("caav").data()[0]),
                    &(T2_.block("cava").data()[0]),
                    &(T2_.block("cavv").data()[0]),
                    &(T2_.block("acaa").data()[0]),
                    &(T2_.block("acav").data()[0]),
                    &(T2_.block("acva").data()[0]),
                    &(T2_.block("acvv").data()[0]),
                    &(T2_.block("aaav").data()[0]),
                    &(T2_.block("aava").data()[0]),
                    &(T2_.block("aavv").data()[0]),
                    &(T2_.block("cCaA").data()[0]),
                    &(T2_.block("cCaV").data()[0]),
                    &(T2_.block("cCvA").data()[0]),
                    &(T2_.block("cCvV").data()[0]),
                    &(T2_.block("cAaA").data()[0]),
                    &(T2_.block("cAaV").data()[0]),
                    &(T2_.block("cAvA").data()[0]),
                    &(T2_.block("cAvV").data()[0]),
                    &(T2_.block("aCaA").data()[0]),
                    &(T2_.block("aCaV").data()[0]),
                    &(T2_.block("aCvA").data()[0]),
                    &(T2_.block("aCvV").data()[0]),
                    &(T2_.block("aAaV").data()[0]),
                    &(T2_.block("aAvA").data()[0]),
                    &(T2_.block("aAvV").data()[0]),
                    &(T2_.block("CCAA").data()[0]),
                    &(T2_.block("CCAV").data()[0]),
                    &(T2_.block("CCVA").data()[0]),
                    &(T2_.block("CCVV").data()[0]),
                    &(T2_.block("CAAA").data()[0]),
                    &(T2_.block("CAAV").data()[0]),
                    &(T2_.block("CAVA").data()[0]),
                    &(T2_.block("CAVV").data()[0]),
                    &(T2_.block("ACAA").data()[0]),
                    &(T2_.block("ACAV").data()[0]),
                    &(T2_.block("ACVA").data()[0]),
                    &(T2_.block("ACVV").data()[0]),
                    &(T2_.block("AAAV").data()[0]),
                    &(T2_.block("AAVA").data()[0]),
                    &(T2_.block("AAVV").data()[0]));
            }
            if (cycle > max_diis_vectors){
                if (diis_manager->subspace_size() >= min_diis_vectors && cycle){
                    outfile->Printf(" -> DIIS");
                    outfile->Flush();
                    diis_manager->extrapolate(51,
                        &(T1_.block("cv").data()[0]),
                        &(T1_.block("CV").data()[0]),
                        &(T1_.block("ca").data()[0]),
                        &(T1_.block("CA").data()[0]),
                        &(T1_.block("av").data()[0]),
                        &(T1_.block("AV").data()[0]),
                        &(T2_.block("ccaa").data()[0]),
                        &(T2_.block("ccav").data()[0]),
                        &(T2_.block("ccva").data()[0]),
                        &(T2_.block("ccvv").data()[0]),
                        &(T2_.block("caaa").data()[0]),
                        &(T2_.block("caav").data()[0]),
                        &(T2_.block("cava").data()[0]),
                        &(T2_.block("cavv").data()[0]),
                        &(T2_.block("acaa").data()[0]),
                        &(T2_.block("acav").data()[0]),
                        &(T2_.block("acva").data()[0]),
                        &(T2_.block("acvv").data()[0]),
                        &(T2_.block("aaav").data()[0]),
                        &(T2_.block("aava").data()[0]),
                        &(T2_.block("aavv").data()[0]),
                        &(T2_.block("cCaA").data()[0]),
                        &(T2_.block("cCaV").data()[0]),
                        &(T2_.block("cCvA").data()[0]),
                        &(T2_.block("cCvV").data()[0]),
                        &(T2_.block("cAaA").data()[0]),
                        &(T2_.block("cAaV").data()[0]),
                        &(T2_.block("cAvA").data()[0]),
                        &(T2_.block("cAvV").data()[0]),
                        &(T2_.block("aCaA").data()[0]),
                        &(T2_.block("aCaV").data()[0]),
                        &(T2_.block("aCvA").data()[0]),
                        &(T2_.block("aCvV").data()[0]),
                        &(T2_.block("aAaV").data()[0]),
                        &(T2_.block("aAvA").data()[0]),
                        &(T2_.block("aAvV").data()[0]),
                        &(T2_.block("CCAA").data()[0]),
                        &(T2_.block("CCAV").data()[0]),
                        &(T2_.block("CCVA").data()[0]),
                        &(T2_.block("CCVV").data()[0]),
                        &(T2_.block("CAAA").data()[0]),
                        &(T2_.block("CAAV").data()[0]),
                        &(T2_.block("CAVA").data()[0]),
                        &(T2_.block("CAVV").data()[0]),
                        &(T2_.block("ACAA").data()[0]),
                        &(T2_.block("ACAV").data()[0]),
                        &(T2_.block("ACVA").data()[0]),
                        &(T2_.block("ACVV").data()[0]),
                        &(T2_.block("AAAV").data()[0]),
                        &(T2_.block("AAVA").data()[0]),
                        &(T2_.block("AAVV").data()[0]));
                }
            }
        }

        // printing
        outfile->Printf("\n    %5d  %16.12f %10.3e  %10.3e %10.3e  %10.3e %10.3e  %8.3f %8.3f",
                        cycle, Ecorr, Edelta, Hbar1od, Hbar2od, T1rms_, T2rms_, time_hbar, time_amp);

        // test convergence
        double rms = T1rms_ > T2rms_ ? T1rms_ : T2rms_;
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
    energy.push_back({"E0 (reference)", Eref_});
    energy.push_back({"MR-LDSRG(2) correlation energy", Ecorr});
    energy.push_back({"MR-LDSRG(2) total energy", Eref_ + Ecorr});
    for (auto& str_dim: energy){
        outfile->Printf("\n    %-30s = %23.15f", str_dim.first.c_str(), str_dim.second);
    }

    // analyze converged amplitudes
    analyze_amplitudes("Final",T1_,T2_);

    Hbar0_ = Ecorr;
    return Ecorr;
}

double MRDSRG::Hbar1od_norm(const std::vector<std::string>& blocks){
    double norm = 0.0;

    for(auto& block: blocks){
        double norm_block = Hbar1_.block(block).norm();
        norm += 2.0 * norm_block * norm_block;
    }
    norm = std::sqrt(norm);

    return norm;
}

double MRDSRG::Hbar2od_norm(const std::vector<std::string>& blocks){
    double norm = 0.0;

    for(auto& block: blocks){
        double norm_block = Hbar2_.block(block).norm();
        norm += 2.0 * norm_block * norm_block;
    }
    norm = std::sqrt(norm);

    return norm;
}

}}
