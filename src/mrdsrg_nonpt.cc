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

    // iteration variables
    bool converged = false;
    int maxn = options_.get_int("SRG_RSC_NCOMM");
    double ct_threshold = options_.get_double("SRG_RSC_THRESHOLD");
    std::string dsrg_op = options_.get_str("DSRG_TRANS_TYPE");

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
        if(dsrg_op == "UNITARY"){
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
        }

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
        if(print_ > 2){
            outfile->Printf("\n  n = %3d, C1norm = %20.15f, C2norm = %20.15f", n, norm_C1, norm_C2);
        }
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
    if(options_.get_str("THREEPDC") == "ZERO"){
        outfile->Printf("\n    Skip Lambda3 contributions in [Hbar2, T2].");
    }
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
    std::vector<std::string> blocks1 = od_one_labels();

    // figure out off-diagonal block labels for Hbar2
    std::vector<std::string> blocks2 = od_two_labels();

    // iteration variables
    double Ecorr = 0.0;
    int cycle = 0, maxiter = options_.get_int("MAXITER");
    double e_conv = options_.get_double("E_CONVERGENCE");
    double r_conv = options_.get_double("R_CONVERGENCE");
    bool converged = false, failed = false;
    Hbar1_ = BTF_->build(tensor_type_,"Hbar1",spin_cases({"gg"}));
    Hbar2_ = BTF_->build(tensor_type_,"Hbar2",spin_cases({"gggg"}));
    O1_ = BTF_->build(tensor_type_,"O1",spin_cases({"gg"}));
    O2_ = BTF_->build(tensor_type_,"O2",spin_cases({"gggg"}));
    C1_ = BTF_->build(tensor_type_,"C1",spin_cases({"gg"}));
    C2_ = BTF_->build(tensor_type_,"C2",spin_cases({"gggg"}));
    DT1_ = BTF_->build(tensor_type_,"DT1",spin_cases({"hp"}));
    DT2_ = BTF_->build(tensor_type_,"DT2",spin_cases({"hhpp"}));
    std::vector<double> big_T, big_DT;
    size_t numel = vector_size_diis(T1_,blocks1,T2_,blocks2);

    // setup DIIS
    std::shared_ptr<DIISManager> diis_manager;
    int max_diis_vectors = options_.get_int("DIIS_MAX_VECS");
    int min_diis_vectors = options_.get_int("DIIS_MIN_VECS");
    if (max_diis_vectors > 0){
        diis_manager = std::shared_ptr<DIISManager>(new DIISManager(max_diis_vectors, "LDSRG2 DIIS T", DIISManager::LargestError, DIISManager::InCore));
        diis_manager->set_error_vector_size(1,DIISEntry::Pointer,numel);
        diis_manager->set_vector_size(1,DIISEntry::Pointer,numel);
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
        update_t();
        double time_amp = t_amp.elapsed();

        // copy amplitudes to the big vector
        big_T = copy_amp_diis(T1_,blocks1,T2_,blocks2);
        big_DT = copy_amp_diis(DT1_,blocks1,DT2_,blocks2);

        // DIIS amplitudes
        if(diis_manager){
            if(cycle >= min_diis_vectors){
                diis_manager->add_entry(2, &(big_DT[0]), &(big_T[0]));
            }
            if (cycle > max_diis_vectors){
                if (diis_manager->subspace_size() >= min_diis_vectors && cycle){
                    outfile->Printf(" -> DIIS");
                    outfile->Flush();
                    diis_manager->extrapolate(1, &(big_T[0]));
                    return_amp_diis(T1_,blocks1,T2_,blocks2,big_T);
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
            outfile->Printf("\n\n    The computation does not converge in %d iterations! Quitting.\n", maxiter);
            converged = true;
            failed = true;
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

    // fail to converge
    if(failed){
        throw PSIEXCEPTION("The MR-LDSRG(2) computation does not converge.");
    }

    Hbar0_ = Ecorr;
    return Ecorr;
}

double MRDSRG::compute_energy_cepa0(){
    // print title
    outfile->Printf("\n\n  ==> Computing MR-DSRG-CEPA0 Energy <==\n");
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

    // figure out off-diagonal blocks labels
    std::vector<std::string> blocks1 = od_one_labels();
    std::vector<std::string> blocks2 = od_two_labels();

    // iteration variables
    double Ecorr = 0.0;
    int cycle = 0, maxiter = options_.get_int("MAXITER");
    double e_conv = options_.get_double("E_CONVERGENCE");
    double r_conv = options_.get_double("R_CONVERGENCE");
    bool converged = false, failed = false;
    Hbar1_ = BTF_->build(tensor_type_,"Hbar1",spin_cases({"gg"}));
    Hbar2_ = BTF_->build(tensor_type_,"Hbar2",spin_cases({"gggg"}));
    O1_ = BTF_->build(tensor_type_,"O1",spin_cases({"gg"}));
    O2_ = BTF_->build(tensor_type_,"O2",spin_cases({"gggg"}));
    C1_ = BTF_->build(tensor_type_,"C1",spin_cases({"gg"}));
    C2_ = BTF_->build(tensor_type_,"C2",spin_cases({"gggg"}));
    DT1_ = BTF_->build(tensor_type_,"DT1",spin_cases({"hp"}));
    DT2_ = BTF_->build(tensor_type_,"DT2",spin_cases({"hhpp"}));
    std::vector<double> big_T, big_DT;
    size_t numel = vector_size_diis(T1_,blocks1,T2_,blocks2);
    std::string dsrg_op = options_.get_str("DSRG_TRANS_TYPE");

    // setup DIIS
    std::shared_ptr<DIISManager> diis_manager;
    int max_diis_vectors = options_.get_int("DIIS_MAX_VECS");
    int min_diis_vectors = options_.get_int("DIIS_MIN_VECS");
    if (max_diis_vectors > 0){
        diis_manager = std::shared_ptr<DIISManager>(new DIISManager(max_diis_vectors, "LDSRG2 DIIS T", DIISManager::LargestError, DIISManager::InCore));
        diis_manager->set_error_vector_size(1,DIISEntry::Pointer,numel);
        diis_manager->set_vector_size(1,DIISEntry::Pointer,numel);
    }

    // start iteration
    do{
        // compute Hbar
        boost::timer t_hbar;

        // initialize Hbar with bare H
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

        // compute F contribution to O1 and O2
        C1_.zero();
        C2_.zero();
        H1_T1_C1(F_,T1_,0.5,C1_);
        H1_T2_C1(F_,T2_,0.5,C1_);
        H1_T2_C2(F_,T2_,0.5,C2_);

        O1_["pq"] += C1_["pq"];
        O1_["PQ"] += C1_["PQ"];
        O2_["pqrs"] += C2_["pqrs"];
        O2_["pQrS"] += C2_["pQrS"];
        O2_["PQRS"] += C2_["PQRS"];

        if(dsrg_op == "UNITARY"){
            O1_["pq"] += C1_["qp"];
            O1_["PQ"] += C1_["QP"];
            O2_["pqrs"] += C2_["rspq"];
            O2_["pQrS"] += C2_["rSpQ"];
            O2_["PQRS"] += C2_["RSPQ"];
        }

        // Compute [H, T] + 0.5 * [[F, A], T]
        double C0 = 0.0;
        C1_.zero();
        C2_.zero();

        // zero-body
        H1_T1_C0(O1_,T1_,1.0,C0);
        H1_T2_C0(O1_,T2_,1.0,C0);
        H2_T1_C0(O2_,T1_,1.0,C0);
        H2_T2_C0(O2_,T2_,1.0,C0);
        // one-body
        H1_T1_C1(O1_,T1_,1.0,C1_);
        H1_T2_C1(O1_,T2_,1.0,C1_);
        H2_T1_C1(O2_,T1_,1.0,C1_);
        H2_T2_C1(O2_,T2_,1.0,C1_);
        // two-body
        H1_T2_C2(O1_,T2_,1.0,C2_);
        H2_T1_C2(O2_,T1_,1.0,C2_);
        H2_T2_C2(O2_,T2_,1.0,C2_);

        // Hbar += [H, T] + [H, T]^dagger
        if(dsrg_op == "UNITARY"){
            Hbar0_ += 2.0 * C0;
            Hbar1_["pq"] += C1_["pq"];
            Hbar1_["PQ"] += C1_["PQ"];
            Hbar1_["pq"] += C1_["qp"];
            Hbar1_["PQ"] += C1_["QP"];
            Hbar2_["pqrs"] += C2_["pqrs"];
            Hbar2_["pQrS"] += C2_["pQrS"];
            Hbar2_["PQRS"] += C2_["PQRS"];
            Hbar2_["pqrs"] += C2_["rspq"];
            Hbar2_["pQrS"] += C2_["rSpQ"];
            Hbar2_["PQRS"] += C2_["RSPQ"];
        }else{
            Hbar0_ += C0;
            Hbar1_["pq"] += C1_["pq"];
            Hbar1_["PQ"] += C1_["PQ"];
            Hbar2_["pqrs"] += C2_["pqrs"];
            Hbar2_["pQrS"] += C2_["pQrS"];
            Hbar2_["PQRS"] += C2_["PQRS"];
        }

        double Edelta = Hbar0_ - Ecorr;
        Ecorr = Hbar0_;
        double time_hbar = t_hbar.elapsed();

        // compute norms of off-diagonal Hbar
        double Hbar1od = Hbar1od_norm(blocks1);
        double Hbar2od = Hbar2od_norm(blocks2);

        // update amplitudes
        boost::timer t_amp;
        update_t();
        double time_amp = t_amp.elapsed();

        // copy amplitudes to the big vector
        big_T = copy_amp_diis(T1_,blocks1,T2_,blocks2);
        big_DT = copy_amp_diis(DT1_,blocks1,DT2_,blocks2);

        // DIIS amplitudes
        if(diis_manager){
            if(cycle >= min_diis_vectors){
                diis_manager->add_entry(2, &(big_DT[0]), &(big_T[0]));
            }
            if (cycle > max_diis_vectors){
                if (diis_manager->subspace_size() >= min_diis_vectors && cycle){
                    outfile->Printf(" -> DIIS");
                    outfile->Flush();
                    diis_manager->extrapolate(1, &(big_T[0]));
                    return_amp_diis(T1_,blocks1,T2_,blocks2,big_T);
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
            outfile->Printf("\n\n    The computation does not converge in %d iterations! Quitting.\n", maxiter);
            converged = true;
            failed = true;
        }
        outfile->Flush();
        ++cycle;
    } while (!converged);

    // print summary
    outfile->Printf("\n    %s", dash.c_str());
    outfile->Printf("\n\n  ==> MR-DSRG-CEPA0 Energy Summary <==\n");
    std::vector<std::pair<std::string,double>> energy;
    energy.push_back({"E0 (reference)", Eref_});
    energy.push_back({"MR-DSRG-CEPA0 correlation energy", Ecorr});
    energy.push_back({"MR-DSRG-CEPA0 total energy", Eref_ + Ecorr});
    for (auto& str_dim: energy){
        outfile->Printf("\n    %-35s = %23.15f", str_dim.first.c_str(), str_dim.second);
    }

    // analyze converged amplitudes
    analyze_amplitudes("Final",T1_,T2_);

    // fail to converge
    if(failed){
        throw PSIEXCEPTION("The MR-DSRG-CEPA0 computation does not converge.");
    }

    Hbar0_ = Ecorr;
    return Ecorr;
}

std::vector<std::string> MRDSRG::od_one_labels(){
    std::vector<std::string> blocks1 (T1_.block_labels());
    std::vector<std::string> actv_blocks {aactv_label_ + aactv_label_, bactv_label_ + bactv_label_};
    blocks1.erase(std::remove_if(blocks1.begin(), blocks1.end(),
                                 [&](std::string i) {return std::find(actv_blocks.begin(), actv_blocks.end(), i) != actv_blocks.end();}),
            blocks1.end());
    return blocks1;
}

std::vector<std::string> MRDSRG::od_two_labels(){
    std::vector<std::string> blocks2 (T2_.block_labels());
    std::vector<std::string> actv_blocks {aactv_label_ + aactv_label_ + aactv_label_ + aactv_label_,
                aactv_label_ + bactv_label_ + aactv_label_ + bactv_label_,
                bactv_label_ + bactv_label_ + bactv_label_ + bactv_label_};
    blocks2.erase(std::remove_if(blocks2.begin(), blocks2.end(),
                                 [&](std::string i) {return std::find(actv_blocks.begin(), actv_blocks.end(), i) != actv_blocks.end();}),
            blocks2.end());
    return blocks2;
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

std::vector<double> MRDSRG::copy_amp_diis(BlockedTensor& T1, const std::vector<std::string>& label1,
                                          BlockedTensor& T2, const std::vector<std::string>& label2){
    std::vector<double> out;

    for(const auto& block: label1){
        out.insert(out.end(), T1.block(block).data().begin(), T1.block(block).data().end());
    }
    for(const auto& block: label2){
        out.insert(out.end(), T2.block(block).data().begin(), T2.block(block).data().end());
    }

    return out;
}

size_t MRDSRG::vector_size_diis(BlockedTensor& T1, const std::vector<std::string>& label1,
                                BlockedTensor& T2, const std::vector<std::string>& label2){
    size_t total_elements = 0;
    for(const auto& block: label1){
        total_elements += T1.block(block).numel();
    }
    for(const auto& block: label2){
        total_elements += T2.block(block).numel();
    }
    return total_elements;
}

void MRDSRG::return_amp_diis(BlockedTensor& T1, const std::vector<std::string>& label1,
                             BlockedTensor& T2, const std::vector<std::string>& label2,
                             const std::vector<double> &data){
    // test data
    std::map<std::string, size_t> num_elements;
    size_t total_elements = 0;

    for(const auto& block: label1){
        size_t numel = T1.block(block).numel();
        num_elements[block] = total_elements;
        total_elements += numel;
    }
    for(const auto& block: label2){
        size_t numel = T2.block(block).numel();
        num_elements[block] = total_elements;
        total_elements += numel;
    }

    if(data.size() != total_elements){
        throw PSIEXCEPTION("Number of elements in T1 and T2 do not match the bid data vector");
    }

    // transfer data
    for(const auto& block: label1){
        std::vector<double>::const_iterator start = data.begin() + num_elements[block];
        std::vector<double>::const_iterator end = start + T1.block(block).numel();
        std::vector<double> T1_this_block(start, end);
        T1.block(block).data() = T1_this_block;
    }
    for(const auto& block: label2){
        std::vector<double>::const_iterator start = data.begin() + num_elements[block];
        std::vector<double>::const_iterator end = start + T2.block(block).numel();
        std::vector<double> T2_this_block(start, end);
        T2.block(block).data() = T2_this_block;
    }
}

}}
