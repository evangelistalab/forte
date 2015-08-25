#include <algorithm>
#include <vector>
#include <map>
#include <boost/format.hpp>

#include "helpers.h"
#include "mrdsrg.h"

namespace psi{ namespace libadaptive{

double MRDSRG::compute_energy_pt2(){
    // print title
    outfile->Printf("\n\n  ==> Second-Order Perturbation DSRG-MRPT2 <==\n");
    outfile->Printf("\n    Reference:");
    outfile->Printf("\n      J. Chem. Theory Comput. 2015, 11, 2097-2108.\n");

    // create a copy of original F and V in case we want them later
    Hbar1 = BTF->build(tensor_type_,"Hbar1",spin_cases({"gg"}));
    Hbar2 = BTF->build(tensor_type_,"Hbar2",spin_cases({"gggg"}));
    Hbar1["pq"] = F["pq"];
    Hbar1["PQ"] = F["PQ"];
    Hbar2["pqrs"] = V["pqrs"];
    Hbar2["pQrS"] = V["pQrS"];
    Hbar2["PQRS"] = V["PQRS"];

    // create zeroth-order Hamiltonian
    H0th = BTF->build(tensor_type_,"Zeroth-order H",spin_cases({"gg"}));
    H0th.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if(i[0] == i[1]){
            if(spin[0] == AlphaSpin){
                value = Fa[i[0]];
            }else{
                value = Fb[i[0]];
            }
        }
    });

    // test orbitals are semi-canonicalized
    check_semicanonical();

    // compute H0th contribution to H1 and H2
    BlockedTensor temp1 = BTF->build(tensor_type_,"temp1",spin_cases({"gg"}));
    BlockedTensor temp2 = BTF->build(tensor_type_,"temp2",spin_cases({"gggg"}));
    H1_T1_C1(H0th,T1,0.5,temp1);
    H1_T2_C1(H0th,T2,0.5,temp1);
    H1_T2_C2(H0th,T2,0.5,temp2);

    // [H, A] = [H, T] + [H, T]^dagger
    Hbar1["pq"] += temp1["pq"];
    Hbar1["pq"] += temp1["qp"];
    Hbar1["PQ"] += temp1["PQ"];
    Hbar1["PQ"] += temp1["QP"];
    Hbar2["pqrs"] += temp2["pqrs"];
    Hbar2["pqrs"] += temp2["rspq"];
    Hbar2["pQrS"] += temp2["pQrS"];
    Hbar2["pQrS"] += temp2["rSpQ"];
    Hbar2["PQRS"] += temp2["PQRS"];
    Hbar2["PQRS"] += temp2["RSPQ"];

    // compute PT2 energy
    std::vector<std::pair<std::string,double>> energy;
    energy.push_back({"E0 (reference)", Eref});
    double Ecorr = 0.0, Etemp = 0.0;

    H1_T1_C0(Hbar1,T1,1.0,Ecorr);
    energy.push_back({"<[F, T1]>", 2 * (Ecorr - Etemp)});
    Etemp = Ecorr;

    H1_T2_C0(Hbar1,T2,1.0,Ecorr);
    energy.push_back({"<[F, T2]>", 2 * (Ecorr - Etemp)});
    Etemp = Ecorr;

    H2_T1_C0(Hbar2,T1,1.0,Ecorr);
    energy.push_back({"<[V, T1]>", 2 * (Ecorr - Etemp)});
    Etemp = Ecorr;

    H2_T2_C0(Hbar2,T2,1.0,Ecorr);
    energy.push_back({"<[V, T2]>", 2 * (Ecorr - Etemp)});
    Etemp = Ecorr;

    // <[H, A]> = 2 * <[H, T]>
    Ecorr *= 2.0;

    energy.push_back({"DSRG-MRPT2 correlation energy", Ecorr});
    energy.push_back({"DSRG-MRPT2 total energy", Eref + Ecorr});

    outfile->Printf("\n\n  ==> DSRG-MRPT2 Energy Summary <==\n");
    for (auto& str_dim : energy){
        outfile->Printf("\n    %-30s = %22.15f",str_dim.first.c_str(),str_dim.second);
    }

    return Ecorr;
}

double MRDSRG::compute_energy_pt3(){
    // compute DSRG-MRPT2 energy and initialize Hbar and H0th
    // Hbar is the modified first-order Hamiltonian, T is the first-order amplitude
    double pt2 = compute_energy_pt2();

    // print title
    outfile->Printf("\n\n  ==> Third-Order Perturbation DSRG-MRPT3 <==\n");
    outfile->Printf("\n    Reference:");
    outfile->Printf("\n      J. Chem. Phys. (in preparation)\n");

    // compute one- and two-body [H~1st, A1st] for second-order amplitudes
    BlockedTensor temp1 = BTF->build(tensor_type_,"temp1",spin_cases({"gg"}));
    BlockedTensor temp2 = BTF->build(tensor_type_,"temp2",spin_cases({"gggg"}));
    H1_T1_C1(Hbar1,T1,1.0,temp1);
    H1_T2_C1(Hbar1,T2,1.0,temp1);
    H2_T1_C1(Hbar2,T1,1.0,temp1);
    H2_T2_C1(Hbar2,T2,1.0,temp1);
    H1_T2_C2(Hbar1,T2,1.0,temp2);
    H2_T1_C2(Hbar2,T1,1.0,temp2);
    H2_T2_C2(Hbar2,T2,1.0,temp2);

    BlockedTensor H1_2nd = BTF->build(tensor_type_,"H2nd one-body",spin_cases({"gg"}));
    BlockedTensor H2_2nd = BTF->build(tensor_type_,"H2nd two-body",spin_cases({"gggg"}));
    H1_2nd["pq"] += temp1["pq"];
    H1_2nd["pq"] += temp1["qp"];
    H1_2nd["PQ"] += temp1["PQ"];
    H1_2nd["PQ"] += temp1["QP"];
    H2_2nd["pqrs"] += temp2["pqrs"];
    H2_2nd["pqrs"] += temp2["rspq"];
    H2_2nd["pQrS"] += temp2["pQrS"];
    H2_2nd["pQrS"] += temp2["rSpQ"];
    H2_2nd["PQRS"] += temp2["PQRS"];
    H2_2nd["PQRS"] += temp2["RSPQ"];

    // compute the second-order amplitude
    BlockedTensor T1_2nd = BTF->build(tensor_type_,"temp1",spin_cases({"hp"}));
    BlockedTensor T2_2nd = BTF->build(tensor_type_,"temp2",spin_cases({"hhpp"}));
    guess_t2(H2_2nd,T2_2nd);
    guess_t1(H1_2nd,T2_2nd,T1_2nd);
    analyze_amplitudes("Second-Order ",T1_2nd,T2_2nd);

//    // compute modified first-order Hamiltonian H_1st = Hbare_1st + 0.5 * [H0th, A_1st]
//    BlockedTensor H1_1st = BTF->build(tensor_type_,"H1st one-body",spin_cases({"gg"}));
//    BlockedTensor H2_1st = BTF->build(tensor_type_,"H1st two-body",spin_cases({"gggg"}));
//    H1_1st["pq"] = F["pq"];
//    H1_1st["PQ"] = F["PQ"];
//    H2_1st["pqrs"] = V["pqrs"];
//    H2_1st["pQrS"] = V["pQrS"];
//    H2_1st["PQRS"] = V["PQRS"];

//    // compute H0th contribution to H1st
//    BlockedTensor temp1 = BTF->build(tensor_type_,"temp1",spin_cases({"gg"}));
//    BlockedTensor temp2 = BTF->build(tensor_type_,"temp2",spin_cases({"gggg"}));
//    H1_T1_C1(H0th,T1,0.5,temp1);
//    H1_T2_C1(H0th,T2,0.5,temp1);
//    H1_T2_C2(H0th,T2,0.5,temp2);

//    H1_1st["pq"] += temp1["pq"];
//    H1_1st["pq"] += temp1["qp"];
//    H1_1st["PQ"] += temp1["PQ"];
//    H1_1st["PQ"] += temp1["QP"];
//    H2_1st["pqrs"] += temp2["pqrs"];
//    H2_1st["pqrs"] += temp2["rspq"];
//    H2_1st["pQrS"] += temp2["pQrS"];
//    H2_1st["pQrS"] += temp2["rSpQ"];
//    H2_1st["PQRS"] += temp2["PQRS"];
//    H2_1st["PQRS"] += temp2["RSPQ"];

//    // compute one- and two-body [H_1st, T_1st]
//    temp1.zero();
//    temp2.zero();
//    H1_T1_C1(H1_1st,T1,1.0,temp1);
//    H1_T2_C1(H1_1st,T2,1.0,temp1);
//    H2_T1_C1(H2_1st,T1,1.0,temp1);
//    H2_T2_C1(H2_1st,T2,1.0,temp1);
//    H1_T2_C2(H1_1st,T2,1.0,temp2);
//    H2_T1_C2(H2_1st,T1,1.0,temp2);
//    H2_T2_C2(H2_1st,T2,1.0,temp2);

//    // [H_1st, A_1st] = [H_1st, T_1st] + [H_1st, T_1st]^dagger
//    BlockedTensor H1_2nd = BTF->build(tensor_type_,"H2nd one-body",spin_cases({"gg"}));
//    BlockedTensor H2_2nd = BTF->build(tensor_type_,"H2nd two-body",spin_cases({"gggg"}));
//    H1_2nd["pq"] += temp1["pq"];
//    H1_2nd["pq"] += temp1["qp"];
//    H1_2nd["PQ"] += temp1["PQ"];
//    H1_2nd["PQ"] += temp1["QP"];
//    H2_2nd["pqrs"] += temp2["pqrs"];
//    H2_2nd["pqrs"] += temp2["rspq"];
//    H2_2nd["pQrS"] += temp2["pQrS"];
//    H2_2nd["pQrS"] += temp2["rSpQ"];
//    H2_2nd["PQRS"] += temp2["PQRS"];
//    H2_2nd["PQRS"] += temp2["RSPQ"];

//    // compute second-order amplitudes
//    BlockedTensor T1_2nd = BTF->build(tensor_type_,"temp1",spin_cases({"hp"}));
//    BlockedTensor T2_2nd = BTF->build(tensor_type_,"temp2",spin_cases({"hhpp"}));
//    guess_t2(H2_2nd,T2_2nd);
//    guess_t1(H1_2nd,T2_2nd,T1_2nd);
//    analyze_amplitudes("Second-Order ");

//    // compute <[H_1st, T_2nd]> contribution to PT3 energy
//    std::vector<std::pair<std::string,double>> energy;
//    energy.push_back({"E0 (reference)", Eref});
//    double Ecorr1 = 0.0, Etemp1 = 0.0;

//    H1_T1_C0(H1_1st,T1_2nd,1.0,Ecorr1);
//    energy.push_back({"<[F_1st, T1_2nd]>", 2 * (Ecorr1 - Etemp1)});
//    Etemp1 = Ecorr1;

//    H1_T2_C0(H1_1st,T2_2nd,1.0,Ecorr1);
//    energy.push_back({"<[F_1st, T2_2nd]>", 2 * (Ecorr1 - Etemp1)});
//    Etemp1 = Ecorr1;

//    H2_T1_C0(H2_1st,T1_2nd,1.0,Ecorr1);
//    energy.push_back({"<[V_1st, T1_2nd]>", 2 * (Ecorr1 - Etemp1)});
//    Etemp1 = Ecorr1;

//    H2_T2_C0(H2_1st,T2_2nd,1.0,Ecorr1);
//    energy.push_back({"<[V_1st, T2_2nd]>", 2 * (Ecorr1 - Etemp1)});
//    Etemp1 = Ecorr1;

//    // <[H_1st, A_2nd]> = 2 * <[H_1st, T2_nd]>
//    Ecorr1 *= 2.0;
//    energy.push_back({"<[H_1st, A_2nd]>", Ecorr1});

//    // compute modified first-order Hamiltonian H_1st' = Hbare_1st + 1/3 * [H0th, A_1st]
//    H1_1st["pq"] = F["pq"];
//    H1_1st["PQ"] = F["PQ"];
//    H2_1st["pqrs"] = V["pqrs"];
//    H2_1st["pQrS"] = V["pQrS"];
//    H2_1st["PQRS"] = V["PQRS"];

//    // compute H0th contribution to H1st'
//    temp1.zero();
//    temp2.zero();
//    double alpha = 1.0 / 3;
//    H1_T1_C1(H0th,T1,alpha,temp1);
//    H1_T2_C1(H0th,T2,alpha,temp1);
//    H1_T2_C2(H0th,T2,alpha,temp2);

//    H1_1st["pq"] += temp1["pq"];
//    H1_1st["pq"] += temp1["qp"];
//    H1_1st["PQ"] += temp1["PQ"];
//    H1_1st["PQ"] += temp1["QP"];
//    H2_1st["pqrs"] += temp2["pqrs"];
//    H2_1st["pqrs"] += temp2["rspq"];
//    H2_1st["pQrS"] += temp2["pQrS"];
//    H2_1st["pQrS"] += temp2["rSpQ"];
//    H2_1st["PQRS"] += temp2["PQRS"];
//    H2_1st["PQRS"] += temp2["RSPQ"];

//    // compute modified second-order Hamiltonian H_2nd = 0.5 * [H_1st', A_1st] + 0.5 * [H0th, A_2nd]
//    temp1.zero();
//    temp2.zero();
//    H1_T1_C1(H1_1st,T1,0.5,temp1);
//    H1_T2_C1(H1_1st,T2,0.5,temp1);
//    H2_T1_C1(H2_1st,T1,0.5,temp1);
//    H2_T2_C1(H2_1st,T2,0.5,temp1);
//    H1_T2_C2(H1_1st,T2,0.5,temp2);
//    H2_T1_C2(H2_1st,T1,0.5,temp2);
//    H2_T2_C2(H2_1st,T2,0.5,temp2);

//    H1_T1_C1(H0th,T1_2nd,0.5,temp1);
//    H1_T2_C1(H0th,T2_2nd,0.5,temp1);
//    H1_T2_C2(H0th,T2_2nd,0.5,temp2);

//    H1_2nd.zero();
//    H2_2nd.zero();
//    H1_2nd["pq"] += temp1["pq"];
//    H1_2nd["pq"] += temp1["qp"];
//    H1_2nd["PQ"] += temp1["PQ"];
//    H1_2nd["PQ"] += temp1["QP"];
//    H2_2nd["pqrs"] += temp2["pqrs"];
//    H2_2nd["pqrs"] += temp2["rspq"];
//    H2_2nd["pQrS"] += temp2["pQrS"];
//    H2_2nd["pQrS"] += temp2["rSpQ"];
//    H2_2nd["PQRS"] += temp2["PQRS"];
//    H2_2nd["PQRS"] += temp2["RSPQ"];

//    // compute <[H_2nd, A_1st]> contribution to PT3 energy
//    double Ecorr2 = 0.0, Etemp2 = 0.0;

//    H1_T1_C0(H1_2nd,T1,1.0,Ecorr2);
//    energy.push_back({"<[F_2nd, T1_1st]>", 2 * (Ecorr2 - Etemp2)});
//    Etemp2 = Ecorr2;

//    H1_T2_C0(H1_2nd,T2,1.0,Ecorr2);
//    energy.push_back({"<[F_2nd, T2_1st]>", 2 * (Ecorr2 - Etemp2)});
//    Etemp2 = Ecorr2;

//    H2_T1_C0(H2_2nd,T1,1.0,Ecorr2);
//    energy.push_back({"<[V_2nd, T1_1st]>", 2 * (Ecorr2 - Etemp2)});
//    Etemp2 = Ecorr2;

//    H2_T2_C0(H2_2nd,T2,1.0,Ecorr2);
//    energy.push_back({"<[V_2nd, T2_1st]>", 2 * (Ecorr2 - Etemp2)});
//    Etemp2 = Ecorr2;

//    // <[H_2nd, A_1st]> = 2 * <[H_2nd, T_1st]>
//    Ecorr2 *= 2.0;
//    energy.push_back({"<[H_2nd, A_1st]>", Ecorr2});

//    double Ecorr = Ecorr1 + Ecorr2;
//    energy.push_back({"DSRG-MRPT3 correlation energy", Ecorr});
//    energy.push_back({"DSRG-MRPT3 total energy", Eref + Ecorr});

//    outfile->Printf("\n\n  ==> DSRG-MRPT3 Energy Summary <==\n");
//    for (auto& str_dim : energy){
//        outfile->Printf("\n    %-30s = %22.15f",str_dim.first.c_str(),str_dim.second);
//    }

    double Ecorr = 0.0;
    return Ecorr;
}

void MRDSRG::check_semicanonical(){
    outfile->Printf("\n    Checking if orbitals are semi-canonicalized ...");
    std::vector<std::string> blocks = {"cc","aa","vv","CC","AA","VV"};
    std::vector<double> Foff;
    double Foff_sum = 0.0;
    for(auto& block: blocks){
        double value = std::pow(F.block(block).norm(), 2.0) - std::pow(H0th.block(block).norm(), 2.0);
        value = std::sqrt(value);
        Foff.push_back(value);
        Foff_sum += value;
    }
    double threshold = 10.0 * options_.get_double("D_CONVERGENCE");
    if(Foff_sum > threshold){
        std::string sep(3 + 16 * 3, '-');
        outfile->Printf("\n    Warning! Orbitals are not semi-canonicalized!");
        outfile->Printf("\n    DSRG-MRPT2 is currently only formulated using semi-canonical orbitals!");
        outfile->Printf("\n    Off-Diagonal norms of the core, active, virtual blocks of Fock matrix");
        outfile->Printf("\n       %15s %15s %15s", "core", "active", "virtual");
        outfile->Printf("\n    %s", sep.c_str());
        outfile->Printf("\n    Fa %15.10f %15.10f %15.10f", Foff[0], Foff[1], Foff[2]);
        outfile->Printf("\n    Fb %15.10f %15.10f %15.10f", Foff[3], Foff[4], Foff[5]);
        outfile->Printf("\n    %s\n", sep.c_str());
        throw PSIEXCEPTION("Orbitals are not semi-canonicalized.");
    }else{
        outfile->Printf("     OK.");
    }
}

}}
