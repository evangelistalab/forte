#include <algorithm>
#include <vector>
#include <map>
#include <boost/format.hpp>

#include "helpers.h"
#include "mrdsrg.h"

namespace psi{ namespace libadaptive{

void MRDSRG::guess_t2(BlockedTensor& V, BlockedTensor& T2)
{
    Timer timer;
    std::string str = "Computing T2 amplitudes ...";
    outfile->Printf("\n    %-35s", str.c_str());
    T2max = 0.0, t2aa_norm = 0.0, t2ab_norm = 0.0, t2bb_norm = 0.0;

    T2["ijab"] = V["ijab"];
    T2["iJaB"] = V["iJaB"];
    T2["IJAB"] = V["IJAB"];

    T2.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if ((spin[0] == AlphaSpin) && (spin[1] == AlphaSpin)){
            value *= renormalized_denominator(Fa[i[0]] + Fa[i[1]] - Fa[i[2]] - Fa[i[3]]);
            t2aa_norm += value * value;
        }else if ((spin[0] == AlphaSpin) && (spin[1] == BetaSpin)){
            value *= renormalized_denominator(Fa[i[0]] + Fb[i[1]] - Fa[i[2]] - Fb[i[3]]);
            t2ab_norm += value * value;
        }else if ((spin[0] == BetaSpin)  && (spin[1] == BetaSpin)){
            value *= renormalized_denominator(Fb[i[0]] + Fb[i[1]] - Fb[i[2]] - Fb[i[3]]);
            t2bb_norm += value * value;
        }

        if (std::fabs(value) > std::fabs(T2max)) T2max = value;
    });

    // zero internal amplitudes
    T2.block("aaaa").iterate([&](const std::vector<size_t>& i,double& value){
        t2aa_norm -= value * value;
        value = 0.0;
    });
    T2.block("aAaA").iterate([&](const std::vector<size_t>& i,double& value){
        t2ab_norm -= value * value;
        value = 0.0;
    });
    T2.block("AAAA").iterate([&](const std::vector<size_t>& i,double& value){
        t2bb_norm -= value * value;
        value = 0.0;
    });

    // norms
    T2norm = std::sqrt(t2aa_norm + t2bb_norm + 4 * t2ab_norm);
    t2aa_norm = std::sqrt(t2aa_norm);
    t2ab_norm = std::sqrt(t2ab_norm);
    t2bb_norm = std::sqrt(t2bb_norm);
    T2rms = 0.0;

    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void MRDSRG::guess_t1(BlockedTensor& F, BlockedTensor& T2, BlockedTensor& T1)
{
    Timer timer;
    std::string str = "Computing initial T1 amplitudes ...";
    outfile->Printf("\n    %-35s", str.c_str());
    T1max = 0.0, t1a_norm = 0.0, t1b_norm = 0.0;

    BlockedTensor temp = BTF->build(tensor_type_,"temp",spin_cases({"aa"}));
    temp["xu"] = Gamma1["xu"];
    temp["XU"] = Gamma1["XU"];
    temp.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0] == AlphaSpin){
            value *= Fa[i[0]] - Fa[i[1]];
        }else{
            value *= Fb[i[0]] - Fb[i[1]];
        }
    });

    T1["ia"]  = F["ia"];
    T1["ia"] += temp["xu"] * T2["iuax"];
    T1["ia"] += temp["XU"] * T2["iUaX"];

    T1["IA"]  = F["IA"];
    T1["IA"] += temp["xu"] * T2["uIxA"];
    T1["IA"] += temp["XU"] * T2["IUAX"];

    T1.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0]  == AlphaSpin){
            value *= renormalized_denominator(Fa[i[0]] - Fa[i[1]]);
            t1a_norm += value * value;
        }else{
            value *= renormalized_denominator(Fb[i[0]] - Fb[i[1]]);
            t1b_norm += value * value;
        }

        if (std::fabs(value) > std::fabs(T1max)) T1max = value;
    });

    // zero internal amplitudes
    T1.block("aa").iterate([&](const std::vector<size_t>& i,double& value){
        t1a_norm -= value * value;
        value = 0.0;
    });
    T1.block("AA").iterate([&](const std::vector<size_t>& i,double& value){
        t1b_norm -= value * value;
        value = 0.0;
    });

    // norms
    T1norm = std::sqrt(t1a_norm + t1b_norm);
    t1a_norm = std::sqrt(t1a_norm);
    t1b_norm = std::sqrt(t1b_norm);
    T1rms = 0.0;

    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void MRDSRG::update_t2(){
    T2max = 0.0, t2aa_norm = 0.0, t2ab_norm = 0.0, t2bb_norm = 0.0;

    // create a temporary tensor
    BlockedTensor R2 = ambit::BlockedTensor::build(tensor_type_,"R2",spin_cases({"hhpp"}));
    R2["ijab"] = T2["ijab"];
    R2["iJaB"] = T2["iJaB"];
    R2["IJAB"] = T2["IJAB"];
    R2.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if ((spin[0] == AlphaSpin) && (spin[1] == AlphaSpin)){
            value *= Fa[i[0]] + Fa[i[1]] - Fa[i[2]] - Fa[i[3]];
        }else if ((spin[0] == AlphaSpin) && (spin[1] == BetaSpin)){
            value *= Fa[i[0]] + Fb[i[1]] - Fa[i[2]] - Fb[i[3]];
        }else if ((spin[0] == BetaSpin)  && (spin[1] == BetaSpin)){
            value *= Fb[i[0]] + Fb[i[1]] - Fb[i[2]] - Fb[i[3]];
        }
    });
    R2["ijab"] += Hbar2["ijab"];
    R2["iJaB"] += Hbar2["iJaB"];
    R2["IJAB"] += Hbar2["IJAB"];
    R2.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if ((spin[0] == AlphaSpin) && (spin[1] == AlphaSpin)){
            value *= renormalized_denominator(Fa[i[0]] + Fa[i[1]] - Fa[i[2]] - Fa[i[3]]);
            t2aa_norm += value * value;
        }else if ((spin[0] == AlphaSpin) && (spin[1] == BetaSpin)){
            value *= renormalized_denominator(Fa[i[0]] + Fb[i[1]] - Fa[i[2]] - Fb[i[3]]);
            t2ab_norm += value * value;
        }else if ((spin[0] == BetaSpin)  && (spin[1] == BetaSpin)){
            value *= renormalized_denominator(Fb[i[0]] + Fb[i[1]] - Fb[i[2]] - Fb[i[3]]);
            t2bb_norm += value * value;
        }

        if (std::fabs(value) > std::fabs(T2max)) T2max = value;
    });

    // zero internal amplitudes
    R2.block("aaaa").iterate([&](const std::vector<size_t>& i,double& value){
        t2aa_norm -= value * value;
        value = 0.0;
    });
    R2.block("aAaA").iterate([&](const std::vector<size_t>& i,double& value){
        t2ab_norm -= value * value;
        value = 0.0;
    });
    R2.block("AAAA").iterate([&](const std::vector<size_t>& i,double& value){
        t2bb_norm -= value * value;
        value = 0.0;
    });

    // compute RMS
    DT2["ijab"] = T2["ijab"] - R2["ijab"];
    DT2["iJaB"] = T2["iJaB"] - R2["iJaB"];
    DT2["IJAB"] = T2["IJAB"] - R2["IJAB"];
    T2rms = DT2.norm();

    // copy R2 to T2
    T2["ijab"] = R2["ijab"];
    T2["iJaB"] = R2["iJaB"];
    T2["IJAB"] = R2["IJAB"];

    // norms
    T2norm = std::sqrt(t2aa_norm + t2bb_norm + 4 * t2ab_norm);
    t2aa_norm = std::sqrt(t2aa_norm);
    t2ab_norm = std::sqrt(t2ab_norm);
    t2bb_norm = std::sqrt(t2bb_norm);
}

void MRDSRG::update_t1(){
    T1max = 0.0, t1a_norm = 0.0, t1b_norm = 0.0;

    // create a temporary tensor
    BlockedTensor R1 = ambit::BlockedTensor::build(tensor_type_,"R1",spin_cases({"hp"}));
    R1["ia"] = T1["ia"];
    R1["IA"] = T1["IA"];
    R1.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0] == AlphaSpin){
            value *= Fa[i[0]] - Fa[i[1]];
        }else{
            value *= Fb[i[0]] - Fb[i[1]];
        }
    });
    R1["ia"] += Hbar1["ia"];
    R1["IA"] += Hbar1["IA"];
    R1.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0] == AlphaSpin){
            value *= renormalized_denominator(Fa[i[0]] - Fa[i[1]]);
            t1a_norm += value * value;
        }else{
            value *= renormalized_denominator(Fb[i[0]] - Fb[i[1]]);
            t1b_norm += value * value;
        }

        if (std::fabs(value) > std::fabs(T1max)) T1max = value;
    });

    // zero internal amplitudes
    R1.block("aa").iterate([&](const std::vector<size_t>& i,double& value){
        t1a_norm -= value * value;
        value = 0.0;
    });
    R1.block("AA").iterate([&](const std::vector<size_t>& i,double& value){
        t1b_norm -= value * value;
        value = 0.0;
    });

    // compute RMS
    DT1["ia"] = T1["ia"] - R1["ia"];
    DT1["IA"] = T1["IA"] - R1["IA"];
    T1rms = DT1.norm();

    // copy R1 to T1
    T1["ia"] = R1["ia"];
    T1["IA"] = R1["IA"];

    // norms
    T1norm = std::sqrt(t1a_norm + t1b_norm);
    t1a_norm = std::sqrt(t1a_norm);
    t1b_norm = std::sqrt(t1b_norm);
}

void MRDSRG::analyze_amplitudes(const std::string& name, BlockedTensor& T1, BlockedTensor& T2){
    outfile->Printf("\n\n  ==> %sExcitation Amplitudes Summary <==\n", name.c_str());
    outfile->Printf("\n    Active Indices: ");
    int c = 0;
    for(const auto& idx: aactv_mos){
        outfile->Printf("%4zu ", idx);
        if(++c % 10 == 0)
            outfile->Printf("\n    %16c", ' ');
    }
    check_t1(T1);
    check_t2(T2);

    outfile->Printf("\n\n  ==> Possible Intruders <==\n");
    print_intruder("A", lt1a);
    print_intruder("B", lt1b);
    print_intruder("AA", lt2aa);
    print_intruder("AB", lt2ab);
    print_intruder("BB", lt2bb);
}

// Binary function to achieve sorting a vector of pair<vector, double>
// according to the double value in decending order
template <class T1, class T2, class G3 = std::greater<T2> >
struct rsort_pair_second {
    bool operator()(const std::pair<T1,T2>& left, const std::pair<T1,T2>& right) {
        G3 p;
        return p(fabs(left.second), fabs(right.second));
    }
};

void MRDSRG::check_t2(BlockedTensor& T2)
{
    size_t nonzero_aa = 0, nonzero_ab = 0, nonzero_bb = 0;
    std::vector<std::pair<std::vector<size_t>, double>> t2aa, t2ab, t2bb;

    // create all knids of spin maps; 0: aa, 1: ab, 2:bb
    std::map<int, double> spin_to_nonzero;
    std::map<int, std::vector<std::pair<std::vector<size_t>, double>>> spin_to_t2;
    std::map<int, std::vector<std::pair<std::vector<size_t>, double>>> spin_to_lt2;

    for(const std::string& block: T2.block_labels()){
        int spin = isupper(block[0]) + isupper(block[1]);

        // create a reference to simplify the syntax
        std::vector<std::pair<std::vector<size_t>, double>>& temp_t2 = spin_to_t2[spin];
        std::vector<std::pair<std::vector<size_t>, double>>& temp_lt2 = spin_to_lt2[spin];

        T2.block(block).citerate([&](const std::vector<size_t>& i, const double& value){
            if(fabs(value) != 0.0){
                size_t idx0 = label_to_spacemo[block[0]][i[0]];
                size_t idx1 = label_to_spacemo[block[1]][i[1]];
                size_t idx2 = label_to_spacemo[block[2]][i[2]];
                size_t idx3 = label_to_spacemo[block[3]][i[3]];

                ++spin_to_nonzero[spin];

                if((idx0 <= idx1) && (idx2 <= idx3)){
                    std::vector<size_t> indices = {idx0, idx1, idx2, idx3};
                    std::pair<std::vector<size_t>, double> idx_value = std::make_pair(indices, value);

                    temp_t2.push_back(idx_value);
                    std::sort(temp_t2.begin(), temp_t2.end(), rsort_pair_second<std::vector<size_t>, double>());
                    if(temp_t2.size() == ntamp_ + 1){
                        temp_t2.pop_back();
                    }

                    if(fabs(value) > fabs(intruder_tamp_)){
                        temp_lt2.push_back(idx_value);
                    }
                    std::sort(temp_lt2.begin(), temp_lt2.end(), rsort_pair_second<std::vector<size_t>, double>());
                }
            }
        });
    }

    // update values
    nonzero_aa = spin_to_nonzero[0];
    nonzero_ab = spin_to_nonzero[1];
    nonzero_bb = spin_to_nonzero[2];

    t2aa = spin_to_t2[0];
    t2ab = spin_to_t2[1];
    t2bb = spin_to_t2[2];

    lt2aa = spin_to_lt2[0];
    lt2ab = spin_to_lt2[1];
    lt2bb = spin_to_lt2[2];

    // print summary
    print_amp_summary("AA", t2aa, t2aa_norm, nonzero_aa);
    print_amp_summary("AB", t2ab, t2ab_norm, nonzero_ab);
    print_amp_summary("BB", t2bb, t2bb_norm, nonzero_bb);
}

void MRDSRG::check_t1(BlockedTensor &T1)
{
    size_t nonzero_a = 0, nonzero_b = 0;
    std::vector<std::pair<std::vector<size_t>, double>> t1a, t1b;

    // create all kinds of spin maps; true: a, false: b
    std::map<bool, double> spin_to_nonzero;
    std::map<bool, std::vector<std::pair<std::vector<size_t>, double>>> spin_to_t1;
    std::map<bool, std::vector<std::pair<std::vector<size_t>, double>>> spin_to_lt1;

    for(const std::string& block: T1.block_labels()){
        bool spin_alpha = islower(block[0]) ? true : false;

        // create a reference to simplify the syntax
        std::vector<std::pair<std::vector<size_t>, double>>& temp_t1 = spin_to_t1[spin_alpha];
        std::vector<std::pair<std::vector<size_t>, double>>& temp_lt1 = spin_to_lt1[spin_alpha];

        T1.block(block).citerate([&](const std::vector<size_t>& i, const double& value){
            if(fabs(value) != 0.0){
                size_t idx0 = label_to_spacemo[block[0]][i[0]];
                size_t idx1 = label_to_spacemo[block[1]][i[1]];

                std::vector<size_t> indices = {idx0, idx1};
                std::pair<std::vector<size_t>, double> idx_value = std::make_pair(indices, value);

                ++spin_to_nonzero[spin_alpha];

                temp_t1.push_back(idx_value);
                std::sort(temp_t1.begin(), temp_t1.end(), rsort_pair_second<std::vector<size_t>, double>());
                if(temp_t1.size() == ntamp_ + 1){
                    temp_t1.pop_back();
                }

                if(fabs(value) > fabs(intruder_tamp_)){
                    temp_lt1.push_back(idx_value);
                }
                std::sort(temp_lt1.begin(), temp_lt1.end(), rsort_pair_second<std::vector<size_t>, double>());
            }
        });
    }

    // update value
    nonzero_a = spin_to_nonzero[true];
    nonzero_b = spin_to_nonzero[false];

    t1a = spin_to_t1[true];
    t1b = spin_to_t1[false];

    lt1a = spin_to_lt1[true];
    lt1b = spin_to_lt1[false];

    // print summary
    print_amp_summary("A", t1a, t1a_norm, nonzero_a);
    print_amp_summary("B", t1b, t1b_norm, nonzero_b);
}

void MRDSRG::print_amp_summary(const std::string &name,
                                   const std::vector<std::pair<std::vector<size_t>, double> > &list,
                                   const double &norm, const size_t &number_nonzero)
{
    int rank = name.size();
    std::map<char, std::string> spin_case;
    spin_case['A'] = " ";
    spin_case['B'] = "_";

    std::string indent(4, ' ');
    std::string title = indent + "Largest T" + std::to_string(rank)
            + " amplitudes for spin case " + name + ":";
    std::string spin_title;
    std::string mo_title;
    std::string line;
    std::string output;
    std::string summary;

    auto extendstr = [&](std::string s, int n){
        std::string o(s);
        while((--n) > 0) o += s;
        return o;
    };

    if(rank == 1){
        spin_title += str(boost::format(" %3c %3c %3c %3c %9c ") % spin_case[name[0]] % ' ' % spin_case[name[0]] % ' ' % ' ');
        if(spin_title.find_first_not_of(' ') != std::string::npos){
            spin_title = "\n" + indent + extendstr(spin_title, 3);
        }else{
            spin_title = "";
        }
        mo_title += str(boost::format(" %3c %3c %3c %3c %9c ") % 'i' % ' ' % 'a' % ' ' % ' ');
        mo_title = "\n" + indent + extendstr(mo_title, 3);
        for(size_t n = 0; n != list.size(); ++n){
            if(n % 3 == 0) output += "\n" + indent;
            const auto& datapair = list[n];
            std::vector<size_t> idx = datapair.first;
            output += str(boost::format("[%3d %3c %3d %3c]%9.6f ") % idx[0] % ' ' % idx[1] % ' ' % datapair.second);
        }
    }
    else if(rank == 2){
        spin_title += str(boost::format(" %3c %3c %3c %3c %9c ") % spin_case[name[0]] % spin_case[name[1]] % spin_case[name[0]] % spin_case[name[1]] % ' ');
        if(spin_title.find_first_not_of(' ') != std::string::npos){
            spin_title = "\n" + indent + extendstr(spin_title, 3);
        }else{
            spin_title = "";
        }
        mo_title += str(boost::format(" %3c %3c %3c %3c %9c ") % 'i' % 'j' % 'a' % 'b' % ' ');
        mo_title = "\n" + indent + extendstr(mo_title, 3);
        for(size_t n = 0; n != list.size(); ++n){
            if(n % 3 == 0) output += "\n" + indent;
            const auto& datapair = list[n];
            std::vector<size_t> idx = datapair.first;
            output += str(boost::format("[%3d %3d %3d %3d]%9.6f ") % idx[0] % idx[1] % idx[2] % idx[3] % datapair.second);
        }
    }else{
        outfile->Printf("\n    Printing of amplitude is implemented only for T1 and T2!");
        return;
    }

    if(output.size() != 0){
        int linesize = mo_title.size() - 2;
        line = "\n" + indent + std::string(linesize - indent.size(), '-');
        summary = "\n" + indent + "Norm of T" + std::to_string(rank) + name
                + " vector: (nonzero elements: " + std::to_string(number_nonzero) + ")";
        std::string strnorm = str(boost::format("%.15f.") % norm);
        std::string blank(linesize - summary.size() - strnorm.size() + 1, ' ');
        summary += blank + strnorm;

        output = title + spin_title + mo_title + line + output + line + summary + line;
    }
    outfile->Printf("\n%s", output.c_str());
}

void MRDSRG::print_intruder(const std::string &name,
                                const std::vector<std::pair<std::vector<size_t>, double> > &list)
{
    int rank = name.size();
    std::map<char, std::vector<double>> spin_to_F;
    spin_to_F['A'] = Fa;
    spin_to_F['B'] = Fb;

    std::string indent(4, ' ');
    std::string title = indent + "T" + std::to_string(rank) + " amplitudes larger than "
            + str(boost::format("%.4f") % intruder_tamp_) + " for spin case "  + name + ":";
    std::string col_title;
    std::string line;
    std::string output;

    if(rank == 1){
        int x = 30 + 2 * 3 + 2 - 11;
        std::string blank(x / 2, ' ');
        col_title += "\n" + indent + "    Amplitude         Value     "
                + blank + "Denominator" + std::string(x - blank.size(), ' ');
        line = "\n" + indent + std::string(col_title.size() - indent.size() - 1, '-');

        for(size_t n = 0; n != list.size(); ++n){
            const auto& datapair = list[n];
            std::vector<size_t> idx = datapair.first;
            size_t i = idx[0], a = idx[1];
            double fi = spin_to_F[name[0]][i], fa = spin_to_F[name[0]][a];
            double down = fi - fa;
            double v = datapair.second;

            output += "\n" + indent
                    + str(boost::format("[%3d %3c %3d %3c] %13.8f (%10.6f - %10.6f = %10.6f)")
                          % i % ' ' % a % ' ' % v % fi % fa % down);
        }
    }
    else if(rank == 2){
        int x = 50 + 4 * 3 + 2 - 11;
        std::string blank(x / 2, ' ');
        col_title += "\n" + indent + "    Amplitude         Value     "
                + blank + "Denominator" + std::string(x - blank.size(), ' ');
        line = "\n" + indent + std::string(col_title.size() - indent.size() - 1, '-');
        for(size_t n = 0; n != list.size(); ++n){
            const auto& datapair = list[n];
            std::vector<size_t> idx = datapair.first;
            size_t i = idx[0], j = idx[1], a = idx[2], b = idx[3];
            double fi = spin_to_F[name[0]][i], fj = spin_to_F[name[1]][j];
            double fa = spin_to_F[name[0]][a], fb = spin_to_F[name[1]][b];
            double down = fi + fj - fa - fb;
            double v = datapair.second;

            output += "\n" + indent
                    + str(boost::format("[%3d %3d %3d %3d] %13.8f (%10.6f + %10.6f - %10.6f - %10.6f = %10.6f)")
                          % i % j % a % b % v % fi % fj % fa % fb % down);
        }
    }else{
        outfile->Printf("\n    Printing of amplitude is implemented only for T1 and T2!");
        return;
    }

    if(output.size() != 0){
        output = title + col_title + line + output + line;
    }else{
        output = title + " NULL";
    }
    outfile->Printf("\n%s", output.c_str());
}

}}
