/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#include <algorithm>
#include "boost/format.hpp"

#include "psi4/libpsi4util/PsiOutStream.h"

#include "dsrg_mrpt.h"
#include "helpers/timer.h"
#include "helpers/printing.h"


using namespace psi;

namespace forte {

void DSRG_MRPT::compute_T_1st(BlockedTensor& V, BlockedTensor& T2, BlockedTensor& F,
                              BlockedTensor& T1) {
    print_h2("Build Initial Amplitude from DSRG-MRPT2");
    compute_T2_1st(V, T2);
    compute_T1_1st(F, T2, T1);

    // check initial amplitudes
    analyze_amplitudes(T1, T2, "First-Order");
}

void DSRG_MRPT::compute_T2_1st(BlockedTensor& V, BlockedTensor& T2) {
    local_timer timer;
    std::string str = "Computing T2 amplitudes ...";
    outfile->Printf("\n    %-35s", str.c_str());
    T2max_ = 0.0, T2norm_ = 0.0;

    // copy V to T2
    for (const auto& block : T2.block_labels()) {
        T2.block(block)("pqrs") = V.block(block)("pqrs");
    }

    // scale T2 by renormalized denominator
    T2.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        if (std::fabs(value) > 1.0e-15) {
            value *= dsrg_source_->compute_renormalized_denominator(Fdiag_[i[0]] + Fdiag_[i[1]] -
                                                                    Fdiag_[i[2]] - Fdiag_[i[3]]);
            T2norm_ += value * value;
            if (std::fabs(value) > std::fabs(T2max_))
                T2max_ = value;
        } else {
            value = 0.0;
        }
    });

    // because the aaaa block is not stored, it is not zeroed.

    T2norm_ = std::sqrt(T2norm_);
    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void DSRG_MRPT::compute_T1_1st(BlockedTensor& F, BlockedTensor& T2, BlockedTensor& T1) {
    local_timer timer;
    std::string str = "Computing T1 amplitudes ...";
    outfile->Printf("\n    %-35s", str.c_str());
    T1max_ = 0.0, T1norm_ = 0.0;

    // copy F to T1
    for (const auto& block : T1.block_labels()) {
        T1.block(block)("pq") = F.block(block)("pq");
    }

    // create a temp BlockedTensor
    ambit::BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "Temp", {"aa"});
    temp["xu"] = 0.5 * L1_["xu"];

    // scale temp by denominator
    temp.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= Fdiag_[i[0]] - Fdiag_[i[1]];
    });

    // contribution from temp and T2 (extra work because we do not have aaaa of
    // T2)
    T1["ie"] += 2.0 * T2["iuex"] * temp["xu"];
    T1["ie"] -= T2["iuxe"] * temp["xu"];
    T1["my"] += 2.0 * T2["muyx"] * temp["xu"];
    T1["my"] -= T2["muxy"] * temp["xu"];

    // scale T1 by renormalized denominator
    T1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        if (std::fabs(value) > 1.0e-15) {
            value *= dsrg_source_->compute_renormalized_denominator(Fdiag_[i[0]] - Fdiag_[i[1]]);
            T1norm_ += value * value;
            if (std::fabs(value) > std::fabs(T1max_))
                T1max_ = value;
        } else {
            value = 0.0;
        }
    });

    // no need to zero internal amplitudes, because we do not store them

    T1norm_ = std::sqrt(T1norm_);
    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void DSRG_MRPT::analyze_amplitudes(BlockedTensor& T1, BlockedTensor& T2, std::string name) {
    if (!name.empty())
        name += " ";
    outfile->Printf("\n\n  ==> %sExcitation Amplitudes Summary <==\n", name.c_str());
    outfile->Printf("\n    Active Indices: ");
    int c = 0;
    for (const auto& idx : actv_mos_) {
        outfile->Printf("%4zu ", idx);
        if (++c % 10 == 0)
            outfile->Printf("\n    %16c", ' ');
    }

    check_t1(T1);
    check_t2(T2);

    outfile->Printf("\n\n  ==> Possible Intruders <==\n");
    print_intruder("A", lt1_);
    print_intruder("AB", lt2_);
}

// Binary function to achieve sorting a vector of pair<vector, double>
// according to the double value in decending order
template <class T1, class T2, class G3 = std::greater<T2>> struct rsort_pair_second {
    bool operator()(const std::pair<T1, T2>& left, const std::pair<T1, T2>& right) {
        G3 p;
        return p(std::fabs(left.second), std::fabs(right.second));
    }
};

void DSRG_MRPT::check_t2(BlockedTensor& T2) {
    size_t nonzero = 0;
    std::vector<std::pair<std::vector<size_t>, double>> t2_idx_pair;

    for (const std::string& block : T2.block_labels()) {
        T2.block(block).citerate([&](const std::vector<size_t>& i, const double& value) {
            if (std::fabs(value) > 1.0e-15) {
                size_t idx0 = label_to_spacemo_[block[0]][i[0]];
                size_t idx1 = label_to_spacemo_[block[1]][i[1]];
                size_t idx2 = label_to_spacemo_[block[2]][i[2]];
                size_t idx3 = label_to_spacemo_[block[3]][i[3]];

                ++nonzero;

                if ((idx0 <= idx1) && (idx2 <= idx3)) {
                    std::vector<size_t> indices = {idx0, idx1, idx2, idx3};
                    std::pair<std::vector<size_t>, double> idx_value =
                        std::make_pair(indices, value);

                    t2_idx_pair.push_back(idx_value);
                    std::sort(t2_idx_pair.begin(), t2_idx_pair.end(),
                              rsort_pair_second<std::vector<size_t>, double>());
                    if (t2_idx_pair.size() == ntamp_ + 1) {
                        t2_idx_pair.pop_back();
                    }

                    if (std::fabs(value) > std::fabs(intruder_tamp_)) {
                        lt2_.push_back(idx_value);
                    }
                    std::sort(lt2_.begin(), lt2_.end(),
                              rsort_pair_second<std::vector<size_t>, double>());
                }
            }
        });
    }

    // print summary
    print_amp_summary("AB", t2_idx_pair, T2norm_, nonzero);
}

void DSRG_MRPT::check_t1(BlockedTensor& T1) {
    size_t nonzero = 0;
    std::vector<std::pair<std::vector<size_t>, double>> t1_idx_pair;

    for (const std::string& block : T1.block_labels()) {
        T1.block(block).citerate([&](const std::vector<size_t>& i, const double& value) {
            if (std::fabs(value) > 1.0e-15) {
                size_t idx0 = label_to_spacemo_[block[0]][i[0]];
                size_t idx1 = label_to_spacemo_[block[1]][i[1]];

                std::vector<size_t> indices = {idx0, idx1};
                std::pair<std::vector<size_t>, double> idx_value = std::make_pair(indices, value);

                ++nonzero;

                t1_idx_pair.push_back(idx_value);
                std::sort(t1_idx_pair.begin(), t1_idx_pair.end(),
                          rsort_pair_second<std::vector<size_t>, double>());
                if (t1_idx_pair.size() == ntamp_ + 1) {
                    t1_idx_pair.pop_back();
                }

                if (std::fabs(value) > std::fabs(intruder_tamp_)) {
                    lt1_.push_back(idx_value);
                }
                std::sort(lt1_.begin(), lt1_.end(),
                          rsort_pair_second<std::vector<size_t>, double>());
            }
        });
    }

    // print summary
    print_amp_summary("A", t1_idx_pair, T1norm_, nonzero);
}

void DSRG_MRPT::print_amp_summary(const std::string& name,
                                  const std::vector<std::pair<std::vector<size_t>, double>>& list,
                                  const double& norm, const size_t& number_nonzero) {
    int rank = name.size();
    std::map<char, std::string> spin_case;
    spin_case['A'] = " ";
    spin_case['B'] = "_";

    std::string indent(4, ' ');
    std::string title =
        indent + "Largest T" + std::to_string(rank) + " amplitudes for spin case " + name + ":";
    std::string spin_title;
    std::string mo_title;
    std::string line;
    std::string output;
    std::string summary;

    auto extendstr = [&](std::string s, int n) {
        std::string o(s);
        while ((--n) > 0)
            o += s;
        return o;
    };

    if (rank == 1) {
        spin_title += str(boost::format(" %3c %3c %3c %3c %9c ") % spin_case[name[0]] % ' ' %
                          spin_case[name[0]] % ' ' % ' ');
        if (spin_title.find_first_not_of(' ') != std::string::npos) {
            spin_title = "\n" + indent + extendstr(spin_title, 3);
        } else {
            spin_title = "";
        }
        mo_title += str(boost::format(" %3c %3c %3c %3c %9c ") % 'i' % ' ' % 'a' % ' ' % ' ');
        mo_title = "\n" + indent + extendstr(mo_title, 3);
        for (size_t n = 0; n != list.size(); ++n) {
            if (n % 3 == 0)
                output += "\n" + indent;
            const auto& datapair = list[n];
            std::vector<size_t> idx = datapair.first;
            output += str(boost::format("[%3d %3c %3d %3c]%9.6f ") % idx[0] % ' ' % idx[1] % ' ' %
                          datapair.second);
        }
    } else if (rank == 2) {
        spin_title += str(boost::format(" %3c %3c %3c %3c %9c ") % spin_case[name[0]] %
                          spin_case[name[1]] % spin_case[name[0]] % spin_case[name[1]] % ' ');
        if (spin_title.find_first_not_of(' ') != std::string::npos) {
            spin_title = "\n" + indent + extendstr(spin_title, 3);
        } else {
            spin_title = "";
        }
        mo_title += str(boost::format(" %3c %3c %3c %3c %9c ") % 'i' % 'j' % 'a' % 'b' % ' ');
        mo_title = "\n" + indent + extendstr(mo_title, 3);
        for (size_t n = 0; n != list.size(); ++n) {
            if (n % 3 == 0)
                output += "\n" + indent;
            const auto& datapair = list[n];
            std::vector<size_t> idx = datapair.first;
            output += str(boost::format("[%3d %3d %3d %3d]%9.6f ") % idx[0] % idx[1] % idx[2] %
                          idx[3] % datapair.second);
        }
    } else {
        outfile->Printf("\n    Printing of amplitude is implemented only for T1 and T2!");
        return;
    }

    if (output.size() != 0) {
        int linesize = mo_title.size() - 2;
        line = "\n" + indent + std::string(linesize - indent.size(), '-');
        summary = "\n" + indent + "Norm of T" + std::to_string(rank) + name +
                  " vector: (nonzero elements: " + std::to_string(number_nonzero) + ")";
        std::string strnorm = str(boost::format("%.15f.") % norm);
        std::string blank(linesize - summary.size() - strnorm.size() + 1, ' ');
        summary += blank + strnorm;

        output = title + spin_title + mo_title + line + output + line + summary + line;
    } else {
        output = title + " NULL";
    }
    outfile->Printf("\n%s", output.c_str());
}

void DSRG_MRPT::print_intruder(const std::string& name,
                               const std::vector<std::pair<std::vector<size_t>, double>>& list) {
    int rank = name.size();

    std::string indent(4, ' ');
    std::string title = indent + "T" + std::to_string(rank) + " amplitudes larger than " +
                        str(boost::format("%.4f") % intruder_tamp_) + " for spin case " + name +
                        ":";
    std::string col_title;
    std::string line;
    std::string output;

    if (rank == 1) {
        int x = 30 + 2 * 3 + 2 - 11;
        std::string blank(x / 2, ' ');
        col_title += "\n" + indent + "    Amplitude         Value     " + blank + "Denominator" +
                     std::string(x - blank.size(), ' ');
        line = "\n" + indent + std::string(col_title.size() - indent.size() - 1, '-');

        for (size_t n = 0; n != list.size(); ++n) {
            const auto& datapair = list[n];
            std::vector<size_t> idx = datapair.first;
            size_t i = idx[0], a = idx[1];
            double fi = Fdiag_[i], fa = Fdiag_[a];
            double down = fi - fa;
            double v = datapair.second;

            output += "\n" + indent +
                      str(boost::format("[%3d %3c %3d %3c] %13.8f (%10.6f - %10.6f = %10.6f)") % i %
                          ' ' % a % ' ' % v % fi % fa % down);
        }
    } else if (rank == 2) {
        int x = 50 + 4 * 3 + 2 - 11;
        std::string blank(x / 2, ' ');
        col_title += "\n" + indent + "    Amplitude         Value     " + blank + "Denominator" +
                     std::string(x - blank.size(), ' ');
        line = "\n" + indent + std::string(col_title.size() - indent.size() - 1, '-');
        for (size_t n = 0; n != list.size(); ++n) {
            const auto& datapair = list[n];
            std::vector<size_t> idx = datapair.first;
            size_t i = idx[0], j = idx[1], a = idx[2], b = idx[3];
            double fi = Fdiag_[i], fj = Fdiag_[j];
            double fa = Fdiag_[a], fb = Fdiag_[b];
            double down = fi + fj - fa - fb;
            double v = datapair.second;

            output += "\n" + indent + str(boost::format("[%3d %3d %3d %3d] %13.8f (%10.6f + "
                                                        "%10.6f - %10.6f - %10.6f = %10.6f)") %
                                          i % j % a % b % v % fi % fj % fa % fb % down);
        }
    } else {
        outfile->Printf("\n    Printing of amplitude is implemented only for T1 and T2!");
        return;
    }

    if (output.size() != 0) {
        output = title + col_title + line + output + line;
    } else {
        output = title + " NULL";
    }
    outfile->Printf("\n%s", output.c_str());
}
} // namespace forte

