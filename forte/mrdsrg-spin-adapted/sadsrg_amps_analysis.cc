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

#include "psi4/libpsi4util/PsiOutStream.h"

#include "sadsrg.h"

using namespace psi;

namespace forte {

void SADSRG::internal_amps_T1(BlockedTensor& T1) {
    if (internal_amp_.find("SINGLES") != std::string::npos) {
        // TODO: to be filled
    } else {
        T1.block("aa").zero();
    }
}

void SADSRG::internal_amps_T2(BlockedTensor& T2) {
    if (internal_amp_.find("DOUBLES") != std::string::npos) {
        // TODO: to be filled
    } else {
        T2.block("aaaa").zero();
    }
}

void SADSRG::analyze_amplitudes(std::string name, BlockedTensor& T1, BlockedTensor& T2) {
    if (!name.empty())
        name += " ";
    outfile->Printf("\n\n  ==> %sExcitation Amplitudes Summary <==\n", name.c_str());
    outfile->Printf("\n    Active Indices:");
    int c = 0;
    for (const auto& idx : actv_mos_) {
        outfile->Printf(" %4zu", idx);
        if (++c % 10 == 0)
            outfile->Printf("\n    %16c", ' ');
    }
    auto lt1 = check_t1(T1);
    auto lt2 = check_t2(T2);

    outfile->Printf("\n\n  ==> Possible Intruders <==\n");
    print_t1_intruder(lt1);
    print_t2_intruder(lt2);
    if (!semi_canonical_) {
        outfile->Printf(
            "\n  Warning: Amplitudes are not in semicanonical basis, but denominators are!");
    }
}

std::vector<std::pair<std::vector<size_t>, double>> SADSRG::check_t2(BlockedTensor& T2) {
    size_t nonzero = 0;
    T2.citerate([&](const std::vector<size_t>&, const std::vector<SpinType>&, const double& value) {
        if (std::fabs(value) > 1.0e-15) {
            nonzero += 1;
        }
    });

    std::vector<std::pair<std::vector<size_t>, double>> t2(ntamp_ < nonzero ? ntamp_ : nonzero);
    std::vector<std::pair<std::vector<size_t>, double>> lt2;

    // check blocks
    std::vector<std::string> T2blocks;
    std::vector<std::vector<std::string>> equivalent_blocks{
        {"aaaa"},         {"aavv"},         {"ccaa"},         {"ccvv"},         {"aaav", "aava"},
        {"ccav", "ccva"}, {"caaa", "acaa"}, {"acvv", "cavv"}, {"caav", "acva"}, {"acav", "cava"}};
    for (const auto& blocks : equivalent_blocks) {
        for (const std::string& block : blocks) {
            if (T2.is_block(block)) {
                T2blocks.push_back(block);
                break;
            }
        }
    }

    for (const std::string& block : T2blocks) {
        bool sym = (block[0] == block[1]) and (block[2] == block[3]);

        T2.block(block).citerate([&](const std::vector<size_t>& i, const double& value) {
            if (std::fabs(value) > 1.0e-15) {
                size_t idx0 = label_to_spacemo_[block[0]][i[0]];
                size_t idx1 = label_to_spacemo_[block[1]][i[1]];
                size_t idx2 = label_to_spacemo_[block[2]][i[2]];
                size_t idx3 = label_to_spacemo_[block[3]][i[3]];

                // for blocks like ccvv, only test c0 < c1 or (c0 = c1 and v0 <= v1)
                if ((!sym) or (sym && (i[0] <= i[1]) && (i[0] != i[1] or i[2] <= i[3]))) {
                    std::vector<size_t> indices{idx0, idx1, idx2, idx3};
                    std::pair<std::vector<size_t>, double> idx_value =
                        std::make_pair(indices, value);

                    if (std::fabs(value) >= std::fabs(t2[0].second)) {
                        std::pop_heap(t2.begin(), t2.end(), sort_pair_second_descend);
                        t2.pop_back();

                        t2.push_back(idx_value);
                        std::push_heap(t2.begin(), t2.end(), sort_pair_second_descend);
                    }

                    if (std::fabs(value) > std::fabs(intruder_tamp_)) {
                        lt2.push_back(idx_value);
                    }
                }
            }
        });
    }

    std::sort(t2.begin(), t2.end(), sort_pair_second_descend);
    std::sort(lt2.begin(), lt2.end(), sort_pair_second_descend);

    // print summary
    if (t2.size())
        print_t2_summary(t2, T2.norm(), nonzero);

    return lt2;
}

std::vector<std::pair<std::vector<size_t>, double>> SADSRG::check_t1(BlockedTensor& T1) {
    size_t nonzero = 0;
    T1.citerate([&](const std::vector<size_t>&, const std::vector<SpinType>&, const double& value) {
        if (std::fabs(value) > 1.0e-15) {
            nonzero += 1;
        }
    });

    std::vector<std::pair<std::vector<size_t>, double>> t1(ntamp_ < nonzero ? ntamp_ : nonzero);
    std::vector<std::pair<std::vector<size_t>, double>> lt1;

    for (const std::string& block : T1.block_labels()) {
        T1.block(block).citerate([&](const std::vector<size_t>& i, const double& value) {
            if (std::fabs(value) > 1.0e-15) {
                size_t idx0 = label_to_spacemo_[block[0]][i[0]];
                size_t idx1 = label_to_spacemo_[block[1]][i[1]];

                std::vector<size_t> indices{idx0, idx1};
                std::pair<std::vector<size_t>, double> idx_value = std::make_pair(indices, value);

                if (std::fabs(value) >= std::fabs(t1[0].second)) {
                    std::pop_heap(t1.begin(), t1.end(), sort_pair_second_descend);
                    t1.pop_back();

                    t1.push_back(idx_value);
                    std::push_heap(t1.begin(), t1.end(), sort_pair_second_descend);
                }

                if (std::fabs(value) > std::fabs(intruder_tamp_)) {
                    lt1.push_back(idx_value);
                }
            }
        });
    }

    std::sort(t1.begin(), t1.end(), sort_pair_second_descend);
    std::sort(lt1.begin(), lt1.end(), sort_pair_second_descend);

    // print summary
    if (t1.size())
        print_t1_summary(t1, T1.norm(), nonzero);

    return lt1;
}

void SADSRG::print_t1_summary(const std::vector<std::pair<std::vector<size_t>, double>>& list,
                              const double& norm, const size_t& number_nonzero) {
    outfile->Printf("\n    Largest T1 amplitudes (absolute values):");
    std::string dash(65, '-');

    std::string temp = "    i    a           ";
    outfile->Printf("\n    %s %s     i    a", temp.c_str(), temp.c_str());
    outfile->Printf("\n    %s", dash.c_str());

    for (size_t n = 0, n_size = list.size(); n != n_size; ++n) {
        if (n % 3 == 0)
            outfile->Printf("\n    ");
        else
            outfile->Printf(" ");
        const auto& datapair = list[n];
        std::vector<size_t> idx = datapair.first;
        outfile->Printf("[%4zu %4zu]%10.7f", idx[0], idx[1], std::fabs(datapair.second));
    }
    outfile->Printf("\n    %s", dash.c_str());

    outfile->Printf("\n    2-Norm of T1 vector: %44.15f", norm);
    outfile->Printf("\n    Number of nonzero elements: %37zu", number_nonzero);

    outfile->Printf("\n    %s", dash.c_str());
}

void SADSRG::print_t2_summary(const std::vector<std::pair<std::vector<size_t>, double>>& list,
                              const double& norm, const size_t& number_nonzero) {
    outfile->Printf("\n    Largest T2 amplitudes (absolute values):");
    std::string dash(95, '-');

    std::string temp = "    i    j    a    b           ";
    outfile->Printf("\n    %s %s     i    j    a    b", temp.c_str(), temp.c_str());
    outfile->Printf("\n    %s", dash.c_str());

    for (size_t n = 0, n_size = list.size(); n != n_size; ++n) {
        if (n % 3 == 0)
            outfile->Printf("\n    ");
        else
            outfile->Printf(" ");
        const auto& datapair = list[n];
        std::vector<size_t> idx = datapair.first;
        outfile->Printf("[%4zu %4zu %4zu %4zu]%10.7f", idx[0], idx[1], idx[2], idx[3],
                        std::fabs(datapair.second));
    }
    outfile->Printf("\n    %s", dash.c_str());

    outfile->Printf("\n    2-Norm of T2 vector: %74.15f", norm);
    outfile->Printf("\n    Number of nonzero elements: %67zu", number_nonzero);

    outfile->Printf("\n    %s", dash.c_str());
}

void SADSRG::print_t1_intruder(const std::vector<std::pair<std::vector<size_t>, double>>& list) {
    outfile->Printf("\n    T1 amplitudes larger than %.4f:", intruder_tamp_);

    size_t nele = list.size();
    if (nele == 0) {
        outfile->Printf(" NULL");
    } else {
        std::string dash(64, '-');
        outfile->Printf("\n     Amplitudes      Value                   Denominator");
        outfile->Printf("\n    %s", dash.c_str());
        for (size_t n = 0; n != nele; ++n) {
            const auto& datapair = list[n];
            std::vector<size_t> idx = datapair.first;
            size_t i = idx[0], a = idx[1];
            double fi = Fdiag_[i], fa = Fdiag_[a];
            double denominator = fi - fa;
            double v = datapair.second;
            outfile->Printf("\n    [%4zu %4zu] %13.9f (%10.6f - %10.6f = %10.6f)", i, a, v, fi, fa,
                            denominator);
        }
        outfile->Printf("\n    %s", dash.c_str());
    }
}

void SADSRG::print_t2_intruder(const std::vector<std::pair<std::vector<size_t>, double>>& list) {
    outfile->Printf("\n    T2 amplitudes larger than %.4f:", intruder_tamp_);

    size_t nele = list.size();
    if (nele == 0) {
        outfile->Printf(" NULL");
    } else {
        std::string dash(100, '-');
        outfile->Printf("\n     Amplitudes      Value                   Denominator");
        outfile->Printf("\n    %s", dash.c_str());
        for (size_t n = 0; n != nele; ++n) {
            const auto& datapair = list[n];
            std::vector<size_t> idx = datapair.first;
            size_t i = idx[0], j = idx[1], a = idx[2], b = idx[3];
            double fi = Fdiag_[i], fj = Fdiag_[j];
            double fa = Fdiag_[a], fb = Fdiag_[b];
            double denominator = fi + fj - fa - fb;
            double v = datapair.second;
            outfile->Printf(
                "\n    [%4zu %4zu %4zu %4zu] %13.9f (%10.6f + %10.6f - %10.6f - %10.6f = %10.6f)",
                i, j, a, b, v, fi, fj, fa, fb, denominator);
        }
        outfile->Printf("\n    %s", dash.c_str());
    }
}
} // namespace forte
