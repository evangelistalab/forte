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
#include "sadsrg.h"

using namespace psi;

namespace forte {

std::vector<std::string> SADSRG::diag_one_labels() {
    std::vector<std::string> labels;
    for (const std::string& p : {core_label_, actv_label_, virt_label_}) {
        labels.push_back(p + p);
    }
    return labels;
}

std::vector<std::string> SADSRG::od_one_labels_hp() {
    std::vector<std::string> labels;
    for (const std::string& p : {core_label_, actv_label_}) {
        for (const std::string& q : {actv_label_, virt_label_}) {
            if (p == actv_label_ && q == actv_label_) {
                continue;
            }
            labels.push_back(p + q);
        }
    }
    return labels;
}

std::vector<std::string> SADSRG::od_one_labels_ph() {
    std::vector<std::string> blocks1(od_one_labels_hp());
    for (auto& block : blocks1) {
        std::swap(block[0], block[1]);
    }
    return blocks1;
}

std::vector<std::string> SADSRG::od_one_labels() {
    std::vector<std::string> labels(od_one_labels_hp());
    std::vector<std::string> temp(od_one_labels_ph());
    labels.insert(std::end(labels), std::begin(temp), std::end(temp));
    return labels;
}

std::vector<std::string> SADSRG::od_two_labels_hhpp() {
    std::vector<std::string> labels;
    for (const std::string& p : {core_label_, actv_label_}) {
        for (const std::string& q : {core_label_, actv_label_}) {
            for (const std::string& r : {actv_label_, virt_label_}) {
                for (const std::string& s : {actv_label_, virt_label_}) {
                    if (p == actv_label_ && q == actv_label_ && r == actv_label_ &&
                        s == actv_label_) {
                        continue;
                    }
                    labels.push_back(p + q + r + s);
                }
            }
        }
    }
    return labels;
}

std::vector<std::string> SADSRG::od_two_labels_pphh() {
    std::vector<std::string> labels(od_two_labels_hhpp());
    for (auto& block : labels) {
        std::swap(block[0], block[2]);
        std::swap(block[1], block[3]);
    }
    return labels;
}

std::vector<std::string> SADSRG::od_two_labels() {
    std::vector<std::string> labels(od_two_labels_hhpp());
    std::vector<std::string> temp(od_two_labels_pphh());
    labels.insert(std::end(labels), std::begin(temp), std::end(temp));
    return labels;
}

std::vector<std::string> SADSRG::diag_two_labels() {
    std::vector<std::string> general{core_label_, actv_label_, virt_label_};

    std::vector<std::string> all;
    for (const std::string& p : general) {
        for (const std::string& q : general) {
            for (const std::string& r : general) {
                for (const std::string& s : general) {
                    all.push_back(p + q + r + s);
                }
            }
        }
    }

    std::vector<std::string> od(od_two_labels());
    std::sort(od.begin(), od.end());
    std::sort(all.begin(), all.end());

    std::vector<std::string> labels;
    std::set_symmetric_difference(all.begin(), all.end(), od.begin(), od.end(),
                                  std::back_inserter(labels));

    return labels;
}

std::vector<std::string> SADSRG::re_two_labels() {
    std::vector<std::vector<std::string>> half_labels{
        {core_label_ + core_label_},
        {actv_label_ + actv_label_},
        {virt_label_ + virt_label_},
        {core_label_ + actv_label_, actv_label_ + core_label_},
        {core_label_ + virt_label_, virt_label_ + core_label_},
        {actv_label_ + virt_label_, virt_label_ + actv_label_}};

    std::vector<std::string> labels;
    for (const auto& half : half_labels) {
        for (const std::string& half1 : half) {
            for (const std::string& half2 : half) {
                labels.push_back(half1 + half2);
            }
        }
    }

    return labels;
}

std::vector<std::string> SADSRG::nivo_labels() {
    std::vector<std::string> elementary_labels{core_label_, actv_label_, virt_label_};
    std::vector<std::string> blocks_exclude_V3;

    for (std::string s0 : elementary_labels) {
        for (std::string s1 : elementary_labels) {
            for (std::string s2 : elementary_labels) {
                for (std::string s3 : elementary_labels) {
                    std::string s = s0 + s1 + s2 + s3;
                    if (std::count(s.begin(), s.end(), virt_label_[0]) < 3) {
                        blocks_exclude_V3.push_back(s);
                    }
                }
            }
        }
    }

    return blocks_exclude_V3;
}

} // namespace forte
