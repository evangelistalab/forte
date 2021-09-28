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

#include "base_classes/mo_space_info.h"
#include "helpers/printing.h"
#include "boost/format.hpp"
#include "dsrg_time.h"

using namespace psi;

namespace forte {

DSRG_TIME::DSRG_TIME() {
    // fill in code
    code_ = {"110", "120", "210", "220", "111", "121", "211", "221", "122", "212", "222"};

    // initialize timing vector
    timing_ = std::vector<double>(code_.size(), 0.0);

    // map code to time index
    for (size_t i = 0; i < code_.size(); ++i) {
        code_to_tidx_[code_[i]] = i;
    }
}

void DSRG_TIME::add(const std::string& code, const double& t) {
    if (test_code(code)) {
        auto iter = std::find(code_.begin(), code_.end(), code);
        if (iter != code_.end()) {
            timing_[code_to_tidx_[code]] += t;
        } else {
            create_code(code);
            timing_[code_to_tidx_[code]] += t;
        }
    }
}

void DSRG_TIME::subtract(const std::string& code, const double& t) {
    if (test_code(code)) {
        auto iter = std::find(code_.begin(), code_.end(), code);
        if (iter != code_.end()) {
            timing_[code_to_tidx_[code]] -= t;
        } else {
            print_h2("Echo from DSRG_TIME", "!!!", "!!!");
            outfile->Printf("  Cannot find time code %s. Substract nothing.", code.c_str());
        }
    }
}

void DSRG_TIME::reset() {
    for (auto& t : timing_) {
        t = 0.0;
    }
}

void DSRG_TIME::reset(const std::string& code) {
    if (test_code(code)) {
        auto iter = std::find(code_.begin(), code_.end(), code);
        if (iter != code_.end()) {
            timing_[code_to_tidx_[code]] = 0.0;
        } else {
            print_h2("Echo from DSRG_TIME", "!!!", "!!!");
            outfile->Printf("  Cannot find time code %s. Reset nothing.", code.c_str());
        }
    }
}

void DSRG_TIME::create_code(const std::string& code) {
    if (test_code(code)) {
        auto iter = std::find(code_.begin(), code_.end(), code);
        if (iter == code_.end()) {
            code_.emplace_back(code);
            timing_.emplace_back(0.0);
            size_t size = code_.size();
            code_to_tidx_[code_.back()] = size - 1;
        } else {
            print_h2("Echo from DSRG_TIME", "!!!", "!!!");
            outfile->Printf("  Time code %s is already there. Create nothing.", code.c_str());
        }
    }
}

void DSRG_TIME::delete_code(const std::string& code) {
    if (test_code(code)) {
        auto iter = std::find(code_.begin(), code_.end(), code);
        if (iter == code_.end()) {
            print_h2("Echo from DSRG_TIME", "!!!", "!!!");
            outfile->Printf("  Cannot find time code %s. Delete nothing.", code.c_str());
        } else {
            code_.erase(iter);
            const size_t offset = iter - code_.begin();
            timing_.erase(timing_.begin() + offset);
            auto it = code_to_tidx_.find(code);
            code_to_tidx_.erase(it, code_to_tidx_.end());
            for (size_t i = offset; i < code_.size(); ++i) {
                code_to_tidx_[code_[i]] = i;
            }
        }
    }
}

void DSRG_TIME::print(const std::string& code) {
    if (test_code(code)) {
        auto iter = std::find(code_.begin(), code_.end(), code);
        if (iter == code_.end()) {
            print_h2("Echo from DSRG_TIME", "!!!", "!!!");
            outfile->Printf("  Cannot find time code %s. Print nothing.", code.c_str());
        } else {
            outfile->Printf("\n    Time for [A%c,B%c] -> C%c: %10.3f s.", code[0], code[1], code[2],
                            timing_[code_to_tidx_[code]]);
        }
    }
}

void DSRG_TIME::print() {
    print_h2("DSRG_TIME for Computing Commutators in Seconds");

    std::string dash(34, '=');
    outfile->Printf("\n    %s", dash.c_str());
    for (const auto& code : code_) {
        outfile->Printf("\n    Time for [A%c,B%c] -> C%c: %10.3f", code[0], code[1], code[2],
                        timing_[code_to_tidx_[code]]);
    }
    outfile->Printf("\n    %s", dash.c_str());
}

void DSRG_TIME::print_comm_time() {
    if (timing_.size() == 11) {
        outfile->Printf("\n\n  ==> Total Timings (s) for Computing Commutators <==\n");
        std::string indent(4, ' ');
        std::string dash(53, '-');
        std::string output;
        output += indent + str(boost::format("%5c  %10s  %10s  %10s  %10s\n") % ' ' % "[H1, T1]" %
                               "[H1, T2]" % "[H2, T1]" % "[H2, T2]");
        output += indent + dash + "\n";
        output += indent + str(boost::format("%5s  %10.3f  %10.3f  %10.3f  %10.3f\n") % "-> C0" %
                               timing_[0] % timing_[1] % timing_[2] % timing_[3]);
        output += indent + str(boost::format("%5s  %10.3f  %10.3f  %10.3f  %10.3f\n") % "-> C1" %
                               timing_[4] % timing_[5] % timing_[6] % timing_[7]);
        output += indent + str(boost::format("%5s  %10c  %10.3f  %10.3f  %10.3f\n") % "-> C2" %
                               ' ' % timing_[8] % timing_[9] % timing_[10]);
        output += indent + dash + "\n";
        outfile->Printf("\n%s", output.c_str());
    } else {
        //        print_h2("Echo from DSRG_TIME", "!!!", "!!!");
        //        outfile->Printf("  Wrong size of \"timing\". Print nothing.");
        print();
    }
}

bool DSRG_TIME::test_code(const std::string& code) {
    bool out = true;
    if (code.size() != 3) {
        out = false;
        print_h2("Echo from DSRG_TIME", "!!!", "!!!");
        outfile->Printf("  Wrong code size: %s contains %d digits.", code.c_str(), code.size());
    }
    return out;
}
} // namespace forte
