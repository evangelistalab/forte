/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2025 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "helpers/helpers.h"

#include "base_classes/forte_options.h"

#include "state_info.h"

namespace forte {

StateInfo::StateInfo(int na, int nb, int multiplicity, int twice_ms, int irrep,
                     const std::string& irrep_label, const std::vector<size_t> gas_min,
                     const std::vector<size_t> gas_max)
    : na_(na), nb_(nb), multiplicity_(multiplicity), twice_ms_(twice_ms), irrep_(irrep),
      irrep_label_(irrep_label), gas_min_(gas_min), gas_max_(gas_max) {}

const std::vector<std::string> StateInfo::multiplicity_labels{
    "Singlet", "Doublet", "Triplet", "Quartet", "Quintet", "Sextet", "Septet", "Octet",
    "Nonet",   "Decaet",  "11-et",   "12-et",   "13-et",   "14-et",  "15-et",  "16-et",
    "17-et",   "18-et",   "19-et",   "20-et",   "21-et",   "22-et",  "23-et",  "24-et"};

int StateInfo::na() const { return na_; }

int StateInfo::nb() const { return nb_; }

int StateInfo::multiplicity() const { return multiplicity_; }

int StateInfo::twice_ms() const { return twice_ms_; }

int StateInfo::irrep() const { return irrep_; }

const std::string& StateInfo::irrep_label() const { return irrep_label_; }

const std::string& StateInfo::multiplicity_label() const {
    return multiplicity_labels[multiplicity_ - 1];
}

const std::vector<size_t>& StateInfo::gas_min() const { return gas_min_; }

const std::vector<size_t>& StateInfo::gas_max() const { return gas_max_; }

bool StateInfo::operator<(const StateInfo& rhs) const {
    // Make sure the roots are in increasing energy order for core-excited state calcualtions
    if ((gas_min_ == rhs.gas_min_) && (gas_max_ == rhs.gas_max_)) {
        return std::tie(na_, nb_, multiplicity_, twice_ms_, irrep_) <
               std::tie(rhs.na_, rhs.nb_, rhs.multiplicity_, rhs.twice_ms_, rhs.irrep_);
    } else if (gas_max_ == rhs.gas_max_) {
        // The state with a smaller gas occupation in the first gas space is 'bigger'.
        // Ground state is smaller than core-excited state under this definition.
        return gas_min_ > rhs.gas_min_;
    } else {
        return gas_max_ > rhs.gas_max_;
    }
}

bool StateInfo::operator!=(const StateInfo& rhs) const {
    return std::tie(na_, nb_, multiplicity_, twice_ms_, irrep_, gas_min_, gas_max_) !=
           std::tie(rhs.na_, rhs.nb_, rhs.multiplicity_, rhs.twice_ms_, rhs.irrep_, rhs.gas_min_,
                    rhs.gas_max_);
}

bool StateInfo::operator==(const StateInfo& rhs) const {
    return std::tie(na_, nb_, multiplicity_, twice_ms_, irrep_, gas_min_, gas_max_) ==
           std::tie(rhs.na_, rhs.nb_, rhs.multiplicity_, rhs.twice_ms_, rhs.irrep_, rhs.gas_min_,
                    rhs.gas_max_);
}

StateInfo make_state_info_from_options(std::shared_ptr<ForteOptions> options,
                                       const Symmetry& symmetry) {
    int nel = options->get_int("NEL");

    size_t multiplicity = options->get_int("MULTIPLICITY");

    // If the user did not specify ms determine the value from the input or
    // take the lowest value consistent with the value of "MULTIPLICITY"
    // For example:
    //    singlet: multiplicity = 1 -> twice_ms = 0 (ms = 0)
    //    doublet: multiplicity = 2 -> twice_ms = 1 (ms = 1/2)
    //    triplet: multiplicity = 3 -> twice_ms = 0 (ms = 0)
    size_t twice_ms = (multiplicity + 1) % 2;
    if (not options->is_none("MS")) {
        twice_ms = std::lround(2.0 * options->get_double("MS"));
    }

    if (((nel - twice_ms) % 2) != 0) {
        throw std::runtime_error("\n\n  make_state_info_from_options: Wrong value of M_s.\n\n");
    }

    size_t na = (nel + twice_ms) / 2;
    size_t nb = nel - na;

    size_t irrep = 0;

    if (not options->is_none("ROOT_SYM")) {
        irrep = options->get_int("ROOT_SYM");
    }

    std::string irrep_label = symmetry.irrep_label(irrep);
    return StateInfo(na, nb, multiplicity, twice_ms, irrep, irrep_label);
}

std::string StateInfo::str() const {
    std::string gas_restrictions;
    if (gas_min_.size() > 0) {
        gas_restrictions += " GAS min: ";
        for (size_t i : gas_min_)
            gas_restrictions += std::to_string(i) + " ";
        gas_restrictions += ";";
    }

    if (gas_max_.size() > 0) {
        gas_restrictions += " GAS max: ";
        for (size_t i : gas_max_)
            gas_restrictions += std::to_string(i) + " ";
        gas_restrictions += ";";
    }
    return str_minimum() + gas_restrictions;
}

std::string StateInfo::str_minimum() const {
    std::string irrep_label1 =
        irrep_label_.empty() ? "Irrep " + std::to_string(irrep_) : irrep_label();
    return multiplicity_label() + " (Ms = " + get_ms_string(twice_ms_) + ") " + irrep_label1;
}

std::string StateInfo::str_short() const {
    std::string multi = "m" + std::to_string(multiplicity_) + ".z" + std::to_string(twice_ms_);
    std::string sym = ".h" + std::to_string(irrep_);
    std::string gmin, gmax;
    if (gas_min_.size() > 0) {
        gmin = ".g";
        for (size_t i : gas_min_)
            gmin += "_" + std::to_string(i);
    }
    if (gas_max_.size() > 0) {
        gmax = ".g";
        for (size_t i : gas_max_)
            gmax += "_" + std::to_string(i);
    }
    return multi + sym + gmin + gmax;
}

std::size_t StateInfo::hash() const {
    // here we form a string representation and then call std::hash<std::string> on it.
    std::string repr = std::to_string(na_);
    repr += "_" + std::to_string(nb_);
    repr += "_" + std::to_string(multiplicity_);
    repr += "_" + std::to_string(twice_ms_);
    repr += "_" + std::to_string(irrep_);
    for (size_t i : gas_min_)
        repr += "_" + std::to_string(i);
    for (size_t i : gas_max_)
        repr += "_" + std::to_string(i);
    return std::hash<std::string>{}(repr);
}

} // namespace forte
