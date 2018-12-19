/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifndef _forte_options_h_
#define _forte_options_h_

#include "psi4/liboptions/liboptions.h"
#include <string>
#include <vector>

namespace forte {

// Types to store options

// For the bool, int, and double types store:
// ("label", default value, "description")
using bool_opt_t = std::tuple<std::string, bool, std::string>;
using int_opt_t = std::tuple<std::string, int, std::string>;
using double_opt_t = std::tuple<std::string, double, std::string>;

// For the string type stores:
// ("label", default value, "description",vector<"allowed values">)
using str_opt_t = std::tuple<std::string, std::string, std::string, std::vector<std::string>>;

// For the array type stores:
// ("label", "description")
using array_opt_t = std::tuple<std::string, std::string>;

/**
 * @brief The ForteOptions class
 */
class ForteOptions {
  public:
    /**
     * @brief ForteOptions
     */
    ForteOptions();

    /**
     * @brief ForteOptions
     * @param options a psi4 Options object
     */
    ForteOptions(psi::Options& options);

    /**
     * @brief Add a boolean option
     * @param label Option label
     * @param value Default value of the option
     * @param description Description of the option
     */
    void add_bool(const std::string& label, bool value, const std::string& description = "");

    /**
     * @brief Add a integer option
     * @param label Option label
     * @param value Default value of the option
     * @param description Description of the option
     */
    void add_int(const std::string& label, int value, const std::string& description = "");

    /**
     * @brief Add a double option
     * @param label Option label
     * @param value Default value of the option
     * @param description Description of the option
     */
    void add_double(const std::string& label, double value, const std::string& description = "");

    /**
     * @brief Add a string option
     * @param label Option label
     * @param value Default value of the option
     * @param description Description of the option
     */
    void add_str(const std::string& label, const std::string& value,
                 const std::string& description = "");

    /**
     * @brief Add a string option and provide a list of allowed option values
     * @param label Option label
     * @param value Default value of the option
     * @param description Description of the option
     * @param allowed_values An array of allowed option values
     */
    void add_str(const std::string& label, const std::string& value,
                 const std::vector<std::string>& allowed_values,
                 const std::string& description = "");

    /**
     * @brief Add an array option
     * @param label Option label
     * @param description Description of the option
     */
    void add_array(const std::string& label, const std::string& description = "");

    /**
     * @brief Get a boolean option
     * @param label Option label
     */
    bool get_bool(const std::string& label);

    /**
     * @brief Get a integer option
     * @param label Option label
     */
    int get_int(const std::string& label);

    /**
     * @brief Get a double option
     * @param label Option label
     */
    double get_double(const std::string& label);

    /**
     * @brief Get a string option
     * @param label Option label
     */
    const std::string get_str(const std::string& label);

    /**
     * @brief If an option is changed
     * @param label Option label
     */
    bool has_changed(const std::string& label);

    /// Add the options to psi4's options class
    void push_options_to_psi4(psi::Options& options);

    /**
     * @brief Generate documentation for the options registered with this object
     * @return A string with a list of options
     */
    std::string generate_documentation() const;

    /**
     * @brief Update the local copy of psi_options_
     */
    void update_psi_options(psi::Options& options);

  private:
    std::vector<bool_opt_t> bool_opts_;
    std::vector<int_opt_t> int_opts_;
    std::vector<double_opt_t> double_opts_;
    std::vector<str_opt_t> str_opts_;
    std::vector<array_opt_t> array_opts_;
    psi::Options psi_options_;
};
} // namespace forte

#endif // _forte_options_h_
