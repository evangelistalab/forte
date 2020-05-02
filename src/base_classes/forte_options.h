/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include <pybind11/pybind11.h>

#include <string>
#include <vector>

namespace py = pybind11;

namespace psi {
class Options;
}

namespace forte {

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
     * @brief Set the group to which options are added
     * @param group a string with the group name (default = "")
     */
    void set_group(const std::string& group = "");

    /**
     * @brief Get the group to which options are added
     * @return the group name
     */
    const std::string& get_group();

    /**
     * @brief Add a python object option
     * @param label Option label
     * @param type the option type
     * @param default_value default value of the option
     * @param description description of the option
     */
    void add(const std::string& label, const std::string& type, pybind11::object default_value,
             const std::string& description);

    /**
     * @brief Add a python object option
     * @param label Option label
     * @param type the option type
     * @param default_value default value of the option
     * @param description description of the option
     */
    void add(const std::string& label, const std::string& type, pybind11::object default_value,
             pybind11::list allowed_values, const std::string& description);

    /**
     * @brief Get a python object option
     * @param label Option label
     * @return a py::object containing the result
     */
    std::pair<py::object, std::string> get(const std::string& label) const;

    /**
     * @brief Set a python object option
     * @param label Option label
     * @param a py::object containing the value to be set
     */
    void set(const std::string& label, const py::object val);

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
    void add_int_array(const std::string& label, const std::string& description = "");
    void add_double_array(const std::string& label, const std::string& description = "");

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
    std::string get_str(const std::string& label);

    /**
     * @brief Get a general python list
     * @param label
     * @return a py list
     */
    py::list get_gen_list(const std::string& label);

    /**
     * @brief Get a vector of int option
     * @param label Option label
     */
    std::vector<int> get_int_vec(const std::string& label);

    /**
     * @brief Get a vector of int option
     * @param label Option label
     */
    std::vector<double> get_double_vec(const std::string& label);

    /**
     * @brief Set a boolean option
     * @param label Option label
     * @param val Option value
     */
    void set_bool(const std::string& label, bool val);

    /**
     * @brief Set a integer option
     * @param label Option label
     * @param val Option value
     */
    void set_int(const std::string& label, int val);

    /**
     * @brief Set a double option
     * @param label Option label
     * @param val Option value
     */
    void set_double(const std::string& label, double val);

    /**
     * @brief Set a string option
     * @param label Option label
     * @param val Option value
     */
    void set_str(const std::string& label, const std::string& val);

    /**
     * @brief Set a general python list
     * @param label Option label
     * @param val Option value (a python list)
     */
    void set_gen_list(const std::string& label, py::list val);

    /**
     * @brief Set a vector of int option
     * @param label Option label
     * @param val Option value
     */
    void set_int_vec(const std::string& label, const std::vector<int>& val);

    /**
     * @brief Set a vector of int option
     * @param label Option label
     * @param val Option value
     */
    void set_double_vec(const std::string& label, const std::vector<double>& val);

    /**
     * @brief Register the options with Psi4's options object
     * @param options a Psi4 option object
     */
    void push_options_to_psi4(psi::Options& options);

    /**
     * @brief Read options from a Psi4's options object
     * @param options a Psi4 option object
     */
    void get_options_from_psi4(psi::Options& options);

    /**
     * @brief Generate documentation for the options registered with this object
     * @return A string with a list of options
     */
    std::string generate_documentation() const;

    /**
     * @brief Return a python dictionary with all the options registered
     */
    pybind11::dict dict();

    void set_dict(pybind11::dict dict) { dict_ = dict; }

  private:
    pybind11::dict dict_;
    std::string group_ = "";
};
} // namespace forte

#endif // _forte_options_h_
