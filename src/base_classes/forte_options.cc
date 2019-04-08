#include "base_classes/forte_options.h"
#include "base_classes/mo_space_info.h"
#include "helpers/helpers.h"

#include "psi4/libpsi4util/PsiOutStream.h"

using namespace pybind11::literals;

using namespace psi;

namespace forte {

std::string rst_bold(const std::string& s);

std::string option_formatter(const std::string& type, const std::string& label,
                             const std::string& default_value, const std::string& description,
                             const std::string& allowed_values);

ForteOptions::ForteOptions() {}

ForteOptions::ForteOptions(psi::Options& options) : psi_options_(options) {}

pybind11::dict ForteOptions::dict() { return dict_; }

py::dict make_option(const std::string& type, const std::string& group, py::object default_value,
                     const std::string& description) {
    return py::dict("type"_a = type, "group"_a = py::str(group), "value"_a = default_value,
                    "default_value"_a = default_value, "description"_a = description.c_str());
}

py::dict make_option(const std::string& type, const std::string& group, py::object default_value,
                     py::list allowed_values, const std::string& description) {
    return py::dict("type"_a = type, "group"_a = py::str(group), "value"_a = default_value,
                    "default_value"_a = default_value, "allowed_values"_a = allowed_values,
                    "description"_a = description.c_str());
}

void ForteOptions::set_group(const std::string& group) {
    group_ = group;
    // outfile->Printf("Setting group to %s",group_.c_str());
}

const std::string& ForteOptions::get_group() { return group_; }

void ForteOptions::add(const std::string& label, const std::string& type, py::object default_value,
                       const std::string& description) {
    dict_[label.c_str()] = make_option(type, group_, default_value, description);
}

void ForteOptions::add(const std::string& label, const std::string& type, py::object default_value,
                       py::list allowed_values, const std::string& description) {
    dict_[label.c_str()] = make_option(type, group_, default_value, allowed_values, description);
}

py::object ForteOptions::get(const std::string& label) {
    py::object result = py::cast<py::none>(Py_None);
    if (dict_.contains(label.c_str())) {
        result = dict_[label.c_str()]["value"];
    }
    return result;
}

void ForteOptions::add_bool(const std::string& label, bool value, const std::string& description) {
    add(label, "bool", py::bool_(value), description);
}

void ForteOptions::add_int(const std::string& label, int value, const std::string& description) {
    int_opts_.push_back(std::make_tuple(label, value, description));
//    add(label, "int", py::int_(value), description);
}

void ForteOptions::add_double(const std::string& label, double value,
                              const std::string& description) {
    double_opts_.push_back(std::make_tuple(label, value, description));
//    add(label, "float", py::float_(value), description);
}

void ForteOptions::add_str(const std::string& label, const std::string& value,
                           const std::string& description) {
    str_opts_.push_back(std::make_tuple(label, value, description, std::vector<std::string>()));
//    add(label, "str", py::str(value), description);
}

void ForteOptions::add_str(const std::string& label, const std::string& value,
                           const std::vector<std::string>& allowed_values,
                           const std::string& description) {
    auto allowed_values_list = py::list();
    for (const auto& s : allowed_values) {
        allowed_values_list.append(py::str(s));
    }
    add(label, "str", py::str(value), allowed_values_list, description);
}

void ForteOptions::add_array(const std::string& label, const std::string& description) {
    array_opts_.push_back(std::make_tuple(label, description));
//    add(label, "list", py::str("empty"), description);
}

// py::object ForteOptions::get(const std::string& label) { dict_[label.c_str()]["value"]; }

bool ForteOptions::get_bool(const std::string& label) { return py::cast<bool>(get(label)); }

int ForteOptions::get_int(const std::string& label) { return psi_options_.get_int(label); }

double ForteOptions::get_double(const std::string& label) { return psi_options_.get_double(label); }

std::string ForteOptions::get_str(const std::string& label) { return psi_options_.get_str(label); }

std::vector<int> ForteOptions::get_int_vec(const std::string& label) {
    return psi_options_.get_int_vector(label);
}

std::vector<double> ForteOptions::get_double_vec(const std::string& label) {
    return psi_options_.get_double_vector(label);
}

bool ForteOptions::has_changed(const std::string& label) {
    return psi_options_[label].has_changed();
}

void ForteOptions::push_options_to_psi4(psi::Options& options) {
    for (auto item : dict_) {
        auto label = py::cast<std::string>(item.first);
        auto type = py::cast<std::string>(item.second["type"]);
        auto py_default_value = item.second["default_value"];
        if (type == "bool") {
            options.add_bool(label, py::cast<bool>(py_default_value));
        }
    }

    for (const auto& opt : int_opts_) {
        options.add_int(std::get<0>(opt), std::get<1>(opt));
    }

    for (const auto& opt : double_opts_) {
        options.add_double(std::get<0>(opt), std::get<1>(opt));
    }

    for (const auto& opt : str_opts_) {
        if (std::get<3>(opt).size() > 0) {
            const std::vector<std::string>& str_vec = std::get<3>(opt);
            std::string allowed = to_string(str_vec, " ");
            options.add_str(std::get<0>(opt), std::get<1>(opt), allowed);
        } else {
            options.add_str(std::get<0>(opt), std::get<1>(opt));
        }
    }

    for (const auto& opt : array_opts_) {
        options.add(std::get<0>(opt), new psi::ArrayType());
    }
}

void ForteOptions::get_options_from_psi4(psi::Options& options) {
    //    for (auto item : dict_) {
    //        auto label = py::cast<std::string>(item.first);
    //        options.get_bool(label);
    //    }

    for (auto item : dict_) {
        auto label = py::cast<std::string>(item.first);
        auto type = py::cast<std::string>(item.second["type"]);
        if (type == "bool") {
            bool value = options.get_bool(label);
            item.second["value"] = py::cast(value);
        }
    }
}

void ForteOptions::update_psi_options(psi::Options& options) { psi_options_ = options; }

std::string ForteOptions::generate_documentation() const {
    std::vector<std::pair<std::string, std::string>> option_docs_list;

    //    for (const auto& opt : bool_opts_) {
    //        const std::string& label = std::get<0>(opt);
    //        const std::string& default_value = std::get<1>(opt) ? "True" : "False";
    //        const std::string& description = std::get<2>(opt);
    //        std::string option_text =
    //            option_formatter("Boolean", label, default_value, description, "");
    //        outfile->Printf("\n %s", label.c_str());
    //        option_docs_list.push_back(std::make_pair(label, option_text));
    //    }

    for (const auto& opt : int_opts_) {
        const std::string& label = std::get<0>(opt);
        const std::string& default_value = std::to_string(std::get<1>(opt));
        const std::string& description = std::get<2>(opt);
        std::string option_text =
            option_formatter("Integer", label, default_value, description, "");
        outfile->Printf("\n %s", label.c_str());
        option_docs_list.push_back(std::make_pair(label, option_text));
    }

    for (const auto& opt : double_opts_) {
        const std::string& label = std::get<0>(opt);
        const std::string& default_value = std::to_string(std::get<1>(opt));
        const std::string& description = std::get<2>(opt);
        std::string option_text = option_formatter("Double", label, default_value, description, "");
        outfile->Printf("\n %s", label.c_str());
        option_docs_list.push_back(std::make_pair(label, option_text));
    }

    for (const auto& opt : str_opts_) {
        const std::string& label = std::get<0>(opt);
        const std::string& default_value = std::get<1>(opt);
        const std::string& description = std::get<2>(opt);
        const std::string& allowed_values = to_string(std::get<3>(opt), ", ");
        std::string option_text =
            option_formatter("String", label, default_value, description, allowed_values);
        outfile->Printf("\n %s", label.c_str());
        option_docs_list.push_back(std::make_pair(label, option_text));
    }

    for (const auto& opt : array_opts_) {
        const std::string& label = std::get<0>(opt);
        const std::string& description = std::get<1>(opt);
        std::string option_text = option_formatter("Array", label, "[]", description, "");
        outfile->Printf("\n %s", label.c_str());
        option_docs_list.push_back(std::make_pair(label, option_text));
    }

    std::sort(option_docs_list.begin(), option_docs_list.end());
    std::vector<std::string> options_lines;

    options_lines.push_back(".. _`sec:options`:\n");
    options_lines.push_back("List of Forte options");
    options_lines.push_back("=====================\n");
    options_lines.push_back(".. sectionauthor:: Francesco A. Evangelista\n");
    for (const auto& p : option_docs_list) {
        options_lines.push_back(p.second);
    }

    return to_string(options_lines, "\n");
}

std::string rst_bold(const std::string& s) { return "**" + s + "**"; }

std::string option_formatter(const std::string& type, const std::string& label,
                             const std::string& default_value, const std::string& description,
                             const std::string& allowed_values) {
    std::string s;

    s += rst_bold(label) + "\n\n";
    s += description + "\n\n";
    s += "* Type: " + type + "\n\n";
    s += "* Default value: " + default_value + "\n\n";
    if (allowed_values.size() > 0) {
        s += "* Allowed values: " + allowed_values;
    }

    return s;
}

} // namespace forte
