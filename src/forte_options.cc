#include "forte_options.h"
#include "helpers/mo_space_info.h"
#include "helpers/helpers.h"

#include "psi4/libpsi4util/PsiOutStream.h"

using namespace psi;

namespace forte {

std::string rst_bold(const std::string& s);

std::string option_formatter(const std::string& type, const std::string& label,
                             const std::string& default_value, const std::string& description,
                             const std::string& allowed_values);

ForteOptions::ForteOptions(psi::Options& options): psi_options_(options) {}

void ForteOptions::add_bool(const std::string& label, bool value, const std::string& description) {
    bool_opts_.push_back(std::make_tuple(label, value, description));
}

void ForteOptions::add_int(const std::string& label, int value, const std::string& description) {
    int_opts_.push_back(std::make_tuple(label, value, description));
}

void ForteOptions::add_double(const std::string& label, double value,
                              const std::string& description) {
    double_opts_.push_back(std::make_tuple(label, value, description));
}

void ForteOptions::add_str(const std::string& label, const std::string& value,
                           const std::string& description) {
    str_opts_.push_back(std::make_tuple(label, value, description, std::vector<std::string>()));
}

void ForteOptions::add_str(const std::string& label, const std::string& value,
                           const std::vector<std::string>& allowed_values,
                           const std::string& description) {
    str_opts_.push_back(std::make_tuple(label, value, description, allowed_values));
}

void ForteOptions::add_array(const std::string& label, const std::string& description) {
    array_opts_.push_back(std::make_tuple(label, description));
}

bool ForteOptions::get_bool(const std::string& label) const { return psi_options_.get_bool(label); }

int ForteOptions::get_int(const std::string& label) const { return psi_options_.get_int(label); }

double ForteOptions::get_double(const std::string& label) const { return psi_options_.get_double(label); }

const std::string& ForteOptions::get_str(const std::string& label) const { return psi_options_.get_str(label); }

bool ForteOptions::has_changed(const std::string& label) const { return psi_options_[label].has_changed(); }

void ForteOptions::add_psi4_options(psi::Options& options) {
    for (const auto& opt : bool_opts_) {
        options.add_bool(std::get<0>(opt), std::get<1>(opt));
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

std::string ForteOptions::generate_documentation() const {
    std::vector<std::pair<std::string, std::string>> option_docs_list;

    for (const auto& opt : bool_opts_) {
        const std::string& label = std::get<0>(opt);
        const std::string& default_value = std::get<1>(opt) ? "True" : "False";
        const std::string& description = std::get<2>(opt);
        std::string option_text =
            option_formatter("Boolean", label, default_value, description, "");
        outfile->Printf("\n %s", label.c_str());
        option_docs_list.push_back(std::make_pair(label, option_text));
    }

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
