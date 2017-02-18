#include "forte_options.h"

namespace psi {
namespace forte {

void ForteOptions::add_bool(const std::string& label, bool value,
                            const std::string& description) {
    bool_opts_.push_back(std::make_tuple(label, value, description));
}

void ForteOptions::add_int(const std::string& label, int value,
                           const std::string& description) {
    int_opts_.push_back(std::make_tuple(label, value, description));
}

void ForteOptions::add_double(const std::string& label, double value,
                              const std::string& description) {
    double_opts_.push_back(std::make_tuple(label, value, description));
}

void ForteOptions::add_str(const std::string& label, std::string value,
                           const std::string& description,
                           const std::vector<std::string>& allowed_values) {
    str_opts_.push_back(
        std::make_tuple(label, value, description, allowed_values));
}

void ForteOptions::add_array(const std::string& label,
                             const std::string& description) {
    array_opts_.push_back(std::make_tuple(label, description));
}

void ForteOptions::add_psi4_options(Options& options) {
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
        if (std::get<2>(opt).size() > 0) {
            options.add_str(std::get<0>(opt), std::get<1>(opt),
                            std::get<2>(opt));
        } else {
            options.add_str(std::get<0>(opt), std::get<1>(opt));
        }
    }

    for (const auto& opt : array_opts_) {
        options.add(std::get<0>(opt), new ArrayType());
    }
}
}
}
