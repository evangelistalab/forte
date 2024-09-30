#pragma once

#include <string>

namespace forte {

/// @brief Class to represent a data type
class DataType {
  private:
    std::string name_;
    char kind_;
    size_t itemsize_;

  public:
    DataType(const std::string& name, char kind, size_t itemsize)
        : name_(name), kind_(kind), itemsize_(itemsize) {}

    std::string name() const { return name_; }

    char kind() const { return kind_; }

    size_t itemsize() const { return itemsize_; }

    std::string to_string() const {
        return "DataType(name='" + name_ + "', kind='" + kind_ +
               "', itemsize=" + std::to_string(itemsize_) + ")";
    }

    bool operator==(const DataType& other) const {
        return name_ == other.name_ and kind_ == other.kind_ and itemsize_ == other.itemsize_;
    }

    bool operator!=(const DataType& other) const { return !(*this == other); }
};

} // namespace forte