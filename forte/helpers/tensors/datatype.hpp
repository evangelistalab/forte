#pragma once

#include <string>

/// @brief Class to represent a data type
class DataType {
private:
  std::string name_;
  char kind_;
  size_t itemsize_;

public:
  DataType(const std::string &name, char kind, size_t itemsize)
      : name_(name), kind_(kind), itemsize_(itemsize) {}
  std::string name() const { return name_; }
  char kind() const { return kind_; }
  size_t itemsize() const { return itemsize_; }
};
