#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_PLUGIN(pybind_plugin) {
  py::module m("pybind_plugin", "pybind11 example plugin");

  m.def("myplugin", &myplugin, "Run my plugin");

  return m.ptr();
}
