// kaldi-hmm-gmm/python/csrc/eigen-test.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-hmm-gmm/python/csrc/eigen-test.h"

namespace khg {

namespace {

class Foo {
 public:
  Foo() {
    m1_.resize(2, 3);
    m2_.resize(5, 6);

    m1_.setOnes();
    m2_.setZero();
  }

  Eigen::MatrixXf m1_;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m2_;
};

}  // namespace

void PybindEigenTest(py::module *m) {
  using PyClass = Foo;

  // see
  // https://stackoverflow.com/questions/57926648/eigenmatrix-contained-inside-a-struct-is-marked-as-non-writeable-by-pybind11
  // https://github.com/pybind/pybind11/blob/master/include/pybind11/pybind11.h#L1709
  py::class_<PyClass>(*m, "TestEigen")
      .def(py::init<>())
      .def_property(
          "m1", [](PyClass &self) -> Eigen::MatrixXf & { return self.m1_; },
          [](PyClass &self, const Eigen::MatrixXf &m) { self.m1_ = m; },
          py::return_value_policy::reference_internal);
  // .def_readwrite("m1", &PyClass::m1_);
  // .def_readwrite("m2", &PyClass::m2_);
}

}  // namespace khg
