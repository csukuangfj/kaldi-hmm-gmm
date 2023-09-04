// kaldi-hmm-gmm/python/csrc/clusterable-classes.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-hmm-gmm/python/csrc/clusterable-classes.h"

#include "kaldi-hmm-gmm/csrc/cluster-utils.h"
#include "kaldi-hmm-gmm/csrc/clusterable-classes.h"
#include "kaldi-hmm-gmm/csrc/clusterable-itf.h"

namespace khg {

static constexpr const char *kScalarClusterableDoc = R"doc(
Maximize the negated sum of squared distances from the cluster center
to each scalar.

Suppose there are n scalars :math:`x_1`, :math:`x_2`, ..., :math:`x_n`,
the cluster center is defined as:

.. math::

    x = \frac{x_1 + x_2 + ... + x_n}{n}

The objective function is:

.. math::

    -1 *\left( (x_1 - x)^2 + (x_2 - x)^2 + ... + (x_n - x)^2  \right) &= -1 *\left( \sum_i^n(x_i - x)^2 \right) \\
    &= -1 * \left( \sum_{i=1}^n (x_i - \frac{\sum_{k=1}^n x_k}{n})^2 \right)\\
    & = -1 * \left( \sum_{i=1}^n (x_i^2 + \frac{\left( \sum_{k=1}^n x_k \right)^2}{n^2} - 2 x_i \frac{\sum_{k=1}^n x_k}{n}  \right)\\
    &= -1 * \left( \sum_{i=1}^n x_i^2+ n \frac{\left( \sum_{k=1}^n x_k \right)^2}{n^2} - 2 \sum_{i=1}^n x_i \frac{\sum_{k=1}^n x_k}{n} \right)\\
    &= -1 * \left( \sum_{i=1}^n x_i^2+ n \frac{\left( \sum_{k=1}^n x_k \right)^2}{n^2} - 2\left( \sum_{i=1}^n x_i \right)\frac{\sum_{k=1}^n x_k}{n} \right) \\
    &= -1 * \left( \sum_{i=1}^n x_i^2 + \frac{\left( \sum_{k=1}^n x_k \right)^2}{n} - 2 \frac{\left( \sum_{k=1}^n x_k \right)^2}{n} \right)\\
    &= -1 * \left( \sum_{i=1}^n x_i^2 - \frac{\left( \sum_{k=1}^n x_k \right)^2}{n} \right)\\

The goal is to maximize the objective function.

Its ``objf`` is defined as:

.. code-block:: c++

    return -(x2_ - x_ * x_ / count_);

where ``x2`` is the second order statistics, ``x_`` is the first order
statistics, and ``count_`` is the number of scalars.

When adding two :class:`ScalarClusterable`, we use:

.. code-block:: c++

    x_ += other->x_;
    x2_ += other->x2_;
    count_ += other->count_;

When subtracting one :class:`ScalarClusterable` from another, we use:

.. code-block:: c++

  x_ -= other->x_;
  x2_ -= other->x2_;
  count_ -= other->count_;
)doc";

static void PybindClusterable(py::module *m) {
  using PyClass = Clusterable;
  py::class_<PyClass>(*m, "Clusterable")
      .def("copy", &PyClass::Copy, py::return_value_policy::take_ownership)
      .def("objf", &PyClass::Objf)
      .def("normalizer", &PyClass::Normalizer)
      .def("set_zero", &PyClass::SetZero)
      .def("add", &PyClass::Add, py::arg("other"))
      .def("sub", &PyClass::Sub, py::arg("other"))
      .def("scale", &PyClass::Scale, py::arg("f"))
      .def("type", &PyClass::Type)
      .def("objf_plus", &PyClass::ObjfPlus, py::arg("other"))
      .def("objf_minus", &PyClass::ObjfMinus, py::arg("other"))
      .def("distance", &PyClass::Distance, py::arg("oter"));
}

static void PybindScalarClusterable(py::module *m) {
  using PyClass = ScalarClusterable;
  py::class_<PyClass, Clusterable>(*m, "ScalarClusterable",
                                   kScalarClusterableDoc)
      .def(py::init<>())
      .def(py::init<float>(), py::arg("x"))
      .def("info", &PyClass::Info)
      .def("mean", &PyClass::Mean);
}

static void PybindGaussClusterable(py::module *m) {
  using PyClass = GaussClusterable;
  py::class_<PyClass, Clusterable>(*m, "GaussClusterable")
      .def(py::init<>())
      .def(py::init<int32_t, float>(), py::arg("dim"), py::arg("var_floor"))
      .def(py::init<const DoubleVector &, const DoubleVector &, float, float>(),
           py::arg("x_stats"), py::arg("x2_stats"), py::arg("var_floor"),
           py::arg("count"))
      .def("add_stats", &PyClass::AddStats, py::arg("vec"), py::arg("weight"))
      .def("count", &PyClass::count)
      .def("x_stats", &PyClass::x_stats)
      .def("x2_stats", &PyClass::x2_stats);
}

void PybindClusterableClass(py::module *m) {
  PybindClusterable(m);
  PybindScalarClusterable(m);
  PybindGaussClusterable(m);
  m->def("sum_clusterable_objf", &SumClusterableObjf, py::arg("vec"));
  m->def("sum_clusterable_normalizer", &SumClusterableNormalizer,
         py::arg("vec"));
  m->def("sum_clusterable", &SumClusterable, py::arg("vec"),
         py::return_value_policy::take_ownership);
}

}  // namespace khg
