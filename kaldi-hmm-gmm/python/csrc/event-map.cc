// kaldi-hmm-gmm/python/csrc/event-map.cc
//
// Copyright (c)  2023  Xiaomi Corporation
#include "kaldi-hmm-gmm/python/csrc/event-map.h"

#include <string>
#include <utility>
#include <vector>

#include "kaldi-hmm-gmm/csrc/event-map.h"

namespace khg {

static void PybindEventMapBase(py::module *m) {
  using Class = EventMap;
  using PyClass = EventMap;
  py::class_<PyClass>(*m, "EventMap")
      .def_static("check", &PyClass::Check, py::arg("event"))
      .def_static(
          "lookup",
          [](const EventType &event,
             EventKeyType key) -> std::pair<bool, int32_t> {
            EventValueType anwser = -1;
            bool found = PyClass::Lookup(event, key, &anwser);
            return std::make_pair(found, anwser);
          },
          py::arg("event"), py::arg("key"))
      .def(
          "map",
          [](PyClass &self,
             const EventType &event) -> std::pair<bool, int32_t> {
            int32_t anwser = -1;
            bool found = self.Map(event, &anwser);
            return std::make_pair(found, anwser);
          },
          py::arg("event"))
      .def(
          "multimap",
          [](PyClass &self, const EventType &event,
             std::vector<int32_t> answers = {}) -> std::vector<int32_t> {
            self.MultiMap(event, &answers);
            return answers;
          },
          py::arg("event"), py::arg("answers") = std::vector<int32_t>{})
      .def("__str__", [](PyClass &self) -> std::string {
        std::ostringstream os;
        self.Write(os, /*binary*/ false);
        return os.str();
      });
}

static void PybindConstantEvent(py::module *m) {
  using PyClass = ConstantEventMap;
  py::class_<PyClass, EventMap>(*m, "ConstantEventMap")
      .def(py::init<int32_t>(), py::arg("anwser"));
}

void PybindEventMap(py::module *m) {
  PybindEventMapBase(m);
  PybindConstantEvent(m);
}

}  // namespace khg
