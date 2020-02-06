/* Copyright 2020 The TensorFlow Quantum Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow_quantum/core/src/circuit.h"

#include <vector>

#include "tensorflow_quantum/core/src/gates_def.h"

namespace tfq{

Circuit::Circuit() : num_qubits(0) {}
Circuit::Circuit(unsigned int num_qubits, std::vector<Gate>& gates)
    : num_qubits(num_qubits), gates(gates) {}

Circuit::bool operator==(const Circuit& r) const {
    if (this->num_qubits != r.num_qubits) {
      return false;
    }
    if (this->gates.size() != r.gates.size()) {
      return false;
    }
    for (size_t i = 0; i < this->gates.size(); i++) {
      if (this->gates.at(i) != r.gates.at(i)) {
        return false;
      }
    }
    return true;
  }

Circuit::bool operator!=(const Circuit& r) const { return !(*this == r); }

}  //namespace tfq
