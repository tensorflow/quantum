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

#ifndef TFQ_CORE_SRC_CIRCUIT_H_
#define TFQ_CORE_SRC_CIRCUIT_H_

#include <array>
#include <vector>

#include "tensorflow_quantum/core/src/gates_def.h"

namespace tfq {

class Circuit {
 public:
  unsigned int num_qubits;
  std::vector<Gate> gates;

  // provided to ease serialization testing
  bool operator==(const Circuit& r) const {
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

  bool operator!=(const Circuit& r) const { return !(*this == r); }
};

}  // namespace tfq

#endif  // TFQ_CORE_SRC_CIRCUIT_H_
