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

#ifndef TFQ_CORE_QSIM_FUSER_BASIC_H_
#define TFQ_CORE_QSIM_FUSER_BASIC_H_

#include <vector>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_quantum/core/src/circuit.h"

namespace tfq {
namespace qsim {

class GateFused {
 public:
  unsigned int time;
  unsigned int num_qubits;
  std::array<unsigned int, 2> qubits;
  const Gate* pmaster;
  std::vector<const Gate*> gates;
};

bool operator==(const GateFused& l, const GateFused& r);
bool operator!=(const GateFused& l, const GateFused& r);

tensorflow::Status FuseGates(const Circuit& circuit,
                             std::vector<GateFused>* fused);

}  // namespace qsim
}  // namespace tfq

#endif  // TFQ_CORE_QSIM_FUSER_BASIC_H_
