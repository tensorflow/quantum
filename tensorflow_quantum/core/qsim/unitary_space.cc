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

#include "tensorflow_quantum/core/qsim/unitary_space.h"

#include <complex>
#include <memory>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_quantum/core/qsim/fuser_basic.h"
#include "tensorflow_quantum/core/qsim/matrix.h"
#include "tensorflow_quantum/core/src/circuit.h"
#include "tensorflow_quantum/core/src/circuit_parser.h"

namespace tfq {
namespace qsim {

tensorflow::Status UnitarySpace::Update(const Circuit& circuit) {
  tensorflow::Status status;
  // Special case for single qubit;
  // derived classes free to return an error.
  if (GetNumQubits() <= 1) {
    for (uint64_t i = 0; i < circuit.gates.size(); i++) {
      const auto& gate = circuit.gates[i];
      if (gate.num_qubits == 1) {
        float matrix[8];
        Matrix2Set(gate.matrix, matrix);
        status = ApplyGate1(matrix);
        if (!status.ok()) {
          return status;
        }
      } else {
        return tensorflow::Status(
            tensorflow::error::INVALID_ARGUMENT,
            "Got a multi-qubit gate in a 1 qubit circuit.");
      }
    }
    return tensorflow::Status::OK();
  }

  std::vector<GateFused> fused_gates;
  status = FuseGates(circuit, &fused_gates);
  if (!status.ok()) {
    return status;
  }

  for (const GateFused& gate : fused_gates) {
    float matrix[32];
    const std::vector<const Gate*>* ref;
    gate.GetAllGates(&ref);
    CalcMatrix4(gate.GetQubit0(), gate.GetQubit1(), *ref, matrix);
    ApplyGate2(gate.GetQubit0(), gate.GetQubit1(), matrix);
  }

  return tensorflow::Status::OK();
}

bool UnitarySpace::Valid() const {
  // TODO: more roubust test?
  return unitary_ != nullptr;
}

float* UnitarySpace::GetRawUnitary() const { return unitary_; };

uint64_t UnitarySpace::GetNumQubits() const { return num_qubits_; }

uint64_t UnitarySpace::GetNumThreads() const { return num_threads_; }

}  // namespace qsim
}  // namespace tfq
