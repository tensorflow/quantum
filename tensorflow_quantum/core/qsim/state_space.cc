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

#include "tensorflow_quantum/core/qsim/state_space.h"

#include <complex>
#include <memory>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_quantum/core/proto/pauli_sum.pb.h"
#include "tensorflow_quantum/core/qsim/fuser_basic.h"
#include "tensorflow_quantum/core/src/circuit.h"
#include "tensorflow_quantum/core/src/circuit_parser.h"
#include "tensorflow_quantum/core/src/matrix.h"

namespace tfq {
namespace qsim {

tensorflow::Status StateSpace::Update(const Circuit& circuit) {
  tensorflow::Status status;
  // Special case for single qubit;
  // derived classes free to return an error.
  if (GetDimension() <= 2) {
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
    CalcMatrix4(gate.qubits[0], gate.qubits[1], gate.gates, matrix);
    ApplyGate2(gate.qubits[0], gate.qubits[1], matrix);
  }

  return tensorflow::Status::OK();
}

tensorflow::Status StateSpace::ComputeExpectation(
    const tfq::proto::PauliSum& p_sum, float* expectation_value) {
  // apply the  gates of the pauliterms to a new copy of the wavefunction
  // and add up expectation value term by term.
  tensorflow::Status status = tensorflow::Status::OK();
  std::unique_ptr<StateSpace> transformed_state =
      std::unique_ptr<StateSpace>(Clone());
  transformed_state->CreateState();
  for (const tfq::proto::PauliTerm& term : p_sum.terms()) {
    // catch identity terms
    if (term.paulis_size() == 0) {
      *expectation_value += term.coefficient_real();
      // TODO(zaqqwerty): error somewhere if identities have any imaginary part
      continue;
    }

    Circuit measurement_circuit;

    status = CircuitFromPauliTerm(term, num_qubits_, &measurement_circuit);
    if (!status.ok()) {
      return status;
    }
    transformed_state->CopyFrom(*this);
    status = transformed_state->Update(measurement_circuit);
    if (!status.ok()) {
      return status;
    }
    *expectation_value +=
        term.coefficient_real() * GetRealInnerProduct(*transformed_state);
  }
  return status;
}

bool StateSpace::Valid() const {
  // TODO: more roubust test?
  return state_ != nullptr;
}

float* StateSpace::GetRawState() const { return state_; };

void StateSpace::SetRawState(float* state_update) { state_ = state_update; }

uint64_t StateSpace::GetNumEntries() const { return size_; }

uint64_t StateSpace::GetDimension() const { return size_ / 2; }

uint64_t StateSpace::GetNumQubits() const { return num_qubits_; }

uint64_t StateSpace::GetNumThreads() const { return num_threads_; }

}  // namespace qsim
}  // namespace tfq
