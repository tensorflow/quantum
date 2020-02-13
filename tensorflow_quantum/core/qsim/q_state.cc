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

#include "tensorflow_quantum/core/qsim/q_state.h"

#include <complex>
#include <memory>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_quantum/core/proto/pauli_sum.pb.h"
#include "tensorflow_quantum/core/qsim/fuser_basic.h"
#include "tensorflow_quantum/core/qsim/mux.h"
#include "tensorflow_quantum/core/qsim/simulator.h"
#include "tensorflow_quantum/core/qsim/simulator2_slow.h"
#include "tensorflow_quantum/core/src/circuit.h"
#include "tensorflow_quantum/core/src/circuit_parser.h"
#include "tensorflow_quantum/core/src/matrix.h"

namespace tfq {
namespace qsim {
namespace {

// TODO(zaqqwerty): Make thread count adaptive to machine
int GetAvailableThreads() { return 1; }

}  // namespace

QState::QState(const int num_qubits) : num_qubits_(num_qubits) {
  const int num_threads = GetAvailableThreads();
  simulator_ = GetSimulator(num_qubits, num_threads);

  state_ = simulator_->CreateState();
  simulator_->SetStateZero(state_);
}

QState::~QState() {
  simulator_->DeleteState(state_);
  delete state_;
}

tensorflow::Status QState::Update(const Circuit& circuit) {
  if (!simulator_->Valid(*state_)) {
    return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                              "Invalid space.");
  }

  // Delegate to single qubit workaround.
  // This delegation is only supported with the
  // slow simulator for now.
  if (state_space_->Size() <= 2) {
    for (uint64_t i = 0; i < circuit.gates.size(); i++) {
      const auto& gate = circuit.gates[i];

      if (gate.num_qubits == 1) {
        float matrix[8];
        Matrix2Set(gate.matrix, matrix);
        simulator_->ApplyGate1(matrix, state_);
      } else {
        return tensorflow::Status(
            tensorflow::error::INVALID_ARGUMENT,
            "Got a multi qubit gate in a 1 qubit circuit.");
      }
    }
    return tensorflow::Status::OK();
  }

  std::vector<GateFused> fused_gates;
  auto status = FuseGates(circuit, &fused_gates);
  if (!status.ok()) {
    return status;
  }

  for (const GateFused& gate : fused_gates) {
    float matrix[32];
    CalcMatrix4(gate.qubits[0], gate.qubits[1], gate.gates, matrix);
    simulator_->ApplyGate2(gate.qubits[0], gate.qubits[1], matrix, state_);
  }

  return tensorflow::Status::OK();
}

void QState::CopyOnto(QState* other) const {
  simulator_->CopyState(*state_, other->state_);
}

std::complex<float> QState::GetAmplitude(const uint64_t i) const {
  return simulator_->GetAmpl(*state_, i);
}

void QState::SetAmplitude(const uint64_t i,
                          const std::complex<float>& val) const {
  simulator_->SetAmpl(state_, i, val);
}

float QState::GetRealInnerProduct(const QState& other) const {
  // TODO (zaqwerty): investigate const-ness of input arguments here.
  return simulator_->GetRealInnerProduct(*state_, *(other.state_));
}

tensorflow::Status QState::ComputeExpectation(const tfq::proto::PauliSum& p_sum,
                                              float* expectation_value) {
  // apply the  gates of the pauliterms to a new copy of the wavefunction
  // and add up expectation value term by term.
  for (const tfq::proto::PauliTerm& term : p_sum.terms()) {
    // catch identity terms
    if (term.paulis_size() == 0) {
      *expectation_value += term.coefficient_real();
      // TODO(zaqqwerty): error somewhere if identities have any imaginary part
      continue;
    }

    tensorflow::Status status;
    Circuit measurement_circuit;
    // TODO (zaqqwerty): profile whether creating a copy, evolving the copy
    // and then deleting the copy is better OR evolving the referencing
    // computing the number and then un-evolving the reference is faster.
    status =
        CircuitFromPauliTerm(term, this->num_qubits_, &measurement_circuit);
    if (!status.ok()) {
      return status;
    }
    tfq::qsim::QState transformed_state(this->num_qubits_);
    this->CopyOnto(&transformed_state);
    status = transformed_state.Update(measurement_circuit);
    if (!status.ok()) {
      return status;
    }
    *expectation_value +=
        term.coefficient_real() * this->GetRealInnerProduct(transformed_state);
  }
  return tensorflow::Status::OK();
}

}  // namespace qsim
}  // namespace tfq
