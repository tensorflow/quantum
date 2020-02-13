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

#ifndef TFQ_CORE_QSIM_Q_STATE_H_
#define TFQ_CORE_QSIM_Q_STATE_H_

#include <complex>
#include <memory>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_quantum/core/proto/pauli_sum.pb.h"
#include "tensorflow_quantum/core/qsim/simulator.h"
#include "tensorflow_quantum/core/src/circuit.h"

namespace tfq {
namespace qsim {

class QState {
 public:
  // Selects the proper Simulator based on the runtime
  // environment and initializes the associated state.
  QState(const int num_qubits);

  // Creates the state with a pre-selected Simulator.
  QState(std::unique_ptr<Simulator> simulator, const int num_qubits);

  // Cleans up allocated memory.
  ~QState();

  // Updates the state of this object by executing the current state through the
  // given circuit.
  tensorflow::Status Update(const Circuit& circuit);

  // Copies the state from this object into the other.
  void CopyOnto(QState* other) const;

  // Returns the amplitude of this state at the given index.
  std::complex<float> GetAmplitude(const uint64_t i) const;

  // Sets the amplitude of this state at the given index.
  void SetAmplitude(const uint64_t i, const std::complex<float>& val) const;

  // Returns the real inner product between the two states.
  float GetRealInnerProduct(const QState& other) const;

  // Compute the expectation value for a given state vector and PauliSum.
  tensorflow::Status ComputeExpectation(const tfq::proto::PauliSum& p_sum,
                                        float* expectation_value);

 private:
  std::unique_ptr<Simulator> simulator_;

  // TODO(pmassey): Consider moving to a unique_ptr to clean up memory.
  std::unique_ptr<float, decltype(&free)>* state_;
  int num_qubits_;
};

}  // namespace qsim
}  // namespace tfq

#endif  // TFQ_CORE_QSIM_Q_STATE_H_
