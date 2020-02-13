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

#ifndef TFQ_CORE_QSIM_STATE_SPACE_H_
#define TFQ_CORE_QSIM_STATE_SPACE_H_

#include <complex>
#include <memory>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_quantum/core/proto/pauli_sum.pb.h"
#include "tensorflow_quantum/core/src/circuit.h"

namespace tfq {
namespace qsim {

class StateSpace {
 public:
  StateSpace(const unsigned int num_qubits, const unsigned int num_threads)
      : state_(NULL),
        size_(2 * (uint64_t{1} << num_qubits)),
        num_qubits_(num_qubits),
        num_threads_(num_threads) {}

  // Updates the state by applying the given circuit.
  tensorflow::Status Update(const Circuit& circuit);

  // Computes the expectation value for a given state vector and PauliSum.
  tensorflow::Status ComputeExpectation(const tfq::proto::PauliSum& p_sum,
                                        float* expectation_value);

  // Reserve the memory associated with the state in this space
  void CreateState();

  // Free the memory associated with the state in this space
  void DeleteState();

  // Returns true if memory for the state has been succesfully allocated
  bool Valid() const;

  // Load a copy of this StateSpace into `other`
  tensorflow::Status Copy(StateSpace* other) const;

  // Dimension of the complex Hilbert space represented by this StateSpace
  uint64_t Dimension() const;

  virtual ~StateSpace() {}

  // Function to apply a two qubit gate to the state on indices q0 and q1.
  virtual void ApplyGate2(const unsigned int q0, const unsigned int q1,
                          const float* matrix) = 0;

  // Function to apply a one-qubit gate if there is only one qubit in the state.
  // Implementations are given the option to return an error.
  virtual tensorflow::Status ApplyGate1(const float* matrix) = 0;

  // Set all entries in the state to zero
  virtual void SetStateZero() = 0;

  // Get the inner product between this state and the state in `other`
  virtual float GetRealInnerProduct(const StateSpace* other) const = 0;

  // Get the amplitude at the given state index
  virtual std::complex<float> GetAmpl(const uint64_t i) const = 0;

  // Set the amplitude at the given state index
  virtual void SetAmpl(const uint64_t i, const std::complex<float>& val) = 0;

 protected:
  float* state_;
  uint64_t size_;
  unsigned int num_qubits_;
  unsigned int num_threads_;
};

}  // namespace qsim
}  // namespace tfq

#endif  // TFQ_CORE_QSIM_STATE_SPACE_H_
