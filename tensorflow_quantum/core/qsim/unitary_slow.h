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

#ifndef TFQ_CORE_QSIM_UNITARY_SLOW_H_
#define TFQ_CORE_QSIM_UNITARY_SLOW_H_

#include <complex>
#include <memory>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_quantum/core/src/circuit.h"

namespace tfq {
namespace qsim {

// Handles calculations of unitary matrices that a circuit enacts.
class UnitarySlow : public Unitary {
 public:
  UnitarySlow(const uint64_t num_qubits, const uint64_t num_threads);

  virtual ~Unitary() {}

  // Get the Unitary type.
  virtual UnitaryType GetType() const = 0;

  // Reserve the memory associated with this Unitary.
  virtual void CreateState() = 0;

  // Free the memory associated with this Unitary.
  virtual void DeleteState() = 0;

  // Function to apply a two qubit gate to the Unitary on indices q0 and q1.
  // Must adhere to big-endian convention of Cirq.
  virtual void ApplyGate2(const unsigned int q0, const unsigned int q1,
                          const float* matrix) = 0;

  // Function to apply a one-qubit gate if there is one qubit in the Unitary.
  // Implementations are given the option to return an error.
  virtual tensorflow::Status ApplyGate1(const float* matrix) = 0;

  // Set to identity matrix.
  virtual void SetIdentity() = 0;

  // Get the amplitude at the given state index.
  virtual std::complex<float> GetEntry(const uint64_t i, const uint64_t j) const = 0;

  // Set the amplitude at the given state index
  virtual void SetEntry(const uint64_t i, const uint64_t j, const std::complex<float>& val) = 0;

 protected:
  float* state_;
  uint64_t size_;
  uint64_t num_qubits_;
  uint64_t num_threads_;
};

}  // namespace qsim
}  // namespace tfq

#endif  // TFQ_CORE_QSIM_UNITARY_SLOW_H_
