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

#ifndef TFQ_CORE_QSIM_UNITARY_SPACE_SLOW_H_
#define TFQ_CORE_QSIM_UNITARY_SPACE_SLOW_H_

#include <complex>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_quantum/core/qsim/unitary_space.h"
#include "tensorflow_quantum/core/src/circuit.h"

namespace tfq {
namespace qsim_old {

// Handles calculations of unitary matrices that a circuit enacts.
class UnitarySpaceSlow : public UnitarySpace {
 public:
  UnitarySpaceSlow(const uint64_t num_qubits, const uint64_t num_threads);

  virtual ~UnitarySpaceSlow();

  // Get the Unitary type.
  virtual UnitarySpaceType GetType() const override;

  // Reserve the memory associated with this Unitary.
  virtual void CreateUnitary() override;

  // Free the memory associated with this Unitary.
  virtual void DeleteUnitary() override;

  // Function to apply a two qubit gate to the Unitary on indices q0 and q1.
  // Must adhere to big-endian convention of Cirq.
  virtual void ApplyGate2(const unsigned int q0, const unsigned int q1,
                          const float* matrix) override;

  // Function to apply a one-qubit gate if there is one qubit in the Unitary.
  // Implementations are given the option to return an error.
  virtual tensorflow::Status ApplyGate1(const float* matrix) override;

  // Set to identity matrix.
  virtual void SetIdentity() override;

  // Get the matrix entry at the given index.
  virtual std::complex<float> GetEntry(const uint64_t i,
                                       const uint64_t j) const override;

  // Set the matrix entry at the given index
  virtual void SetEntry(const uint64_t i, const uint64_t j,
                        const std::complex<float>& val) override;
};

}  // namespace qsim_old
}  // namespace tfq

#endif  // TFQ_CORE_QSIM_UNITARY_SPACE_SLOW_H_
