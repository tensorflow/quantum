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

#ifndef TFQ_CORE_QSIM_STATE_SPACE_SLOW_H_
#define TFQ_CORE_QSIM_STATE_SPACE_SLOW_H_

#include <complex>
#include <memory>

#include "tensorflow_quantum/core/qsim/state_space.h"

namespace tfq {
namespace qsim {

class StateSpaceSlow : public StateSpace {
 public:
  StateSpaceSlow(const uint64_t num_qubits, const uint64_t num_threads);

  virtual ~StateSpaceSlow();

  StateSpaceType GetType() const override;

  // Reserve the memory associated with the state in this space
  virtual void CreateState() override;

  // Free the memory associated with the state in this space
  virtual void DeleteState() override;

  // Return a pointer to a copy of this StateSpace.
  // NOTE: user is responsible for deleting the returned copy.
  virtual StateSpace* Clone() const override;

  // Copy the state information from another statespace.
  // Assumes the state has been initialized/created.
  virtual void CopyFrom(const StateSpace& other) const override;

  // Function to apply a two qubit gate to the state on indices q0 and q1.
  virtual void ApplyGate2(const unsigned int q0, const unsigned int q1,
                          const float* matrix) override;

  // Function to apply a one-qubit gate if there is only one qubit in the state.
  // Implementations are given the option to return an error.
  virtual tensorflow::Status ApplyGate1(const float* matrix) override;

  // Set state to the all zero |000...0> state
  virtual void SetStateZero() override;

  // Get the inner product between this state and the state in `other`
  virtual float GetRealInnerProduct(const StateSpace& other) const override;

  // Get the amplitude at the given state index
  virtual std::complex<float> GetAmpl(const uint64_t i) const override;

  // Set the amplitude at the given state index
  virtual void SetAmpl(const uint64_t i,
                       const std::complex<float>& val) override;
};

}  // namespace qsim
}  // namespace tfq

#endif  // TFQ_CORE_QSIM_STATE_SPACE_SLOW_H_
