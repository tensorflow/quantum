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
  StateSpaceSlow(const unsigned int num_qubits, const unsigned int num_threads);

  virtual ~StateSpaceSlow() {DeleteState();}

  // Function to apply a two qubit gate to the state on indices q0 and q1.
  virtual void ApplyGate2(const unsigned int q0, const unsigned int q1,
                          const float* matrix) override;

  // Function to apply a one-qubit gate if there is only one qubit in the state.
  // Implementations are given the option to return an error.
  virtual tensorflow::Status ApplyGate1(const float* matrix) override;

    // Return a StateSpace which is a copy of this StateSpace
  virtual std::shared_ptr<StateSpace> Copy() const override;

  // Set all entries in the state to zero
  virtual void SetStateZero() override;

  // Get the inner product between the state in this StateSpace and
  // the state in `other`.
  virtual float GetRealInnerProduct(const std::shared_ptr<StateSpace> other) const override;

  // Get the amplitude at the given state index
  virtual std::complex<float> GetAmpl(const uint64_t i) const override;

  // Set the amplitude at the given state index
  virtual void SetAmpl(const uint64_t i, const std::complex<float>& val) override;
};

}  // namespace qsim
}  // namespace tfq

#endif  // TFQ_CORE_QSIM_STATE_SPACE_SLOW_H_
