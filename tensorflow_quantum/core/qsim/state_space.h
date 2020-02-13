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

#ifndef STATE_SPACE_H_
#define STATE_SPACE_H_

#include <complex>
#include <memory>

namespace tfq {
namespace qsim {

class StateSpace {
 public:
  // TODO: clean up the abusive use of state here
  // the virtual functions rely on it, but then overwrite it.
  using State = std::unique_ptr<float, decltype(&free)>;

  StateSpace(const unsigned int num_qubits, const unsigned int num_threads)
      : size_(2 * (uint64_t{1} << num_qubits)),
        num_qubits_(num_qubits),
        num_threads_(num_threads) {}

  // Function to apply a two qubit gate to the state on indices q0
  // and q1.
  virtual void ApplyGate2(const unsigned int q0, const unsigned int q1,
                          const float* matrix) const = 0;

  // Function to apply updates to state if there is only one qubit in
  // the state.
  virtual void ApplyGate1(const float* matrix) const = 0;

  // Reserve the memory required to represent a state in this space
  virtual void InitState() const = 0;

  // Return true if memory for this state has already been allocated,
  // else return false
  static bool Valid() = 0;

  // Return a StateSpace which is a copy of this StateSpace
  virtual StateSpace Copy() const = 0;

  // Set all entries in the state to zero
  virtual void SetStateZero() const = 0;

  // Get the inner product between the state in this StateSpace and
  // the state in `other`.
  virtual float GetRealInnerProduct(const StateSpace& other) const = 0;

  // Get the amplitude at the given state index
  virtual std::complex<float> GetAmpl(const uint64_t i) const = 0;

  // Set the amplitude at the given state index
  virtual void SetAmpl(const uint64_t i,
                       const std::complex<float>& val) const = 0;

  // Dimension of the complex Hilbert space represented by this StateSpace
  virtual uint64_t Size() const = 0;

  virtual ~Simulator() {}

 protected:
  uint64_t size_;
  unsigned int num_qubits_;
  unsigned int num_threads_;
};

}  // namespace qsim
}  // namespace tfq

#endif  // STATE_SPACE_H_
