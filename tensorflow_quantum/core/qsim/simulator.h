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

#ifndef SIMULATOR_H_
#define SIMULATOR_H_

#include <memory>
#include "glog/logging.h"
#include "tensorflow_quantum/core/qsim/statespace.h"

namespace tfq {
namespace qsim {

class Simulator {
 public:
  // TODO: clean up the abusive use of state here
  // the virtual functions rely on it, but then overwrite it.
  using State = std::unique_ptr<float, decltype(&free)>;

  Simulator(const unsigned int num_qubits, const unsigned int num_threads)
      : num_qubits_(num_qubits), num_threads_(num_threads) {}

  // Function to apply a two qubit gate to the state on indices q0
  // and q1.
  virtual void ApplyGate2(const unsigned int q0, const unsigned int q1,
                          const float* matrix, State* state) const = 0;

  // Function to apply updates to state if there is only one qubit in
  // the state.
  virtual void ApplyGate1(const float* matrix, State* state) const = 0;

  virtual ~Simulator() {}

 protected:
  unsigned int num_qubits_;
  unsigned int num_threads_;
};

}  // namespace qsim
}  // namespace tfq

#endif  // SIMULATOR_H_
