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

#include "tensorflow_quantum/core/qsim/statespace.h"

#include <complex>
#include <memory>

#include "tensorflow_quantum/core/qsim/util.h"

namespace tfq {
namespace qsim {

using State = std::unique_ptr<float, decltype(&free)>;

StateSpace::StateSpace(unsigned num_qubits, unsigned num_threads)
    : size_(2 * (uint64_t{1} << num_qubits)), num_threads_(num_threads) {}

State* StateSpace::CreateState() const {
  return new State((float*)qsim::_aligned_malloc(sizeof(float) * size_), &free);
}

void StateSpace::DeleteState(State* state) {
  qsim::_aligned_free(state->release());
}

uint64_t StateSpace::Size() const { return size_ / 2; }

uint64_t StateSpace::RawSize() const { return size_; }

float* StateSpace::RawData(State* state) { return state->get(); }

const float* StateSpace::RawData(const State& state) { return state.get(); }

bool StateSpace::Valid(const State& state) {
  // TODO: more roubust test?
  return state.get() != nullptr;
}

}  // namespace qsim
}  // namespace tfq
