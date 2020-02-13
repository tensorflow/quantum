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

#include "tensorflow_quantum/core/qsim/simulator.h"

#include <memory>

#include "tensorflow_quantum/core/qsim/util.h"

namespace tfq {
namespace qsim {

using State = std::unique_ptr<float, decltype(&free)>;

State* Simulator::CreateState() const {
  return new State((float*)qsim::_aligned_malloc(sizeof(float) * size_), &free);
}

void Simulator::DeleteState(State* state) {
  qsim::_aligned_free(state->release());
}

uint64_t Simulator::Size() const { return size_ / 2; }

bool Simulator::Valid(const State& state) {
  // TODO: more roubust test?
  return state.get() != nullptr;
}

}  // namespace qsim
}  // namespace tfq
