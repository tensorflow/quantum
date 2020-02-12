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

#ifndef TFQ_CORE_QSIM_STATESPACE_AVX_H_
#define TFQ_CORE_QSIM_STATESPACE_AVX_H_

#include <immintrin.h>
#include <stdlib.h>

#include <complex>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>

#include "tensorflow_quantum/core/qsim/statespace.h"
#include "tensorflow_quantum/core/qsim/util.h"

namespace tfq {
namespace qsim {

class StateSpaceAVX : public StateSpace {
 public:
  using State = std::unique_ptr<float, decltype(&free)>;

  StateSpaceAVX(const unsigned int num_qubits, const unsigned int num_threads);

  void CopyState(const State& src, State* dest) const override;

  void SetStateZero(State* state) const override;

  float GetRealInnerProduct(const State& a, const State& b) const override;
};

}  // namespace qsim
}  // namespace tfq

#endif  // TFQ_CORE_QSIM_STATESPACE_AVX_H_
