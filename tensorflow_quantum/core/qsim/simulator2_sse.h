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

#ifndef TFQ_CORE_QSIM_SIMULATOR2_SSE_H_
#define TFQ_CORE_QSIM_SIMULATOR2_SSE_H_

#include <immintrin.h>
#include <smmintrin.h>

#include <cmath>
#include <cstdint>

#include "tensorflow_quantum/core/qsim/simulator.h"

namespace tfq {
namespace qsim {

class Simulator2SSE : public Simulator {
 public:
  using State = Simulator::State;

  Simulator2SSE(const unsigned int num_qubits, const unsigned int num_threads);

  void ApplyGate2(const unsigned int q0, const unsigned int q1,
                  const float* matrix, State* state) const override;

  void ApplyGate1(const float* matrix, State* state) const override;

  void CopyState(const State& src, State* dest) const override;

  void SetStateZero(State* state) const override;

  float GetRealInnerProduct(const State& a, const State& b) const override;

  std::complex<float> GetAmpl(const State& state,
                              const uint64_t i) const override;

  void SetAmpl(State* state, const uint64_t i,
               const std::complex<float>& val) const override;

 private:
  void ApplyGate2HH(const unsigned int q0, const unsigned int q1,
                    const float* matrix, State* state) const;

  void ApplyGate2LL(const unsigned int q0, const unsigned int q1,
                    const float* matrix, State* state) const;

  void ApplyGate2HL(const unsigned int q0, const unsigned int q1,
                    const float* matrix, State* state) const;
};

}  // namespace qsim
}  // namespace tfq

#endif  // TFQ_CORE_QSIM_SIMULATOR2_SSE_H_
