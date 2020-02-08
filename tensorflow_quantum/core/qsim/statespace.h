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

#ifndef STATESPACE_H_
#define STATESPACE_H_

#include <complex>
#include <memory>

namespace tfq {
namespace qsim {

class StateSpace {
 public:
  // TODO: clean up the abusive use of state here
  // the virtual functions rely on it, but then overwrite it.
  using State = std::unique_ptr<float, decltype(&free)>;

  StateSpace(const unsigned int num_qubits, const unsigned int num_threads);

  virtual ~StateSpace() {}

  State* CreateState() const;

  static void DeleteState(State* state);

  uint64_t Size() const;

  uint64_t RawSize() const;

  static float* RawData(State* state);
  static const float* RawData(const State& state);

  static bool Valid(const State& state);

  virtual void CopyState(const State& src, State* dest) const = 0;
  virtual void SetStateZero(State* state) const = 0;
  virtual float GetRealInnerProduct(const State& a, const State& b) const = 0;
  virtual std::complex<float> GetAmpl(const State& state,
                                      const uint64_t i) const = 0;
  virtual void SetAmpl(State* state, const uint64_t i,
                       const std::complex<float>& val) const = 0;

 protected:
  uint64_t size_;
  unsigned int num_threads_;
};

}  // namespace qsim
}  // namespace tfq

#endif  // STATESPACE_H_
