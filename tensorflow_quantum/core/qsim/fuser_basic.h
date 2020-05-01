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

#ifndef TFQ_CORE_QSIM_FUSER_BASIC_H_
#define TFQ_CORE_QSIM_FUSER_BASIC_H_

#include <vector>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_quantum/core/src/circuit.h"

namespace tfq {
namespace qsim {

class GateFused {
 public:
    GateFused(const unsigned int time, const unsigned int q0,
            const unsigned int q1, const Gate* anchor);

  ~GateFused() {}

  void AddGate(const Gate* gate);
  std::vector<const Gate*> GetAllGates() const;
  unsigned int GetNumGates() const;
  const Gate* GetGate(unsigned int gate_index) const;
  void SetGate(unsigned int gate_index, const Gate* gate);

  unsigned int GetTime() const;
  void SetTime(unsigned int time);

  unsigned int GetQubit0() const;
  void SetQubit0(unsigned int q0);

  unsigned int GetQubit1() const;
  void SetQubit1(unsigned int q1);

  const Gate* GetAnchor() const;
  void SetAnchor(const Gate* anchor);

 private:
  unsigned int time_;
  std::array<unsigned int, 2> qubits_;
  const Gate* anchor_;
  std::vector<const Gate*> gates_;
};

bool operator==(const GateFused& l, const GateFused& r);
bool operator!=(const GateFused& l, const GateFused& r);

tensorflow::Status FuseGates(const Circuit& circuit,
                             std::vector<GateFused>* fused);

}  // namespace qsim
}  // namespace tfq

#endif  // TFQ_CORE_QSIM_FUSER_BASIC_H_
