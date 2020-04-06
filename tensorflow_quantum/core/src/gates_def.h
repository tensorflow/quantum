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

#ifndef TFQ_CORE_SRC_GATES_DEF_H_
#define TFQ_CORE_SRC_GATES_DEF_H_

// These gates were originally designed to take a template fp_type variable, but
// using that causes linking problems around the matrix element in the Gate
// class. Only floats are implemented now as a result.
//
// TODO(pmassey): Add tests to assert that the swapped versions of matrices are
// the same as applying swap gates and the original gate.

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/lib/core/status.h"

namespace tfq {

// Permute 4x4 matrix to switch between two qubits.
template <typename Array2>
static void SwapQubits(Array2& mat) {
  std::swap(mat[2], mat[4]);
  std::swap(mat[3], mat[5]);
  std::swap(mat[8], mat[16]);
  std::swap(mat[9], mat[17]);
  std::swap(mat[10], mat[20]);
  std::swap(mat[11], mat[21]);
  std::swap(mat[12], mat[18]);
  std::swap(mat[13], mat[19]);
  std::swap(mat[14], mat[22]);
  std::swap(mat[15], mat[23]);
  std::swap(mat[26], mat[28]);
  std::swap(mat[27], mat[29]);
}

class Gate {
 public:
  unsigned int time;
  unsigned int num_qubits;
  std::array<unsigned int, 2> qubits;
  std::array<float, 32> matrix;

  // Overload for empty gates (to be assigned to)
  Gate();

  // Overload for one-qubit gates
  Gate(const unsigned int time_in, const unsigned int qubit_in,
       const std::array<float, 8>& matrix_in);

  // Overload for two-qubit gates
  Gate(const unsigned int time_in, const unsigned int q1, const unsigned int q2,
       const std::array<float, 32>& matrix_in);

  ~Gate() {}
};

bool operator==(const Gate& l, const Gate& r);
bool operator!=(const Gate& l, const Gate& r);

using Matrix1q = std::array<float, 8>;
using Matrix2q = std::array<float, 32>;

// Constructor base class that takes in Gate parameters and returns a Gate.
class GateBuilder {
 public:
  virtual tensorflow::Status Build(
      const unsigned int time, const std::vector<unsigned int>& locations,
      const absl::flat_hash_map<std::string, float>& args, Gate* gate) = 0;

  virtual ~GateBuilder() = default;
};

// Fill the pointed-to gate using the passed parameters.
// Returns UNIMPLEMENTED error if the given gate_id matches no TFQuantum gate,
// or INVALID_ARGUMENT error if a parameter is invalid for the given gate_id.
tensorflow::Status InitGate(const std::string& gate_name,
                            const unsigned int time,
                            const std::vector<unsigned int>& locations,
                            const absl::flat_hash_map<std::string, float>& args,
                            Gate* gate);

// ============================================================================
// GateBuilder Interfaces.
// ============================================================================

class OneQubitGateBuilder : public GateBuilder {
 public:
  virtual tensorflow::Status Build(
      const unsigned int time, const std::vector<unsigned int>& locations,
      const absl::flat_hash_map<std::string, float>& args, Gate* gate) override;

  virtual Matrix1q GetMatrix(const float exponent,
                             const float global_shift) = 0;
};

class OneQubitConstantGateBuilder : public GateBuilder {
 public:
  virtual tensorflow::Status Build(
      const unsigned int time, const std::vector<unsigned int>& locations,
      const absl::flat_hash_map<std::string, float>& args, Gate* gate) override;

  virtual Matrix1q GetMatrix() = 0;
};

class OneQubitPhasedGateBuilder : public GateBuilder {
 public:
  virtual tensorflow::Status Build(
      const unsigned int time, const std::vector<unsigned int>& locations,
      const absl::flat_hash_map<std::string, float>& args, Gate* gate) override;

  virtual Matrix1q GetMatrix(const float exponent, const float phase_exponent,
                             const float global_shift) = 0;
};

class TwoQubitGateBuilder : public GateBuilder {
 public:
  virtual tensorflow::Status Build(
      const unsigned int time, const std::vector<unsigned int>& locations,
      const absl::flat_hash_map<std::string, float>& args, Gate* gate) override;

  virtual Matrix2q GetMatrix(const float exponent,
                             const float global_shift) = 0;
};

class TwoQubitPhasedGateBuilder : public GateBuilder {
 public:
  virtual tensorflow::Status Build(
      const unsigned int time, const std::vector<unsigned int>& locations,
      const absl::flat_hash_map<std::string, float>& args, Gate* gate) override;

  virtual Matrix2q GetMatrix(const float exponent, const float phase_exponent,
                             const float global_shift) = 0;
};

class TwoQubitConstantGateBuilder : public GateBuilder {
 public:
  virtual tensorflow::Status Build(
      const unsigned int time, const std::vector<unsigned int>& locations,
      const absl::flat_hash_map<std::string, float>& args, Gate* gate) override;

  virtual Matrix2q GetMatrix() = 0;
};

// ============================================================================
// GateBuilder implementations.
// ============================================================================

class XPowGateBuilder : public OneQubitGateBuilder {
 public:
  Matrix1q GetMatrix(const float exponent, const float global_shift) override;
};

class YPowGateBuilder : public OneQubitGateBuilder {
 public:
  Matrix1q GetMatrix(const float exponent, const float global_shift) override;
};

class ZPowGateBuilder : public OneQubitGateBuilder {
 public:
  Matrix1q GetMatrix(const float exponent, const float global_shift) override;
};

class HPowGateBuilder : public OneQubitGateBuilder {
 public:
  Matrix1q GetMatrix(const float exponent, const float global_shift) override;
};

class IGateBuilder : public OneQubitConstantGateBuilder {
 public:
  Matrix1q GetMatrix() override;
};

class PhasedXPowGateBuilder : public OneQubitPhasedGateBuilder {
 public:
  Matrix1q GetMatrix(const float exponent, const float phase_exponent,
                     const float global_shift) override;
};

class XXPowGateBuilder : public TwoQubitGateBuilder {
 public:
  Matrix2q GetMatrix(const float exponent, const float global_shift) override;
};

class YYPowGateBuilder : public TwoQubitGateBuilder {
 public:
  Matrix2q GetMatrix(const float exponent, const float global_shift) override;
};

class ZZPowGateBuilder : public TwoQubitGateBuilder {
 public:
  Matrix2q GetMatrix(const float exponent, const float global_shift) override;
};

class CZPowGateBuilder : public TwoQubitGateBuilder {
 public:
  Matrix2q GetMatrix(const float exponent, const float global_shift) override;
};

class CNotPowGateBuilder : public TwoQubitGateBuilder {
 public:
  Matrix2q GetMatrix(const float exponent, const float global_shift) override;
};

class SwapPowGateBuilder : public TwoQubitGateBuilder {
 public:
  Matrix2q GetMatrix(const float exponent, const float global_shift) override;
};

class ISwapPowGateBuilder : public TwoQubitGateBuilder {
 public:
  Matrix2q GetMatrix(const float exponent, const float global_shift) override;
};

class PhasedISwapPowGateBuilder : public TwoQubitPhasedGateBuilder {
 public:
  Matrix2q GetMatrix(const float exponent, const float phase_exponent,
                     const float global_shift) override;
};

class I2GateBuilder : public TwoQubitConstantGateBuilder {
 public:
  Matrix2q GetMatrix() override;
};

class FSimGateBuilder : public GateBuilder {
 public:
  virtual tensorflow::Status Build(
      const unsigned int time, const std::vector<unsigned int>& locations,
      const absl::flat_hash_map<std::string, float>& args, Gate* gate) override;

  Matrix2q GetMatrix(const float theta, const float phi);
};

}  // namespace tfq

#endif  // TFQ_CORE_QSIM_GATES_DEF_H_
