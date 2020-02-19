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

#include "tensorflow_quantum/core/src/gates_def.h"

#define _USE_MATH_DEFINES
#include <complex>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/lib/core/status.h"

namespace tfq {
namespace {

using ::tensorflow::Status;

static const float DENOM_PLUS = 4.0f + 2.0f * std::sqrt(2);
static const float DENOM_MINUS = 4.0f - 2.0f * std::sqrt(2);
static const std::complex<float> PRE_00_PLUS =
    std::complex<float>((3 + 2 * std::sqrt(2)) / DENOM_PLUS, 0);
static const std::complex<float> PRE_00_MINUS =
    std::complex<float>((3 - 2 * std::sqrt(2)) / DENOM_MINUS, 0);
static const std::complex<float> PRE_0110_PLUS =
    std::complex<float>((1 + std::sqrt(2)) / DENOM_PLUS, 0);
static const std::complex<float> PRE_0110_MINUS =
    std::complex<float>((1 - std::sqrt(2)) / DENOM_MINUS, 0);
static const std::complex<float> PRE_11_PLUS =
    std::complex<float>(1 / DENOM_PLUS, 0);
static const std::complex<float> PRE_11_MINUS =
    std::complex<float>(1 / DENOM_MINUS, 0);

static constexpr std::complex<float> I_UNIT = std::complex<float>(0, 1);
static constexpr float pi = static_cast<float>(M_PI);
inline std::complex<float> global_phase(const float exponent,
                                        const float global_shift) {
  return std::exp(I_UNIT * pi * global_shift * exponent);
}

// Caller must free returned object
GateBuilder* GateNameMapper(const std::string& gate_name) {
  // clang-format off
  if (gate_name == "I") {return new IGateBuilder();}
  if (gate_name == "HP") {return new HPowGateBuilder();}
  if (gate_name == "XP") {return new XPowGateBuilder();}
  if (gate_name == "XXP") {return new XXPowGateBuilder();}
  if (gate_name == "YP") {return new YPowGateBuilder();}
  if (gate_name == "YYP") {return new YYPowGateBuilder();}
  if (gate_name == "ZP") {return new ZPowGateBuilder();}
  if (gate_name == "ZZP") {return new ZZPowGateBuilder();}
  if (gate_name == "CZP") {return new CZPowGateBuilder();}
  if (gate_name == "I2") {return new I2GateBuilder();}
  if (gate_name == "CNP") {return new CNotPowGateBuilder();}
  if (gate_name == "SP") {return new SwapPowGateBuilder();}
  if (gate_name == "ISP") {return new ISwapPowGateBuilder();}
  if (gate_name == "PXP") {return new PhasedXPowGateBuilder();}
  if (gate_name == "FSIM") {return new FSimGateBuilder();}
  if (gate_name == "PISP") {return new PhasedISwapPowGateBuilder();}
  // clang-format on
  return NULL;
}

}  // namespace

Gate::Gate() : time(0), num_qubits(0) {}

Gate::Gate(const unsigned int time_in, const unsigned int qubit_in,
           const std::array<float, 8>& matrix_in)
    : time(time_in), num_qubits(1) {
  qubits[0] = qubit_in;
  std::copy(matrix_in.begin(), matrix_in.end(), matrix.begin());
}

Gate::Gate(const unsigned int time_in, const unsigned int q1,
           const unsigned int q2, const std::array<float, 32>& matrix_in)
    : time(time_in), num_qubits(2) {
  qubits[0] = q1;
  qubits[1] = q2;
  std::copy(matrix_in.begin(), matrix_in.end(), matrix.begin());
}

bool operator==(const Gate& l, const Gate& r) {
  if (l.time != r.time) {
    return false;
  }
  if (l.num_qubits != r.num_qubits) {
    return false;
  }
  for (unsigned int i = 0; i < l.num_qubits; i++) {
    if (l.qubits.at(i) != r.qubits.at(i)) {
      return false;
    }
  }
  if (l.num_qubits > 0) {
    // real and imaginary component for each matrix site
    const unsigned int true_mat_size =
        2 * (1 << l.num_qubits) * (1 << l.num_qubits);
    for (unsigned int i = 0; i < true_mat_size; i++) {
      if (std::fabs(l.matrix[i] - r.matrix[i]) > 1e-6) {
        return false;
      }
    }
  }
  return true;
}

bool operator!=(const Gate& l, const Gate& r) { return !(l == r); }

Status InitGate(const std::string& gate_name, const unsigned int time,
                const std::vector<unsigned int>& locations,
                const absl::flat_hash_map<std::string, float>& args,
                Gate* gate) {
  GateBuilder* builder;
  builder = GateNameMapper(gate_name);
  if (builder == NULL) {
    return Status(tensorflow::error::UNIMPLEMENTED,
                  "The given gate id, " + gate_name +
                      ", does not match any available TFQ gate.");
  }
  Status status = builder->Build(time, locations, args, gate);
  delete builder;
  return status;
}

Status OneQubitGateBuilder::Build(
    const unsigned int time, const std::vector<unsigned int>& locations,
    const absl::flat_hash_map<std::string, float>& args, Gate* gate) {
  if (locations.size() != 1) {
    return Status(tensorflow::error::INVALID_ARGUMENT,
                  "Only one qubit location should be provided.");
  }

  float exponent;
  float global_shift;
  const auto itr_exponent = args.find("exponent");
  const auto itr_global_shift = args.find("global_shift");
  // Workaround to support scalar multiplication of symbols. See serialize.py.
  const auto itr_exponent_scalar = args.find("exponent_scalar");
  if (itr_exponent == args.end() || itr_global_shift == args.end() ||
      itr_exponent_scalar == args.end()) {
    return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                              "Eigen gates require exponent and "
                              "global_shift args.");
  }
  exponent = itr_exponent->second * itr_exponent_scalar->second;
  global_shift = itr_global_shift->second;
  *gate = Gate(time, locations[0], GetMatrix(exponent, global_shift));
  return Status::OK();
}

Status OneQubitConstantGateBuilder::Build(
    const unsigned int time, const std::vector<unsigned int>& locations,
    const absl::flat_hash_map<std::string, float>& args, Gate* gate) {
  if (locations.size() != 1) {
    return Status(tensorflow::error::INVALID_ARGUMENT,
                  "Only one qubit location should be provided.");
  }
  if (!args.empty()) {
    return Status(tensorflow::error::INVALID_ARGUMENT,
                  "Constant gates take no arguments, " +
                      std::to_string(args.size()) + " were given.");
  }
  *gate = Gate(time, locations[0], GetMatrix());
  return Status::OK();
}

Status OneQubitPhasedGateBuilder::Build(
    const unsigned int time, const std::vector<unsigned int>& locations,
    const absl::flat_hash_map<std::string, float>& args, Gate* gate) {
  if (locations.size() != 1) {
    return Status(tensorflow::error::INVALID_ARGUMENT,
                  "Only one qubit location should be provided.");
  }
  float exponent;
  float phase_exponent;
  float global_shift;
  const auto itr_exponent = args.find("exponent");
  const auto itr_phase_exponent = args.find("phase_exponent");
  const auto itr_global_shift = args.find("global_shift");
  // Workaround to support scalar multiplication of symbols. See serialize.py.
  const auto itr_exponent_scalar = args.find("exponent_scalar");
  const auto itr_phase_exponent_scalar = args.find("phase_exponent_scalar");
  if (itr_exponent == args.end() || itr_global_shift == args.end() ||
      itr_exponent_scalar == args.end()) {
    return Status(tensorflow::error::INVALID_ARGUMENT,
                  "Eigen gates require exponent and global_shift args.");
  }
  exponent = itr_exponent->second * itr_exponent_scalar->second;
  phase_exponent =
      itr_phase_exponent->second * itr_phase_exponent_scalar->second;
  global_shift = itr_global_shift->second;
  *gate = Gate(time, locations[0],
               GetMatrix(exponent, phase_exponent, global_shift));
  return Status::OK();
}

Status TwoQubitGateBuilder::Build(
    const unsigned int time, const std::vector<unsigned int>& locations,
    const absl::flat_hash_map<std::string, float>& args, Gate* gate) {
  if (locations.size() != 2) {
    return Status(tensorflow::error::INVALID_ARGUMENT,
                  "Only two qubit locations should be provided.");
  }
  float exponent;
  float global_shift;
  const auto itr_exponent = args.find("exponent");
  const auto itr_global_shift = args.find("global_shift");
  // Workaround to support scalar multiplication of symbols. See serialize.py.
  const auto itr_exponent_scalar = args.find("exponent_scalar");
  if (itr_exponent == args.end() || itr_global_shift == args.end() ||
      itr_exponent_scalar == args.end()) {
    return Status(tensorflow::error::INVALID_ARGUMENT,
                  "Eigen gates require exponent and global_shift args.");
  }
  exponent = itr_exponent->second * itr_exponent_scalar->second;
  global_shift = itr_global_shift->second;
  if (locations[0] < locations[1]) {
    *gate = Gate(time, locations[0], locations[1],
                 GetMatrix(exponent, global_shift));
  } else {
    *gate = Gate(time, locations[1], locations[0],
                 GetSwappedMatrix(exponent, global_shift));
  }
  return Status::OK();
}

Status TwoQubitPhasedGateBuilder::Build(
    const unsigned int time, const std::vector<unsigned int>& locations,
    const absl::flat_hash_map<std::string, float>& args, Gate* gate) {
  if (locations.size() != 2) {
    return Status(tensorflow::error::INVALID_ARGUMENT,
                  "Only two qubit locations should be provided.");
  }
  float exponent;
  float global_shift;
  float phase_exponent;
  const auto itr_exponent = args.find("exponent");
  const auto itr_phase_exponent = args.find("phase_exponent");
  const auto itr_global_shift = args.find("global_shift");
  // Workaround to support scalar multiplication of symbols. See serialize.py.
  const auto itr_exponent_scalar = args.find("exponent_scalar");
  const auto itr_phase_exponent_scalar = args.find("phase_exponent_scalar");
  if (itr_exponent == args.end() || itr_global_shift == args.end() ||
      itr_exponent_scalar == args.end()) {
    return Status(tensorflow::error::INVALID_ARGUMENT,
                  "Eigen gates require exponent and global_shift args.");
  }
  exponent = itr_exponent->second * itr_exponent_scalar->second;
  phase_exponent =
      itr_phase_exponent->second * itr_phase_exponent_scalar->second;
  global_shift = itr_global_shift->second;
  if (locations[0] < locations[1]) {
    *gate = Gate(time, locations[0], locations[1],
                 GetMatrix(exponent, phase_exponent, global_shift));
  } else {
    *gate = Gate(time, locations[1], locations[0],
                 GetSwappedMatrix(exponent, phase_exponent, global_shift));
  }
  return Status::OK();
}

Status TwoQubitConstantGateBuilder::Build(
    const unsigned int time, const std::vector<unsigned int>& locations,
    const absl::flat_hash_map<std::string, float>& args, Gate* gate) {
  if (locations.size() != 2) {
    return Status(tensorflow::error::INVALID_ARGUMENT,
                  "Only two qubit locations should be provided.");
  }
  if (!args.empty()) {
    return Status(tensorflow::error::INVALID_ARGUMENT,
                  "Constant gates take no arguments, " +
                      std::to_string(args.size()) + " were given.");
  }

  if (locations[0] < locations[1]) {
    *gate = Gate(time, locations[0], locations[1], GetMatrix());
  } else {
    *gate = Gate(time, locations[1], locations[0], GetSwappedMatrix());
  }

  return Status::OK();
}

Matrix1q XPowGateBuilder::GetMatrix(const float exponent,
                                    const float global_shift) {
  const std::complex<float> g = global_phase(exponent, global_shift);
  const std::complex<float> w = global_phase(exponent, 1);
  const std::complex<float> plus = 0.5f * g * (1.0f + w);
  const std::complex<float> minus = 0.5f * g * (1.0f - w);

  return {{plus.real(), plus.imag(), minus.real(), minus.imag(), minus.real(),
           minus.imag(), plus.real(), plus.imag()}};
}

Matrix1q YPowGateBuilder::GetMatrix(const float exponent,
                                    const float global_shift) {
  const std::complex<float> g = global_phase(exponent, global_shift);
  const std::complex<float> w = global_phase(exponent, 1);
  const std::complex<float> plus = 0.5f * g * (1.0f + w);
  const std::complex<float> v01 = 0.5f * g * (-1.0f + w) * I_UNIT;
  const std::complex<float> v10 = 0.5f * g * (1.0f - w) * I_UNIT;

  return {{plus.real(), plus.imag(), v01.real(), v01.imag(), v10.real(),
           v10.imag(), plus.real(), plus.imag()}};
}

Matrix1q ZPowGateBuilder::GetMatrix(const float exponent,
                                    const float global_shift) {
  const std::complex<float> g = global_phase(exponent, global_shift);
  const std::complex<float> w = global_phase(exponent, 1);
  const std::complex<float> prod = g * w;

  return {
      {g.real(), g.imag(), 0.0f, 0.0f, 0.0f, 0.0f, prod.real(), prod.imag()}};
}

Matrix1q HPowGateBuilder::GetMatrix(const float exponent,
                                    const float global_shift) {
  const std::complex<float> g = global_phase(exponent, global_shift);
  const std::complex<float> w = global_phase(exponent, 1);

  const std::complex<float> v00 = g * (PRE_00_PLUS + PRE_00_MINUS * w);
  const std::complex<float> v01_and_10 =
      g * (PRE_0110_PLUS + PRE_0110_MINUS * w);
  const std::complex<float> v11 = g * (PRE_11_PLUS + PRE_11_MINUS * w);

  return {{v00.real(), v00.imag(), v01_and_10.real(), v01_and_10.imag(),
           v01_and_10.real(), v01_and_10.imag(), v11.real(), v11.imag()}};
}

Matrix1q IGateBuilder::GetMatrix() {
  static constexpr Matrix1q matrix = {{1, 0, 0, 0, 0, 0, 1, 0}};
  return matrix;
}

Matrix1q PhasedXPowGateBuilder::GetMatrix(const float exponent,
                                          const float phase_exponent,
                                          const float global_shift) {
  const std::complex<float> g = global_phase(exponent, global_shift);
  const std::complex<float> w = global_phase(exponent, 1);
  const std::complex<float> w2 = global_phase(phase_exponent, 1);
  const std::complex<float> w2_n = global_phase(-1.0f * phase_exponent, 1);

  const std::complex<float> v00_and_11 = 0.5f * g * (1.0f + w);
  const std::complex<float> v01 = 0.5f * g * (1.0f - w) * w2_n;
  const std::complex<float> v10 = 0.5f * g * (1.0f - w) * w2;

  return {{v00_and_11.real(), v00_and_11.imag(), v01.real(), v01.imag(),
           v10.real(), v10.imag(), v00_and_11.real(), v00_and_11.imag()}};
}

Matrix2q XXPowGateBuilder::GetMatrix(const float exponent,
                                     const float global_shift) {
  const std::complex<float> g = global_phase(exponent, global_shift);
  const std::complex<float> w = global_phase(exponent, 1);
  const std::complex<float> pc = 0.5f * g * (1.0f + w);
  const std::complex<float> mc = 0.5f * g * (1.0f - w);

  // clang-format off
  return {{pc.real(), pc.imag(), 0, 0, 0, 0, mc.real(), mc.imag(),
           0, 0, pc.real(), pc.imag(), mc.real(), mc.imag(), 0, 0,
           0, 0, mc.real(), mc.imag(), pc.real(), pc.imag(), 0, 0,
           mc.real(), mc.imag(), 0, 0, 0, 0, pc.real(), pc.imag()}};
  // clang-format on
}

Matrix2q XXPowGateBuilder::GetSwappedMatrix(const float exponent,
                                            const float global_shift) {
  return GetMatrix(exponent, global_shift);
}

Matrix2q YYPowGateBuilder::GetMatrix(const float exponent,
                                     const float global_shift) {
  const std::complex<float> g = global_phase(exponent, global_shift);
  const std::complex<float> w = global_phase(exponent, 1);
  const std::complex<float> pc = 0.5f * g * (1.0f + w);
  const std::complex<float> mc = 0.5f * g * (1.0f - w);
  const std::complex<float> n_mc = -1.0f * mc;

  // clang-format off
  return {{pc.real(), pc.imag(), 0, 0, 0, 0, n_mc.real(), n_mc.imag(),
           0, 0, pc.real(), pc.imag(), mc.real(), mc.imag(), 0, 0,
           0, 0, mc.real(), mc.imag(), pc.real(), pc.imag(), 0, 0,
           n_mc.real(), n_mc.imag(), 0, 0, 0, 0, pc.real(), pc.imag()}};
  // clang-format on
}

Matrix2q YYPowGateBuilder::GetSwappedMatrix(const float exponent,
                                            const float global_shift) {
  return GetMatrix(exponent, global_shift);
}

Matrix2q ZZPowGateBuilder::GetMatrix(const float exponent,
                                     const float global_shift) {
  const std::complex<float> g = global_phase(exponent, global_shift);
  const std::complex<float> w = global_phase(exponent, 1);
  const std::complex<float> gw = g * w;

  // clang-format off
  return {{g.real(), g.imag(), 0, 0, 0, 0, 0, 0,
           0, 0, gw.real(), gw.imag(), 0, 0, 0, 0,
           0, 0, 0, 0, gw.real(), gw.imag(), 0, 0,
           0, 0, 0, 0, 0, 0, g.real(), g.imag()}};
  // clang-format on
}

Matrix2q ZZPowGateBuilder::GetSwappedMatrix(const float exponent,
                                            const float global_shift) {
  return GetMatrix(exponent, global_shift);
}

Matrix2q CZPowGateBuilder::GetMatrix(const float exponent,
                                     const float global_shift) {
  const std::complex<float> g = global_phase(exponent, global_shift);
  const std::complex<float> w = global_phase(exponent, 1);
  const std::complex<float> gw = g * w;

  // clang-format off
  return {{g.real(), g.imag(), 0, 0, 0, 0, 0, 0,
           0, 0, g.real(), g.imag(), 0, 0, 0, 0,
           0, 0, 0, 0, g.real(), g.imag(), 0, 0,
           0, 0, 0, 0, 0, 0, gw.real(), gw.imag()}};
  // clang-format on
}

Matrix2q CZPowGateBuilder::GetSwappedMatrix(const float exponent,
                                            const float global_shift) {
  return GetMatrix(exponent, global_shift);
}

Matrix2q CNotPowGateBuilder::GetMatrix(const float exponent,
                                       const float global_shift) {
  const std::complex<float> g = global_phase(exponent, global_shift);
  const std::complex<float> w = global_phase(exponent, 1);
  const std::complex<float> plus = 0.5f * g * (1.0f + w);
  const std::complex<float> minus = 0.5f * g * (1.0f - w);

  // clang-format off
  return {{g.real(), g.imag(), 0, 0, 0, 0, 0, 0,
           0, 0, g.real(), g.imag(), 0, 0, 0, 0,
           0, 0, 0, 0, plus.real(), plus.imag(), minus.real(), minus.imag(),
           0, 0, 0, 0, minus.real(), minus.imag(), plus.real(), plus.imag()}};
  // clang-format on
}

Matrix2q CNotPowGateBuilder::GetSwappedMatrix(const float exponent,
                                              const float global_shift) {
  const std::complex<float> g = global_phase(exponent, global_shift);
  const std::complex<float> w = global_phase(exponent, 1);
  const std::complex<float> plus = 0.5f * g * (1.0f + w);
  const std::complex<float> minus = 0.5f * g * (1.0f - w);

  // clang-format off
  return {{g.real(), g.imag(), 0, 0, 0, 0, 0, 0,
           0, 0, plus.real(), plus.imag(), 0, 0, minus.real(), minus.imag(),
           0, 0, 0, 0, g.real(), g.imag(), 0, 0,
           0, 0, minus.real(), minus.imag(), 0, 0, plus.real(), plus.imag()}};
  // clang-format on
}

Matrix2q SwapPowGateBuilder::GetMatrix(const float exponent,
                                       const float global_shift) {
  const std::complex<float> g = global_phase(exponent, global_shift);
  const std::complex<float> w = global_phase(exponent, 1);
  const std::complex<float> plus = 0.5f * g * (1.0f + w);
  const std::complex<float> minus = 0.5f * g * (1.0f - w);

  // clang-format off
  return {{g.real(), g.imag(), 0, 0, 0, 0, 0, 0,
           0, 0, plus.real(), plus.imag(), minus.real(), minus.imag(), 0, 0,
           0, 0, minus.real(), minus.imag(), plus.real(), plus.imag(), 0, 0,
           0, 0, 0, 0, 0, 0, g.real(), g.imag()}};
  // clang-format on
}

Matrix2q SwapPowGateBuilder::GetSwappedMatrix(const float exponent,
                                              const float global_shift) {
  return GetMatrix(exponent, global_shift);
}

Matrix2q ISwapPowGateBuilder::GetMatrix(const float exponent,
                                        const float global_shift) {
  const std::complex<float> g = global_phase(exponent, global_shift);
  const std::complex<float> wp = global_phase(exponent, 0.5);
  const std::complex<float> wm = global_phase(exponent, -0.5);
  const std::complex<float> plus = 0.5f * g * (wp + wm);
  const std::complex<float> minus = 0.5f * g * (wp - wm);

  // clang-format off
  return {{g.real(), g.imag(), 0, 0, 0, 0, 0, 0,
           0, 0, plus.real(), plus.imag(), minus.real(), minus.imag(), 0, 0,
           0, 0, minus.real(), minus.imag(), plus.real(), plus.imag(), 0, 0,
           0, 0, 0, 0, 0, 0, g.real(), g.imag()}};
  // clang-format on
}

Matrix2q ISwapPowGateBuilder::GetSwappedMatrix(const float exponent,
                                               const float global_shift) {
  return GetMatrix(exponent, global_shift);
}

Matrix2q PhasedISwapPowGateBuilder::GetMatrix(const float exponent,
                                              const float phase_exponent,
                                              const float global_shift) {
  const std::complex<float> g = global_phase(exponent, global_shift);
  const std::complex<float> wp = global_phase(exponent, 0.5);
  const std::complex<float> wm = global_phase(exponent, -0.5);
  const std::complex<float> plus = 0.5f * g * (wp + wm);
  const std::complex<float> minus = 0.5f * g * (wp - wm);
  const std::complex<float> f = global_phase(phase_exponent, 2.0);
  const std::complex<float> f_star = std::conj(f);
  const std::complex<float> ur = minus * f;
  const std::complex<float> bl = minus * f_star;

  return {
      {g.real(),  g.imag(),    0,           0,           0,         0, 0, 0, 0,
       0,         plus.real(), plus.imag(), ur.real(),   ur.imag(), 0, 0, 0, 0,
       bl.real(), bl.imag(),   plus.real(), plus.imag(), 0,         0, 0, 0, 0,
       0,         0,           0,           g.real(),    g.imag()}};
}

Matrix2q PhasedISwapPowGateBuilder::GetSwappedMatrix(const float exponent,
                                                     const float phase_exponent,
                                                     const float global_shift) {
  const std::complex<float> g = global_phase(exponent, global_shift);
  const std::complex<float> wp = global_phase(exponent, 0.5);
  const std::complex<float> wm = global_phase(exponent, -0.5);
  const std::complex<float> plus = 0.5f * g * (wp + wm);
  const std::complex<float> minus = 0.5f * g * (wp - wm);
  const std::complex<float> f = global_phase(phase_exponent, 2.0);
  const std::complex<float> f_star = std::conj(f);
  const std::complex<float> ur = minus * f;
  const std::complex<float> bl = minus * f_star;

  return {
      {g.real(),  g.imag(),    0,           0,           0,         0, 0, 0, 0,
       0,         plus.real(), plus.imag(), bl.real(),   bl.imag(), 0, 0, 0, 0,
       ur.real(), ur.imag(),   plus.real(), plus.imag(), 0,         0, 0, 0, 0,
       0,         0,           0,           g.real(),    g.imag()}};
}

Matrix2q I2GateBuilder::GetMatrix() {
  // clang-format off
  static constexpr Matrix2q matrix = {{1, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 1, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 1, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 1, 0}};
  // clang-format on
  return matrix;
}

Matrix2q I2GateBuilder::GetSwappedMatrix() { return GetMatrix(); }

Status FSimGateBuilder::Build(
    const unsigned int time, const std::vector<unsigned int>& locations,
    const absl::flat_hash_map<std::string, float>& args, Gate* gate) {
  if (locations.size() != 2) {
    return Status(tensorflow::error::INVALID_ARGUMENT,
                  "Two qubit locations should be provided.");
  }
  double theta;
  double phi;
  const auto itr_theta = args.find("theta");
  const auto itr_phi = args.find("phi");
  const auto itr_theta_scalar = args.find("theta_scalar");
  const auto itr_phi_scalar = args.find("phi_scalar");
  if (itr_theta == args.end() || itr_phi == args.end() ||
      itr_theta_scalar == args.end() || itr_phi_scalar == args.end()) {
    return Status(tensorflow::error::INVALID_ARGUMENT,
                  "FSimGate requires theta and phi args.");
  }
  theta = itr_theta->second * itr_theta_scalar->second;
  phi = itr_phi->second * itr_phi_scalar->second;

  if (locations[0] < locations[1]) {
    *gate = Gate(time, locations[0], locations[1], GetMatrix(theta, phi));
  } else {
    *gate =
        Gate(time, locations[1], locations[0], GetSwappedMatrix(theta, phi));
  }
  return Status::OK();
}

Matrix2q FSimGateBuilder::GetMatrix(const float theta, const float phi) {
  const std::complex<float> minus_i_unit = std::complex<float>(0, -1.0);
  const std::complex<float> a = std::cos(theta);
  const std::complex<float> b = minus_i_unit * std::sin(theta);
  const std::complex<float> c = std::exp(minus_i_unit * phi);
  // clang-format off
  return {{1, 0, 0, 0, 0, 0, 0, 0,
          0, 0, a.real(), a.imag(), b.real(), b.imag(), 0, 0,
          0, 0, b.real(), b.imag(), a.real(), a.imag(), 0, 0,
          0, 0, 0, 0, 0, 0, c.real(), c.imag()}};
  // clang-format on
}

Matrix2q FSimGateBuilder::GetSwappedMatrix(const float theta, const float phi) {
  return GetMatrix(theta, phi);
}

}  // namespace tfq
