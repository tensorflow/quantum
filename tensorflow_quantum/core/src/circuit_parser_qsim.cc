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

#include "tensorflow_quantum/core/src/circuit_parser_qsim.h"

#include <string>
#include <vector>

#include "../qsim/lib/circuit.h"
#include "../qsim/lib/fuser.h"
#include "../qsim/lib/fuser_basic.h"
#include "../qsim/lib/gates_cirq.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/numbers.h"
#include "cirq/google/api/v2/program.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_quantum/core/proto/pauli_sum.pb.h"

namespace tfq {

using ::cirq::google::api::v2::Moment;
using ::cirq::google::api::v2::Operation;
using ::cirq::google::api::v2::Program;
using ::tensorflow::Status;

namespace {

typedef absl::flat_hash_map<std::string, std::pair<int, float>> SymbolMap;
typedef qsim::Cirq::GateCirq<float> QsimGate;
typedef qsim::Circuit<QsimGate> QsimCircuit;

// series of fixed signature gate builders.
// there is no need to error check for unparseable symbols
// or proto args not being present. Those errors are caught
// upstream.

// single qubit gate Create(time, q0)
inline void SingleConstantGate(
    const Operation& op, const SymbolMap& param_map,
    const std::function<QsimGate(unsigned, unsigned)>& create_f,
    const unsigned time, QsimCircuit* circuit) {
  int q0;
  bool unused = absl::SimpleAtoi(op.qubits(0).id(), &q0);
  circuit->gates.push_back(create_f(time, q0));
}

// two qubit gate Create(time, q0, q1)
inline void TwoConstantGate(
    const Operation& op, const SymbolMap& param_map,
    const std::function<QsimGate(unsigned, unsigned, unsigned)>& create_f,
    const unsigned time, QsimCircuit* circuit) {
  int q0, q1;
  bool unused = absl::SimpleAtoi(op.qubits(0).id(), &q0);
  unused = absl::SimpleAtoi(op.qubits(1).id(), &q1);
  circuit->gates.push_back(create_f(time, q0, q1));
}

// single qubit eigen -> Create(time, q0, exponent, global_shift)
inline void SingleEigenGate(
    const Operation& op, const SymbolMap& param_map,
    const std::function<QsimGate(unsigned, unsigned, float, float)>& create_f,
    const unsigned time, QsimCircuit* circuit) {
  int q0;
  bool unused = absl::SimpleAtoi(op.qubits(0).id(), &q0);
  const auto expi = param_map.find("exponent");
  const auto exp_s_i = param_map.find("exponent_scalar");
  const auto gs_i = param_map.find("global_shift");
  circuit->gates.push_back(
      create_f(time, q0, expi->second.second * exp_s_i->second.second,
               gs_i->second.second));
}

// two qubit eigen -> Create(time, q0, q1, exp, gs)
inline void TwoEigenGate(
    const Operation& op, const SymbolMap& param_map,
    const std::function<QsimGate(unsigned, unsigned, unsigned, float, float)>&
        create_f,
    const unsigned time, QsimCircuit* circuit) {
  int q0, q1;
  bool unused = absl::SimpleAtoi(op.qubits(0).id(), &q0);
  unused = absl::SimpleAtoi(op.qubits(1).id(), &q1);
  const auto expi = param_map.find("exponent");
  const auto exp_s_i = param_map.find("exponent_scalar");
  const auto gs_i = param_map.find("global_shift");
  circuit->gates.push_back(
      create_f(time, q0, q1, expi->second.second * exp_s_i->second.second,
               gs_i->second.second));
}

void IGate(const Operation& op, const SymbolMap& param_map, const unsigned time,
           QsimCircuit* circuit) {
  SingleConstantGate(op, param_map, &qsim::Cirq::I<float>::Create, time,
                     circuit);
}

void I2Gate(const Operation& op, const SymbolMap& param_map,
            const unsigned time, QsimCircuit* circuit) {
  TwoConstantGate(op, param_map, &qsim::Cirq::I2<float>::Create, time, circuit);
}

void HGate(const Operation& op, const SymbolMap& param_map, const unsigned time,
           QsimCircuit* circuit) {
  SingleEigenGate(op, param_map, &qsim::Cirq::HPowGate<float>::Create, time,
                  circuit);
}

void XGate(const Operation& op, const SymbolMap& param_map, const unsigned time,
           QsimCircuit* circuit) {
  SingleEigenGate(op, param_map, &qsim::Cirq::XPowGate<float>::Create, time,
                  circuit);
}

void XXGate(const Operation& op, const SymbolMap& param_map,
            const unsigned time, QsimCircuit* circuit) {
  TwoEigenGate(op, param_map, &qsim::Cirq::XXPowGate<float>::Create, time,
               circuit);
}

void YGate(const Operation& op, const SymbolMap& param_map, const unsigned time,
           QsimCircuit* circuit) {
  SingleEigenGate(op, param_map, &qsim::Cirq::YPowGate<float>::Create, time,
                  circuit);
}

void YYGate(const Operation& op, const SymbolMap& param_map,
            const unsigned time, QsimCircuit* circuit) {
  TwoEigenGate(op, param_map, &qsim::Cirq::YYPowGate<float>::Create, time,
               circuit);
}

void ZGate(const Operation& op, const SymbolMap& param_map, const unsigned time,
           QsimCircuit* circuit) {
  SingleEigenGate(op, param_map, &qsim::Cirq::ZPowGate<float>::Create, time,
                  circuit);
}

void ZZGate(const Operation& op, const SymbolMap& param_map,
            const unsigned time, QsimCircuit* circuit) {
  TwoEigenGate(op, param_map, &qsim::Cirq::ZZPowGate<float>::Create, time,
               circuit);
}

void CZGate(const Operation& op, const SymbolMap& param_map,
            const unsigned time, QsimCircuit* circuit) {
  TwoEigenGate(op, param_map, &qsim::Cirq::CZPowGate<float>::Create, time,
               circuit);
}

void CXGate(const Operation& op, const SymbolMap& param_map,
            const unsigned time, QsimCircuit* circuit) {
  TwoEigenGate(op, param_map, &qsim::Cirq::CXPowGate<float>::Create, time,
               circuit);
}

void SwapGate(const Operation& op, const SymbolMap& param_map,
              const unsigned time, QsimCircuit* circuit) {
  TwoEigenGate(op, param_map, &qsim::Cirq::SwapPowGate<float>::Create, time,
               circuit);
}

void ISwapGate(const Operation& op, const SymbolMap& param_map,
               const unsigned time, QsimCircuit* circuit) {
  TwoEigenGate(op, param_map, &qsim::Cirq::ISwapPowGate<float>::Create, time,
               circuit);
}

// single qubit PhasedXPow -> Create(time, q0, pexp, exp)
inline void PhasedXGate(const Operation& op, const SymbolMap& param_map,
                        const unsigned time, QsimCircuit* circuit) {
  int q0;
  bool unused = absl::SimpleAtoi(op.qubits(0).id(), &q0);
  const auto expi = param_map.find("exponent");
  const auto exp_s_i = param_map.find("exponent_scalar");
  const auto pexpi = param_map.find("phase_exponent");
  const auto pexp_s_i = param_map.find("phase_exponent_scalar");
  const auto gs_i = param_map.find("global_shift");
  circuit->gates.push_back(qsim::Cirq::PhasedXPowGate<float>::Create(
      time, q0, pexpi->second.second * pexp_s_i->second.second,
      expi->second.second * exp_s_i->second.second, gs_i->second.second));
}

// two qubit fsim -> Create(time, q0, q1, theta, phi)
inline void FsimGate(const Operation& op, const SymbolMap& param_map,
                     const unsigned time, QsimCircuit* circuit) {
  int q0, q1;
  bool unused = absl::SimpleAtoi(op.qubits(0).id(), &q0);
  unused = absl::SimpleAtoi(op.qubits(1).id(), &q1);
  const auto theta = param_map.find("theta");
  const auto theta_s = param_map.find("theta_scalar");
  const auto phi = param_map.find("phi");
  const auto phi_s = param_map.find("phi_scalar");
  circuit->gates.push_back(qsim::Cirq::FSimGate<float>::Create(
      time, q0, q1, theta->second.second * theta_s->second.second,
      phi->second.second * phi_s->second.second));
}

// two qubit phase iswap -> Create(time, q0, q1, pexp, exp)
inline void PhasedISwapGate(const Operation& op, const SymbolMap& param_map,
                            const unsigned time, QsimCircuit* circuit) {
  int q0, q1;
  bool unused = absl::SimpleAtoi(op.qubits(0).id(), &q0);
  unused = absl::SimpleAtoi(op.qubits(1).id(), &q1);
  const auto pexp = param_map.find("phase_exponent");
  const auto pexp_s = param_map.find("phase_exponent_scalar");
  const auto exp = param_map.find("exponent");
  const auto exp_s = param_map.find("exponent_scalar");
  circuit->gates.push_back(qsim::Cirq::PhasedISwapPowGate<float>::Create(
      time, q0, q1, pexp->second.second * pexp_s->second.second,
      exp->second.second * exp_s->second.second));
}

tensorflow::Status ParseAppendGate(const Operation& op,
                                   const SymbolMap& param_map,
                                   const unsigned time, QsimCircuit* circuit) {
  // map gate name -> callable to build that qsim gate from operation proto.
  static const absl::flat_hash_map<
      std::string, std::function<void(const Operation&, const SymbolMap&,
                                      const unsigned, QsimCircuit*)>>
      func_map = {{"I", &IGate},       {"HP", &HGate},
                  {"XP", &XGate},      {"XXP", &XXGate},
                  {"YP", &YGate},      {"YYP", &YYGate},
                  {"ZP", &ZGate},      {"ZZP", &ZZGate},
                  {"CZP", &CZGate},    {"I2", &I2Gate},
                  {"CNP", &CXGate},    {"SP", &SwapGate},
                  {"ISP", &ISwapGate}, {"PXP", &PhasedXGate},
                  {"FSIM", &FsimGate}, {"PSIP", &PhasedISwapGate}};

  auto build_f = func_map.find(op.gate().id());
  if (build_f == func_map.end()) {
    return Status(tensorflow::error::INVALID_ARGUMENT,
                  "Could not parse gate id: " + op.gate().id());
  }
  build_f->second(op, param_map, time, circuit);
  return Status::OK();
}

}  // namespace

tensorflow::Status QsimCircuitFromProgram(
    const Program& program, const SymbolMap& param_map, const int num_qubits,
    QsimCircuit* circuit,
    std::vector<qsim::GateFused<QsimGate>>* fused_circuit) {
  // Convert proto to qsim internal representation.
  circuit->num_qubits = num_qubits;
  int time = 0;
  for (const Moment& moment : program.circuit().moments()) {
    for (const Operation& op : moment.operations()) {
      Status status = ParseAppendGate(op, param_map, time, circuit);
      if (!status.ok()) {
        return status;
      }
    }
    time++;
  }

  // Build fused circuit.
  *fused_circuit = qsim::BasicGateFuser<QsimGate>().FuseGates(
      circuit->num_qubits, circuit->gates, time);
  return Status::OK();
}

}  // namespace tfq
