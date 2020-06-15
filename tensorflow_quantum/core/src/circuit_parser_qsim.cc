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

#include <algorithm>
#include <iostream>
#include <memory>
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
tensorflow::Status ParseAppendGate(
    const Operation& op,
    const absl::flat_hash_map<std::string, std::pair<int, float>>& param_map,
    const unsigned time, qsim::Circuit<qsim::Cirq::GateCirq<float>>* circuit) {
  bool unused;
  const std::string& gate_name = op.gate().id();
  if (gate_name == "I") {
    // Safe to not error check since this has been done
    //  upstream with ResolveQubitIDs.
    int q0;
    unused = absl::SimpleAtoi(op.qubits(0).id(), &q0);
    circuit->gates.push_back(qsim::Cirq::I<float>().Create(time, q0));
    return Status::OK();
  }
  if (gate_name == "HP") {
    int q0;
    unused = absl::SimpleAtoi(op.qubits(0).id(), &q0);
    const auto expi = param_map.find("exponent");
    const auto exp_s_i = param_map.find("exponent_scalar");
    const auto gs_i = param_map.find("global_shift");
    circuit->gates.push_back(qsim::Cirq::HPowGate<float>().Create(
        time, q0, expi->second.second * exp_s_i->second.second,
        gs_i->second.second));
    return Status::OK();
  }
  if (gate_name == "XP") {
    int q0;
    unused = absl::SimpleAtoi(op.qubits(0).id(), &q0);
    const auto expi = param_map.find("exponent");
    const auto exp_s_i = param_map.find("exponent_scalar");
    const auto gs_i = param_map.find("global_shift");
    circuit->gates.push_back(qsim::Cirq::XPowGate<float>().Create(
        time, q0, expi->second.second * exp_s_i->second.second,
        gs_i->second.second));
    return Status::OK();
  }
  if (gate_name == "XXP") {
    int q0, q1;
    unused = absl::SimpleAtoi(op.qubits(0).id(), &q0);
    unused = absl::SimpleAtoi(op.qubits(1).id(), &q1);
    const auto expi = param_map.find("exponent");
    const auto exp_s_i = param_map.find("exponent_scalar");
    const auto gs_i = param_map.find("global_shift");
    circuit->gates.push_back(qsim::Cirq::XXPowGate<float>().Create(
        time, q0, q1, expi->second.second * exp_s_i->second.second,
        gs_i->second.second));
    return Status::OK();
  }
  if (gate_name == "YP") {
    int q0;
    unused = absl::SimpleAtoi(op.qubits(0).id(), &q0);
    const auto expi = param_map.find("exponent");
    const auto exp_s_i = param_map.find("exponent_scalar");
    const auto gs_i = param_map.find("global_shift");
    circuit->gates.push_back(qsim::Cirq::YPowGate<float>().Create(
        time, q0, expi->second.second * exp_s_i->second.second,
        gs_i->second.second));
    return Status::OK();
  }
  if (gate_name == "YYP") {
    int q0, q1;
    unused = absl::SimpleAtoi(op.qubits(0).id(), &q0);
    unused = absl::SimpleAtoi(op.qubits(1).id(), &q1);
    const auto expi = param_map.find("exponent");
    const auto exp_s_i = param_map.find("exponent_scalar");
    const auto gs_i = param_map.find("global_shift");
    circuit->gates.push_back(qsim::Cirq::YYPowGate<float>().Create(
        time, q0, q1, expi->second.second * exp_s_i->second.second,
        gs_i->second.second));
    return Status::OK();
  }
  if (gate_name == "ZP") {
    int q0;
    unused = absl::SimpleAtoi(op.qubits(0).id(), &q0);
    const auto expi = param_map.find("exponent");
    const auto exp_s_i = param_map.find("exponent_scalar");
    const auto gs_i = param_map.find("global_shift");
    circuit->gates.push_back(qsim::Cirq::ZPowGate<float>().Create(
        time, q0, expi->second.second * exp_s_i->second.second,
        gs_i->second.second));
    return Status::OK();
  }
  if (gate_name == "ZZP") {
    int q0, q1;
    unused = absl::SimpleAtoi(op.qubits(0).id(), &q0);
    unused = absl::SimpleAtoi(op.qubits(1).id(), &q1);
    const auto expi = param_map.find("exponent");
    const auto exp_s_i = param_map.find("exponent_scalar");
    const auto gs_i = param_map.find("global_shift");
    circuit->gates.push_back(qsim::Cirq::ZZPowGate<float>().Create(
        time, q0, q1, expi->second.second * exp_s_i->second.second,
        gs_i->second.second));
    return Status::OK();
  }
  if (gate_name == "CZP") {
    int q0, q1;
    unused = absl::SimpleAtoi(op.qubits(0).id(), &q0);
    unused = absl::SimpleAtoi(op.qubits(1).id(), &q1);
    const auto expi = param_map.find("exponent");
    const auto exp_s_i = param_map.find("exponent_scalar");
    const auto gs_i = param_map.find("global_shift");
    circuit->gates.push_back(qsim::Cirq::CZPowGate<float>().Create(
        time, q0, q1, expi->second.second * exp_s_i->second.second,
        gs_i->second.second));
    return Status::OK();
  }
  if (gate_name == "I2") {
    int q0, q1;
    unused = absl::SimpleAtoi(op.qubits(0).id(), &q0);
    unused = absl::SimpleAtoi(op.qubits(1).id(), &q1);
    circuit->gates.push_back(qsim::Cirq::I2<float>().Create(time, q0, q1));
    return Status::OK();
  }
  if (gate_name == "CNP") {
    int q0, q1;
    unused = absl::SimpleAtoi(op.qubits(0).id(), &q0);
    unused = absl::SimpleAtoi(op.qubits(1).id(), &q1);
    const auto expi = param_map.find("exponent");
    const auto exp_s_i = param_map.find("exponent_scalar");
    const auto gs_i = param_map.find("global_shift");
    circuit->gates.push_back(qsim::Cirq::CXPowGate<float>().Create(
        time, q0, q1, expi->second.second * exp_s_i->second.second,
        gs_i->second.second));
    return Status::OK();
  }
  if (gate_name == "SP") {
    int q0, q1;
    unused = absl::SimpleAtoi(op.qubits(0).id(), &q0);
    unused = absl::SimpleAtoi(op.qubits(1).id(), &q1);
    const auto expi = param_map.find("exponent");
    const auto exp_s_i = param_map.find("exponent_scalar");
    const auto gs_i = param_map.find("global_shift");
    circuit->gates.push_back(qsim::Cirq::SwapPowGate<float>().Create(
        time, q0, q1, expi->second.second * exp_s_i->second.second,
        gs_i->second.second));
    return Status::OK();
  }
  if (gate_name == "ISP") {
    int q0, q1;
    unused = absl::SimpleAtoi(op.qubits(0).id(), &q0);
    unused = absl::SimpleAtoi(op.qubits(1).id(), &q1);
    const auto expi = param_map.find("exponent");
    const auto exp_s_i = param_map.find("exponent_scalar");
    const auto gs_i = param_map.find("global_shift");
    circuit->gates.push_back(qsim::Cirq::ISwapPowGate<float>().Create(
        time, q0, q1, expi->second.second * exp_s_i->second.second,
        gs_i->second.second));
    return Status::OK();
  }
  if (gate_name == "PXP") {
    int q0;
    unused = absl::SimpleAtoi(op.qubits(0).id(), &q0);
    const auto expi = param_map.find("exponent");
    const auto exp_s_i = param_map.find("exponent_scalar");
    const auto pexpi = param_map.find("phase_exponent");
    const auto pexp_s_i = param_map.find("phase_exponent_scalar");
    const auto gs_i = param_map.find("global_shift");
    circuit->gates.push_back(qsim::Cirq::PhasedXPowGate<float>().Create(
        time, q0, pexpi->second.second * pexp_s_i->second.second,
        expi->second.second * exp_s_i->second.second, gs_i->second.second));
    return Status::OK();
  }
  if (gate_name == "FSIM") {
    int q0, q1;
    unused = absl::SimpleAtoi(op.qubits(0).id(), &q0);
    unused = absl::SimpleAtoi(op.qubits(1).id(), &q1);
    const auto theta = param_map.find("theta");
    const auto theta_s = param_map.find("theta_scalar");
    const auto phi = param_map.find("phi");
    const auto phi_s = param_map.find("phi_scalar");
    circuit->gates.push_back(qsim::Cirq::FSimGate<float>().Create(
        time, q0, q1, theta->second.second * theta_s->second.second,
        phi->second.second * phi_s->second.second));
    return Status::OK();
  }
  if (gate_name == "PISP") {
    int q0, q1;
    unused = absl::SimpleAtoi(op.qubits(0).id(), &q0);
    unused = absl::SimpleAtoi(op.qubits(1).id(), &q1);
    const auto pexp = param_map.find("phase_exponent");
    const auto pexp_s = param_map.find("phase_exponent_scalar");
    const auto exp = param_map.find("exponent");
    const auto exp_s = param_map.find("exponent_scalar");
    circuit->gates.push_back(qsim::Cirq::PhasedISwapPowGate<float>().Create(
        time, q0, q1, pexp->second.second * pexp_s->second.second,
        exp->second.second * exp_s->second.second));
    return Status::OK();
  }

  return Status(tensorflow::error::INVALID_ARGUMENT,
                "Could not parse Gate id: " + op.gate().id());
}

}  // namespace

tensorflow::Status QsimCircuitFromProgram(
    const Program& program,
    const absl::flat_hash_map<std::string, std::pair<int, float>>& param_map,
    const int num_qubits, qsim::Circuit<qsim::Cirq::GateCirq<float>>* circuit,
    std::vector<qsim::GateFused<qsim::Cirq::GateCirq<float>>>* fused_circuit) {
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
  *fused_circuit =
      qsim::BasicGateFuser<qsim::Cirq::GateCirq<float>>().FuseGates(
          circuit->num_qubits, circuit->gates, time);
  return Status::OK();
}

}  // namespace tfq
