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
#include "../qsim/lib/io.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/numbers.h"
#include "absl/types/optional.h"
#include "cirq/google/api/v2/program.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_quantum/core/proto/pauli_sum.pb.h"

namespace tfq {

using ::cirq::google::api::v2::Moment;
using ::cirq::google::api::v2::Operation;
using ::cirq::google::api::v2::Program;
using ::tensorflow::Status;
using ::tfq::proto::PauliTerm;

namespace {

typedef absl::flat_hash_map<std::string, std::pair<int, float>> SymbolMap;
typedef qsim::Cirq::GateCirq<float> QsimGate;
typedef qsim::Circuit<QsimGate> QsimCircuit;

inline Status ParseProtoArg(
    const Operation& op, const std::string& arg_name,
    const SymbolMap& param_map, float* result,
    absl::optional<std::string>* symbol_used = nullptr) {
  // find arg_name in proto.
  // iterator<Map<str, Arg>>
  const auto arg_v = op.args().find(arg_name);
  if (arg_v == op.args().end()) {
    return Status(tensorflow::error::INVALID_ARGUMENT,
                  "Could not find arg: " + arg_name + " in op.");
  }
  // find proto arg field.
  // ::cirq::google::api::v2::Arg
  const auto proto_arg = arg_v->second;
  *result = proto_arg.arg_value().float_value();
  if (!proto_arg.symbol().empty()) {
    // find symbol value in param_map.
    const auto iter = param_map.find(proto_arg.symbol());
    if (iter == param_map.end()) {
      return Status(
          tensorflow::error::INVALID_ARGUMENT,
          "Could not find symbol in parameter map: " + proto_arg.symbol());
    }
    *result = iter->second.second;
    if (symbol_used != nullptr) {
      symbol_used->emplace(iter->first);
    }
  }
  return Status::OK();
}

// series of fixed signature gate builders.
// there is no need to error check for unparseable symbols
// or proto args not being present. Those errors are caught
// upstream.

// single qubit gate Create(time, q0)
inline Status SingleConstantGate(
    const Operation& op, const SymbolMap& param_map,
    const std::function<QsimGate(unsigned int, unsigned int)>& create_f,
    const unsigned int num_qubits, const unsigned int time,
    QsimCircuit* circuit, std::vector<GateMetaData>* metadata) {
  unsigned int q0;
  bool unused = absl::SimpleAtoi(op.qubits(0).id(), &q0);
  circuit->gates.push_back(create_f(time, num_qubits - q0 - 1));

  // check for symbols and track metadata if needed.
  if (metadata != nullptr) {
    GateMetaData info;
    info.index = circuit->gates.size() - 1;
    metadata->push_back(info);
  }
  return Status::OK();
}

// two qubit gate Create(time, q0, q1)
inline Status TwoConstantGate(
    const Operation& op, const SymbolMap& param_map,
    const std::function<QsimGate(unsigned int, unsigned int, unsigned int)>&
        create_f,
    const unsigned int num_qubits, const unsigned int time,
    QsimCircuit* circuit, std::vector<GateMetaData>* metadata) {
  unsigned int q0, q1;
  bool unused = absl::SimpleAtoi(op.qubits(0).id(), &q0);
  unused = absl::SimpleAtoi(op.qubits(1).id(), &q1);
  circuit->gates.push_back(
      create_f(time, num_qubits - q0 - 1, num_qubits - q1 - 1));

  // check for symbols and track metadata if needed.
  if (metadata != nullptr) {
    GateMetaData info;
    info.index = circuit->gates.size() - 1;
    metadata->push_back(info);
  }
  return Status::OK();
}

// single qubit eigen -> Create(time, q0, exponent, global_shift)
inline Status SingleEigenGate(
    const Operation& op, const SymbolMap& param_map,
    const std::function<QsimGate(unsigned int, unsigned int, float, float)>&
        create_f,
    const unsigned int num_qubits, const unsigned int time,
    QsimCircuit* circuit, std::vector<GateMetaData>* metadata) {
  unsigned int q0;
  bool unused;
  float exp, exp_s, gs;
  Status u;
  unused = absl::SimpleAtoi(op.qubits(0).id(), &q0);

  absl::optional<std::string> exponent_symbol;
  u = ParseProtoArg(op, "exponent", param_map, &exp, &exponent_symbol);
  if (!u.ok()) {
    return u;
  }
  u = ParseProtoArg(op, "exponent_scalar", param_map, &exp_s);
  if (!u.ok()) {
    return u;
  }
  u = ParseProtoArg(op, "global_shift", param_map, &gs);
  if (!u.ok()) {
    return u;
  }

  circuit->gates.push_back(
      create_f(time, num_qubits - q0 - 1, exp * exp_s, gs));

  // check for symbols and track metadata if needed.
  if (metadata != nullptr) {
    GateMetaData info;
    info.index = circuit->gates.size() - 1;
    info.gate_params = {exp, exp_s, gs};
    info.create_f1 = create_f;
    if (exponent_symbol.has_value()) {
      info.symbol_values = {exponent_symbol.value()};
      info.placeholder_names = {GateParamNames::kExponent};
    }
    metadata->push_back(info);
  }
  return Status::OK();
}

// two qubit eigen -> Create(time, q0, q1, exp, gs)
inline Status TwoEigenGate(
    const Operation& op, const SymbolMap& param_map,
    const std::function<QsimGate(unsigned int, unsigned int, unsigned int,
                                 float, float)>& create_f,
    const unsigned int num_qubits, const unsigned int time,
    QsimCircuit* circuit, std::vector<GateMetaData>* metadata) {
  unsigned int q0, q1;
  float exp, exp_s, gs;
  bool unused;
  Status u;
  unused = absl::SimpleAtoi(op.qubits(0).id(), &q0);
  unused = absl::SimpleAtoi(op.qubits(1).id(), &q1);

  absl::optional<std::string> exponent_symbol;
  u = ParseProtoArg(op, "exponent", param_map, &exp, &exponent_symbol);
  if (!u.ok()) {
    return u;
  }
  u = ParseProtoArg(op, "exponent_scalar", param_map, &exp_s);
  if (!u.ok()) {
    return u;
  }
  u = ParseProtoArg(op, "global_shift", param_map, &gs);
  if (!u.ok()) {
    return u;
  }
  circuit->gates.push_back(create_f(time, num_qubits - q0 - 1,
                                    num_qubits - q1 - 1, exp * exp_s, gs));

  // check for symbols and track metadata if needed.
  if (metadata != nullptr) {
    GateMetaData info;
    info.index = circuit->gates.size() - 1;
    info.gate_params = {exp, exp_s, gs};
    info.create_f2 = create_f;
    if (exponent_symbol.has_value()) {
      info.symbol_values = {exponent_symbol.value()};
      info.placeholder_names = {GateParamNames::kExponent};
    }
    metadata->push_back(info);
  }
  return Status::OK();
}

Status IGate(const Operation& op, const SymbolMap& param_map,
             const unsigned int num_qubits, const unsigned int time,
             QsimCircuit* circuit, std::vector<GateMetaData>* metadata) {
  return SingleConstantGate(op, param_map, &qsim::Cirq::I<float>::Create,
                            num_qubits, time, circuit, metadata);
}

Status I2Gate(const Operation& op, const SymbolMap& param_map,
              const unsigned int num_qubits, const unsigned int time,
              QsimCircuit* circuit, std::vector<GateMetaData>* metadata) {
  return TwoConstantGate(op, param_map, &qsim::Cirq::I2<float>::Create,
                         num_qubits, time, circuit, metadata);
}

Status HGate(const Operation& op, const SymbolMap& param_map,
             const unsigned int num_qubits, const unsigned int time,
             QsimCircuit* circuit, std::vector<GateMetaData>* metadata) {
  return SingleEigenGate(op, param_map, &qsim::Cirq::HPowGate<float>::Create,
                         num_qubits, time, circuit, metadata);
}

Status XGate(const Operation& op, const SymbolMap& param_map,
             const unsigned int num_qubits, const unsigned int time,
             QsimCircuit* circuit, std::vector<GateMetaData>* metadata) {
  return SingleEigenGate(op, param_map, &qsim::Cirq::XPowGate<float>::Create,
                         num_qubits, time, circuit, metadata);
}

Status XXGate(const Operation& op, const SymbolMap& param_map,
              const unsigned int num_qubits, const unsigned int time,
              QsimCircuit* circuit, std::vector<GateMetaData>* metadata) {
  return TwoEigenGate(op, param_map, &qsim::Cirq::XXPowGate<float>::Create,
                      num_qubits, time, circuit, metadata);
}

Status YGate(const Operation& op, const SymbolMap& param_map,
             const unsigned int num_qubits, const unsigned int time,
             QsimCircuit* circuit, std::vector<GateMetaData>* metadata) {
  return SingleEigenGate(op, param_map, &qsim::Cirq::YPowGate<float>::Create,
                         num_qubits, time, circuit, metadata);
}

Status YYGate(const Operation& op, const SymbolMap& param_map,
              const unsigned int num_qubits, const unsigned int time,
              QsimCircuit* circuit, std::vector<GateMetaData>* metadata) {
  return TwoEigenGate(op, param_map, &qsim::Cirq::YYPowGate<float>::Create,
                      num_qubits, time, circuit, metadata);
}

Status ZGate(const Operation& op, const SymbolMap& param_map,
             const unsigned int num_qubits, const unsigned int time,
             QsimCircuit* circuit, std::vector<GateMetaData>* metadata) {
  return SingleEigenGate(op, param_map, &qsim::Cirq::ZPowGate<float>::Create,
                         num_qubits, time, circuit, metadata);
}

Status ZZGate(const Operation& op, const SymbolMap& param_map,
              const unsigned int num_qubits, const unsigned int time,
              QsimCircuit* circuit, std::vector<GateMetaData>* metadata) {
  return TwoEigenGate(op, param_map, &qsim::Cirq::ZZPowGate<float>::Create,
                      num_qubits, time, circuit, metadata);
}

Status CZGate(const Operation& op, const SymbolMap& param_map,
              const unsigned int num_qubits, const unsigned int time,
              QsimCircuit* circuit, std::vector<GateMetaData>* metadata) {
  return TwoEigenGate(op, param_map, &qsim::Cirq::CZPowGate<float>::Create,
                      num_qubits, time, circuit, metadata);
}

Status CXGate(const Operation& op, const SymbolMap& param_map,
              const unsigned int num_qubits, const unsigned int time,
              QsimCircuit* circuit, std::vector<GateMetaData>* metadata) {
  return TwoEigenGate(op, param_map, &qsim::Cirq::CXPowGate<float>::Create,
                      num_qubits, time, circuit, metadata);
}

Status SwapGate(const Operation& op, const SymbolMap& param_map,
                const unsigned int num_qubits, const unsigned int time,
                QsimCircuit* circuit, std::vector<GateMetaData>* metadata) {
  return TwoEigenGate(op, param_map, &qsim::Cirq::SwapPowGate<float>::Create,
                      num_qubits, time, circuit, metadata);
}

Status ISwapGate(const Operation& op, const SymbolMap& param_map,
                 const unsigned int num_qubits, const unsigned int time,
                 QsimCircuit* circuit, std::vector<GateMetaData>* metadata) {
  return TwoEigenGate(op, param_map, &qsim::Cirq::ISwapPowGate<float>::Create,
                      num_qubits, time, circuit, metadata);
}

// single qubit PhasedXPow -> Create(time, q0, pexp, exp, gs)
inline Status PhasedXGate(const Operation& op, const SymbolMap& param_map,
                          const unsigned int num_qubits,
                          const unsigned int time, QsimCircuit* circuit,
                          std::vector<GateMetaData>* metadata) {
  int q0;
  bool unused;
  float pexp, pexp_s, exp, exp_s, gs;
  Status u;
  unused = absl::SimpleAtoi(op.qubits(0).id(), &q0);

  absl::optional<std::string> exponent_symbol;
  u = ParseProtoArg(op, "exponent", param_map, &exp, &exponent_symbol);
  if (!u.ok()) {
    return u;
  }
  u = ParseProtoArg(op, "exponent_scalar", param_map, &exp_s);
  if (!u.ok()) {
    return u;
  }
  absl::optional<std::string> phase_exponent_symbol;
  u = ParseProtoArg(op, "phase_exponent", param_map, &pexp,
                    &phase_exponent_symbol);
  if (!u.ok()) {
    return u;
  }
  u = ParseProtoArg(op, "phase_exponent_scalar", param_map, &pexp_s);
  if (!u.ok()) {
    return u;
  }
  u = ParseProtoArg(op, "global_shift", param_map, &gs);
  if (!u.ok()) {
    return u;
  }
  circuit->gates.push_back(qsim::Cirq::PhasedXPowGate<float>::Create(
      time, num_qubits - q0 - 1, pexp * pexp_s, exp * exp_s, gs));

  // check for symbols and track metadata if needed.
  if (metadata != nullptr) {
    GateMetaData info;
    info.index = circuit->gates.size() - 1;
    info.gate_params = {pexp, pexp_s, exp, exp_s, gs};
    if (phase_exponent_symbol.has_value()) {
      info.symbol_values.push_back(phase_exponent_symbol.value());
      info.placeholder_names.push_back(GateParamNames::kPhaseExponent);
    }
    if (exponent_symbol.has_value()) {
      info.symbol_values.push_back(exponent_symbol.value());
      info.placeholder_names.push_back(GateParamNames::kExponent);
    }
    metadata->push_back(info);
  }
  return Status::OK();
}

// two qubit fsim -> Create(time, q0, q1, theta, phi)
inline Status FsimGate(const Operation& op, const SymbolMap& param_map,
                       const unsigned int num_qubits, const unsigned int time,
                       QsimCircuit* circuit,
                       std::vector<GateMetaData>* metadata) {
  int q0, q1;
  bool unused;
  float theta, theta_s, phi, phi_s;
  Status u;
  unused = absl::SimpleAtoi(op.qubits(0).id(), &q0);
  unused = absl::SimpleAtoi(op.qubits(1).id(), &q1);

  absl::optional<std::string> theta_symbol;
  u = ParseProtoArg(op, "theta", param_map, &theta, &theta_symbol);
  if (!u.ok()) {
    return u;
  }
  u = ParseProtoArg(op, "theta_scalar", param_map, &theta_s);
  if (!u.ok()) {
    return u;
  }
  absl::optional<std::string> phi_symbol;
  u = ParseProtoArg(op, "phi", param_map, &phi, &phi_symbol);
  if (!u.ok()) {
    return u;
  }
  u = ParseProtoArg(op, "phi_scalar", param_map, &phi_s);
  if (!u.ok()) {
    return u;
  }
  circuit->gates.push_back(qsim::Cirq::FSimGate<float>::Create(
      time, num_qubits - q0 - 1, num_qubits - q1 - 1, theta * theta_s,
      phi * phi_s));

  // check for symbols and track metadata if needed.
  if (metadata != nullptr) {
    GateMetaData info;
    info.index = circuit->gates.size() - 1;
    info.gate_params = {theta, theta_s, phi, phi_s};
    if (theta_symbol.has_value()) {
      info.symbol_values.push_back(theta_symbol.value());
      info.placeholder_names.push_back(GateParamNames::kTheta);
    }
    if (phi_symbol.has_value()) {
      info.symbol_values.push_back(phi_symbol.value());
      info.placeholder_names.push_back(GateParamNames::kPhi);
    }
    metadata->push_back(info);
  }
  return Status::OK();
}

// two qubit phase iswap -> Create(time, q0, q1, pexp, exp)
inline Status PhasedISwapGate(const Operation& op, const SymbolMap& param_map,
                              const unsigned int num_qubits,
                              const unsigned int time, QsimCircuit* circuit,
                              std::vector<GateMetaData>* metadata) {
  int q0, q1;
  bool unused;
  float pexp, pexp_s, exp, exp_s;
  Status u;
  unused = absl::SimpleAtoi(op.qubits(0).id(), &q0);
  unused = absl::SimpleAtoi(op.qubits(1).id(), &q1);

  absl::optional<std::string> exponent_symbol;
  u = ParseProtoArg(op, "exponent", param_map, &exp, &exponent_symbol);
  if (!u.ok()) {
    return u;
  }
  u = ParseProtoArg(op, "exponent_scalar", param_map, &exp_s);
  if (!u.ok()) {
    return u;
  }
  absl::optional<std::string> phase_exponent_symbol;
  u = ParseProtoArg(op, "phase_exponent", param_map, &pexp,
                    &phase_exponent_symbol);
  if (!u.ok()) {
    return u;
  }
  u = ParseProtoArg(op, "phase_exponent_scalar", param_map, &pexp_s);
  if (!u.ok()) {
    return u;
  }
  circuit->gates.push_back(qsim::Cirq::PhasedISwapPowGate<float>::Create(
      time, num_qubits - q0 - 1, num_qubits - q1 - 1, pexp * pexp_s,
      exp * exp_s));

  // check for symbols and track metadata if needed.
  if (metadata != nullptr) {
    GateMetaData info;
    info.index = circuit->gates.size() - 1;
    info.gate_params = {pexp, pexp_s, exp, exp_s};
    if (phase_exponent_symbol.has_value()) {
      info.symbol_values.push_back(phase_exponent_symbol.value());
      info.placeholder_names.push_back(GateParamNames::kPhaseExponent);
    }
    if (exponent_symbol.has_value()) {
      info.symbol_values.push_back(exponent_symbol.value());
      info.placeholder_names.push_back(GateParamNames::kExponent);
    }
    metadata->push_back(info);
  }
  return Status::OK();
}

tensorflow::Status ParseAppendGate(const Operation& op,
                                   const SymbolMap& param_map,
                                   const unsigned int num_qubits,
                                   const unsigned int time,
                                   QsimCircuit* circuit,
                                   std::vector<GateMetaData>* metadata) {
  // map gate name -> callable to build that qsim gate from operation proto.
  static const absl::flat_hash_map<
      std::string,
      std::function<Status(const Operation&, const SymbolMap&,
                           const unsigned int, const unsigned int, QsimCircuit*,
                           std::vector<GateMetaData>*)>>
      func_map = {{"I", &IGate},       {"HP", &HGate},
                  {"XP", &XGate},      {"XXP", &XXGate},
                  {"YP", &YGate},      {"YYP", &YYGate},
                  {"ZP", &ZGate},      {"ZZP", &ZZGate},
                  {"CZP", &CZGate},    {"I2", &I2Gate},
                  {"CNP", &CXGate},    {"SP", &SwapGate},
                  {"ISP", &ISwapGate}, {"PXP", &PhasedXGate},
                  {"FSIM", &FsimGate}, {"PISP", &PhasedISwapGate}};

  auto build_f = func_map.find(op.gate().id());
  if (build_f == func_map.end()) {
    return Status(tensorflow::error::INVALID_ARGUMENT,
                  "Could not parse gate id: " + op.gate().id());
  }
  return build_f->second(op, param_map, num_qubits, time, circuit, metadata);
}

}  // namespace

tensorflow::Status QsimCircuitFromProgram(
    const Program& program, const SymbolMap& param_map, const int num_qubits,
    QsimCircuit* circuit, std::vector<qsim::GateFused<QsimGate>>* fused_circuit,
    std::vector<GateMetaData>* metadata /*=nullptr*/) {
  // Convert proto to qsim internal representation.
  circuit->num_qubits = num_qubits;
  int time = 0;
  // Special case empty.
  if (num_qubits <= 0) {
    return Status::OK();
  }

  circuit->gates.reserve(program.circuit().moments_size() * num_qubits);
  if (metadata != nullptr) {
    metadata->reserve(program.circuit().moments_size() * num_qubits);
  }
  for (const Moment& moment : program.circuit().moments()) {
    for (const Operation& op : moment.operations()) {
      Status status =
          ParseAppendGate(op, param_map, num_qubits, time, circuit, metadata);
      if (!status.ok()) {
        return status;
      }
    }
    time++;
  }

  // Build fused circuit.
  *fused_circuit = qsim::BasicGateFuser<qsim::IO, QsimGate>().FuseGates(
      circuit->num_qubits, circuit->gates);
  return Status::OK();
}

Status QsimCircuitFromPauliTerm(
    const PauliTerm& term, const int num_qubits, QsimCircuit* circuit,
    std::vector<qsim::GateFused<QsimGate>>* fused_circuit) {
  Program measurement_program;
  SymbolMap empty_map;
  measurement_program.mutable_circuit()->set_scheduling_strategy(
      cirq::google::api::v2::Circuit::MOMENT_BY_MOMENT);
  Moment* term_moment = measurement_program.mutable_circuit()->add_moments();
  for (const tfq::proto::PauliQubitPair& pair : term.paulis()) {
    Operation* new_op = term_moment->add_operations();

    // create corresponding eigen gate op.
    new_op->add_qubits()->set_id(pair.qubit_id());
    new_op->mutable_gate()->set_id(pair.pauli_type() + "P");
    (*new_op->mutable_args())["exponent"].mutable_arg_value()->set_float_value(
        1.0);
    (*new_op->mutable_args())["global_shift"]
        .mutable_arg_value()
        ->set_float_value(0.0);
    (*new_op->mutable_args())["exponent_scalar"]
        .mutable_arg_value()
        ->set_float_value(1.0);
  }

  return QsimCircuitFromProgram(measurement_program, empty_map, num_qubits,
                                circuit, fused_circuit);
}

Status QsimZBasisCircuitFromPauliTerm(
    const PauliTerm& term, const int num_qubits, QsimCircuit* circuit,
    std::vector<qsim::GateFused<QsimGate>>* fused_circuit) {
  Program measurement_program;
  SymbolMap empty_map;
  measurement_program.mutable_circuit()->set_scheduling_strategy(
      cirq::google::api::v2::Circuit::MOMENT_BY_MOMENT);
  Moment* term_moment = measurement_program.mutable_circuit()->add_moments();
  float transform_exponent = 0.0;
  std::string gate_type;
  for (const tfq::proto::PauliQubitPair& pair : term.paulis()) {
    if (pair.pauli_type() == "Z") {
      // Z requires no transform.
      continue;
    }

    // Assume it is X and the transform is Y^-0.5
    transform_exponent = -0.5;
    gate_type = "Y";
    if (pair.pauli_type() == "Y") {
      // Y requires X**0.5 transform.
      transform_exponent = 0.5;
      gate_type = "X";
    }

    Operation* new_op = term_moment->add_operations();

    // create corresponding eigen gate op.
    new_op->add_qubits()->set_id(pair.qubit_id());
    new_op->mutable_gate()->set_id(gate_type + "P");
    (*new_op->mutable_args())["exponent"].mutable_arg_value()->set_float_value(
        transform_exponent);
    (*new_op->mutable_args())["global_shift"]
        .mutable_arg_value()
        ->set_float_value(0.0);
    (*new_op->mutable_args())["exponent_scalar"]
        .mutable_arg_value()
        ->set_float_value(1.0);
  }

  return QsimCircuitFromProgram(measurement_program, empty_map, num_qubits,
                                circuit, fused_circuit);
}

}  // namespace tfq
