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
#include "tensorflow_quantum/core/src/adj_util.h"

#include <functional>
#include <string>
#include <vector>

#include "../qsim/lib/circuit.h"
#include "../qsim/lib/fuser.h"
#include "../qsim/lib/fuser_basic.h"
#include "../qsim/lib/gate.h"
#include "../qsim/lib/gates_cirq.h"
#include "../qsim/lib/io.h"
#include "../qsim/lib/matrix.h"
#include "tensorflow_quantum/core/src/circuit_parser_qsim.h"

namespace tfq {

static const float _GRAD_EPS = 5e-3;

typedef qsim::Cirq::GateCirq<float> QsimGate;
typedef qsim::Circuit<QsimGate> QsimCircuit;

void CreateGradientCircuit(
    const QsimCircuit& circuit, const std::vector<GateMetaData>& metadata,
    std::vector<std::vector<qsim::GateFused<QsimGate>>>* partial_fuses,
    std::vector<GradientOfGate>* grad_gates) {
  for (int i = 0; i < metadata.size(); i++) {
    if (metadata[i].symbol_values.size() == 0) {
      continue;
    }
    // found a gate that was constructed with symbols.
    GradientOfGate grad;

    // Single qubit Eigen.
    if (circuit.gates[i].kind == qsim::Cirq::GateKind::kXPowGate ||
        circuit.gates[i].kind == qsim::Cirq::GateKind::kYPowGate ||
        circuit.gates[i].kind == qsim::Cirq::GateKind::kZPowGate ||
        circuit.gates[i].kind == qsim::Cirq::GateKind::kHPowGate) {
      PopulateGradientSingleEigen(
          metadata[i].create_f1, metadata[i].symbol_values[0], i,
          circuit.gates[i].qubits[0], metadata[i].gate_params[0],
          metadata[i].gate_params[1], metadata[i].gate_params[2], &grad);
      grad_gates->push_back(grad);
    }

    // Two qubit Eigen.
    else if (circuit.gates[i].kind == qsim::Cirq::GateKind::kCZPowGate ||
             circuit.gates[i].kind == qsim::Cirq::GateKind::kCXPowGate ||
             circuit.gates[i].kind == qsim::Cirq::GateKind::kXXPowGate ||
             circuit.gates[i].kind == qsim::Cirq::GateKind::kYYPowGate ||
             circuit.gates[i].kind == qsim::Cirq::GateKind::kZZPowGate ||
             circuit.gates[i].kind == qsim::Cirq::GateKind::kISwapPowGate ||
             circuit.gates[i].kind == qsim::Cirq::GateKind::kSwapPowGate) {
      bool swapq = circuit.gates[i].swapped;
      PopulateGradientTwoEigen(
          metadata[i].create_f2, metadata[i].symbol_values[0], i,
          swapq ? circuit.gates[i].qubits[1] : circuit.gates[i].qubits[0],
          swapq ? circuit.gates[i].qubits[0] : circuit.gates[i].qubits[1],
          metadata[i].gate_params[0], metadata[i].gate_params[1],
          metadata[i].gate_params[2], &grad);
      grad_gates->push_back(grad);
    }

    // Three qubit Eigen.
    else if (circuit.gates[i].kind == qsim::Cirq::GateKind::kCCZPowGate ||
             circuit.gates[i].kind == qsim::Cirq::GateKind::kCCXPowGate) {
      bool swapq = circuit.gates[i].swapped;
      PopulateGradientThreeEigen(
          metadata[i].create_f3, metadata[i].symbol_values[0], i,
          swapq ? circuit.gates[i].qubits[2] : circuit.gates[i].qubits[0],
          circuit.gates[i].qubits[1],
          swapq ? circuit.gates[i].qubits[0] : circuit.gates[i].qubits[2],
          metadata[i].gate_params[0], metadata[i].gate_params[1],
          metadata[i].gate_params[2], &grad);
      grad_gates->push_back(grad);
    }

    // PhasedX
    else if (circuit.gates[i].kind == qsim::Cirq::GateKind::kPhasedXPowGate) {
      // Process potentially several symbols.
      for (int j = 0; j < metadata[i].symbol_values.size(); j++) {
        if (metadata[i].placeholder_names[j] ==
            GateParamNames::kPhaseExponent) {
          PopulateGradientPhasedXPhasedExponent(
              metadata[i].symbol_values[j], i, circuit.gates[i].qubits[0],
              metadata[i].gate_params[0], metadata[i].gate_params[1],
              metadata[i].gate_params[2], metadata[i].gate_params[3],
              metadata[i].gate_params[4], &grad);
        } else if (metadata[i].placeholder_names[j] ==
                   GateParamNames::kExponent) {
          PopulateGradientPhasedXExponent(
              metadata[i].symbol_values[j], i, circuit.gates[i].qubits[0],
              metadata[i].gate_params[0], metadata[i].gate_params[1],
              metadata[i].gate_params[2], metadata[i].gate_params[3],
              metadata[i].gate_params[4], &grad);
        }
      }
      grad_gates->push_back(grad);
    }

    // Fsim
    else if (circuit.gates[i].kind == qsim::Cirq::GateKind::kFSimGate) {
      // Process potentially several symbols.

      bool swapq = circuit.gates[i].swapped;
      for (int j = 0; j < metadata[i].symbol_values.size(); j++) {
        if (metadata[i].placeholder_names[j] == GateParamNames::kTheta) {
          PopulateGradientFsimTheta(
              metadata[i].symbol_values[j], i,
              swapq ? circuit.gates[i].qubits[1] : circuit.gates[i].qubits[0],
              swapq ? circuit.gates[i].qubits[0] : circuit.gates[i].qubits[1],
              metadata[i].gate_params[0], metadata[i].gate_params[1],
              metadata[i].gate_params[2], metadata[i].gate_params[3], &grad);
        } else if (metadata[i].placeholder_names[j] == GateParamNames::kPhi) {
          PopulateGradientFsimPhi(
              metadata[i].symbol_values[j], i,
              swapq ? circuit.gates[i].qubits[1] : circuit.gates[i].qubits[0],
              swapq ? circuit.gates[i].qubits[0] : circuit.gates[i].qubits[1],
              metadata[i].gate_params[0], metadata[i].gate_params[1],
              metadata[i].gate_params[2], metadata[i].gate_params[3], &grad);
        }
      }
      grad_gates->push_back(grad);
    }

    // PhasedISwap
    else if (circuit.gates[i].kind ==
             qsim::Cirq::GateKind::kPhasedISwapPowGate) {
      // Process potentially several symbols.
      bool swapq = circuit.gates[i].swapped;
      for (int j = 0; j < metadata[i].symbol_values.size(); j++) {
        if (metadata[i].placeholder_names[j] ==
            GateParamNames::kPhaseExponent) {
          PopulateGradientPhasedISwapPhasedExponent(
              metadata[i].symbol_values[j], i,
              swapq ? circuit.gates[i].qubits[1] : circuit.gates[i].qubits[0],
              swapq ? circuit.gates[i].qubits[0] : circuit.gates[i].qubits[1],
              metadata[i].gate_params[0], metadata[i].gate_params[1],
              metadata[i].gate_params[2], metadata[i].gate_params[3], &grad);

        } else if (metadata[i].placeholder_names[j] ==
                   GateParamNames::kExponent) {
          PopulateGradientPhasedISwapExponent(
              metadata[i].symbol_values[j], i,
              swapq ? circuit.gates[i].qubits[1] : circuit.gates[i].qubits[0],
              swapq ? circuit.gates[i].qubits[0] : circuit.gates[i].qubits[1],
              metadata[i].gate_params[0], metadata[i].gate_params[1],
              metadata[i].gate_params[2], metadata[i].gate_params[3], &grad);
        }
      }
      grad_gates->push_back(grad);
    }
  }

  // Produce partial fuses around the gradient gates.
  auto fuser = qsim::BasicGateFuser<qsim::IO, QsimGate>();
  auto left = circuit.gates.begin();
  auto right = left;

  partial_fuses->assign(grad_gates->size() + 1,
                        std::vector<qsim::GateFused<QsimGate>>({}));
  for (int i = 0; i < grad_gates->size(); i++) {
    right = circuit.gates.begin() + (*grad_gates)[i].index;
    (*partial_fuses)[i] =
        fuser.FuseGates(qsim::BasicGateFuser<qsim::IO, QsimGate>::Parameter(),
                        circuit.num_qubits, left, right);
    left = right + 1;
  }
  right = circuit.gates.end();
  (*partial_fuses)[grad_gates->size()] =
      fuser.FuseGates(qsim::BasicGateFuser<qsim::IO, QsimGate>::Parameter(),
                      circuit.num_qubits, left, right);
}

void PopulateGradientSingleEigen(
    const std::function<QsimGate(unsigned int, unsigned int, float, float)>&
        create_f,
    const std::string& symbol, unsigned int location, unsigned int qid,
    float exp, float exp_s, float gs, GradientOfGate* grad) {
  grad->params.push_back(symbol);
  grad->index = location;
  auto left = create_f(0, qid, (exp + _GRAD_EPS) * exp_s, gs);
  auto right = create_f(0, qid, (exp - _GRAD_EPS) * exp_s, gs);
  Matrix2Diff(right.matrix,
              left.matrix);  // left's entries have right subtracted.
  qsim::MatrixScalarMultiply(0.5 / _GRAD_EPS, left.matrix);
  grad->grad_gates.push_back(left);
}

void PopulateGradientTwoEigen(
    const std::function<QsimGate(unsigned int, unsigned int, unsigned int,
                                 float, float)>& create_f,
    const std::string& symbol, unsigned int location, unsigned int qid,
    unsigned int qid2, float exp, float exp_s, float gs, GradientOfGate* grad) {
  grad->params.push_back(symbol);
  grad->index = location;
  auto left = create_f(0, qid, qid2, (exp + _GRAD_EPS) * exp_s, gs);
  auto right = create_f(0, qid, qid2, (exp - _GRAD_EPS) * exp_s, gs);
  Matrix4Diff(right.matrix,
              left.matrix);  // left's entries have right subtracted.
  qsim::MatrixScalarMultiply(0.5 / _GRAD_EPS, left.matrix);
  grad->grad_gates.push_back(left);
}

void PopulateGradientThreeEigen(
    const std::function<qsim::Cirq::GateCirq<float>(unsigned int, unsigned int,
                                                    unsigned int, unsigned int,
                                                    float, float)>& create_f,
    const std::string& symbol, unsigned int location, unsigned int qid,
    unsigned int qid2, unsigned int qid3, float exp, float exp_s, float gs,
    GradientOfGate* grad) {
  grad->params.push_back(symbol);
  grad->index = location;
  auto left = create_f(0, qid, qid2, qid3, (exp + _GRAD_EPS) * exp_s, gs);
  auto right = create_f(0, qid, qid2, qid3, (exp - _GRAD_EPS) * exp_s, gs);
  Matrix8Diff(right.matrix,
              left.matrix);  // left's entries have right subtracted.
  qsim::MatrixScalarMultiply(0.5 / _GRAD_EPS, left.matrix);
  grad->grad_gates.push_back(left);
}

void PopulateGradientPhasedXPhasedExponent(const std::string& symbol,
                                           unsigned int location,
                                           unsigned int qid, float pexp,
                                           float pexp_s, float exp, float exp_s,
                                           float gs, GradientOfGate* grad) {
  grad->params.push_back(symbol);
  grad->index = location;
  auto left = qsim::Cirq::PhasedXPowGate<float>::Create(
      0, qid, (pexp + _GRAD_EPS) * pexp_s, exp * exp_s, gs);
  auto right = qsim::Cirq::PhasedXPowGate<float>::Create(
      0, qid, (pexp - _GRAD_EPS) * pexp_s, exp * exp_s, gs);
  Matrix2Diff(right.matrix,
              left.matrix);  // left's entries have right subtracted.
  qsim::MatrixScalarMultiply(0.5 / _GRAD_EPS, left.matrix);
  grad->grad_gates.push_back(left);
}

void PopulateGradientPhasedXExponent(const std::string& symbol,
                                     unsigned int location, unsigned int qid,
                                     float pexp, float pexp_s, float exp,
                                     float exp_s, float gs,
                                     GradientOfGate* grad) {
  grad->params.push_back(symbol);
  grad->index = location;
  auto left = qsim::Cirq::PhasedXPowGate<float>::Create(
      0, qid, pexp * pexp_s, (exp + _GRAD_EPS) * exp_s, gs);
  auto right = qsim::Cirq::PhasedXPowGate<float>::Create(
      0, qid, pexp * pexp_s, (exp - _GRAD_EPS) * exp_s, gs);
  Matrix2Diff(right.matrix,
              left.matrix);  // left's entries have right subtracted.
  qsim::MatrixScalarMultiply(0.5 / _GRAD_EPS, left.matrix);
  grad->grad_gates.push_back(left);
}

void PopulateGradientFsimTheta(const std::string& symbol, unsigned int location,
                               unsigned int qid, unsigned qid2, float theta,
                               float theta_s, float phi, float phi_s,
                               GradientOfGate* grad) {
  grad->params.push_back(symbol);
  grad->index = location;
  auto left = qsim::Cirq::FSimGate<float>::Create(
      0, qid, qid2, (theta + _GRAD_EPS) * theta_s, phi * phi_s);
  auto right = qsim::Cirq::FSimGate<float>::Create(
      0, qid, qid2, (theta - _GRAD_EPS) * theta_s, phi * phi_s);
  Matrix4Diff(right.matrix,
              left.matrix);  // left's entries have right subtracted.
  qsim::MatrixScalarMultiply(0.5 / _GRAD_EPS, left.matrix);
  grad->grad_gates.push_back(left);
}

void PopulateGradientFsimPhi(const std::string& symbol, unsigned int location,
                             unsigned int qid, unsigned qid2, float theta,
                             float theta_s, float phi, float phi_s,
                             GradientOfGate* grad) {
  grad->params.push_back(symbol);
  grad->index = location;
  auto left = qsim::Cirq::FSimGate<float>::Create(0, qid, qid2, theta * theta_s,
                                                  (phi + _GRAD_EPS) * phi_s);
  auto right = qsim::Cirq::FSimGate<float>::Create(
      0, qid, qid2, theta * theta_s, (phi - _GRAD_EPS) * phi_s);
  Matrix4Diff(right.matrix,
              left.matrix);  // left's entries have right subtracted.
  qsim::MatrixScalarMultiply(0.5 / _GRAD_EPS, left.matrix);
  grad->grad_gates.push_back(left);
}

void PopulateGradientPhasedISwapPhasedExponent(
    const std::string& symbol, unsigned int location, unsigned int qid,
    unsigned int qid2, float pexp, float pexp_s, float exp, float exp_s,
    GradientOfGate* grad) {
  grad->params.push_back(symbol);
  grad->index = location;
  auto left = qsim::Cirq::PhasedISwapPowGate<float>::Create(
      0, qid, qid2, (pexp + _GRAD_EPS) * pexp_s, exp * exp_s);
  auto right = qsim::Cirq::PhasedISwapPowGate<float>::Create(
      0, qid, qid2, (pexp - _GRAD_EPS) * pexp_s, exp * exp_s);
  Matrix4Diff(right.matrix,
              left.matrix);  // left's entries have right subtracted.
  qsim::MatrixScalarMultiply(0.5 / _GRAD_EPS, left.matrix);
  grad->grad_gates.push_back(left);
}

void PopulateGradientPhasedISwapExponent(const std::string& symbol,
                                         unsigned int location,
                                         unsigned int qid, unsigned int qid2,
                                         float pexp, float pexp_s, float exp,
                                         float exp_s, GradientOfGate* grad) {
  grad->params.push_back(symbol);
  grad->index = location;
  auto left = qsim::Cirq::PhasedISwapPowGate<float>::Create(
      0, qid, qid2, pexp * pexp_s, (exp + _GRAD_EPS) * exp_s);
  auto right = qsim::Cirq::PhasedISwapPowGate<float>::Create(
      0, qid, qid2, pexp * pexp_s, (exp - _GRAD_EPS) * exp_s);
  Matrix4Diff(right.matrix,
              left.matrix);  // left's entries have right subtracted.
  qsim::MatrixScalarMultiply(0.5 / _GRAD_EPS, left.matrix);
  grad->grad_gates.push_back(left);
}

}  // namespace tfq
