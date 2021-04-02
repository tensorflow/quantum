/* Copyright 2021 The TensorFlow Quantum Authors. All Rights Reserved.

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
#include "tensorflow_quantum/core/src/adj_hessian_util.h"

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

typedef qsim::Cirq::GateCirq<float> QsimGate;
typedef qsim::Circuit<QsimGate> QsimCircuit;

void CreateHessianCircuit(
    const QsimCircuit& circuit, const std::vector<GateMetaData>& metadata,
    std::vector<std::vector<qsim::GateFused<QsimGate>>>* partial_fuses,
    std::vector<GradientOfGate>* grad_gates) {
  for (std::vector<tfq::GateMetaData>::size_type i = 0; i < metadata.size(); i++) {
    if (metadata[i].symbol_values.empty()) {
      continue;
    }
    // found a gate that was constructed with symbols.
    GradientOfGate grad;

    // Single qubit Eigen.
    if (circuit.gates[i].kind == qsim::Cirq::GateKind::kXPowGate ||
        circuit.gates[i].kind == qsim::Cirq::GateKind::kYPowGate ||
        circuit.gates[i].kind == qsim::Cirq::GateKind::kZPowGate ||
        circuit.gates[i].kind == qsim::Cirq::GateKind::kHPowGate) {
      PopulateHessianSingleEigen(
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
      PopulateHessianTwoEigen(
          metadata[i].create_f2, metadata[i].symbol_values[0], i,
          swapq ? circuit.gates[i].qubits[1] : circuit.gates[i].qubits[0],
          swapq ? circuit.gates[i].qubits[0] : circuit.gates[i].qubits[1],
          metadata[i].gate_params[0], metadata[i].gate_params[1],
          metadata[i].gate_params[2], &grad);
      grad_gates->push_back(grad);
    }

    // PhasedX
    else if (circuit.gates[i].kind == qsim::Cirq::GateKind::kPhasedXPowGate) {
      // Process potentially several symbols.
      bool symbolic_pexp = false;
      bool symbolic_exp = false;
      for (std::vector<std::basic_string<char> >::size_type j = 0; j < metadata[i].symbol_values.size(); j++) {
        if (metadata[i].placeholder_names[j] ==
            GateParamNames::kPhaseExponent) {
          symbolic_pexp = true;
          PopulateHessianPhasedXPhasedExponent(
              metadata[i].symbol_values[j], i, circuit.gates[i].qubits[0],
              metadata[i].gate_params[0], metadata[i].gate_params[1],
              metadata[i].gate_params[2], metadata[i].gate_params[3],
              metadata[i].gate_params[4], &grad);
        } else if (metadata[i].placeholder_names[j] ==
                   GateParamNames::kExponent) {
          symbolic_exp = true;
          PopulateHessianPhasedXExponent(
              metadata[i].symbol_values[j], i, circuit.gates[i].qubits[0],
              metadata[i].gate_params[0], metadata[i].gate_params[1],
              metadata[i].gate_params[2], metadata[i].gate_params[3],
              metadata[i].gate_params[4], &grad);
        }
      }
      if (symbolic_pexp && symbolic_exp) {
        PopulateCrossTermPhasedXPhasedExponentExponent(
            i, circuit.gates[i].qubits[0], metadata[i].gate_params[0],
            metadata[i].gate_params[1], metadata[i].gate_params[2],
            metadata[i].gate_params[3], metadata[i].gate_params[4], &grad);
      }
      grad_gates->push_back(grad);
    }

    // Fsim
    else if (circuit.gates[i].kind == qsim::Cirq::GateKind::kFSimGate) {
      // Process potentially several symbols.

      bool swapq = circuit.gates[i].swapped;
      bool symbolic_theta = false;
      bool symbolic_phi = false;
      for (std::vector<std::basic_string<char> >::size_type j = 0; j < metadata[i].symbol_values.size(); j++) {
        if (metadata[i].placeholder_names[j] == GateParamNames::kTheta) {
          symbolic_theta = true;
          PopulateHessianFsimTheta(
              metadata[i].symbol_values[j], i,
              swapq ? circuit.gates[i].qubits[1] : circuit.gates[i].qubits[0],
              swapq ? circuit.gates[i].qubits[0] : circuit.gates[i].qubits[1],
              metadata[i].gate_params[0], metadata[i].gate_params[1],
              metadata[i].gate_params[2], metadata[i].gate_params[3], &grad);
        } else if (metadata[i].placeholder_names[j] == GateParamNames::kPhi) {
          symbolic_phi = true;
          PopulateHessianFsimPhi(
              metadata[i].symbol_values[j], i,
              swapq ? circuit.gates[i].qubits[1] : circuit.gates[i].qubits[0],
              swapq ? circuit.gates[i].qubits[0] : circuit.gates[i].qubits[1],
              metadata[i].gate_params[0], metadata[i].gate_params[1],
              metadata[i].gate_params[2], metadata[i].gate_params[3], &grad);
        }
      }
      if (symbolic_theta && symbolic_phi) {
        PopulateCrossTermFsimThetaPhi(
            i, swapq ? circuit.gates[i].qubits[1] : circuit.gates[i].qubits[0],
            swapq ? circuit.gates[i].qubits[0] : circuit.gates[i].qubits[1],
            metadata[i].gate_params[0], metadata[i].gate_params[1],
            metadata[i].gate_params[2], metadata[i].gate_params[3], &grad);
      }

      grad_gates->push_back(grad);
    }

    // PhasedISwap
    else if (circuit.gates[i].kind ==
             qsim::Cirq::GateKind::kPhasedISwapPowGate) {
      // Process potentially several symbols.
      bool swapq = circuit.gates[i].swapped;
      bool symbolic_pexp = false;
      bool symbolic_exp = false;
      for (std::vector<std::basic_string<char> >::size_type j = 0; j < metadata[i].symbol_values.size(); j++) {
        if (metadata[i].placeholder_names[j] ==
            GateParamNames::kPhaseExponent) {
          symbolic_pexp = true;
          PopulateHessianPhasedISwapPhasedExponent(
              metadata[i].symbol_values[j], i,
              swapq ? circuit.gates[i].qubits[1] : circuit.gates[i].qubits[0],
              swapq ? circuit.gates[i].qubits[0] : circuit.gates[i].qubits[1],
              metadata[i].gate_params[0], metadata[i].gate_params[1],
              metadata[i].gate_params[2], metadata[i].gate_params[3], &grad);

        } else if (metadata[i].placeholder_names[j] ==
                   GateParamNames::kExponent) {
          symbolic_exp = true;
          PopulateHessianPhasedISwapExponent(
              metadata[i].symbol_values[j], i,
              swapq ? circuit.gates[i].qubits[1] : circuit.gates[i].qubits[0],
              swapq ? circuit.gates[i].qubits[0] : circuit.gates[i].qubits[1],
              metadata[i].gate_params[0], metadata[i].gate_params[1],
              metadata[i].gate_params[2], metadata[i].gate_params[3], &grad);
        }
      }
      if (symbolic_pexp && symbolic_exp) {
        PopulateCrossTermPhasedISwapPhasedExponentExponent(
            i, swapq ? circuit.gates[i].qubits[1] : circuit.gates[i].qubits[0],
            swapq ? circuit.gates[i].qubits[0] : circuit.gates[i].qubits[1],
            metadata[i].gate_params[0], metadata[i].gate_params[1],
            metadata[i].gate_params[2], metadata[i].gate_params[3], &grad);
      }
      grad_gates->push_back(grad);
    }
  }

  // Produce partial fuses around the hessian gates.
  auto fuser = qsim::BasicGateFuser<qsim::IO, QsimGate>();
  auto left = circuit.gates.begin();
  auto right = left;

  partial_fuses->assign(grad_gates->size() + 1,
                        std::vector<qsim::GateFused<QsimGate>>({}));
  for (std::vector<GradientOfGate>::size_type i = 0; i < grad_gates->size(); i++) {
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

void PopulateHessianSingleEigen(
    const std::function<QsimGate(unsigned int, unsigned int, float, float)>&
        create_f,
    const std::string& symbol, unsigned int location, unsigned int qid,
    float exp, float exp_s, float gs, GradientOfGate* grad) {
  grad->params.push_back(symbol);
  grad->index = location;
  auto left = create_f(0, qid, (exp + _HESS_EPS) * exp_s, gs);
  auto center = create_f(0, qid, exp * exp_s, gs);
  auto right = create_f(0, qid, (exp - _HESS_EPS) * exp_s, gs);
  // Due to precision issue, (1) multiplies weights first rather than last.
  // and (2) doesn't use _INVERSE_HESS_EPS_SQUARE
  qsim::MatrixScalarMultiply(_INVERSE_HESS_EPS_SQUARE, left.matrix);
  qsim::MatrixScalarMultiply(_INVERSE_HESS_EPS_SQUARE, right.matrix);
  qsim::MatrixScalarMultiply(_INVERSE_HESS_EPS_SQUARE, center.matrix);
  std::cout << "left = [";
  int size = 8;
  for (int i = 0; i < size; i++) {
    char buf[100];
    auto val = left.matrix[i];
    std::sprintf(buf, "%.7e", val);
    auto ending = (
        (i % 2 == 0) ? ((val > 0) ? "+" : "") :
                       ((i == size - 1) ? "j" : "j,"));
    std::cout << buf << ending;
  }
  std::cout << "]" << std::endl;
  std::cout << "right = [";
  for (int i = 0; i < size; i++) {
    char buf[100];
    auto val = right.matrix[i];
    std::sprintf(buf, "%.7e", val);
    auto ending = (
        (i % 2 == 0) ? ((val > 0) ? "+" : "") :
                       ((i == size - 1) ? "j" : "j,"));
    std::cout << buf << ending;
  }
  std::cout << "]" << std::endl;
  std::cout << "center = [";
  for (int i = 0; i < size; i++) {
    char buf[100];
    auto val = center.matrix[i];
    std::sprintf(buf, "%.7e", val);
    auto ending = (
        (i % 2 == 0) ? ((val > 0) ? "+" : "") :
                       ((i == size - 1) ? "j" : "j,"));
    std::cout << buf << ending;
  }
  std::cout << "]" << std::endl;
  Matrix2Add(right.matrix,
             left.matrix);  // left's entries have right added
  std::cout << "left_plus_right = [";
  for (int i = 0; i < size; i++) {
    char buf[100];
    auto val = left.matrix[i];
    std::sprintf(buf, "%.7e", val);
    auto prefix = (val > 0) ? ((i % 2 == 1) ? "+" : "") : "";
    auto ending = (i % 2 == 0) ? "" : ((i == size - 1) ? "j" : "j,");
    std::cout << prefix << buf << ending;
  }
  std::cout << "]" << std::endl;
  qsim::MatrixScalarMultiply(2.0, center.matrix);
  std::cout << "twice_center = [";
  for (int i = 0; i < size; i++) {
    char buf[100];
    auto val = center.matrix[i];
    std::sprintf(buf, "%.7e", val);
    auto ending = (
        (i % 2 == 0) ? ((val > 0) ? "+" : "") :
                       ((i == size - 1) ? "j" : "j,"));
    std::cout << buf << ending;
  }
  std::cout << "]" << std::endl;
  Matrix2Diff(center.matrix,
              left.matrix);  // left's entries have center subtracted.
  std::cout << "left_plus_right_minus_twice_center = [";
  for (int i = 0; i < size; i++) {
    char buf[100];
    auto val = left.matrix[i];
    std::sprintf(buf, "%.7e", val);
    auto ending = (
        (i % 2 == 0) ? ((val > 0) ? "+" : "") :
                       ((i == size - 1) ? "j" : "j,"));
    std::cout << buf << ending;
  }
  std::cout << "]" << std::endl;
  grad->grad_gates.push_back(left);
}

void PopulateHessianTwoEigen(
    const std::function<QsimGate(unsigned int, unsigned int, unsigned int,
                                 float, float)>& create_f,
    const std::string& symbol, unsigned int location, unsigned int qid,
    unsigned int qid2, float exp, float exp_s, float gs, GradientOfGate* grad) {
  grad->params.push_back(symbol);
  grad->index = location;
  auto left = create_f(0, qid, qid2, (exp + _HESS_EPS) * exp_s, gs);
  auto center = create_f(0, qid, qid2, exp * exp_s, gs);
  auto right = create_f(0, qid, qid2, (exp - _HESS_EPS) * exp_s, gs);
  // Due to precision issue, multiply weights first.
  qsim::MatrixScalarMultiply(_INVERSE_HESS_EPS_SQUARE, left.matrix);
  qsim::MatrixScalarMultiply(_INVERSE_HESS_EPS_SQUARE, right.matrix);
  qsim::MatrixScalarMultiply(_INVERSE_HESS_EPS_SQUARE, center.matrix);
  std::cout << "PopulateHessianTwoEigen" << std::endl;
  int size = 32;
  std::cout << "left = [";
  for (int i = 0; i < size; i++) {
    char buf[100];
    auto val = left.matrix[i];
    std::sprintf(buf, "%.7e", val);
    auto ending = (
        (i % 2 == 0) ? ((val > 0) ? "+" : "") :
                       ((i == size - 1) ? "j" : "j,"));
    std::cout << buf << ending;
  }
  std::cout << "]" << std::endl;
  std::cout << "center = [";
  for (int i = 0; i < size; i++) {
    char buf[100];
    auto val = center.matrix[i];
    std::sprintf(buf, "%.7e", val);
    auto ending = (
        (i % 2 == 0) ? ((val > 0) ? "+" : "") :
                       ((i == size - 1) ? "j" : "j,"));
    std::cout << buf << ending;
  }
  std::cout << "]" << std::endl;
  std::cout << "right = [";
  for (int i = 0; i < size; i++) {
    char buf[100];
    auto val = right.matrix[i];
    std::sprintf(buf, "%.7e", val);
    auto ending = (
        (i % 2 == 0) ? ((val > 0) ? "+" : "") :
                       ((i == size - 1) ? "j" : "j,"));
    std::cout << buf << ending;
  }
  std::cout << "]" << std::endl;
  Matrix4Add(right.matrix,
             left.matrix);  // left's entries have right added.
  std::cout << "left_plus_right = [";
  for (int i = 0; i < size; i++) {
    char buf[100];
    auto val = left.matrix[i];
    std::sprintf(buf, "%.7e", val);
    auto ending = (
        (i % 2 == 0) ? ((val > 0) ? "+" : "") :
                       ((i == size - 1) ? "j" : "j,"));
    std::cout << buf << ending;
  }
  std::cout << "]" << std::endl;
  qsim::MatrixScalarMultiply(2.0, center.matrix);
  std::cout << "twice_center = [";
  for (int i = 0; i < size; i++) {
    char buf[100];
    auto val = center.matrix[i];
    std::sprintf(buf, "%.7e", val);
    auto ending = (
        (i % 2 == 0) ? ((val > 0) ? "+" : "") :
                       ((i == size - 1) ? "j" : "j,"));
    std::cout << buf << ending;
  }
  std::cout << "]" << std::endl;
  Matrix4Diff(center.matrix,
              left.matrix);  // left's entries have center subtracted.
  std::cout << "left_plus_right_minus_twice_center = [";
  for (int i = 0; i < size; i++) {
    char buf[100];
    auto val = left.matrix[i];
    std::sprintf(buf, "%.7e", val);
    auto ending = (
        (i % 2 == 0) ? ((val > 0) ? "+" : "") :
                       ((i == size - 1) ? "j" : "j,"));
    std::cout << buf << ending;
  }
  std::cout << "]" << std::endl;
  grad->grad_gates.push_back(left);
}

void PopulateHessianPhasedXPhasedExponent(const std::string& symbol,
                                          unsigned int location,
                                          unsigned int qid, float pexp,
                                          float pexp_s, float exp, float exp_s,
                                          float gs, GradientOfGate* grad) {
  grad->params.push_back(symbol);
  grad->index = location;
  auto left = qsim::Cirq::PhasedXPowGate<float>::Create(
      0, qid, (pexp + _HESS_EPS) * pexp_s, exp * exp_s, gs);
  auto center = qsim::Cirq::PhasedXPowGate<float>::Create(0, qid, pexp * pexp_s,
                                                          exp * exp_s, gs);
  auto right = qsim::Cirq::PhasedXPowGate<float>::Create(
      0, qid, (pexp - _HESS_EPS) * pexp_s, exp * exp_s, gs);
  // Due to precision issue, multiply weights first.
  qsim::MatrixScalarMultiply(_INVERSE_HESS_EPS_SQUARE, left.matrix);
  qsim::MatrixScalarMultiply(_INVERSE_HESS_EPS_SQUARE, right.matrix);
  qsim::MatrixScalarMultiply(_INVERSE_HESS_EPS_SQUARE, center.matrix);
  Matrix2Add(right.matrix,
             left.matrix);  // left's entries have right added.
  qsim::MatrixScalarMultiply(2.0, center.matrix);
  Matrix2Diff(center.matrix,
              left.matrix);  // left's entries have center subtracted.
  grad->grad_gates.push_back(left);
}

void PopulateHessianPhasedXExponent(const std::string& symbol,
                                    unsigned int location, unsigned int qid,
                                    float pexp, float pexp_s, float exp,
                                    float exp_s, float gs,
                                    GradientOfGate* grad) {
  grad->params.push_back(symbol);
  grad->index = location;
  auto left = qsim::Cirq::PhasedXPowGate<float>::Create(
      0, qid, pexp * pexp_s, (exp + _HESS_EPS) * exp_s, gs);
  auto center = qsim::Cirq::PhasedXPowGate<float>::Create(0, qid, pexp * pexp_s,
                                                          exp * exp_s, gs);
  auto right = qsim::Cirq::PhasedXPowGate<float>::Create(
      0, qid, pexp * pexp_s, (exp - _HESS_EPS) * exp_s, gs);
  // Due to precision issue, multiply weights first.
  qsim::MatrixScalarMultiply(_INVERSE_HESS_EPS_SQUARE, left.matrix);
  qsim::MatrixScalarMultiply(_INVERSE_HESS_EPS_SQUARE, right.matrix);
  qsim::MatrixScalarMultiply(_INVERSE_HESS_EPS_SQUARE, center.matrix);
  Matrix2Add(right.matrix,
             left.matrix);  // left's entries have right added.
  qsim::MatrixScalarMultiply(2.0, center.matrix);
  Matrix2Diff(center.matrix,
              left.matrix);  // left's entries have center subtracted.
  grad->grad_gates.push_back(left);
}

void PopulateCrossTermPhasedXPhasedExponentExponent(
    unsigned int location, unsigned int qid, float pexp, float pexp_s,
    float exp, float exp_s, float gs, GradientOfGate* grad) {
  grad->params.push_back(kUsePrevTwoSymbols);
  grad->index = location;
  auto left = qsim::Cirq::PhasedXPowGate<float>::Create(
      0, qid, (pexp + _GRAD_EPS) * pexp_s, (exp + _GRAD_EPS) * exp_s, gs);
  auto left_center = qsim::Cirq::PhasedXPowGate<float>::Create(
      0, qid, (pexp + _GRAD_EPS) * pexp_s, (exp - _GRAD_EPS) * exp_s, gs);
  auto right_center = qsim::Cirq::PhasedXPowGate<float>::Create(
      0, qid, (pexp - _GRAD_EPS) * pexp_s, (exp + _GRAD_EPS) * exp_s, gs);
  auto right = qsim::Cirq::PhasedXPowGate<float>::Create(
      0, qid, (pexp - _GRAD_EPS) * pexp_s, (exp - _GRAD_EPS) * exp_s, gs);
  // Due to precision issue, multiply weights first.
  qsim::MatrixScalarMultiply(_INVERSE_HESS_EPS_SQUARE, left.matrix);
  qsim::MatrixScalarMultiply(_INVERSE_HESS_EPS_SQUARE, right.matrix);
  qsim::MatrixScalarMultiply(_INVERSE_HESS_EPS_SQUARE, left_center.matrix);
  qsim::MatrixScalarMultiply(_INVERSE_HESS_EPS_SQUARE, right_center.matrix);
  Matrix2Add(right.matrix,
             left.matrix);  // left's entries have right added.
  Matrix2Add(right_center.matrix, left_center.matrix);
  Matrix2Diff(left_center.matrix,
              left.matrix);  // left's entries have left_center subtracted.
  grad->grad_gates.push_back(left);
}

void PopulateHessianFsimTheta(const std::string& symbol, unsigned int location,
                              unsigned int qid, unsigned qid2, float theta,
                              float theta_s, float phi, float phi_s,
                              GradientOfGate* grad) {
  grad->params.push_back(symbol);
  grad->index = location;
  auto left = qsim::Cirq::FSimGate<float>::Create(
      0, qid, qid2, (theta + _HESS_EPS) * theta_s, phi * phi_s);
  auto center = qsim::Cirq::FSimGate<float>::Create(
      0, qid, qid2, theta * theta_s, phi * phi_s);
  auto right = qsim::Cirq::FSimGate<float>::Create(
      0, qid, qid2, (theta - _HESS_EPS) * theta_s, phi * phi_s);
  // Due to precision issue, multiply weights first.
  qsim::MatrixScalarMultiply(_INVERSE_HESS_EPS_SQUARE, left.matrix);
  qsim::MatrixScalarMultiply(_INVERSE_HESS_EPS_SQUARE, right.matrix);
  qsim::MatrixScalarMultiply(_INVERSE_HESS_EPS_SQUARE, center.matrix);
  Matrix4Add(right.matrix,
             left.matrix);  // left's entries have right added.
  qsim::MatrixScalarMultiply(2.0, center.matrix);
  Matrix4Diff(center.matrix,
              left.matrix);  // left's entries have center subtracted.
  grad->grad_gates.push_back(left);
}

void PopulateHessianFsimPhi(const std::string& symbol, unsigned int location,
                            unsigned int qid, unsigned qid2, float theta,
                            float theta_s, float phi, float phi_s,
                            GradientOfGate* grad) {
  grad->params.push_back(symbol);
  grad->index = location;
  auto left = qsim::Cirq::FSimGate<float>::Create(0, qid, qid2, theta * theta_s,
                                                  (phi + _HESS_EPS) * phi_s);
  auto center = qsim::Cirq::FSimGate<float>::Create(
      0, qid, qid2, theta * theta_s, phi * phi_s);
  auto right = qsim::Cirq::FSimGate<float>::Create(
      0, qid, qid2, theta * theta_s, (phi - _HESS_EPS) * phi_s);
  // Due to precision issue, multiply weights first.
  qsim::MatrixScalarMultiply(_INVERSE_HESS_EPS_SQUARE, left.matrix);
  qsim::MatrixScalarMultiply(_INVERSE_HESS_EPS_SQUARE, right.matrix);
  qsim::MatrixScalarMultiply(_INVERSE_HESS_EPS_SQUARE, center.matrix);
  Matrix4Add(right.matrix,
             left.matrix);  // left's entries have right added.
  qsim::MatrixScalarMultiply(2.0, center.matrix);
  Matrix4Diff(center.matrix,
              left.matrix);  // left's entries have center subtracted.
  grad->grad_gates.push_back(left);
}

void PopulateCrossTermFsimThetaPhi(unsigned int location, unsigned int qid,
                                   unsigned qid2, float theta, float theta_s,
                                   float phi, float phi_s,
                                   GradientOfGate* grad) {
  grad->params.push_back(kUsePrevTwoSymbols);
  grad->index = location;
  auto left = qsim::Cirq::FSimGate<float>::Create(
      0, qid, qid2, (theta + _GRAD_EPS) * theta_s, (phi + _GRAD_EPS) * phi_s);
  auto left_center = qsim::Cirq::FSimGate<float>::Create(
      0, qid, qid2, (theta + _GRAD_EPS) * theta_s, (phi - _GRAD_EPS) * phi_s);
  auto right_center = qsim::Cirq::FSimGate<float>::Create(
      0, qid, qid2, (theta - _GRAD_EPS) * theta_s, (phi + _GRAD_EPS) * phi_s);
  auto right = qsim::Cirq::FSimGate<float>::Create(
      0, qid, qid2, (theta - _GRAD_EPS) * theta_s, (phi - _GRAD_EPS) * phi_s);
  // Due to precision issue, multiply weights first.
  qsim::MatrixScalarMultiply(_INVERSE_HESS_EPS_SQUARE, left.matrix);
  qsim::MatrixScalarMultiply(_INVERSE_HESS_EPS_SQUARE, right.matrix);
  qsim::MatrixScalarMultiply(_INVERSE_HESS_EPS_SQUARE, left_center.matrix);
  qsim::MatrixScalarMultiply(_INVERSE_HESS_EPS_SQUARE, right_center.matrix);
  Matrix4Add(right.matrix,
             left.matrix);  // left's entries have right added.
  Matrix4Add(right_center.matrix, left_center.matrix);
  Matrix4Diff(left_center.matrix,
              left.matrix);  // left's entries have left_center subtracted.
  grad->grad_gates.push_back(left);
}

void PopulateHessianPhasedISwapPhasedExponent(
    const std::string& symbol, unsigned int location, unsigned int qid,
    unsigned int qid2, float pexp, float pexp_s, float exp, float exp_s,
    GradientOfGate* grad) {
  grad->params.push_back(symbol);
  grad->index = location;
  auto left = qsim::Cirq::PhasedISwapPowGate<float>::Create(
      0, qid, qid2, (pexp + _HESS_EPS) * pexp_s, exp * exp_s);
  auto center = qsim::Cirq::PhasedISwapPowGate<float>::Create(
      0, qid, qid2, pexp * pexp_s, exp * exp_s);
  auto right = qsim::Cirq::PhasedISwapPowGate<float>::Create(
      0, qid, qid2, (pexp - _HESS_EPS) * pexp_s, exp * exp_s);
  // Due to precision issue, multiply weights first.
  qsim::MatrixScalarMultiply(_INVERSE_HESS_EPS_SQUARE, left.matrix);
  qsim::MatrixScalarMultiply(_INVERSE_HESS_EPS_SQUARE, right.matrix);
  qsim::MatrixScalarMultiply(_INVERSE_HESS_EPS_SQUARE, center.matrix);
  Matrix4Add(right.matrix,
             left.matrix);  // left's entries have right added.
  qsim::MatrixScalarMultiply(2.0, center.matrix);
  Matrix4Diff(center.matrix,
              left.matrix);  // left's entries have center subtracted.
  grad->grad_gates.push_back(left);
}

void PopulateHessianPhasedISwapExponent(const std::string& symbol,
                                        unsigned int location, unsigned int qid,
                                        unsigned int qid2, float pexp,
                                        float pexp_s, float exp, float exp_s,
                                        GradientOfGate* grad) {
  grad->params.push_back(symbol);
  grad->index = location;
  auto left = qsim::Cirq::PhasedISwapPowGate<float>::Create(
      0, qid, qid2, pexp * pexp_s, (exp + _HESS_EPS) * exp_s);
  auto center = qsim::Cirq::PhasedISwapPowGate<float>::Create(
      0, qid, qid2, pexp * pexp_s, exp * exp_s);
  auto right = qsim::Cirq::PhasedISwapPowGate<float>::Create(
      0, qid, qid2, pexp * pexp_s, (exp - _HESS_EPS) * exp_s);
  // Due to precision issue, multiply weights first.
  qsim::MatrixScalarMultiply(_INVERSE_HESS_EPS_SQUARE, left.matrix);
  qsim::MatrixScalarMultiply(_INVERSE_HESS_EPS_SQUARE, right.matrix);
  qsim::MatrixScalarMultiply(_INVERSE_HESS_EPS_SQUARE, center.matrix);
  Matrix4Add(right.matrix,
             left.matrix);  // left's entries have right added.
  qsim::MatrixScalarMultiply(2.0, center.matrix);
  Matrix4Diff(center.matrix,
              left.matrix);  // left's entries have center subtracted.
  grad->grad_gates.push_back(left);
}

void PopulateCrossTermPhasedISwapPhasedExponentExponent(
    unsigned int location, unsigned int qid, unsigned int qid2, float pexp,
    float pexp_s, float exp, float exp_s, GradientOfGate* grad) {
  grad->params.push_back(kUsePrevTwoSymbols);
  grad->index = location;
  auto left = qsim::Cirq::PhasedISwapPowGate<float>::Create(
      0, qid, qid2, (pexp + _GRAD_EPS) * pexp_s, (exp + _GRAD_EPS) * exp_s);
  auto left_center = qsim::Cirq::PhasedISwapPowGate<float>::Create(
      0, qid, qid2, (pexp + _GRAD_EPS) * pexp_s, (exp - _GRAD_EPS) * exp_s);
  auto right_center = qsim::Cirq::PhasedISwapPowGate<float>::Create(
      0, qid, qid2, (pexp - _GRAD_EPS) * pexp_s, (exp + _GRAD_EPS) * exp_s);
  auto right = qsim::Cirq::PhasedISwapPowGate<float>::Create(
      0, qid, qid2, (pexp - _GRAD_EPS) * pexp_s, (exp - _GRAD_EPS) * exp_s);
  // Due to precision issue, multiply weights first.
  qsim::MatrixScalarMultiply(_INVERSE_HESS_EPS_SQUARE, left.matrix);
  qsim::MatrixScalarMultiply(_INVERSE_HESS_EPS_SQUARE, right.matrix);
  qsim::MatrixScalarMultiply(_INVERSE_HESS_EPS_SQUARE, left_center.matrix);
  qsim::MatrixScalarMultiply(_INVERSE_HESS_EPS_SQUARE, right_center.matrix);
  Matrix4Add(right.matrix,
             left.matrix);  // left's entries have right added.
  Matrix4Add(right_center.matrix, left_center.matrix);
  Matrix4Diff(left_center.matrix,
              left.matrix);  // left's entries have center subtracted.
  grad->grad_gates.push_back(left);
}

}  // namespace tfq
