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

#ifndef TFQ_CORE_SRC_ADJ_UTIL_H_
#define TFQ_CORE_SRC_ADJ_UTIL_H_

#include <functional>
#include <string>
#include <vector>

#include "../qsim/lib/circuit.h"
#include "../qsim/lib/fuser.h"
#include "../qsim/lib/fuser_basic.h"
#include "../qsim/lib/gates_cirq.h"
#include "../qsim/lib/io.h"
#include "../qsim/lib/matrix.h"
#include "tensorflow_quantum/core/src/circuit_parser_qsim.h"

namespace tfq {

static const float _GRAD_EPS = 5e-3;

struct GradientOfGate {
  // name of parameters used by gate.
  std::vector<std::string> params;

  // index of gate in original circuit.
  // Not if multiple calls to Populate* are used
  // on the same object ensure that the index is
  // the same for all of them!
  int index;

  // Gates for gradients. Has a 1:1 mapping with params.
  std::vector<qsim::Cirq::GateCirq<float>> grad_gates;
};

// Computes all gates who's gradient will need to be taken, in addition
// fuses all gates around those gates for faster circuit execution.
void CreateGradientCircuit(
    const qsim::Circuit<qsim::Cirq::GateCirq<float>>& circuit,
    const std::vector<GateMetaData>& metadata,
    std::vector<std::vector<qsim::GateFused<qsim::Cirq::GateCirq<float>>>>*
        partial_fuses,
    std::vector<GradientOfGate>* grad_gates);

void PopulateGradientSingleEigen(
    const std::function<qsim::Cirq::GateCirq<float>(unsigned int, unsigned int,
                                                    float, float)>& create_f,
    const std::string& symbol, unsigned int location, unsigned int qid,
    float exp, float exp_s, float gs, GradientOfGate* grad);

void PopulateGradientTwoEigen(
    const std::function<qsim::Cirq::GateCirq<float>(
        unsigned int, unsigned int, unsigned int, float, float)>& create_f,
    const std::string& symbol, unsigned int location, unsigned int qid,
    unsigned int qid2, float exp, float exp_s, float gs, GradientOfGate* grad);

// Note: all methods below expect gate qubit indices to have been swapped so
// qid < qid2.
void PopulateGradientPhasedXPhasedExponent(const std::string& symbol,
                                           unsigned int location,
                                           unsigned int qid, float pexp,
                                           float pexp_s, float exp, float exp_s,
                                           float gs, GradientOfGate* grad);

void PopulateGradientPhasedXExponent(const std::string& symbol,
                                     unsigned int location, unsigned int qid,
                                     float pexp, float pexp_s, float exp,
                                     float exp_s, float gs,
                                     GradientOfGate* grad);

void PopulateGradientFsimTheta(const std::string& symbol, unsigned int location,
                               unsigned int qid, unsigned qid2, float theta,
                               float theta_s, float phi, float phi_s,
                               GradientOfGate* grad);

void PopulateGradientFsimPhi(const std::string& symbol, unsigned int location,
                             unsigned int qid, unsigned qid2, float theta,
                             float theta_s, float phi, float phi_s,
                             GradientOfGate* grad);

void PopulateGradientPhasedISwapPhasedExponent(
    const std::string& symbol, unsigned int location, unsigned int qid,
    unsigned int qid2, float pexp, float pexp_s, float exp, float exp_s,
    GradientOfGate* grad);

void PopulateGradientPhasedISwapExponent(const std::string& symbol,
                                         unsigned int location,
                                         unsigned int qid, unsigned int qid2,
                                         float pexp, float pexp_s, float exp,
                                         float exp_s, GradientOfGate* grad);

// does matrix elementwise subtraction dest -= source.
template <typename Array2>
void Matrix2Diff(Array2& source, Array2& dest) {
  for (unsigned i = 0; i < 8; i++) {
    dest[i] -= source[i];
  }
}

// does matrix elementwise subtraction dest -= source.
template <typename Array2>
void Matrix4Diff(Array2& source, Array2& dest) {
  for (unsigned i = 0; i < 32; i++) {
    dest[i] -= source[i];
  }
}

}  // namespace tfq

#endif  // TFQ_CORE_SRC_ADJ_UTIL_H_
