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

#ifndef TFQ_CORE_SRC_ADJ_HESSIAN_UTIL_H_
#define TFQ_CORE_SRC_ADJ_HESSIAN_UTIL_H_

#include <functional>
#include <string>
#include <vector>

#include "../qsim/lib/circuit.h"
#include "../qsim/lib/fuser.h"
#include "../qsim/lib/fuser_basic.h"
#include "../qsim/lib/gates_cirq.h"
#include "../qsim/lib/io.h"
#include "../qsim/lib/matrix.h"
#include "tensorflow_quantum/core/src/adj_util.h"
#include "tensorflow_quantum/core/src/circuit_parser_qsim.h"

namespace tfq {

static const float _HESS_EPS = 1e-2;
static const float _INVERSE_HESS_EPS_SQUARE = 1e4;
static const std::string kUsePrevTwoSymbols = "use_prev_two_symbols";

// Computes all gates who's hessian will need to be taken, in addition
// fuses all gates around those gates for faster circuit execution.
void CreateHessianCircuit(
    const qsim::Circuit<qsim::Cirq::GateCirq<float>>& circuit,
    const std::vector<GateMetaData>& metadata,
    std::vector<std::vector<qsim::GateFused<qsim::Cirq::GateCirq<float>>>>*
        partial_fuses,
    std::vector<GradientOfGate>* grad_gates);

void PopulateHessianSingleEigen(
    const std::function<qsim::Cirq::GateCirq<float>(unsigned int, unsigned int,
                                                    float, float)>& create_f,
    const std::string& symbol, unsigned int location, unsigned int qid,
    float exp, float exp_s, float gs, GradientOfGate* grad);

void PopulateHessianTwoEigen(
    const std::function<qsim::Cirq::GateCirq<float>(
        unsigned int, unsigned int, unsigned int, float, float)>& create_f,
    const std::string& symbol, unsigned int location, unsigned int qid,
    unsigned int qid2, float exp, float exp_s, float gs, GradientOfGate* grad);

// Note: all methods below expect gate qubit indices to have been swapped so
// qid < qid2.
void PopulateHessianPhasedXPhasedExponent(const std::string& symbol,
                                          unsigned int location,
                                          unsigned int qid, float pexp,
                                          float pexp_s, float exp, float exp_s,
                                          float gs, GradientOfGate* grad);

void PopulateHessianPhasedXExponent(const std::string& symbol,
                                    unsigned int location, unsigned int qid,
                                    float pexp, float pexp_s, float exp,
                                    float exp_s, float gs,
                                    GradientOfGate* grad);

void PopulateCrossTermPhasedXPhasedExponentExponent(
    unsigned int location, unsigned int qid, float pexp, float pexp_s,
    float exp, float exp_s, float gs, GradientOfGate* grad);

void PopulateHessianFsimTheta(const std::string& symbol, unsigned int location,
                              unsigned int qid, unsigned qid2, float theta,
                              float theta_s, float phi, float phi_s,
                              GradientOfGate* grad);

void PopulateHessianFsimPhi(const std::string& symbol, unsigned int location,
                            unsigned int qid, unsigned qid2, float theta,
                            float theta_s, float phi, float phi_s,
                            GradientOfGate* grad);

void PopulateCrossTermFsimThetaPhi(unsigned int location, unsigned int qid,
                                   unsigned qid2, float theta, float theta_s,
                                   float phi, float phi_s,
                                   GradientOfGate* grad);

void PopulateHessianPhasedISwapPhasedExponent(
    const std::string& symbol, unsigned int location, unsigned int qid,
    unsigned int qid2, float pexp, float pexp_s, float exp, float exp_s,
    GradientOfGate* grad);

void PopulateHessianPhasedISwapExponent(const std::string& symbol,
                                        unsigned int location, unsigned int qid,
                                        unsigned int qid2, float pexp,
                                        float pexp_s, float exp, float exp_s,
                                        GradientOfGate* grad);

void PopulateCrossTermPhasedISwapPhasedExponentExponent(
    unsigned int location, unsigned int qid, unsigned int qid2, float pexp,
    float pexp_s, float exp, float exp_s, GradientOfGate* grad);

template <typename Array2>
void Matrix2Add(Array2& source, Array2& dest) {
  for (unsigned i = 0; i < 8; i++) {
    dest[i] += source[i];
  }
}

// does matrix elementwise addition dest += source.
template <typename Array2>
void Matrix4Add(Array2& source, Array2& dest) {
  for (unsigned i = 0; i < 32; i++) {
    dest[i] += source[i];
  }
}

// Due to the large error from the finite differencing PhasedXPowGate,
// Here we introduce analytically differentiated version of PhasedXPowGate up to
// 2nd order with double precision.

template <typename fp_type>
struct D2PhasedExponentPhasedXPowGate {
  static constexpr qsim::Cirq::GateKind kind = qsim::Cirq::kPhasedXPowGate;
  static constexpr char name[] = "D2PhasedExponentPhasedXPowGate";
  static constexpr unsigned num_qubits = 1;
  static constexpr bool symmetric = true;

  static qsim::Cirq::GateCirq<fp_type> Create(unsigned time, unsigned q0,
                                  fp_type phase_exponent,
                                  fp_type phase_exponent_scalar,
                                  fp_type exponent = 1,
                                  fp_type global_shift = 0) {
    double pi = qsim::Cirq::pi_double;
    double pexp = static_cast<double>(phase_exponent);
    double pexp_s = static_cast<double>(phase_exponent_scalar);
    double exp = static_cast<double>(exponent);
    double ec = std::cos(pi * exp);
    double es = std::sin(pi * exp);
    double gc = std::cos(pi * exp * global_shift);
    double gs = std::sin(pi * exp * global_shift);

    double d2p_pc = -pi*pi*pexp_s*pexp_s*std::cos(pi * pexp_s * pexp);
    double d2p_ps = -pi*pi*pexp_s*pexp_s*std::sin(pi * pexp_s * pexp);

    double br = -0.5 * ((-1 + ec) * gc - es * gs);
    double bi = -0.5 * ((-1 + ec) * gs + es * gc);

    return qsim::CreateGate<qsim::Cirq::GateCirq<fp_type>, D2PhasedExponentPhasedXPowGate>(
        time, {q0}, {0., 0., static_cast<fp_type>(d2p_pc * br + d2p_ps * bi),
                     static_cast<fp_type>(d2p_pc * bi - d2p_ps * br),
                     static_cast<fp_type>(d2p_pc * br - d2p_ps * bi),
                     static_cast<fp_type>(d2p_pc * bi + d2p_ps * br), 0., 0.},
        {phase_exponent, exponent, global_shift});
  }
};

template <typename fp_type>
struct D2ExponentPhasedXPowGate {
  static constexpr qsim::Cirq::GateKind kind = qsim::Cirq::kPhasedXPowGate;
  static constexpr char name[] = "D2ExponentPhasedXPowGate";
  static constexpr unsigned num_qubits = 1;
  static constexpr bool symmetric = true;

  static constexpr fp_type pi = static_cast<fp_type>(qsim::Cirq::pi_double);

  static qsim::Cirq::GateCirq<fp_type> Create(unsigned time, unsigned q0,
                                  fp_type phase_exponent,
                                  fp_type exponent = 1,
                                  fp_type exponent_scalar = 1,
                                  fp_type global_shift = 0) {
    double pi = qsim::Cirq::pi_double;
    double pexp = static_cast<double>(phase_exponent);
    double exp = static_cast<double>(exponent);
    double exp_s = static_cast<double>(exponent_scalar);
    double pc = std::cos(pi * pexp);
    double ps = std::sin(pi * pexp);
    double ec = std::cos(pi * exp_s *  exp);
    double es = std::sin(pi * exp_s * exp);
    double gc = std::cos(pi * exp_s * exp * global_shift);
    double gs = std::sin(pi * exp_s * exp * global_shift);
    double dec = -pi * exp_s * std::sin(pi * exp_s *  exp);
    double des = pi * exp_s * std::cos(pi * exp_s * exp);
    double dgc = -pi * exp_s * global_shift * std::sin(pi * exp_s * exp * global_shift);
    double dgs = pi * exp_s * global_shift * std::cos(pi * exp_s * exp * global_shift);
    double d2ec = - pi * exp_s * pi * exp_s * std::cos(pi * exp_s * exp);
    double d2es = - pi * exp_s * pi * exp_s * std::sin(pi * exp_s * exp);
    double d2gc = - pi * exp_s * pi * exp_s * global_shift * global_shift * std::cos(pi * exp_s * exp * global_shift);
    double d2gs = - pi * exp_s * pi * exp_s * global_shift * global_shift * std::sin(pi * exp_s * exp * global_shift);
    double common_r_front = ec * d2gc + 2.0 * dec * dgc + d2ec * gc;
    double common_r_back = d2es * gs + 2.0 * des * dgs + es * d2gs;
    double common_i_front = ec * d2gs + 2.0 * dec * dgs + d2ec * gs;
    double common_i_back = d2es * gc + 2.0 * des * dgc + es * d2gc;

    fp_type d2ar = static_cast<fp_type>(0.5 * (d2gc + common_r_front - common_r_back));
    fp_type d2ai = static_cast<fp_type>(0.5 * (d2gs + common_i_front + common_i_back));
    double d2br = -0.5 * (-d2gc + common_r_front - common_r_back);
    double d2bi = -0.5 * (-d2gs + common_i_front + common_i_back);

    return qsim::CreateGate<qsim::Cirq::GateCirq<fp_type>, D2ExponentPhasedXPowGate>(
        time, {q0}, {d2ar, d2ai, static_cast<fp_type>(pc * d2br + ps * d2bi),
                     static_cast<fp_type>(pc * d2bi - ps * d2br),
                     static_cast<fp_type>(pc * d2br - ps * d2bi),
                     static_cast<fp_type>(pc * d2bi + ps * d2br), d2ar, d2ai},
        {phase_exponent, exponent, global_shift});
  }
};

template <typename fp_type>
struct DPhasedExponentDExponentPhasedXPowGate {
  static constexpr qsim::Cirq::GateKind kind = qsim::Cirq::kPhasedXPowGate;
  static constexpr char name[] = "DPhasedExponentDExponentPhasedXPowGate";
  static constexpr unsigned num_qubits = 1;
  static constexpr bool symmetric = true;

  static constexpr fp_type pi = static_cast<fp_type>(qsim::Cirq::pi_double);

  static qsim::Cirq::GateCirq<fp_type> Create(unsigned time, unsigned q0,
                                  fp_type phase_exponent,
                                  fp_type phase_exponent_scalar,
                                  fp_type exponent = 1,
                                  fp_type exponent_scalar = 1,
                                  fp_type global_shift = 0) {
    double pi = qsim::Cirq::pi_double;
    double pexp = static_cast<double>(phase_exponent);
    double pexp_s = static_cast<double>(phase_exponent_scalar);
    double exp = static_cast<double>(exponent);
    double exp_s = static_cast<double>(exponent_scalar);
    double ec = std::cos(pi * exp_s * exp);
    double es = std::sin(pi * exp_s * exp);
    double gc = std::cos(pi * exp_s * exp * global_shift);
    double gs = std::sin(pi * exp_s * exp * global_shift);
   
    double dp_pc = -pi * pexp_s * std::sin(pi * pexp_s * pexp);
    double dp_ps = pi * pexp_s * std::cos(pi * pexp_s * pexp);
    double de_ec = -pi * exp_s * std::sin(pi * exp_s * exp);
    double de_es = pi * exp_s * std::cos(pi * exp_s * exp);
    double de_gc = -pi * exp_s * global_shift * std::sin(pi * exp_s * global_shift * exp);
    double de_gs = pi * exp_s * global_shift * std::cos(pi * exp_s * global_shift * exp);

    double de_br = -0.5 * ((-1 + ec) * de_gc + de_ec * gc - de_es * gs - es * de_gs);
    double de_bi = -0.5 * ((-1 + ec) * de_gs + de_ec * gs + de_es * gc + es * de_gc);

    return qsim::CreateGate<qsim::Cirq::GateCirq<fp_type>, DPhasedExponentDExponentPhasedXPowGate>(
        time, {q0}, {0.,0., static_cast<fp_type>(dp_pc * de_br + dp_ps * de_bi),
                     static_cast<fp_type>(dp_pc * de_bi - dp_ps * de_br),
                     static_cast<fp_type>(dp_pc * de_br - dp_ps * de_bi),
                     static_cast<fp_type>(dp_pc * de_bi + dp_ps * de_br), 0.,0.},
        {phase_exponent, exponent, global_shift});
  }
};

}  // namespace tfq

#endif  // TFQ_CORE_SRC_ADJ_UTIL_H_
