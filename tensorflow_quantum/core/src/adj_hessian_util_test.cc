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
#include "tensorflow_quantum/core/src/adj_hessian_util.h"

#include <string>
#include <vector>

#include "../qsim/lib/circuit.h"
#include "../qsim/lib/fuser.h"
#include "../qsim/lib/fuser_basic.h"
#include "../qsim/lib/gates_cirq.h"
#include "../qsim/lib/io.h"
#include "../qsim/lib/matrix.h"
#include "gtest/gtest.h"

namespace tfq {
namespace {

void Matrix2Equal(const std::vector<float>& v,
                  const std::vector<float>& expected, float eps) {
  for (int i = 0; i < 8; i++) {
    EXPECT_NEAR(v[i], expected[i], eps);
  }
}

void Matrix4Equal(const std::vector<float>& v,
                  const std::vector<float>& expected, float eps) {
  for (int i = 0; i < 32; i++) {
    EXPECT_NEAR(v[i], expected[i], eps);
  }
}

typedef absl::flat_hash_map<std::string, std::pair<int, float>> SymbolMap;
typedef qsim::Cirq::GateCirq<float> QsimGate;
typedef qsim::Circuit<QsimGate> QsimCircuit;

class OneQubitEigenFixture
    : public ::testing::TestWithParam<
          std::function<QsimGate(unsigned int, unsigned int, float, float)>> {};

TEST_P(OneQubitEigenFixture, CreateHessianSingleEigen) {
  QsimCircuit circuit;
  std::vector<GateMetaData> metadata;
  std::vector<std::vector<qsim::GateFused<QsimGate>>> fuses;
  std::vector<GradientOfGate> grad_gates;

  // Create a symbolized gate.
  std::function<QsimGate(unsigned int, unsigned int, float, float)> given_f =
      GetParam();

  circuit.num_qubits = 2;
  circuit.gates.push_back(given_f(0, 1, 1.0, 2.0));
  GateMetaData meta;
  meta.index = 0;
  meta.symbol_values.push_back("TheSymbol");
  meta.placeholder_names.push_back(GateParamNames::kExponent);
  meta.gate_params = {1.0, 1.0, 2.0};
  meta.create_f1 = given_f;
  metadata.push_back(meta);

  CreateHessianCircuit(circuit, metadata, &fuses, &grad_gates);
  EXPECT_EQ(grad_gates.size(), 1);
  EXPECT_EQ(grad_gates[0].index, 0);
  EXPECT_EQ(grad_gates[0].params.size(), 1);
  EXPECT_EQ(grad_gates[0].params[0], "TheSymbol");
  // fuse everything into 2 gates. One fuse before this gate and one after.
  // both wind up being identity since this is the only gate.
  EXPECT_EQ(fuses.size(), 2);

  GradientOfGate tmp;
  PopulateHessianSingleEigen(given_f, "TheSymbol", 0, 1, 1.0, 1.0, 2.0, &tmp);

  Matrix2Equal(tmp.grad_gates[0].matrix, grad_gates[0].grad_gates[0].matrix,
               1e-4);

  // Test with NO symbol.
  metadata.clear();
  meta.symbol_values.clear();
  meta.placeholder_names.clear();
  fuses.clear();
  grad_gates.clear();

  CreateHessianCircuit(circuit, metadata, &fuses, &grad_gates);
  EXPECT_EQ(grad_gates.size(), 0);
  EXPECT_EQ(fuses.size(), 1);
}

INSTANTIATE_TEST_CASE_P(
    OneQubitEigenTests, OneQubitEigenFixture,
    ::testing::Values(&qsim::Cirq::XPowGate<float>::Create,
                      &qsim::Cirq::YPowGate<float>::Create,
                      &qsim::Cirq::ZPowGate<float>::Create,
                      &qsim::Cirq::HPowGate<float>::Create));

class TwoQubitEigenFixture
    : public ::testing::TestWithParam<std::function<QsimGate(
          unsigned int, unsigned int, unsigned int, float, float)>> {};

TEST_P(TwoQubitEigenFixture, CreateHessianTwoEigen) {
  QsimCircuit circuit;
  std::vector<GateMetaData> metadata;
  std::vector<std::vector<qsim::GateFused<QsimGate>>> fuses;
  std::vector<GradientOfGate> grad_gates;

  // Create a symbolized gate.
  std::function<QsimGate(unsigned int, unsigned int, unsigned int, float,
                         float)>
      given_f = GetParam();

  circuit.num_qubits = 2;
  circuit.gates.push_back(given_f(0, 0, 1, 1.0, 2.0));
  GateMetaData meta;
  meta.index = 0;
  meta.symbol_values.push_back("TheSymbol");
  meta.placeholder_names.push_back(GateParamNames::kExponent);
  meta.gate_params = {1.0, 1.0, 2.0};
  meta.create_f2 = given_f;
  metadata.push_back(meta);

  CreateHessianCircuit(circuit, metadata, &fuses, &grad_gates);
  EXPECT_EQ(grad_gates.size(), 1);
  EXPECT_EQ(grad_gates[0].index, 0);
  EXPECT_EQ(grad_gates[0].params.size(), 1);
  EXPECT_EQ(grad_gates[0].params[0], "TheSymbol");

  // fuse everything into 2 gates. One fuse before this gate and one after.
  // both wind up being identity since this is the only gate.
  EXPECT_EQ(fuses.size(), 2);

  GradientOfGate tmp;
  PopulateHessianTwoEigen(given_f, "TheSymbol", 0, 0, 1, 1.0, 1.0, 2.0, &tmp);

  Matrix4Equal(tmp.grad_gates[0].matrix, grad_gates[0].grad_gates[0].matrix,
               1e-4);

  // Test with NO symbol.
  metadata.clear();
  meta.symbol_values.clear();
  meta.placeholder_names.clear();
  fuses.clear();
  grad_gates.clear();

  CreateHessianCircuit(circuit, metadata, &fuses, &grad_gates);
  EXPECT_EQ(grad_gates.size(), 0);
  EXPECT_EQ(fuses.size(), 1);  // fuse everything into 1 gate.
}

INSTANTIATE_TEST_CASE_P(
    TwoQubitEigenTests, TwoQubitEigenFixture,
    ::testing::Values(&qsim::Cirq::CZPowGate<float>::Create,
                      &qsim::Cirq::CXPowGate<float>::Create,
                      &qsim::Cirq::XXPowGate<float>::Create,
                      &qsim::Cirq::YYPowGate<float>::Create,
                      &qsim::Cirq::ZZPowGate<float>::Create,
                      &qsim::Cirq::ISwapPowGate<float>::Create,
                      &qsim::Cirq::SwapPowGate<float>::Create));

TEST(AdjHessianUtilTest, CreateHessianPhasedX) {
  QsimCircuit circuit;
  std::vector<GateMetaData> metadata;
  std::vector<std::vector<qsim::GateFused<QsimGate>>> fuses;
  std::vector<GradientOfGate> grad_gates;

  // Create a symbolized gate.
  circuit.num_qubits = 2;
  circuit.gates.push_back(
      qsim::Cirq::PhasedXPowGate<float>::Create(0, 0, 1.0, 2.0, 3.0));
  GateMetaData meta;
  meta.index = 0;
  meta.symbol_values.push_back("TheSymbol");
  meta.placeholder_names.push_back(GateParamNames::kPhaseExponent);
  meta.symbol_values.push_back("TheSymbol2");
  meta.placeholder_names.push_back(GateParamNames::kExponent);
  meta.gate_params = {1.0, 1.0, 2.0, 1.0, 3.0};
  metadata.push_back(meta);

  CreateHessianCircuit(circuit, metadata, &fuses, &grad_gates);
  EXPECT_EQ(grad_gates.size(), 1);
  EXPECT_EQ(grad_gates[0].index, 0);
  EXPECT_EQ(grad_gates[0].params.size(), 3);
  EXPECT_EQ(grad_gates[0].params[0], "TheSymbol");
  EXPECT_EQ(grad_gates[0].params[1], "TheSymbol2");
  // Third symbol is automatically generated `kUsePrevTwoSymbols`.
  EXPECT_EQ(grad_gates[0].params[2], kUsePrevTwoSymbols);

  // fuse everything into 2 gates. One fuse before this gate and one after.
  // both wind up being identity since this is the only gate.
  EXPECT_EQ(fuses.size(), 2);

  GradientOfGate tmp;
  PopulateHessianPhasedXPhasedExponent("TheSymbol", 0, 0, 1.0, 1.0, 2.0, 1.0,
                                        3.0, &tmp);

  Matrix2Equal(tmp.grad_gates[0].matrix, grad_gates[0].grad_gates[0].matrix,
               1e-4);

  GradientOfGate tmp2;
  PopulateHessianPhasedXExponent("TheSymbol2", 0, 0, 1.0, 1.0, 2.0, 1.0, 3.0,
                                  &tmp2);

  Matrix2Equal(tmp2.grad_gates[0].matrix, grad_gates[0].grad_gates[1].matrix,
               1e-4);

  GradientOfGate tmp3;
  PopulateHessianPhasedXExponent("TheSymbol31", 0, 0, 1.0, 1.0, 2.0, 1.0, 3.0,
                                  &tmp2);

  Matrix2Equal(tmp2.grad_gates[0].matrix, grad_gates[0].grad_gates[1].matrix,
               1e-4);
  metadata.clear();
  meta.symbol_values.clear();
  meta.placeholder_names.clear();
  grad_gates.clear();
  fuses.clear();

  metadata.push_back(meta);

  CreateHessianCircuit(circuit, metadata, &fuses, &grad_gates);
  EXPECT_EQ(grad_gates.size(), 0);
  EXPECT_EQ(fuses.size(), 1);
}

TEST(AdjHessianUtilTest, CreateHessianPhasedISwap) {
  QsimCircuit circuit;
  std::vector<GateMetaData> metadata;
  std::vector<std::vector<qsim::GateFused<QsimGate>>> fuses;
  std::vector<GradientOfGate> grad_gates;

  // Create a symbolized gate.
  circuit.num_qubits = 2;
  circuit.gates.push_back(
      qsim::Cirq::PhasedISwapPowGate<float>::Create(0, 0, 1, 1.0, 2.0));
  GateMetaData meta;
  meta.index = 0;
  meta.symbol_values.push_back("TheSymbol");
  meta.placeholder_names.push_back(GateParamNames::kPhaseExponent);
  meta.symbol_values.push_back("TheSymbol2");
  meta.placeholder_names.push_back(GateParamNames::kExponent);
  meta.gate_params = {1.0, 1.0, 2.0, 1.0};
  metadata.push_back(meta);

  CreateHessianCircuit(circuit, metadata, &fuses, &grad_gates);
  EXPECT_EQ(grad_gates.size(), 1);
  EXPECT_EQ(grad_gates[0].index, 0);
  EXPECT_EQ(grad_gates[0].params.size(), 3);
  EXPECT_EQ(grad_gates[0].params[0], "TheSymbol");
  EXPECT_EQ(grad_gates[0].params[1], "TheSymbol2");
  // Third symbol is automatically generated `kUsePrevTwoSymbols`.
  EXPECT_EQ(grad_gates[0].params[2], kUsePrevTwoSymbols);

  // fuse everything into 2 gates. One fuse before this gate and one after.
  // both wind up being identity since this is the only gate.
  EXPECT_EQ(fuses.size(), 2);

  GradientOfGate tmp;
  PopulateHessianPhasedISwapPhasedExponent("TheSymbol", 0, 0, 1, 1.0, 1.0, 2.0,
                                            1.0, &tmp);

  Matrix4Equal(tmp.grad_gates[0].matrix, grad_gates[0].grad_gates[0].matrix,
               3e-2);

  GradientOfGate tmp2;
  PopulateHessianPhasedISwapExponent("TheSymbol2", 0, 0, 1, 1.0, 1.0, 2.0, 1.0,
                                      &tmp2);

  Matrix4Equal(tmp2.grad_gates[0].matrix, grad_gates[0].grad_gates[1].matrix,
               3e-2);

  metadata.clear();
  meta.symbol_values.clear();
  meta.placeholder_names.clear();
  grad_gates.clear();
  fuses.clear();

  metadata.push_back(meta);

  CreateHessianCircuit(circuit, metadata, &fuses, &grad_gates);
  EXPECT_EQ(grad_gates.size(), 0);
  EXPECT_EQ(fuses.size(), 1);
}

TEST(AdjHessianUtilTest, CreateHessianFSim) {
  QsimCircuit circuit;
  std::vector<GateMetaData> metadata;
  std::vector<std::vector<qsim::GateFused<QsimGate>>> fuses;
  std::vector<GradientOfGate> grad_gates;

  // Create a symbolized gate.
  circuit.num_qubits = 2;
  circuit.gates.push_back(
      qsim::Cirq::FSimGate<float>::Create(0, 0, 1, 1.0, 2.0));
  GateMetaData meta;
  meta.index = 0;
  meta.symbol_values.push_back("TheSymbol");
  meta.placeholder_names.push_back(GateParamNames::kTheta);
  meta.symbol_values.push_back("TheSymbol2");
  meta.placeholder_names.push_back(GateParamNames::kPhi);
  meta.gate_params = {1.0, 1.0, 2.0, 1.0};
  metadata.push_back(meta);

  CreateHessianCircuit(circuit, metadata, &fuses, &grad_gates);
  EXPECT_EQ(grad_gates.size(), 1);
  EXPECT_EQ(grad_gates[0].index, 0);
  EXPECT_EQ(grad_gates[0].params.size(), 3);
  EXPECT_EQ(grad_gates[0].params[0], "TheSymbol");
  EXPECT_EQ(grad_gates[0].params[1], "TheSymbol2");
  // Third symbol is automatically generated `kUsePrevTwoSymbols`.
  EXPECT_EQ(grad_gates[0].params[2], kUsePrevTwoSymbols);

  // fuse everything into 2 gates. One fuse before this gate and one after.
  // both wind up being identity since this is the only gate.
  EXPECT_EQ(fuses.size(), 2);

  GradientOfGate tmp;
  PopulateHessianFsimTheta("TheSymbol", 0, 0, 1, 1.0, 1.0, 2.0, 1.0, &tmp);

  Matrix4Equal(tmp.grad_gates[0].matrix, grad_gates[0].grad_gates[0].matrix,
               1e-4);

  GradientOfGate tmp2;
  PopulateHessianFsimPhi("TheSymbol2", 0, 0, 1, 1.0, 1.0, 2.0, 1.0, &tmp2);

  Matrix4Equal(tmp2.grad_gates[0].matrix, grad_gates[0].grad_gates[1].matrix,
               1e-4);

  metadata.clear();
  meta.symbol_values.clear();
  meta.placeholder_names.clear();
  grad_gates.clear();
  fuses.clear();

  metadata.push_back(meta);

  CreateHessianCircuit(circuit, metadata, &fuses, &grad_gates);
  EXPECT_EQ(grad_gates.size(), 0);
  EXPECT_EQ(fuses.size(), 1);
}

TEST(AdjHessianUtilTest, CreateHessianEmpty) {
  QsimCircuit empty_circuit;
  std::vector<GateMetaData> empty_metadata;
  std::vector<std::vector<qsim::GateFused<QsimGate>>> fuses;
  std::vector<GradientOfGate> grad_gates;

  CreateHessianCircuit(empty_circuit, empty_metadata, &fuses, &grad_gates);

  // Should create a single "empty fuse."
  EXPECT_EQ(fuses.size(), 1);
  EXPECT_EQ(fuses[0].size(), 0);

  // No gradients.
  EXPECT_EQ(grad_gates.size(), 0);
}

TEST(AdjHessianUtilTest, SingleEigenGrad) {
  GradientOfGate grad;

  PopulateHessianSingleEigen(&qsim::Cirq::YPowGate<float>::Create, "hello", 5,
                              2, 0.125, 1.0, 0.0, &grad);

  // Value verified from:
  /*
  (cirq.unitary(cirq.Y**(0.125 + 1e-2))
    + cirq.unitary(cirq.Y**(0.125 - 1e-2))
    - cirq.unitary(cirq.Y**(0.125))
    - cirq.unitary(cirq.Y**(0.125))) * 1e4
  array([[-4.55878779-1.88831173j,  1.88831173-4.55878779j],
         [-1.88831173+4.55878779j, -4.55878779-1.88831173j]])
  */
  std::vector<float> expected{-4.558788 , -1.8883117,  1.8883117, -4.558788 , -1.8883117,
        4.558788 , -4.558788 , -1.8883117};

  EXPECT_EQ(grad.index, 5);
  EXPECT_EQ(grad.params[0], "hello");
  Matrix2Equal(grad.grad_gates[0].matrix, expected, 5e-3);
}

TEST(AdjHessianUtilTest, TwoEigenGrad) {
  GradientOfGate grad;

  PopulateHessianTwoEigen(&qsim::Cirq::XXPowGate<float>::Create, "hi", 5, 2, 3,
                           0.001, 1.0, 0.0, &grad);

  // Value verified from:
  /*
  (cirq.unitary(cirq.XX**(0.001 + 1e-2))
    + cirq.unitary(cirq.XX**(0.001 - 1e-2))
    - cirq.unitary(cirq.XX**(0.001))
    - cirq.unitary(cirq.XX**(0.001))) * 1e4
    array([[-0.00049348-1.55031128e-06j,  0.        +0.00000000e+00j,
             0.        +0.00000000e+00j,  0.00049348+1.55031128e-06j],
           [ 0.        +0.00000000e+00j, -0.00049348-1.55031128e-06j,
             0.00049348+1.55031128e-06j,  0.        +0.00000000e+00j],
           [ 0.        +0.00000000e+00j,  0.00049348+1.55031128e-06j,
            -0.00049348-1.55031128e-06j,  0.        +0.00000000e+00j],
           [ 0.00049348+1.55031128e-06j,  0.        +0.00000000e+00j,
             0.        +0.00000000e+00j, -0.00049348-1.55031128e-06j]])
  */
  std::vector<float> expected{-4.934372  , -0.01550184,  0.        ,  0.        ,  0.        ,
        0.        ,  4.934372  ,  0.01550184,  0.        ,  0.        ,
       -4.934372  , -0.01550184,  4.934372  ,  0.01550184,  0.        ,
        0.        ,  0.        ,  0.        ,  4.934372  ,  0.01550184,
       -4.934372  , -0.01550184,  0.        ,  0.        ,  4.934372  ,
        0.01550184,  0.        ,  0.        ,  0.        ,  0.        ,
       -4.934372  , -0.01550184};

  EXPECT_EQ(grad.index, 5);
  EXPECT_EQ(grad.params[0], "hi");
  Matrix4Equal(grad.grad_gates[0].matrix, expected, 1e-3);
}

TEST(AdjHessianUtilTest, PhasedXPhasedExponent) {
  GradientOfGate grad;

  PopulateHessianPhasedXPhasedExponent("hello2", 5, 2, 0.001, 1.0, 1.0, 1.0,
                                        0.0, &grad);
  /* Value verified from:
  (cirq.unitary(cirq.PhasedXPowGate(exponent=1.0,phase_exponent=0.001 + 1e-2))
    + cirq.unitary(cirq.PhasedXPowGate(exponent=1.0,phase_exponent=0.001 - 1e-2))
    - cirq.unitary(cirq.PhasedXPowGate(exponent=1.0,phase_exponent=0.001))
    - cirq.unitary(cirq.PhasedXPowGate(exponent=1.0,phase_exponent=0.001))) * 1e4
    array([[ 0.        +0.j        , -9.86874398+0.03100368j],
           [-9.86874398-0.03100368j,  0.        +0.j        ]])
  */
  std::vector<float> expected{0.        ,  0.        , -9.868744  ,  0.03100368, -9.868744  ,
       -0.03100368,  0.        ,  0.       };

  EXPECT_EQ(grad.index, 5);
  EXPECT_EQ(grad.params[0], "hello2");
  Matrix2Equal(grad.grad_gates[0].matrix, expected, 5e-4);
}

TEST(AdjHessianUtilTest, PhasedXExponent) {
  GradientOfGate grad;

  PopulateHessianPhasedXExponent("hello3", 5, 2, 10.123, 1.0, 0.789, 1.0, 0.0,
                                  &grad);
  /* Value verified from:
  (cirq.unitary(cirq.PhasedXPowGate(exponent=0.789+1e-2,phase_exponent=10.123))
    + cirq.unitary(cirq.PhasedXPowGate(exponent=0.789-1e-2,phase_exponent=10.123))
    - cirq.unitary(cirq.PhasedXPowGate(exponent=0.789,phase_exponent=10.123))
    - cirq.unitary(cirq.PhasedXPowGate(exponent=0.789,phase_exponent=10.123))) * 1e4
    array([[ 3.88941758-3.03656025j, -2.45824276+4.2784705j ],
           [-4.74702582+1.34685303j,  3.88941758-3.03656025j]])
  */
  std::vector<float> expected{3.8894176, -3.0365603, -2.4582427,  4.2784705, -4.747026 ,
        1.346853 ,  3.8894176, -3.0365603};

  EXPECT_EQ(grad.index, 5);
  EXPECT_EQ(grad.params[0], "hello3");
  Matrix2Equal(grad.grad_gates[0].matrix, expected, 1e-3);
}

TEST(AdjHessianUtilTest, FSimThetaGrad) {
  GradientOfGate grad;
  PopulateHessianFsimTheta("hihi", 5, 2, 3, 0.5, 1.0, 1.2, 1.0, &grad);

  /* Value verified from:
  (cirq.unitary(cirq.FSimGate(theta=0.5 + 1e-2,phi=1.2)) 
    + cirq.unitary(cirq.FSimGate(theta=0.5-1e-2,phi=1.2))
    - cirq.unitary(cirq.FSimGate(theta=0.5,phi=1.2)) 
    - cirq.unitary(cirq.FSimGate(theta=0.5,phi=1.2))) * 1e4
    array([[ 0.        +0.j        ,  0.        +0.j        ,
             0.        +0.j        ,  0.        +0.j        ],
           [ 0.        +0.j        , -0.87757525+0.j        ,
             0.        +0.47942154j,  0.        +0.j        ],
           [ 0.        +0.j        ,  0.        +0.47942154j,
             -0.87757525+0.j        ,  0.        +0.j        ],
           [ 0.        +0.j        ,  0.        +0.j        ,
            0.        +0.j        ,  0.        +0.j        ]])
  */
  std::vector<float> expected{
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
       -0.8775753 ,  0.        ,  0.        ,  0.47942156,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.47942156,
       -0.8775753 ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.       
  };

  EXPECT_EQ(grad.index, 5);
  EXPECT_EQ(grad.params[0], "hihi");
  Matrix4Equal(grad.grad_gates[0].matrix, expected, 1e-3);
}

TEST(AdjHessianUtilTest, FSimPhiGrad) {
  GradientOfGate grad;
  PopulateHessianFsimPhi("hihi2", 5, 2, 3, 0.5, 1.0, 1.2, 1.0, &grad);

  /*
  (cirq.unitary(cirq.FSimGate(theta=0.5,phi=1.2+1e-2)) 
    + cirq.unitary(cirq.FSimGate(theta=0.5,phi=1.2-1e-2))
    - cirq.unitary(cirq.FSimGate(theta=0.5,phi=1.2)) 
    - cirq.unitary(cirq.FSimGate(theta=0.5,phi=1.2))) * 1e4
  array([[ 0.        +0.j        ,  0.        +0.j        ,
           0.        +0.j        ,  0.        +0.j        ],
         [ 0.        +0.j        ,  0.        +0.j        ,
           0.        +0.j        ,  0.        +0.j        ],
         [ 0.        +0.j        ,  0.        +0.j        ,
           0.        +0.j        ,  0.        +0.j        ],
         [ 0.        +0.j        ,  0.        +0.j        ,
           0.        +0.j        , -0.36235473+0.93203132j]])
  */
  std::vector<float> expected{
      0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
       -0.36235473,  0.93203133};

  EXPECT_EQ(grad.index, 5);
  EXPECT_EQ(grad.params[0], "hihi2");
  Matrix4Equal(grad.grad_gates[0].matrix, expected, 5e-4);
}

TEST(AdjHessianUtilTest, PhasedISwapPhasedExponent) {
  GradientOfGate grad;

  PopulateHessianPhasedISwapPhasedExponent("h", 5, 3, 2, 8.9, 1.0, -3.2, 1.0,
                                            &grad);

  /*
  (cirq.unitary(cirq.PhasedISwapPowGate(exponent=-3.2,phase_exponent=8.9+1e-2))
  + cirq.unitary(cirq.PhasedISwapPowGate(exponent=-3.2,phase_exponent=8.9-1e-2))
  - cirq.unitary(cirq.PhasedISwapPowGate(exponent=-3.2,phase_exponent=8.9))
  - cirq.unitary(cirq.PhasedISwapPowGate(exponent=-3.2,phase_exponent=8.9))) * 1e4
  
  array([[  0.         +0.j        ,   0.         +0.j        ,
            0.         +0.j        ,   0.         +0.j        ],
         [  0.         +0.j        ,   0.         +0.j        ,
            -22.06184686-30.36552715j,   0.         +0.j        ],
         [  0.         +0.j        ,  22.06184686-30.36552715j,
            0.         +0.j        ,   0.         +0.j        ],
         [  0.         +0.j        ,   0.         +0.j        ,
            0.         +0.j        ,   0.         +0.j        ]])
  */
  std::vector<float> expected{
      0.      ,   0.      ,   0.      ,   0.      ,   0.      ,
         0.      ,   0.      ,   0.      ,   0.      ,   0.      ,
         0.      ,   0.      , -22.061848, -30.365528,   0.      ,
         0.      ,   0.      ,   0.      ,  22.061848, -30.365528,
         0.      ,   0.      ,   0.      ,   0.      ,   0.      ,
         0.      ,   0.      ,   0.      ,   0.      ,   0.      ,
         0.      ,   0.      };

  EXPECT_EQ(grad.index, 5);
  EXPECT_EQ(grad.params[0], "h");
  Matrix4Equal(grad.grad_gates[0].matrix, expected, 3e-2);
}

TEST(AdjHessianUtilTest, PhasedISwapExponent) {
  GradientOfGate grad;

  PopulateHessianPhasedISwapExponent("h2", 5, 3, 2, 8.9, 1.0, -3.2, 1.0,
                                      &grad);

  /*
  (cirq.unitary(cirq.PhasedISwapPowGate(exponent=-3.2+1e-2,phase_exponent=8.9))
  + cirq.unitary(cirq.PhasedISwapPowGate(exponent=-3.2-1e-2,phase_exponent=8.9))
  - cirq.unitary(cirq.PhasedISwapPowGate(exponent=-3.2,phase_exponent=8.9))
  - cirq.unitary(cirq.PhasedISwapPowGate(exponent=-3.2,phase_exponent=8.9))) * 1e4
  array([[ 0.        +0.j       ,  0.        +0.j       ,
           0.        +0.j       ,  0.        +0.j       ],
         [ 0.        +0.j       , -0.76245319+0.j       ,
          -1.37929079-1.8984309j,  0.        +0.j       ],
         [ 0.        +0.j       ,  1.37929079-1.8984309j,
          -0.76245319+0.j       ,  0.        +0.j       ],
         [ 0.        +0.j       ,  0.        +0.j       ,
           0.        +0.j       ,  0.        +0.j       ]])
  */
  std::vector<float> expected{
      0.       ,  0.       ,  0.       ,  0.       ,  0.       ,
        0.       ,  0.       ,  0.       ,  0.       ,  0.       ,
       -0.7624532,  0.       , -1.3792908, -1.898431 ,  0.       ,
        0.       ,  0.       ,  0.       ,  1.3792908, -1.898431 ,
       -0.7624532,  0.       ,  0.       ,  0.       ,  0.       ,
        0.       ,  0.       ,  0.       ,  0.       ,  0.       ,
        0.       ,  0.       };

  EXPECT_EQ(grad.index, 5);
  EXPECT_EQ(grad.params[0], "h2");
  Matrix4Equal(grad.grad_gates[0].matrix, expected, 2e-3);
}

TEST(AdjHessianUtilTest, Matrix2Add) {
  std::array<float, 8> u{1, 2, 3, 4, 5, 6, 7, 8};
  std::array<float, 8> u2{0, -1, -2, -3, -4, -5, -6, -7};
  Matrix2Add(u, u2);
  for (int i = 0; i < 8; i++) {
    EXPECT_EQ(u2[i], 1);
    EXPECT_EQ(u[i], i + 1);
  }
}

TEST(AdjHessianUtilTest, Matrix4Add) {
  std::array<float, 32> u;
  std::array<float, 32> u2;

  for (int i = 0; i < 32; i++) {
    u2[i] = -i;
    u[i] = i + 1;
  }

  Matrix4Add(u, u2);
  for (int i = 0; i < 32; i++) {
    EXPECT_EQ(u2[i], 1);
    EXPECT_EQ(u[i], i + 1);
  }
}

}  // namespace
}  // namespace tfq
