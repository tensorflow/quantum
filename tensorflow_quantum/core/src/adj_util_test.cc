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

TEST_P(OneQubitEigenFixture, CreateGradientSingleEigen) {
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

  CreateGradientCircuit(circuit, metadata, &fuses, &grad_gates);
  EXPECT_EQ(grad_gates.size(), 1);
  EXPECT_EQ(grad_gates[0].index, 0);
  EXPECT_EQ(grad_gates[0].params.size(), 1);
  EXPECT_EQ(grad_gates[0].params[0], "TheSymbol");
  // fuse everything into 2 gates. One fuse before this gate and one after.
  // both wind up being identity since this is the only gate.
  EXPECT_EQ(fuses.size(), 2);

  GradientOfGate tmp;
  PopulateGradientSingleEigen(given_f, "TheSymbol", 0, 1, 1.0, 1.0, 2.0, &tmp);

  Matrix2Equal(tmp.grad_gates[0].matrix, grad_gates[0].grad_gates[0].matrix,
               1e-4);

  // Test with NO symbol.
  metadata.clear();
  meta.symbol_values.clear();
  meta.placeholder_names.clear();
  fuses.clear();
  grad_gates.clear();

  CreateGradientCircuit(circuit, metadata, &fuses, &grad_gates);
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

TEST_P(TwoQubitEigenFixture, CreateGradientTwoEigen) {
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

  CreateGradientCircuit(circuit, metadata, &fuses, &grad_gates);
  EXPECT_EQ(grad_gates.size(), 1);
  EXPECT_EQ(grad_gates[0].index, 0);
  EXPECT_EQ(grad_gates[0].params.size(), 1);
  EXPECT_EQ(grad_gates[0].params[0], "TheSymbol");

  // fuse everything into 2 gates. One fuse before this gate and one after.
  // both wind up being identity since this is the only gate.
  EXPECT_EQ(fuses.size(), 2);

  GradientOfGate tmp;
  PopulateGradientTwoEigen(given_f, "TheSymbol", 0, 0, 1, 1.0, 1.0, 2.0, &tmp);

  Matrix4Equal(tmp.grad_gates[0].matrix, grad_gates[0].grad_gates[0].matrix,
               1e-4);

  // Test with NO symbol.
  metadata.clear();
  meta.symbol_values.clear();
  meta.placeholder_names.clear();
  fuses.clear();
  grad_gates.clear();

  CreateGradientCircuit(circuit, metadata, &fuses, &grad_gates);
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

TEST(AdjUtilTest, CreateGradientPhasedX) {
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

  CreateGradientCircuit(circuit, metadata, &fuses, &grad_gates);
  EXPECT_EQ(grad_gates.size(), 1);
  EXPECT_EQ(grad_gates[0].index, 0);
  EXPECT_EQ(grad_gates[0].params.size(), 2);
  EXPECT_EQ(grad_gates[0].params[0], "TheSymbol");
  EXPECT_EQ(grad_gates[0].params[1], "TheSymbol2");

  // fuse everything into 2 gates. One fuse before this gate and one after.
  // both wind up being identity since this is the only gate.
  EXPECT_EQ(fuses.size(), 2);

  GradientOfGate tmp;
  PopulateGradientPhasedXPhasedExponent("TheSymbol", 0, 0, 1.0, 1.0, 2.0, 1.0,
                                        3.0, &tmp);

  Matrix2Equal(tmp.grad_gates[0].matrix, grad_gates[0].grad_gates[0].matrix,
               1e-4);

  GradientOfGate tmp2;
  PopulateGradientPhasedXExponent("TheSymbol2", 0, 0, 1.0, 1.0, 2.0, 1.0, 3.0,
                                  &tmp2);

  Matrix2Equal(tmp2.grad_gates[0].matrix, grad_gates[0].grad_gates[1].matrix,
               1e-4);

  metadata.clear();
  meta.symbol_values.clear();
  meta.placeholder_names.clear();
  grad_gates.clear();
  fuses.clear();

  metadata.push_back(meta);

  CreateGradientCircuit(circuit, metadata, &fuses, &grad_gates);
  EXPECT_EQ(grad_gates.size(), 0);
  EXPECT_EQ(fuses.size(), 1);
}

TEST(AdjUtilTest, CreateGradientPhasedISwap) {
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

  CreateGradientCircuit(circuit, metadata, &fuses, &grad_gates);
  EXPECT_EQ(grad_gates.size(), 1);
  EXPECT_EQ(grad_gates[0].index, 0);
  EXPECT_EQ(grad_gates[0].params.size(), 2);
  EXPECT_EQ(grad_gates[0].params[0], "TheSymbol");
  EXPECT_EQ(grad_gates[0].params[1], "TheSymbol2");

  // fuse everything into 2 gates. One fuse before this gate and one after.
  // both wind up being identity since this is the only gate.
  EXPECT_EQ(fuses.size(), 2);

  GradientOfGate tmp;
  PopulateGradientPhasedISwapPhasedExponent("TheSymbol", 0, 0, 1, 1.0, 1.0, 2.0,
                                            1.0, &tmp);

  Matrix4Equal(tmp.grad_gates[0].matrix, grad_gates[0].grad_gates[0].matrix,
               1e-4);

  GradientOfGate tmp2;
  PopulateGradientPhasedISwapExponent("TheSymbol2", 0, 0, 1, 1.0, 1.0, 2.0, 1.0,
                                      &tmp2);

  Matrix4Equal(tmp2.grad_gates[0].matrix, grad_gates[0].grad_gates[1].matrix,
               1e-4);

  metadata.clear();
  meta.symbol_values.clear();
  meta.placeholder_names.clear();
  grad_gates.clear();
  fuses.clear();

  metadata.push_back(meta);

  CreateGradientCircuit(circuit, metadata, &fuses, &grad_gates);
  EXPECT_EQ(grad_gates.size(), 0);
  EXPECT_EQ(fuses.size(), 1);
}

TEST(AdjUtilTest, CreateGradientFSim) {
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

  CreateGradientCircuit(circuit, metadata, &fuses, &grad_gates);
  EXPECT_EQ(grad_gates.size(), 1);
  EXPECT_EQ(grad_gates[0].index, 0);
  EXPECT_EQ(grad_gates[0].params.size(), 2);
  EXPECT_EQ(grad_gates[0].params[0], "TheSymbol");
  EXPECT_EQ(grad_gates[0].params[1], "TheSymbol2");

  // fuse everything into 2 gates. One fuse before this gate and one after.
  // both wind up being identity since this is the only gate.
  EXPECT_EQ(fuses.size(), 2);

  GradientOfGate tmp;
  PopulateGradientFsimTheta("TheSymbol", 0, 0, 1, 1.0, 1.0, 2.0, 1.0, &tmp);

  Matrix4Equal(tmp.grad_gates[0].matrix, grad_gates[0].grad_gates[0].matrix,
               1e-4);

  GradientOfGate tmp2;
  PopulateGradientFsimPhi("TheSymbol2", 0, 0, 1, 1.0, 1.0, 2.0, 1.0, &tmp2);

  Matrix4Equal(tmp2.grad_gates[0].matrix, grad_gates[0].grad_gates[1].matrix,
               1e-4);

  metadata.clear();
  meta.symbol_values.clear();
  meta.placeholder_names.clear();
  grad_gates.clear();
  fuses.clear();

  metadata.push_back(meta);

  CreateGradientCircuit(circuit, metadata, &fuses, &grad_gates);
  EXPECT_EQ(grad_gates.size(), 0);
  EXPECT_EQ(fuses.size(), 1);
}

TEST(AdjUtilTest, CreateGradientEmpty) {
  QsimCircuit empty_circuit;
  std::vector<GateMetaData> empty_metadata;
  std::vector<std::vector<qsim::GateFused<QsimGate>>> fuses;
  std::vector<GradientOfGate> grad_gates;

  CreateGradientCircuit(empty_circuit, empty_metadata, &fuses, &grad_gates);

  // Should create a single "empty fuse."
  EXPECT_EQ(fuses.size(), 1);
  EXPECT_EQ(fuses[0].size(), 0);

  // No gradients.
  EXPECT_EQ(grad_gates.size(), 0);
}

TEST(AdjUtilTest, SingleEigenGrad) {
  GradientOfGate grad;

  PopulateGradientSingleEigen(&qsim::Cirq::YPowGate<float>::Create, "hello", 5,
                              2, 0.125, 1.0, 0.0, &grad);

  // Value verified from:
  /*
  (cirq.unitary(cirq.Y**(0.125 + 1e-4)) -
   cirq.unitary(cirq.Y**(0.125 - 1e-4))) / 2e-4
  array([[-0.60111772+1.45122655j, -1.45122655-0.60111772j],
         [ 1.45122655+0.60111772j, -0.60111772+1.45122655j]])
  */
  std::vector<float> expected{-0.60111, 1.45122, -1.45122, -0.60111,
                              1.45122,  0.60111, -0.60111, 1.45122};

  EXPECT_EQ(grad.index, 5);
  EXPECT_EQ(grad.params[0], "hello");
  Matrix2Equal(grad.grad_gates[0].matrix, expected, 1e-4);
}

TEST(AdjUtilTest, TwoEigenGrad) {
  GradientOfGate grad;

  PopulateGradientTwoEigen(&qsim::Cirq::XXPowGate<float>::Create, "hi", 5, 2, 3,
                           0.001, 1.0, 0.0, &grad);

  // Value verified from:
  /*
  (cirq.unitary(cirq.XX**(0.001 + 1e-4)) -
   cirq.unitary(cirq.XX**(0.001 - 1e-4))) / 2e-4
    array([[-0.00493479+1.57078855j,  0.        +0.j        ,
             0.        +0.j        ,  0.00493479-1.57078855j],
           [ 0.        +0.j        , -0.00493479+1.57078855j,
             0.00493479-1.57078855j,  0.        +0.j        ],
           [ 0.        +0.j        ,  0.00493479-1.57078855j,
            -0.00493479+1.57078855j,  0.        +0.j        ],
           [ 0.00493479-1.57078855j,  0.        +0.j        ,
             0.        +0.j        , -0.00493479+1.57078855j]])
  */
  std::vector<float> expected{
      -0.004934, 1.57078, 0.0,       0.0,       0.0,      0.0,      0.004934,
      -1.57078,  0.0,     0.0,       -0.004934, 1.57078,  0.004934, -1.57078,
      0.0,       0.0,     0.0,       0.0,       0.004934, -1.57078, -0.004934,
      1.57078,   0.0,     0.0,       0.004934,  -1.57078, 0.0,      0.0,
      0.0,       0.0,     -0.004934, 1.57078};

  EXPECT_EQ(grad.index, 5);
  EXPECT_EQ(grad.params[0], "hi");
  Matrix4Equal(grad.grad_gates[0].matrix, expected, 1e-4);
}

TEST(AdjUtilTest, PhasedXPhasedExponent) {
  GradientOfGate grad;

  PopulateGradientPhasedXPhasedExponent("hello2", 5, 2, 10.123, 1.0, 1.0, 1.0,
                                        0.0, &grad);
  /*
  (cirq.unitary(cirq.PhasedXPowGate(exponent=1.0,phase_exponent=0.001 + 1e-4)) -
   cirq.unitary(cirq.PhasedXPowGate(exponent=1.0,phase_exponent=0.001 - 1e-4)))
     / 2e-4
    array([[ 0.        +0.j        , -1.18397518-2.90994963j],
           [-1.18397518+2.90994963j,  0.        +0.j        ]])

  */
  std::vector<float> expected{0.0,      0.0,    -1.18397, -2.9099,
                              -1.18397, 2.9099, 0.0,      0.0};

  EXPECT_EQ(grad.index, 5);
  EXPECT_EQ(grad.params[0], "hello2");
  Matrix2Equal(grad.grad_gates[0].matrix, expected, 1e-4);
}

TEST(AdjUtilTest, PhasedXExponent) {
  GradientOfGate grad;

  PopulateGradientPhasedXExponent("hello3", 5, 2, 10.123, 1.0, 0.789, 1.0, 0.0,
                                  &grad);
  /*
  (cirq.unitary(cirq.PhasedXPowGate(exponent=0.789+1e-4,phase_exponent=10.123))
  -
  cirq.unitary(cirq.PhasedXPowGate(exponent=0.789-1e-4,phase_exponent=10.123)))
  / 2e-4 array([[-0.96664663-1.23814188j,  1.36199145+0.78254732j], [
  0.42875189+1.51114951j, -0.96664663-1.23814188j]])
  */
  std::vector<float> expected{-0.96664, -1.23814, 1.36199,  0.78254,
                              0.42875,  1.51114,  -0.96664, -1.23814};

  EXPECT_EQ(grad.index, 5);
  EXPECT_EQ(grad.params[0], "hello3");
  Matrix2Equal(grad.grad_gates[0].matrix, expected, 1e-4);
}

TEST(AdjUtilTest, FSimThetaGrad) {
  GradientOfGate grad;
  PopulateGradientFsimTheta("hihi", 5, 2, 3, 0.5, 1.0, 1.2, 1.0, &grad);

  /*
  (cirq.unitary(cirq.FSimGate(theta=0.5 + 1e-4,phi=1.2)) -
   cirq.unitary(cirq.FSimGate(theta=0.5-1e-4,phi=1.2))) / 2e-4
    array([[ 0.        +0.j        ,  0.        +0.j        ,
             0.        +0.j        ,  0.        +0.j        ],
           [ 0.        +0.j        , -0.47942554+0.j        ,
             0.        -0.87758256j,  0.        +0.j        ],
           [ 0.        +0.j        ,  0.        -0.87758256j,
            -0.47942554+0.j        ,  0.        +0.j        ],
           [ 0.        +0.j        ,  0.        +0.j        ,
             0.        +0.j        ,  0.        +0.j        ]])
  */
  std::vector<float> expected{
      0.0, 0.0, 0.0,      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,      0.0,      -0.47942,
      0.0, 0.0, -0.87758, 0.0, 0.0, 0.0, 0.0, 0.0, -0.87758, -0.47942, 0.0,
      0.0, 0.0, 0.0,      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,      0.0

  };

  EXPECT_EQ(grad.index, 5);
  EXPECT_EQ(grad.params[0], "hihi");
  Matrix4Equal(grad.grad_gates[0].matrix, expected, 1e-4);
}

TEST(AdjUtilTest, FSimPhiGrad) {
  GradientOfGate grad;
  PopulateGradientFsimPhi("hihi2", 5, 2, 3, 0.5, 1.0, 1.2, 1.0, &grad);

  /*
  (cirq.unitary(cirq.FSimGate(theta=0.5,phi=1.2+1e-4)) -
  cirq.unitary(cirq.FSimGate(theta=0.5,phi=1.2-1e-4))) / 2e-4
  array([[ 0.        +0.j        ,  0.        +0.j        ,
           0.        +0.j        ,  0.        +0.j        ],
         [ 0.        +0.j        ,  0.        +0.j        ,
           0.        +0.j        ,  0.        +0.j        ],
         [ 0.        +0.j        ,  0.        +0.j        ,
           0.        +0.j        ,  0.        +0.j        ],
         [ 0.        +0.j        ,  0.        +0.j        ,
           0.        +0.j        , -0.93203908-0.36235775j]])
  */
  std::vector<float> expected{
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,       0.0,      0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,       0.0,      0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.932039, -0.362357};

  EXPECT_EQ(grad.index, 5);
  EXPECT_EQ(grad.params[0], "hihi2");
  Matrix4Equal(grad.grad_gates[0].matrix, expected, 1e-4);
}

TEST(AdjUtilTest, PhasedISwapPhasedExponent) {
  GradientOfGate grad;

  PopulateGradientPhasedISwapPhasedExponent("h", 5, 3, 2, 8.9, 1.0, -3.2, 1.0,
                                            &grad);

  /*
  (cirq.unitary(cirq.PhasedISwapPowGate(exponent=-3.2,phase_exponent=8.9+1e-4))
  -
  cirq.unitary(cirq.PhasedISwapPowGate(exponent=-3.2,phase_exponent=8.9-1e-4)))
    / 2e-4
  array([[ 0.        +0.j        ,  0.        +0.j        ,
           0.        +0.j        ,  0.        +0.j        ],
         [ 0.        +0.j        ,  0.        +0.j        ,
          -4.83441368+3.51240713j,  0.        +0.j        ],
         [ 0.        +0.j        ,  4.83441368+3.51240713j,
           0.        +0.j        ,  0.        +0.j        ],
         [ 0.        +0.j        ,  0.        +0.j        ,
           0.        +0.j        ,  0.        +0.j        ]])

  */
  std::vector<float> expected{
      0.0, 0.0,      0.0,     0.0, 0.0, 0.0, 0.0, 0.0,     0.0,     0.0, 0.0,
      0.0, -4.83441, 3.51238, 0.0, 0.0, 0.0, 0.0, 4.83441, 3.51238, 0.0, 0.0,
      0.0, 0.0,      0.0,     0.0, 0.0, 0.0, 0.0, 0.0,     0.0,     0.0};

  EXPECT_EQ(grad.index, 5);
  EXPECT_EQ(grad.params[0], "h");
  Matrix4Equal(grad.grad_gates[0].matrix, expected, 1e-3);
}

TEST(AdjUtilTest, PhasedISwapExponent) {
  GradientOfGate grad;

  PopulateGradientPhasedISwapExponent("h2", 5, 3, 2, 8.9, 1.0, -3.2, 1.0,
                                      &grad);

  /*
  (cirq.unitary(cirq.PhasedISwapPowGate(exponent=-3.2+1e-3,phase_exponent=8.9))
  -cirq.unitary(cirq.PhasedISwapPowGate(exponent=-3.2-1e-3,phase_exponent=8.9)))
    / 2e-3
    array([[ 0.        +0.j        ,  0.        +0.j        ,
             0.        +0.j        ,  0.        +0.j        ],
           [ 0.        +0.j        , -1.49391547+0.j        ,
             0.28531247+0.39269892j,  0.        +0.j        ],
           [ 0.        +0.j        , -0.28531247+0.39269892j,
            -1.49391547+0.j        ,  0.        +0.j        ],
           [ 0.        +0.j        ,  0.        +0.j        ,
             0.        +0.j        ,  0.        +0.j        ]])

  */
  std::vector<float> expected{
      0.0, 0.0, 0.0,       0.0,      0.0,      0.0,      0.0, 0.0,
      0.0, 0.0, -1.49391,  0.0,      0.285312, 0.392698, 0.0, 0.0,
      0.0, 0.0, -0.285312, 0.392698, -1.49391, 0.0,      0.0, 0.0,
      0.0, 0.0, 0.0,       0.0,      0.0,      0.0,      0.0, 0.0};

  EXPECT_EQ(grad.index, 5);
  EXPECT_EQ(grad.params[0], "h2");
  Matrix4Equal(grad.grad_gates[0].matrix, expected, 1e-4);
}

TEST(AdjUtilTest, Matrix2Diff) {
  std::array<float, 8> u{1, 2, 3, 4, 5, 6, 7, 8};
  std::array<float, 8> u2{0, 1, 2, 3, 4, 5, 6, 7};
  Matrix2Diff(u, u2);
  for (int i = 0; i < 8; i++) {
    EXPECT_EQ(u2[i], -1);
    EXPECT_EQ(u[i], i + 1);
  }
}

TEST(AdjUtilTest, Matrix4Diff) {
  std::array<float, 32> u;
  std::array<float, 32> u2;

  for (int i = 0; i < 32; i++) {
    u2[i] = i;
    u[i] = i + 1;
  }

  Matrix4Diff(u, u2);
  for (int i = 0; i < 32; i++) {
    EXPECT_EQ(u2[i], -1);
    EXPECT_EQ(u[i], i + 1);
  }
}

}  // namespace
}  // namespace tfq
