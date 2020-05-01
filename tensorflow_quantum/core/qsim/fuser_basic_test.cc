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

#include "tensorflow_quantum/core/qsim/fuser_basic.h"

#include "absl/container/flat_hash_map.h"
#include "gtest/gtest.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_quantum/core/src/circuit.h"
#include "tensorflow_quantum/core/src/gates_def.h"

namespace tfq {
namespace qsim {
namespace {

using ::tensorflow::Status;

TEST(FuserBasicTest, GateFused) {
  Status status;

  std::vector<unsigned int> locations;
  XPowGateBuilder x_pow_builder;
  Gate gate_x, gate_cnot;
  absl::flat_hash_map<std::string, float> arg_map;
  arg_map["global_shift"] = 0.0;
  arg_map["exponent"] = 1.0;
  arg_map["exponent_scalar"] = 1.0;
  locations.push_back(0);
  status = x_pow_builder.Build(0, locations, arg_map, &gate_x);
  ASSERT_EQ(status, Status::OK());
  locations.clear();

  CNotPowGateBuilder cnot_pow_builder;
  absl::flat_hash_map<std::string, float> arg_map_cnot;
  arg_map_cnot["global_shift"] = 0.0;
  arg_map_cnot["exponent"] = 1.0;
  arg_map_cnot["exponent_scalar"] = 1.0;
  locations.push_back(0);
  locations.push_back(1);
  status = cnot_pow_builder.Build(1, locations, arg_map_cnot, &gate_cnot);
  ASSERT_EQ(status, Status::OK());
  locations.clear();

  GateFused test_fused(42, 0, 1, &gate_cnot);
  GateFused real_fused(42, 0, 1, &gate_cnot);
  test_fused.AddGate(&gate_x);
  real_fused.AddGate(&gate_x);
  test_fused.AddGate(&gate_cnot);
  real_fused.AddGate(&gate_cnot);

  // confirm objects are actually equal
  ASSERT_EQ(test_fused.GetTime(), real_fused.GetTime());
  ASSERT_EQ(test_fused.GetQubit0(), real_fused.GetQubit0());
  ASSERT_EQ(test_fused.GetQubit1(), real_fused.GetQubit1());
  ASSERT_EQ(test_fused.GetAnchor(), real_fused.GetAnchor());
  ASSERT_EQ(test_fused.GetGate(0), real_fused.GetGate(0));
  ASSERT_EQ(test_fused.GetGate(1), real_fused.GetGate(1));

  // check equality operator overload
  test_fused.SetTime(real_fused.GetTime() + 1);
  ASSERT_NE(test_fused, real_fused);
  test_fused.SetTime(real_fused.GetTime());

  test_fused.SetQubit0(real_fused.GetQubit0() + 1);
  ASSERT_NE(test_fused, real_fused);
  test_fused.SetQubit0(real_fused.GetQubit0());

  test_fused.SetQubit1(real_fused.GetQubit1() + 1);
  ASSERT_NE(test_fused, real_fused);
  test_fused.SetQubit1(real_fused.GetQubit1());

  test_fused.SetAnchor(&gate_x);
  ASSERT_NE(test_fused, real_fused);
  test_fused.SetAnchor(&gate_cnot);

  test_fused.SetGate(0, &gate_cnot);
  ASSERT_NE(test_fused, real_fused);
  test_fused.SetGate(0, &gate_x);

  test_fused.SetGate(1, &gate_x);
  ASSERT_NE(test_fused, real_fused);
  test_fused.SetGate(1, &gate_cnot);

  ASSERT_EQ(test_fused, real_fused);
}

TEST(FuserBasicTest, FuseGatesMulti) {
  // Tests that many gates are fused correctly.
  //
  // Construct the following test circuit:
  // q0 -- X --   -- |CNOT| --   -- |I|
  // q1 -- Y -- Z -- |CNOT| -- H -- |I|
  // This should all be gathered into one GateFused.
  Status status;
  std::vector<GateFused> test_fused_vec;
  Circuit test_circuit;
  test_circuit.num_qubits = 2;
  test_circuit.gates.reserve(7);

  std::vector<unsigned int> locations;
  XPowGateBuilder x_pow_builder;
  YPowGateBuilder y_pow_builder;
  ZPowGateBuilder z_pow_builder;
  CNotPowGateBuilder cnot_pow_builder;
  HPowGateBuilder h_pow_builder;
  I2GateBuilder i2_builder;
  Gate gate_x, gate_y, gate_z, gate_cnot, gate_h, gate_ident;
  absl::flat_hash_map<std::string, float> arg_map_1q, arg_map_2q, empty_map;
  arg_map_1q["global_shift"] = 0.0;
  arg_map_1q["exponent"] = 1.0;
  arg_map_1q["exponent_scalar"] = 1.0;

  arg_map_2q["global_shift"] = 0.0;
  arg_map_2q["exponent"] = 1.0;
  arg_map_2q["exponent_scalar"] = 1.0;

  unsigned int anchor_t = 2;
  locations.push_back(0);
  locations.push_back(1);
  status = cnot_pow_builder.Build(anchor_t, locations, arg_map_2q, &gate_cnot);
  ASSERT_EQ(status, Status::OK());
  locations.clear();
  GateFused real_fused(anchor_t, 0, 1, &gate_cnot);

  locations.push_back(0);
  status = x_pow_builder.Build(0, locations, arg_map_1q, &gate_x);
  ASSERT_EQ(status, Status::OK());
  test_circuit.gates.push_back(gate_x);
  real_fused.AddGate(&test_circuit.gates.back());
  locations.clear();

  locations.push_back(1);
  status = y_pow_builder.Build(0, locations, arg_map_1q, &gate_y);
  ASSERT_EQ(status, Status::OK());
  test_circuit.gates.push_back(gate_y);
  real_fused.AddGate(&test_circuit.gates.back());
  locations.clear();

  locations.push_back(1);
  status = z_pow_builder.Build(1, locations, arg_map_1q, &gate_z);
  ASSERT_EQ(status, Status::OK());
  test_circuit.gates.push_back(gate_z);
  real_fused.AddGate(&test_circuit.gates.back());
  locations.clear();

  test_circuit.gates.push_back(gate_cnot);
  real_fused.AddGate(&test_circuit.gates.back());

  locations.push_back(0);
  status = h_pow_builder.Build(3, locations, arg_map_1q, &gate_h);
  ASSERT_EQ(status, Status::OK());
  test_circuit.gates.push_back(gate_h);
  real_fused.AddGate(&test_circuit.gates.back());
  locations.clear();

  locations.push_back(0);
  locations.push_back(1);
  status = i2_builder.Build(4, locations, empty_map, &gate_ident);
  ASSERT_EQ(status, Status::OK());
  test_circuit.gates.push_back(gate_ident);
  real_fused.AddGate(&test_circuit.gates.back());
  locations.clear();

  ASSERT_EQ(FuseGates(test_circuit, &test_fused_vec), Status::OK());
  ASSERT_EQ(1, test_fused_vec.size());
  ASSERT_EQ(real_fused, test_fused_vec.at(0));
}

TEST(FuserBasicTest, FuseGatesDisjoint) {
  // Tests that two-qubit gates not sharing both qubits
  // are put into different GateFused.
  //
  // Construct the following test circuit:
  // q0 --   X    --   -- |CNOT| --   -- |I| --
  // q1 -- |CNOT| -- Y -- |CNOT| -- X -- |I| -- |I|
  // q2 -- |CNOT| -- Z --   H    --   --     -- |I|
  // This should be fused into three different GateFused objects.
  // The t = 0 CNOT should fuse the Y, Z, and H gates;
  // the t = 2 CNOT should fuse the X gates and I(q0, q1);
  // and the final I(q1, q2) should be alone.
  Status status;
  std::vector<GateFused> test_fused_vec;
  Circuit test_circuit;
  test_circuit.num_qubits = 3;
  test_circuit.gates.reserve(10);

  std::vector<unsigned int> locations;
  XPowGateBuilder x_pow_builder;
  YPowGateBuilder y_pow_builder;
  ZPowGateBuilder z_pow_builder;
  CNotPowGateBuilder cnot_pow_builder;
  HPowGateBuilder h_pow_builder;
  I2GateBuilder i2_builder;
  Gate gate_x_1, gate_cnot_1, gate_y, gate_z, gate_cnot_2, gate_h, gate_x_2,
      gate_ident_1, gate_ident_2;
  absl::flat_hash_map<std::string, float> arg_map_1q, arg_map_2q, empty_map;
  arg_map_1q["global_shift"] = 0.0;
  arg_map_1q["exponent"] = 1.0;
  arg_map_1q["exponent_scalar"] = 1.0;

  arg_map_2q["global_shift"] = 0.0;
  arg_map_2q["exponent"] = 1.0;
  arg_map_2q["exponent_scalar"] = 1.0;

  // First fused gate
  unsigned int anchor_time_1 = 0;
  locations.push_back(1);
  locations.push_back(2);
  status = cnot_pow_builder.Build(anchor_time_1, locations, arg_map_2q,
                                  &gate_cnot_1);
  ASSERT_EQ(status, Status::OK());
  locations.clear();

  test_circuit.gates.push_back(gate_cnot_1);
  GateFused real_fused_1(anchor_time_1, 1, 2, &gate_cnot_1);
  real_fused_1.AddGate(&test_circuit.gates.back());

  locations.push_back(1);
  status = y_pow_builder.Build(1, locations, arg_map_1q, &gate_y);
  ASSERT_EQ(status, Status::OK());
  test_circuit.gates.push_back(gate_y);
  real_fused_1.AddGate(&test_circuit.gates.back());
  locations.clear();

  locations.push_back(2);
  status = z_pow_builder.Build(1, locations, arg_map_1q, &gate_z);
  ASSERT_EQ(status, Status::OK());
  test_circuit.gates.push_back(gate_z);
  real_fused_1.AddGate(&test_circuit.gates.back());
  locations.clear();

  locations.push_back(2);
  status = h_pow_builder.Build(2, locations, arg_map_1q, &gate_h);
  ASSERT_EQ(status, Status::OK());
  test_circuit.gates.push_back(gate_h);
  real_fused_1.AddGate(&test_circuit.gates.back());
  locations.clear();

  // Second fused gate
  locations.push_back(0);
  status = x_pow_builder.Build(0, locations, arg_map_1q, &gate_x_1);
  ASSERT_EQ(status, Status::OK());
  test_circuit.gates.push_back(gate_x_1);
  locations.clear();

  unsigned int anchor_time_2 = 2;
  locations.push_back(0);
  locations.push_back(1);
  status = cnot_pow_builder.Build(anchor_time_2, locations, arg_map_2q,
                                  &gate_cnot_2);
  ASSERT_EQ(status, Status::OK());
  GateFused real_fused_2(anchor_time_2, 0, 1, &gate_cnot_2);
  real_fused_2.AddGate(&test_circuit.gates.back());
  test_circuit.gates.push_back(gate_cnot_2);
  real_fused_2.AddGate(&test_circuit.gates.back());
  locations.clear();

  locations.push_back(1);
  status = x_pow_builder.Build(3, locations, arg_map_1q, &gate_x_2);
  ASSERT_EQ(status, Status::OK());
  test_circuit.gates.push_back(gate_x_2);
  real_fused_2.AddGate(&test_circuit.gates.back());
  locations.clear();

  locations.push_back(0);
  locations.push_back(1);
  status = i2_builder.Build(4, locations, empty_map, &gate_ident_1);
  ASSERT_EQ(status, Status::OK());
  test_circuit.gates.push_back(gate_ident_1);
  real_fused_2.AddGate(&test_circuit.gates.back());
  locations.clear();

  // Third fused gate
  unsigned int anchor_time_3 = 5;
  locations.push_back(1);
  locations.push_back(2);
  status = i2_builder.Build(anchor_time_3, locations, empty_map, &gate_ident_2);
  ASSERT_EQ(status, Status::OK());
  test_circuit.gates.push_back(gate_ident_2);
  GateFused real_fused_3(anchor_time_3, 1, 2, &gate_ident_2);
  real_fused_3.AddGate(&test_circuit.gates.back());
  locations.clear();

  // Check fused gate equality
  ASSERT_EQ(FuseGates(test_circuit, &test_fused_vec), Status::OK());
  ASSERT_EQ(3, test_fused_vec.size());
  ASSERT_EQ(real_fused_1, test_fused_vec.at(0));
  ASSERT_EQ(real_fused_2, test_fused_vec.at(1));
  ASSERT_EQ(real_fused_3, test_fused_vec.at(2));
}

}  // namespace
}  // namespace qsim
}  // namespace tfq
