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

#include "tensorflow_quantum/core/src/program_resolution.h"

#include <google/protobuf/text_format.h>

#include <string>

#include "absl/container/flat_hash_map.h"
#include "gtest/gtest.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_quantum/core/proto/program.pb.h"

namespace tfq {
namespace {

using tensorflow::Status;
using tfq::proto::PauliSum;
using tfq::proto::Program;

const std::string valid_program = R"(
  circuit {
    moments {
      operations {
        args {
          key: "control_qubits"
          value {
            arg_value {
              string_value: "0_0"
            }
          }
        }
        qubits {
          id: "0_1"
        }
        qubits {
          id: "0_2"
        }
      }
    }
  }
)";

const std::string valid_line_program = R"(
  circuit {
    moments {
      operations {
        args {
          key: "control_qubits"
          value {
            arg_value {
              string_value: "0_1"
            }
          }
        }
        qubits {
          id: "1"
        }
        qubits {
          id: "2"
        }
      }
    }
  }
)";

const std::string valid_psum = R"(
  terms {
    coefficient_real: 1.0
    coefficient_imag: 0.0
    paulis {
      qubit_id: "0_0"
      pauli_type: "X"
    }
  }
  terms {
    coefficient_real: 5.0
    coefficient_imag: 0.0
    paulis {
      qubit_id: "0_2"
      pauli_type: "Y"
    }
    paulis {
      qubit_id: "0_1"
      pauli_type: "Z"
    }
  }
)";

const std::string valid_symbol_program = R"(
  circuit {
    scheduling_strategy: MOMENT_BY_MOMENT
    moments {
      operations {
        args {
          key: "exponent"
          value {
            symbol: "v1"
          }
        }
      }
    }
    moments {
      operations {
        args {
          key: "exponent"
          value {
            symbol: "v2"
          }
        }
      }
    }
  }
)";

const std::string three_qubit_op_program = R"(
  circuit {
    moments {
      operations {
        qubits {
          id: "0_0"
        }
        qubits {
          id: "0_1"
        }
        qubits {
          id: "0_2"
        }
      }
    }
  }
)";

/* Qubit topology:
  1 -- 0 -- 2
       |
       |
       3
*/
const std::string resolved_qubit_program_not_1d = R"(
  circuit {
    moments {
      operations {
        qubits {
          id: "0"
        }
        qubits {
          id: "1"
        }
      }
      operations {
        qubits {
          id: "0"
        }
        qubits {
          id: "2"
        }
      }
      operations {
        qubits {
          id: "0"
        }
        qubits {
          id: "3"
        }
      }
    }
  }
)";

TEST(ProgramResolutionTest, ResolveQubitIdsValid) {
  Program program;
  unsigned int qubit_count;
  ASSERT_TRUE(
      google::protobuf::TextFormat::ParseFromString(valid_program, &program));

  EXPECT_EQ(ResolveQubitIds(&program, &qubit_count), Status());
  EXPECT_EQ(qubit_count, 3);
  EXPECT_EQ(program.circuit().moments(0).operations(0).qubits(0).id(), "1");
  EXPECT_EQ(program.circuit().moments(0).operations(0).qubits(1).id(), "2");
  EXPECT_EQ(program.circuit()
                .moments(0)
                .operations(0)
                .args()
                .at("control_qubits")
                .arg_value()
                .string_value(),
            "0");
}

TEST(ProgramResolutionTest, ResolveQubitIdsValidLine) {
  Program program;
  unsigned int qubit_count;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(valid_line_program,
                                                            &program));

  EXPECT_EQ(ResolveQubitIds(&program, &qubit_count), Status());
  EXPECT_EQ(qubit_count, 3);
  EXPECT_EQ(program.circuit().moments(0).operations(0).qubits(0).id(), "1");
  EXPECT_EQ(program.circuit().moments(0).operations(0).qubits(1).id(), "2");
  EXPECT_EQ(program.circuit()
                .moments(0)
                .operations(0)
                .args()
                .at("control_qubits")
                .arg_value()
                .string_value(),
            "0");
}

TEST(ProgramResolutionTest, ResolveQubitIdsInvalidControlQubit) {
  Program program;
  unsigned int qubit_count;
  ASSERT_TRUE(
      google::protobuf::TextFormat::ParseFromString(valid_program, &program));

  program.mutable_circuit()
      ->mutable_moments(0)
      ->mutable_operations(0)
      ->mutable_args()
      ->at("control_qubits")
      .mutable_arg_value()
      ->set_string_value("junk");
  EXPECT_EQ(ResolveQubitIds(&program, &qubit_count),
            tensorflow::Status(static_cast<tensorflow::errors::Code>(
                                   absl::StatusCode::kInvalidArgument),
                               "Unable to parse qubit: junk"));
}

TEST(ProgramResolutionTest, ResolveQubitIdsInvalidQubit) {
  Program program;
  unsigned int qubit_count;
  ASSERT_TRUE(
      google::protobuf::TextFormat::ParseFromString(valid_program, &program));

  program.mutable_circuit()
      ->mutable_moments(0)
      ->mutable_operations(0)
      ->mutable_qubits(0)
      ->set_id("junk");
  EXPECT_EQ(ResolveQubitIds(&program, &qubit_count),
            tensorflow::Status(static_cast<tensorflow::errors::Code>(
                                   absl::StatusCode::kInvalidArgument),
                               "Unable to parse qubit: junk"));
}

TEST(ProgramResolutionTest, ResolveQubitIdsWithPauliSum) {
  Program program;
  unsigned int qubit_count;
  ASSERT_TRUE(
      google::protobuf::TextFormat::ParseFromString(valid_program, &program));

  PauliSum p_sum;
  ASSERT_TRUE(
      google::protobuf::TextFormat::ParseFromString(valid_psum, &p_sum));
  std::vector<PauliSum> p_sums = {p_sum, p_sum};

  EXPECT_EQ(ResolveQubitIds(&program, &qubit_count, &p_sums), Status());
  EXPECT_EQ(qubit_count, 3);
  EXPECT_EQ(program.circuit().moments(0).operations(0).qubits(0).id(), "1");
  EXPECT_EQ(program.circuit().moments(0).operations(0).qubits(1).id(), "2");
  EXPECT_EQ(program.circuit()
                .moments(0)
                .operations(0)
                .args()
                .at("control_qubits")
                .arg_value()
                .string_value(),
            "0");

  for (int i = 0; i < 2; i++) {
    EXPECT_EQ(p_sums[i].terms(0).paulis(0).qubit_id(), "0");
    EXPECT_EQ(p_sums[i].terms(1).paulis(0).qubit_id(), "2");
    EXPECT_EQ(p_sums[i].terms(1).paulis(1).qubit_id(), "1");
  }
}

TEST(ProgramResolutionTest, ResolveQubitIdsWithInvalidPauliSum) {
  Program program;
  unsigned int qubit_count;
  ASSERT_TRUE(
      google::protobuf::TextFormat::ParseFromString(valid_program, &program));

  PauliSum p_sum;
  ASSERT_TRUE(
      google::protobuf::TextFormat::ParseFromString(valid_psum, &p_sum));
  p_sum.mutable_terms(0)->mutable_paulis(0)->set_qubit_id("1_1");
  std::vector<PauliSum> p_sums = {p_sum, p_sum};

  EXPECT_EQ(ResolveQubitIds(&program, &qubit_count, &p_sums),
            tensorflow::Status(
                static_cast<tensorflow::errors::Code>(
                    absl::StatusCode::kInvalidArgument),
                "Found a Pauli sum operating on qubits not found in circuit."));
}

TEST(ProgramResolutionTest, ResolveQubitIdsMultiProgram) {
  Program program, other;
  unsigned int qubit_count;
  ASSERT_TRUE(
      google::protobuf::TextFormat::ParseFromString(valid_program, &program));
  ASSERT_TRUE(
      google::protobuf::TextFormat::ParseFromString(valid_program, &other));

  // Re-arrange qubits on other.
  other.mutable_circuit()
      ->mutable_moments(0)
      ->mutable_operations(0)
      ->mutable_qubits(1)
      ->set_id("0_0");  // turn 0_2 -> 0_0!
  other.mutable_circuit()
      ->mutable_moments(0)
      ->mutable_operations(0)
      ->mutable_args()
      ->at("control_qubits")
      .mutable_arg_value()
      ->set_string_value("0_2");  // turn 0_0 -> 0_2!

  std::vector<Program> other_programs = {other, other};
  EXPECT_EQ(ResolveQubitIds(&program, &qubit_count, &other_programs), Status());
  EXPECT_EQ(qubit_count, 3);
  EXPECT_EQ(program.circuit().moments(0).operations(0).qubits(0).id(), "1");
  EXPECT_EQ(program.circuit().moments(0).operations(0).qubits(1).id(), "2");
  EXPECT_EQ(program.circuit()
                .moments(0)
                .operations(0)
                .args()
                .at("control_qubits")
                .arg_value()
                .string_value(),
            "0");

  for (int i = 0; i < 2; i++) {
    EXPECT_EQ(
        other_programs[i].circuit().moments(0).operations(0).qubits(0).id(),
        "1");
    EXPECT_EQ(
        other_programs[i].circuit().moments(0).operations(0).qubits(1).id(),
        "0");
    EXPECT_EQ(other_programs[i]
                  .circuit()
                  .moments(0)
                  .operations(0)
                  .args()
                  .at("control_qubits")
                  .arg_value()
                  .string_value(),
              "2");
  }
}

TEST(ProgramResolutionTest, ResolveQubitIdsMultiProgramInvalid) {
  Program program, other;
  unsigned int qubit_count;
  ASSERT_TRUE(
      google::protobuf::TextFormat::ParseFromString(valid_program, &program));
  ASSERT_TRUE(
      google::protobuf::TextFormat::ParseFromString(valid_program, &other));
  program.mutable_circuit()
      ->mutable_moments(0)
      ->mutable_operations(0)
      ->mutable_qubits(0)
      ->set_id("junk");
  std::vector<Program> others = {other, other};
  EXPECT_EQ(ResolveQubitIds(&program, &qubit_count, &others),
            tensorflow::Status(static_cast<tensorflow::errors::Code>(
                                   absl::StatusCode::kInvalidArgument),
                               "Unable to parse qubit: junk"));
}

TEST(ProgramResolutionTest, ResolveQubitIdsMultiProgramInvalidControl) {
  Program program, other;
  unsigned int qubit_count;
  ASSERT_TRUE(
      google::protobuf::TextFormat::ParseFromString(valid_program, &program));
  ASSERT_TRUE(
      google::protobuf::TextFormat::ParseFromString(valid_program, &other));
  program.mutable_circuit()
      ->mutable_moments(0)
      ->mutable_operations(0)
      ->mutable_args()
      ->at("control_qubits")
      .mutable_arg_value()
      ->set_string_value("junk");
  std::vector<Program> others = {other, other};
  EXPECT_EQ(ResolveQubitIds(&program, &qubit_count, &others),
            tensorflow::Status(static_cast<tensorflow::errors::Code>(
                                   absl::StatusCode::kInvalidArgument),
                               "Unable to parse qubit: junk"));
}

TEST(ProgramResolutionTest, ResolveQubitIdsMultiProgramMismatch) {
  Program program, other;
  unsigned int qubit_count;
  ASSERT_TRUE(
      google::protobuf::TextFormat::ParseFromString(valid_program, &program));
  ASSERT_TRUE(
      google::protobuf::TextFormat::ParseFromString(valid_program, &other));
  program.mutable_circuit()
      ->mutable_moments(0)
      ->mutable_operations(0)
      ->mutable_qubits(0)
      ->set_id("0_5");
  std::vector<Program> others = {other, other};
  EXPECT_EQ(
      ResolveQubitIds(&program, &qubit_count, &others),
      tensorflow::Status(
          static_cast<tensorflow::errors::Code>(
              absl::StatusCode::kInvalidArgument),
          "A paired circuit contains qubits not found in reference circuit."));
}

TEST(ProgramResolutionTest, ResolveQubitIdsMultiProgramMismatchControl) {
  Program program, other;
  unsigned int qubit_count;
  ASSERT_TRUE(
      google::protobuf::TextFormat::ParseFromString(valid_program, &program));
  ASSERT_TRUE(
      google::protobuf::TextFormat::ParseFromString(valid_program, &other));
  program.mutable_circuit()
      ->mutable_moments(0)
      ->mutable_operations(0)
      ->mutable_args()
      ->at("control_qubits")
      .mutable_arg_value()
      ->set_string_value("0_5");
  std::vector<Program> others = {other, other};
  EXPECT_EQ(
      ResolveQubitIds(&program, &qubit_count, &others),
      tensorflow::Status(
          static_cast<tensorflow::errors::Code>(
              absl::StatusCode::kInvalidArgument),
          "A paired circuit contains qubits not found in reference circuit."));
}

TEST(ProgramResolutionTest, ResolveQubitIdsMultiProgramSmaller) {
  Program program, other;
  unsigned int qubit_count;
  ASSERT_TRUE(
      google::protobuf::TextFormat::ParseFromString(valid_program, &program));
  ASSERT_TRUE(
      google::protobuf::TextFormat::ParseFromString(valid_program, &other));
  other.mutable_circuit()
      ->mutable_moments(0)
      ->mutable_operations(0)
      ->mutable_qubits(0)
      ->set_id("0_2");
  std::vector<Program> others = {other, other};
  EXPECT_EQ(
      ResolveQubitIds(&program, &qubit_count, &others),
      tensorflow::Status(
          static_cast<tensorflow::errors::Code>(
              absl::StatusCode::kInvalidArgument),
          "A reference circuit contains qubits not found in paired circuit."));
}

TEST(ProgramResolutionTest, ResolveQubitIdsMultiProgramSmallerControl) {
  Program program, other;
  unsigned int qubit_count;
  ASSERT_TRUE(
      google::protobuf::TextFormat::ParseFromString(valid_program, &program));
  ASSERT_TRUE(
      google::protobuf::TextFormat::ParseFromString(valid_program, &other));
  other.mutable_circuit()
      ->mutable_moments(0)
      ->mutable_operations(0)
      ->mutable_args()
      ->at("control_qubits")
      .mutable_arg_value()
      ->set_string_value("0_2");
  std::vector<Program> others = {other, other};
  EXPECT_EQ(
      ResolveQubitIds(&program, &qubit_count, &others),
      tensorflow::Status(
          static_cast<tensorflow::errors::Code>(
              absl::StatusCode::kInvalidArgument),
          "A reference circuit contains qubits not found in paired circuit."));
}

TEST(ProgramResolutionTest, ResolveSymbolsPartial) {
  Program symbol_program;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
      valid_symbol_program, &symbol_program));
  const absl::flat_hash_map<std::string, std::pair<int, float>> param_map = {
      {"v1", {0, 1.0}}};
  EXPECT_EQ(ResolveSymbols(param_map, &symbol_program, false), Status());
  EXPECT_EQ(symbol_program.circuit()
                .moments(0)
                .operations(0)
                .args()
                .at("exponent")
                .arg_value()
                .float_value(),
            1.0);
  EXPECT_EQ(symbol_program.circuit()
                .moments(1)
                .operations(0)
                .args()
                .at("exponent")
                .symbol(),
            "v2");
}

TEST(ProgramResolutionTest, ResolveSymbolsFull) {
  Program symbol_program;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
      valid_symbol_program, &symbol_program));
  const absl::flat_hash_map<std::string, std::pair<int, float>> param_map = {
      {"v1", {0, 1.0}}, {"v2", {1, 2.0f}}};
  EXPECT_EQ(ResolveSymbols(param_map, &symbol_program, false), Status());
  EXPECT_EQ(symbol_program.circuit()
                .moments(0)
                .operations(0)
                .args()
                .at("exponent")
                .arg_value()
                .float_value(),
            1.0);
  EXPECT_EQ(symbol_program.circuit()
                .moments(1)
                .operations(0)
                .args()
                .at("exponent")
                .arg_value()
                .float_value(),
            2.0);
}

TEST(ProgramResolutionTest, ResolveSymbolsStrictPartial) {
  Program symbol_program;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
      valid_symbol_program, &symbol_program));
  const absl::flat_hash_map<std::string, std::pair<int, float>> param_map = {
      {"v1", {0, 1.0}}};
  EXPECT_EQ(ResolveSymbols(param_map, &symbol_program, true),
            Status(static_cast<tensorflow::errors::Code>(
                       absl::StatusCode::kInvalidArgument),
                   "Could not find symbol in parameter map: v2"));
}

TEST(ProgramResolutionTest, ResolveSymbolsStrictFull) {
  Program symbol_program;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
      valid_symbol_program, &symbol_program));
  const absl::flat_hash_map<std::string, std::pair<int, float>> param_map = {
      {"v1", {0, 1.0}}, {"v2", {1, 2.0f}}};
  EXPECT_EQ(ResolveSymbols(param_map, &symbol_program, true), Status());
  EXPECT_EQ(symbol_program.circuit()
                .moments(0)
                .operations(0)
                .args()
                .at("exponent")
                .arg_value()
                .float_value(),
            1.0);
  EXPECT_EQ(symbol_program.circuit()
                .moments(1)
                .operations(0)
                .args()
                .at("exponent")
                .arg_value()
                .float_value(),
            2.0);
}

TEST(ProgramResolutionTest, CheckMPSSupportedEmpty) {
  Program empty;
  EXPECT_EQ(CheckMPSSupported(empty), Status());
}

TEST(ProgramResolutionTest, CheckQubitsIn1DFailedByOpWithMoreThan2Qubits) {
  Program program_with_3qubit_op;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
      three_qubit_op_program, &program_with_3qubit_op));
  EXPECT_EQ(CheckMPSSupported(program_with_3qubit_op),
            Status(static_cast<tensorflow::errors::Code>(
                       absl::StatusCode::kInvalidArgument),
                   "1D operations only support 1 and 2 qubit gates. "
                   "Found: 3 qubit gate."));
}

TEST(ProgramResolutionTest,
     CheckQubitsIn1DFailedByOpWithMoreThan2QubitsOnControlQubits) {
  Program program_with_3qubit_op;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
      valid_program, &program_with_3qubit_op));
  EXPECT_EQ(CheckMPSSupported(program_with_3qubit_op),
            Status(static_cast<tensorflow::errors::Code>(
                       absl::StatusCode::kInvalidArgument),
                   "1D operations only support 1 and 2 qubit gates. "
                   "Found: 3 qubit gate."));
}

TEST(ProgramResolutionTest, CheckQubitsIn1DFailedByNot1DTopology) {
  Program program_not_1d;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
      resolved_qubit_program_not_1d, &program_not_1d));
  EXPECT_EQ(CheckMPSSupported(program_not_1d),
            Status(static_cast<tensorflow::errors::Code>(
                       absl::StatusCode::kInvalidArgument),
                   "A program is not in 1D topology. It contains an"
                   " operation with qubits not neighbors each other."));
}

}  // namespace
}  // namespace tfq
