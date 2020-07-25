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
#include "cirq/google/api/v2/program.pb.h"
#include "gtest/gtest.h"
#include "tensorflow/core/lib/core/status.h"

namespace tfq {
namespace {

using cirq::google::api::v2::Program;

TEST(ProgramResolutionTest, ResolveQubitIdsInvalidArg) {
  const std::string text = R"(
    circuit {
      moments {
        operations {
          qubits {
            id: "0_0"
          }
          qubits {
            id: "1_0"
          }
        }
      }
    }
  )";

  const std::string bad_text = R"(
    circuit {
      moments {
        operations {
          qubits {
            id: "0_0"
          }
          qubits {
            id: "1_junk"
          }
        }
      }
    }
  )";

  const std::string bad_text2 = R"(
    circuit {
      moments {
        operations {
          qubits {
            id: "0_0"
          }
          qubits {
            id: "junk_1"
          }
        }
      }
    }
  )";

  const std::string bad_text3 = R"(
    circuit {
      moments {
        operations {
          qubits {
            id: "0_0"
          }
          qubits {
            id: "1_2_3"
          }
        }
      }
    }
  )";

  const std::string text_good_p_sum = R"(
    terms {
      coefficient_real: 1.0
      coefficient_imag: 0.0
      paulis {
        qubit_id: "0_0"
        pauli_type: "Z"
      }
    }
  )";

  const std::string text_bad_p_sum = R"(
    terms {
      coefficient_real: 1.0
      coefficient_imag: 0.0
      paulis {
        qubit_id: "0_1"
        pauli_type: "X"
      }
    }
  )";

  std::vector<tfq::proto::PauliSum> p_sums;
  tfq::proto::PauliSum p_sum_good, p_sum_bad;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(text_good_p_sum,
                                                            &p_sum_good));
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(text_bad_p_sum,
                                                            &p_sum_bad));
  p_sums.push_back(p_sum_good);
  p_sums.push_back(p_sum_bad);

  Program program;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(text, &program));

  unsigned int num_qubits;
  EXPECT_EQ(ResolveQubitIds(&program, &num_qubits, &p_sums),
            tensorflow::Status(
                tensorflow::error::INVALID_ARGUMENT,
                "Found a Pauli sum operating on qubits not found in circuit."));

  ASSERT_TRUE(
      google::protobuf::TextFormat::ParseFromString(bad_text, &program));
  EXPECT_EQ(ResolveQubitIds(&program, &num_qubits, &p_sums),
            tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                               "Unable to parse qubit: 1_junk"));

  ASSERT_TRUE(
      google::protobuf::TextFormat::ParseFromString(bad_text2, &program));
  EXPECT_EQ(ResolveQubitIds(&program, &num_qubits, &p_sums),
            tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                               "Unable to parse qubit: junk_1"));

  ASSERT_TRUE(
      google::protobuf::TextFormat::ParseFromString(bad_text3, &program));
  EXPECT_EQ(ResolveQubitIds(&program, &num_qubits, &p_sums),
            tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                               "Unable to parse qubit: 1_2_3"));
}

TEST(ProgramResolutionTest, ResolveQubitIds) {
  const std::string text = R"(
    circuit {
      moments {
        operations {
          qubits {
            id: "0_0"
          }
          qubits {
            id: "1_0"
          }
        }
      }
      moments {
        operations {
          qubits {
            id: "0_0"
          }
          qubits {
            id: "0_1"
          }
        }
      }
    }
  )";

  const std::string text_p_sum_0 = R"(
    terms {
      coefficient_real: 1.0
      coefficient_imag: 0.0
      paulis {
        qubit_id: "0_0"
        pauli_type: "Z"
      }
    }
  )";

  const std::string text_p_sum_1 = R"(
    terms {
      coefficient_real: 1.0
      coefficient_imag: 0.0
      paulis {
        qubit_id: "1_0"
        pauli_type: "X"
      }
    }
  )";

  const std::string text_alphabet = R"(
    circuit {
      moments {
        operations {
          qubits {
            id: "0_0"
          }
          qubits {
            id: "0_1"
          }
        }
      }
      moments {
        operations {
          qubits {
            id: "0_2"
          }
          qubits {
            id: "0_3"
          }
        }
      }
    }
  )";

  const std::string text_alphabet_p_sum_0 = R"(
    terms {
      coefficient_real: 1.0
      coefficient_imag: 0.0
      paulis {
        qubit_id: "0_1"
        pauli_type: "Z"
      }
    }
  )";

  const std::string text_alphabet_p_sum_1 = R"(
    terms {
      coefficient_real: 1.0
      coefficient_imag: 0.0
      paulis {
        qubit_id: "0_0"
        pauli_type: "X"
      }
    }
  )";

  const std::string text_empty = R"(
    circuit {
    }
  )";

  std::vector<tfq::proto::PauliSum> p_sums, p_sums_alphabet;
  tfq::proto::PauliSum p_sum_0, p_sum_1;
  ASSERT_TRUE(
      google::protobuf::TextFormat::ParseFromString(text_p_sum_0, &p_sum_0));
  ASSERT_TRUE(
      google::protobuf::TextFormat::ParseFromString(text_p_sum_1, &p_sum_1));
  p_sums.push_back(p_sum_0);
  p_sums.push_back(p_sum_1);
  tfq::proto::PauliSum alphabet_p_sum_0, alphabet_p_sum_1;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
      text_alphabet_p_sum_0, &alphabet_p_sum_0));
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
      text_alphabet_p_sum_1, &alphabet_p_sum_1));
  p_sums_alphabet.push_back(alphabet_p_sum_0);
  p_sums_alphabet.push_back(alphabet_p_sum_1);

  Program program, empty_program, alphabet_program;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(text, &program));
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(text_empty,
                                                            &empty_program));
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(text_alphabet,
                                                            &alphabet_program));

  unsigned int num_qubits, num_qubits_empty, num_qubits_alphabet;
  EXPECT_TRUE(ResolveQubitIds(&program, &num_qubits, &p_sums).ok());
  EXPECT_TRUE(ResolveQubitIds(&empty_program, &num_qubits_empty).ok());
  EXPECT_TRUE(
      ResolveQubitIds(&alphabet_program, &num_qubits_alphabet, &p_sums_alphabet)
          .ok());

  EXPECT_EQ(program.circuit().moments(0).operations(0).qubits(0).id(), "0");
  EXPECT_EQ(program.circuit().moments(0).operations(0).qubits(1).id(), "2");
  EXPECT_EQ(program.circuit().moments(1).operations(0).qubits(0).id(), "0");
  EXPECT_EQ(program.circuit().moments(1).operations(0).qubits(1).id(), "1");

  EXPECT_EQ(alphabet_program.circuit().moments(0).operations(0).qubits(0).id(),
            "0");
  EXPECT_EQ(alphabet_program.circuit().moments(0).operations(0).qubits(1).id(),
            "1");
  EXPECT_EQ(alphabet_program.circuit().moments(1).operations(0).qubits(0).id(),
            "2");
  EXPECT_EQ(alphabet_program.circuit().moments(1).operations(0).qubits(1).id(),
            "3");

  EXPECT_EQ(p_sums.at(0).terms(0).paulis(0).qubit_id(), "0");
  EXPECT_EQ(p_sums.at(1).terms(0).paulis(0).qubit_id(), "2");

  EXPECT_EQ(p_sums_alphabet.at(0).terms(0).paulis(0).qubit_id(), "1");
  EXPECT_EQ(p_sums_alphabet.at(1).terms(0).paulis(0).qubit_id(), "0");

  EXPECT_EQ(num_qubits, 3);
  EXPECT_EQ(num_qubits_empty, 0);
  EXPECT_EQ(num_qubits_alphabet, 4);
}

TEST(ProgramResolutionTest, ResolveSymbolsInvalidArg) {
  const std::string text = R"(
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

  // Test with strict replacement
  Program program_strict;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(text, &program_strict));
  const absl::flat_hash_map<std::string, std::pair<int, float>> param_map = {
      {"v1", {0, 1.0}}};
  EXPECT_EQ(ResolveSymbols(param_map, &program_strict),
            tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                               "Could not find symbol in parameter map: v2"));

  // Test with non-strict replacement
  Program program;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(text, &program));
  EXPECT_TRUE(ResolveSymbols(param_map, &program, false).ok());
  EXPECT_EQ(program.circuit()
                .moments(0)
                .operations(0)
                .args()
                .at("exponent")
                .arg_value()
                .float_value(),
            1.0);
  EXPECT_EQ(program.circuit()
                .moments(1)
                .operations(0)
                .args()
                .at("exponent")
                .symbol(),
            "v2");
  
}

TEST(ProgramResolutionTest, ResolveSymbolsUnused) {
  const std::string text = R"(
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
    }
  )";

  // Test with strict parameter map usage
  Program program_strict;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(text,
                                                            &program_strict));
  const absl::flat_hash_map<std::string, std::pair<int, float>> param_map = {
    {"v1", {0, 1.0}}, {"unused", {0, 1.0}}};
  EXPECT_EQ(ResolveSymbols(param_map, &program_strict),
            tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                               "Parameter map contains symbols not present "
                               "in the program."));

  // Test with non-strict parameter map usage
  Program program;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(text, &program));
  EXPECT_TRUE(ResolveSymbols(param_map, &program, true, false).ok());
  EXPECT_EQ(program.circuit()
                .moments(0)
                .operations(0)
                .args()
                .at("exponent")
                .arg_value()
                .float_value(),
            1.0);

}


}  // namespace
}  // namespace tfq
