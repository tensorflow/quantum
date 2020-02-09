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

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "cirq/google/api/v2/program.pb.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_quantum/core/proto/pauli_sum.pb.h"

namespace tfq {

using cirq::google::api::v2::Arg;
using cirq::google::api::v2::Moment;
using cirq::google::api::v2::Operation;
using cirq::google::api::v2::Program;
using cirq::google::api::v2::Qubit;
using tensorflow::Status;
using tfq::proto::PauliQubitPair;
using tfq::proto::PauliSum;
using tfq::proto::PauliTerm;

Status ResolveQubitIds(Program* program,
                       std::vector<PauliSum>* p_sums /*=nullptr*/) {
  if (program->circuit().moments().empty()) {
    // (#679) Just ignore empty program.
    return Status::OK();
  }

  absl::flat_hash_set<std::string> id_set;
  for (const Moment& moment : program->circuit().moments()) {
    for (const Operation& operation : moment.operations()) {
      for (const Qubit& qubit : operation.qubits()) {
        id_set.insert(qubit.id());
      }
    }
  }

  std::vector<std::string> ids(id_set.begin(), id_set.end());
  std::sort(ids.begin(), ids.end());

  absl::flat_hash_map<std::string, int> id_to_index;
  for (int i = 0; i < ids.size(); i++) {
    id_to_index[ids[i]] = i;
  }

  // Replace the Program Qubit ids with the indices.
  for (Moment& moment : *program->mutable_circuit()->mutable_moments()) {
    for (Operation& operation : *moment.mutable_operations()) {
      for (Qubit& qubit : *operation.mutable_qubits()) {
        const int index = id_to_index.at(qubit.id());
        const std::string new_id = absl::StrCat(index);
        qubit.set_id(new_id);
      }
    }
  }

  if (p_sums) {
    for (int i = 0; i < p_sums->size(); i++) {
      // Replace the PauliSum Qubit ids with the indices.
      for (PauliTerm& term : *(p_sums->at(i)).mutable_terms()) {
        for (PauliQubitPair& pair : *term.mutable_paulis()) {
          const auto result = id_to_index.find(pair.qubit_id());
          if (result == id_to_index.end()) {
            return Status(
                tensorflow::error::INVALID_ARGUMENT,
                "Found a Pauli sum operating on qubits not found in circuit.");
          }
          const int index = result->second;
          const std::string new_id = absl::StrCat(index);
          pair.set_qubit_id(new_id);
        }
      }
    }
  }

  return Status::OK();
}

int GetNumQubits(const Program& program) {
  absl::flat_hash_set<std::string> id_set;
  for (const Moment& moment : program.circuit().moments()) {
    for (const Operation& operation : moment.operations()) {
      for (const Qubit& qubit : operation.qubits()) {
        id_set.insert(qubit.id());
      }
    }
  }
  return id_set.size();
}

Status ResolveSymbols(
    const absl::flat_hash_map<std::string, std::pair<int, double>>& param_map,
    Program* program) {
  for (Moment& moment : *program->mutable_circuit()->mutable_moments()) {
    for (Operation& operation : *moment.mutable_operations()) {
      for (auto& kv : *operation.mutable_args()) {
        Arg& arg = kv.second;
        if (!arg.symbol().empty()) {
          auto iter = param_map.find(arg.symbol());
          if (iter == param_map.end()) {
            return Status(
                tensorflow::error::INVALID_ARGUMENT,
                "Could not find symbol in parameter map: " + arg.symbol());
          }

          arg.mutable_arg_value()->set_float_value(iter->second.second);
        }
      }
    }
  }

  return Status::OK();
}

}  // namespace tfq
