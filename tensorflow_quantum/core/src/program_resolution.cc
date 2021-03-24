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
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
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

Status RegisterQubits(
    const std::string& qb_string,
    absl::flat_hash_set<std::pair<std::pair<int, int>, std::string>>* id_set) {
  // Inserts qubits found in qb_string into id_set.

  if (qb_string.empty()) {
    return Status::OK();  // no control-default value specified in serializer.py
  }

  const std::vector<absl::string_view> qb_list = absl::StrSplit(qb_string, ',');
  for (auto qb : qb_list) {
    int r, c;
    const std::vector<absl::string_view> splits = absl::StrSplit(qb, '_');
    if (splits.size() != 2) {
      return Status(tensorflow::error::INVALID_ARGUMENT,
                    absl::StrCat("Unable to parse qubit: ", qb));
    }
    if (!absl::SimpleAtoi(splits[0], &r)) {
      return Status(tensorflow::error::INVALID_ARGUMENT,
                    absl::StrCat("Unable to parse qubit: ", qb));
    }
    if (!absl::SimpleAtoi(splits[1], &c)) {
      return Status(tensorflow::error::INVALID_ARGUMENT,
                    absl::StrCat("Unable to parse qubit: ", qb));
    }
    auto locs = std::pair<std::pair<int, int>, std::string>(
        std::pair<int, int>(r, c), std::string(qb));
    id_set->insert(locs);
  }
  return Status::OK();
}

Status ResolveQubitIds(Program* program, unsigned int* num_qubits,
                       std::vector<PauliSum>* p_sums /*=nullptr*/) {
  if (program->circuit().moments().empty()) {
    // (#679) Just ignore empty program.
    // Number of qubits in empty programs is zero.
    *num_qubits = 0;
    return Status::OK();
  }

  absl::flat_hash_set<std::pair<std::pair<int, int>, std::string>> id_set;
  for (const Moment& moment : program->circuit().moments()) {
    for (const Operation& operation : moment.operations()) {
      Status s;
      for (const Qubit& qubit : operation.qubits()) {
        s = RegisterQubits(qubit.id(), &id_set);
        if (!s.ok()) {
          return s;
        }
      }
      s = RegisterQubits(
          operation.args().at("control_qubits").arg_value().string_value(),
          &id_set);
      if (!s.ok()) {
        return s;
      }
    }
  }
  *num_qubits = id_set.size();

  // call to std::sort will do (r1 < r2) || ((r1 == r2) && c1 < c2)
  std::vector<std::pair<std::pair<int, int>, std::string>> ids(id_set.begin(),
                                                               id_set.end());
  std::sort(ids.begin(), ids.end());

  absl::flat_hash_map<std::string, std::string> id_to_index;
  for (size_t i = 0; i < ids.size(); i++) {
    id_to_index[ids[i].second] = absl::StrCat(i);
  }

  // Replace the Program Qubit ids with the indices.
  for (Moment& moment : *program->mutable_circuit()->mutable_moments()) {
    for (Operation& operation : *moment.mutable_operations()) {
      // Resolve qubit ids.
      for (Qubit& qubit : *operation.mutable_qubits()) {
        qubit.set_id(id_to_index.at(qubit.id()));
      }
      // Resolve control qubit ids found in the control_qubits arg.
      absl::string_view control_qubits =
          operation.args().at("control_qubits").arg_value().string_value();
      // explicit empty value set in serializer.py.
      if (control_qubits.empty()) {
        continue;
      }
      std::vector<absl::string_view> control_ids =
          absl::StrSplit(control_qubits, ',');
      std::vector<std::string> control_indexs;
      control_indexs.reserve(control_ids.size());
      for (auto id : control_ids) {
        control_indexs.push_back(id_to_index.at(id));
      }
      operation.mutable_args()
          ->at("control_qubits")
          .mutable_arg_value()
          ->set_string_value(absl::StrJoin(control_indexs, ","));
    }
  }

  if (p_sums) {
    for (size_t i = 0; i < p_sums->size(); i++) {
      // Replace the PauliSum Qubit ids with the indices.
      for (PauliTerm& term : *(p_sums->at(i)).mutable_terms()) {
        for (PauliQubitPair& pair : *term.mutable_paulis()) {
          const auto result = id_to_index.find(pair.qubit_id());
          if (result == id_to_index.end()) {
            return Status(
                tensorflow::error::INVALID_ARGUMENT,
                "Found a Pauli sum operating on qubits not found in circuit.");
          }
          pair.set_qubit_id(result->second);
        }
      }
    }
  }

  return Status::OK();
}

Status ResolveQubitIds(Program* program, unsigned int* num_qubits,
                       std::vector<Program>* other_programs) {
  if (program->circuit().moments().empty()) {
    // (#679) Just ignore empty program.
    // Number of qubits in empty programs is zero.
    *num_qubits = 0;
    return Status::OK();
  }

  absl::flat_hash_set<std::pair<std::pair<int, int>, std::string>> id_set;
  for (const Moment& moment : program->circuit().moments()) {
    for (const Operation& operation : moment.operations()) {
      Status s;
      for (const Qubit& qubit : operation.qubits()) {
        s = RegisterQubits(qubit.id(), &id_set);
        if (!s.ok()) {
          return s;
        }
      }
      s = RegisterQubits(
          operation.args().at("control_qubits").arg_value().string_value(),
          &id_set);
      if (!s.ok()) {
        return s;
      }
    }
  }
  *num_qubits = id_set.size();

  // call to std::sort will do (r1 < r2) || ((r1 == r2) && c1 < c2)
  std::vector<std::pair<std::pair<int, int>, std::string>> ids(id_set.begin(),
                                                               id_set.end());
  std::sort(ids.begin(), ids.end());

  absl::flat_hash_map<std::string, std::string> id_to_index;
  absl::flat_hash_set<std::string> id_ref;
  for (size_t i = 0; i < ids.size(); i++) {
    id_to_index[ids[i].second] = absl::StrCat(i);
    id_ref.insert(ids[i].second);
  }

  // Replace the Program Qubit ids with the indices.
  for (Moment& moment : *program->mutable_circuit()->mutable_moments()) {
    for (Operation& operation : *moment.mutable_operations()) {
      for (Qubit& qubit : *operation.mutable_qubits()) {
        qubit.set_id(id_to_index.at(qubit.id()));
      }
      // Resolve control qubit ids found in the control_qubits arg.
      absl::string_view control_qubits =
          operation.args().at("control_qubits").arg_value().string_value();
      // explicit empty value set in serializer.py.
      if (control_qubits.empty()) {
        continue;
      }
      std::vector<absl::string_view> control_ids =
          absl::StrSplit(control_qubits, ',');
      std::vector<std::string> control_indexs;
      control_indexs.reserve(control_ids.size());
      for (auto id : control_ids) {
        control_indexs.push_back(id_to_index.at(id));
      }
      operation.mutable_args()
          ->at("control_qubits")
          .mutable_arg_value()
          ->set_string_value(absl::StrJoin(control_indexs, ","));
    }
  }

  for (size_t i = 0; i < other_programs->size(); i++) {
    // Replace the other_program Qubit ids with the indices.
    absl::flat_hash_set<std::string> visited_qubits(id_ref);
    for (Moment& moment :
         *(other_programs->at(i)).mutable_circuit()->mutable_moments()) {
      for (Operation& operation : *moment.mutable_operations()) {
        // Resolve qubit ids.
        for (Qubit& qubit : *operation.mutable_qubits()) {
          visited_qubits.erase(qubit.id());
          const auto result = id_to_index.find(qubit.id());
          if (result == id_to_index.end()) {
            return Status(tensorflow::error::INVALID_ARGUMENT,
                          "A paired circuit contains qubits not found in "
                          "reference circuit.");
          }
          qubit.set_id(result->second);
        }
        // Resolve control qubit ids.
        absl::string_view control_qubits = operation.mutable_args()
                                               ->at("control_qubits")
                                               .arg_value()
                                               .string_value();
        if (control_qubits.empty()) {  // explicit empty value.
          continue;
        }
        std::vector<absl::string_view> control_ids =
            absl::StrSplit(control_qubits, ',');
        std::vector<std::string> control_indexs;
        control_indexs.reserve(control_ids.size());
        for (auto id : control_ids) {
          visited_qubits.erase(id);
          const auto result = id_to_index.find(id);
          if (result == id_to_index.end()) {
            return Status(tensorflow::error::INVALID_ARGUMENT,
                          "A paired circuit contains qubits not found in "
                          "reference circuit.");
          }
          control_indexs.push_back(result->second);
        }
        operation.mutable_args()
            ->at("control_qubits")
            .mutable_arg_value()
            ->set_string_value(absl::StrJoin(control_indexs, ","));
      }
    }
    if (!visited_qubits.empty()) {
      return Status(
          tensorflow::error::INVALID_ARGUMENT,
          "A reference circuit contains qubits not found in paired circuit.");
    }
  }

  return Status::OK();
}

Status ResolveSymbols(
    const absl::flat_hash_map<std::string, std::pair<int, float>>& param_map,
    Program* program, bool resolve_all /*=true*/) {
  for (Moment& moment : *program->mutable_circuit()->mutable_moments()) {
    for (Operation& operation : *moment.mutable_operations()) {
      for (auto& kv : *operation.mutable_args()) {
        Arg& arg = kv.second;
        if (!arg.symbol().empty()) {
          auto iter = param_map.find(arg.symbol());
          if (iter == param_map.end()) {
            if (resolve_all) {
              return Status(
                  tensorflow::error::INVALID_ARGUMENT,
                  "Could not find symbol in parameter map: " + arg.symbol());
            }
            continue;
          }
          arg.mutable_arg_value()->set_float_value(iter->second.second);
        }
      }
    }
  }

  return Status::OK();
}

}  // namespace tfq
