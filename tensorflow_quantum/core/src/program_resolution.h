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

// Utilties around parsing Cirq Programs into forms that the TensorFlow op can
// better understand.

#ifndef TFQ_CORE_SRC_PROGRAM_RESOLUTION
#define TFQ_CORE_SRC_PROGRAM_RESOLUTION

#include <string>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_quantum/core/proto/pauli_sum.pb.h"
#include "tensorflow_quantum/core/proto/program.pb.h"

namespace tfq {

// Renames the ids of Qubits to be ordered from 0 to n, where n is the number
// of qubits. if p_sum is provided, we will also resolve ordering based on how
// we resolved program. All qubit types are supported, as long as the qubit ids
// are strings; all ids are extracted and lexicographically ordered, then simply
// replaced with their location in that ordering.
//
// The number of qubits in the program is recorded in `num_qubits`.
tensorflow::Status ResolveQubitIds(
    tfq::proto::Program* program, unsigned int* num_qubits,
    std::vector<tfq::proto::PauliSum>* p_sums = nullptr,
    bool swap_endianness = false);

// Overload which allows for strict resolution of multiple programs.
// Will resolve GridQubits in `program` and then double check that
// all qubits in `other_programs` match and resolve them.
// Note: no nullptr default is done here to avoid signature resolutions issues.
tensorflow::Status ResolveQubitIds(
    tfq::proto::Program* program, unsigned int* num_qubits,
    std::vector<tfq::proto::Program>* other_programs);

// Resolves all of the symbols present in the Program. Iterates through all
// operations in all moments, and if any Args have a symbol, replaces the one-of
// with an ArgValue representing the value in the parameter map keyed by the
// symbol. When `resolve_all` is true, returns an error if a symbol does not
// have a correponding value in `param_map`.
// TODO(pmassey): Consider returning an error if a value in the parameter map
// isn't used.
tensorflow::Status ResolveSymbols(
    const absl::flat_hash_map<std::string, std::pair<int, float>>& param_map,
    tfq::proto::Program* program, bool resolve_all = true);

// Checks if the qubits are in 1D topology.
tensorflow::Status CheckMPSSupported(const tfq::proto::Program& program);

}  // namespace tfq

#endif  // TFQ_CORE_SRC_PROGRAM_RESOLUTION
