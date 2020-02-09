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

#include "tensorflow_quantum/core/ops/parse_context.h"

#include <google/protobuf/text_format.h>

#include <string>
#include <vector>

#include "cirq/google/api/v2/program.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow_quantum/core/ops/tfq_simulate_utils.h"
#include "tensorflow_quantum/core/proto/pauli_sum.pb.h"
#include "tensorflow_quantum/core/src/program_resolution.h"

namespace tfq {
namespace {

using ::cirq::google::api::v2::Program;
using ::tensorflow::OpKernelContext;
using ::tensorflow::Status;
using ::tensorflow::Tensor;
using ::tfq::proto::PauliSum;

template <typename T>
Status ParseProto(const std::string& text, T* proto) {
  // First attempt to parse from the binary representation.
  if (proto->ParseFromString(text)) {
    return Status::OK();
  }

  // If that fails, then try to parse from the human readable representation.
  if (google::protobuf::TextFormat::ParseFromString(text, proto)) {
    return Status::OK();
  }

  return Status(tensorflow::error::INVALID_ARGUMENT,
                "Unparseable proto: " + text);
}

}  // namespace

Status ParsePrograms(OpKernelContext* context, const std::string& input_name,
                     std::vector<Program>* programs) {
  const tensorflow::Tensor* input;
  Status status = context->input(input_name, &input);
  if (!status.ok()) {
    return status;
  }

  if (input->dims() != 1) {
    // Never parse anything other than a 1d list of circuits.
    return Status(
        tensorflow::error::INVALID_ARGUMENT,
        absl::StrCat("programs must be rank 1. Got rank ", input->dims(), "."));
  }

  const auto program_strings = input->vec<std::string>();
  const int num_programs = program_strings.dimension(0);
  programs->assign(num_programs, Program());

  auto DoWork = [&](int start, int end) {
    for (int i = start; i < end; i++) {
      OP_REQUIRES_OK(context, ParseProto(program_strings(i), &programs->at(i)));
    }
  };

  const int block_size = GetBlockSize(context, num_programs);
  context->device()
      ->tensorflow_cpu_worker_threads()
      ->workers->TransformRangeConcurrently(block_size, num_programs, DoWork);

  return Status::OK();
}

Status GetProgramsAndProgramsToAppend(
    OpKernelContext* context, std::vector<Program>* programs,
    std::vector<Program>* programs_to_append) {
  Status status = ParsePrograms(context, "programs", programs);
  if (!status.ok()) {
    return status;
  }

  status = ParsePrograms(context, "programs_to_append", programs_to_append);
  if (!status.ok()) {
    return status;
  }

  if (programs->size() != programs_to_append->size()) {
    return Status(tensorflow::error::INVALID_ARGUMENT,
                  "programs and programs_to_append must have matching sizes.");
  }

  return Status::OK();
}

// TODO(pmassey): Add a getter for the case where there is only 1 input program.

Status GetProgramsAndNumQubits(
    OpKernelContext* context, std::vector<Program>* programs,
    std::vector<int>* num_qubits,
    std::vector<std::vector<PauliSum>>* p_sums /*=nullptr*/) {
  Status status = ParsePrograms(context, "programs", programs);
  if (!status.ok()) {
    return status;
  }

  if (p_sums) {
    status = GetPauliSums(context, p_sums);
    if (!status.ok()) {
      return status;
    }
  }

  num_qubits->reserve(programs->size());
  for (int i = 0; i < programs->size(); i++) {
    Program& program = (*programs)[i];
    Status status = Status::OK();
    if (p_sums) {
      status = ResolveQubitIds(&program, &(p_sums->at(i)));
    } else {
      status = ResolveQubitIds(&program);
    }

    if (!status.ok()) {
      return status;
    }

    num_qubits->push_back(GetNumQubits(program));
  }

  return Status::OK();
}

Status GetPauliSums(OpKernelContext* context,
                    std::vector<std::vector<PauliSum>>* p_sums) {
  const Tensor* input;
  Status status = context->input("pauli_sums", &input);
  if (!status.ok()) {
    return status;
  }

  if (input->dims() != 2) {
    return Status(tensorflow::error::INVALID_ARGUMENT,
                  absl::StrCat("pauli_sums must be rank 2. Got rank ",
                               input->dims(), "."));
  }

  const auto sum_specs = input->matrix<std::string>();
  p_sums->reserve(sum_specs.dimension(0));
  for (int i = 0; i < sum_specs.dimension(0); i++) {
    std::vector<PauliSum> sub_ops;
    sub_ops.reserve(sum_specs.dimension(1));
    for (int j = 0; j < sum_specs.dimension(1); j++) {
      const std::string& text = sum_specs(i, j);
      PauliSum p;
      // TODO(pmassey): Consider parsing from the serialized instead of the
      // human readable proto to pass smaller messages.
      status = ParseProto(text, &p);
      if (!status.ok()) {
        return status;
      }
      sub_ops.push_back(p);
    }
    p_sums->push_back(sub_ops);
  }

  return Status::OK();
}

Status GetSymbolMaps(OpKernelContext* context, std::vector<SymbolMap>* maps) {
  const Tensor* input_names;
  Status status = context->input("symbol_names", &input_names);
  if (!status.ok()) {
    return status;
  }

  if (input_names->dims() != 1) {
    return Status(tensorflow::error::INVALID_ARGUMENT,
                  absl::StrCat("symbol_names must be rank 1. Got rank ",
                               input_names->dims(), "."));
  }

  const Tensor* input_values;
  status = context->input("symbol_values", &input_values);
  if (!status.ok()) {
    return status;
  }

  if (input_values->dims() != 2) {
    return Status(tensorflow::error::INVALID_ARGUMENT,
                  absl::StrCat("symbol_values must be rank 2. Got rank ",
                               input_values->dims(), "."));
  }

  const auto symbol_names = input_names->vec<std::string>();
  const auto symbol_values = input_values->matrix<float>();

  if (symbol_names.dimension(0) != symbol_values.dimension(1)) {
    return Status(tensorflow::error::INVALID_ARGUMENT,
                  "Input symbol names and value sizes do not match.");
  }

  maps->reserve(symbol_values.dimension(0));
  for (int i = 0; i < symbol_values.dimension(0); i++) {
    SymbolMap map;
    for (int j = 0; j < symbol_values.dimension(1); j++) {
      const std::string& name = symbol_names(j);
      const double value = (double)symbol_values(i, j);
      map[name] = {j, value};
    }

    maps->push_back(map);
  }

  return Status::OK();
}

// TODO (mbbrough/pmassey/jaeyoo): Should grads return an EigenMatrixXd instead
// of a vector of vectors ?
Status GetGradients(OpKernelContext* context,
                    std::vector<std::vector<float>>* grads) {
  const Tensor* input;
  const Status status = context->input("grad", &input);
  if (!status.ok()) {
    return status;
  }

  const auto input_grads = input->matrix<float>();
  grads->reserve(input_grads.dimension(0));
  for (int i = 0; i < input_grads.dimension(0); i++) {
    std::vector<float> sub_grads;
    sub_grads.reserve(input_grads.dimension(1));
    for (int j = 0; j < input_grads.dimension(1); j++) {
      sub_grads.push_back(input_grads(i, j));
    }
    grads->push_back(sub_grads);
  }

  return Status::OK();
}

}  // namespace tfq
