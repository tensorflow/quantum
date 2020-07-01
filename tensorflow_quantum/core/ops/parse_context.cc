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

  const auto program_strings = input->vec<tensorflow::tstring>();
  const int num_programs = program_strings.dimension(0);
  programs->assign(num_programs, Program());

  auto DoWork = [&](int start, int end) {
    for (int i = start; i < end; i++) {
      OP_REQUIRES_OK(context, ParseProto(program_strings(i), &programs->at(i)));
    }
  };

  // TODO(mbbrough): Determine if this is a good cycle estimate.
  const int cycle_estimate = 1000;
  context->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
      num_programs, cycle_estimate, DoWork);

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

Status GetProgramsAndNumQubits(
    OpKernelContext* context, std::vector<Program>* programs,
    std::vector<int>* num_qubits,
    std::vector<std::vector<PauliSum>>* p_sums /*=nullptr*/) {
  // 1. Parse input programs
  // 2. (Optional) Parse input PauliSums
  // 3. Convert GridQubit locations to integers.
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

  // Resolve qubit ID's in parallel.
  num_qubits->assign(programs->size(), -1);
  auto DoWork = [&](int start, int end) {
    for (int i = start; i < end; i++) {
      Program& program = (*programs)[i];
      unsigned int this_num_qubits;
      if (p_sums) {
        OP_REQUIRES_OK(context, ResolveQubitIds(&program, &this_num_qubits,
                                                &(p_sums->at(i))));
      } else {
        OP_REQUIRES_OK(context, ResolveQubitIds(&program, &this_num_qubits));
      }
      (*num_qubits)[i] = this_num_qubits;
    }
  };

  // TODO(mbbrough): Determine if this is a good cycle estimate.
  const int cycle_estimate = 1000;
  context->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
      num_qubits->size(), cycle_estimate, DoWork);

  return Status::OK();
}

Status GetPauliSums(OpKernelContext* context,
                    std::vector<std::vector<PauliSum>>* p_sums) {
  // 1. Parses PauliSum proto.
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

  const auto sum_specs = input->matrix<tensorflow::tstring>();
  p_sums->assign(sum_specs.dimension(0),
                 std::vector<PauliSum>(sum_specs.dimension(1), PauliSum()));
  auto DoWork = [&](int start, int end) {
    for (int i = start; i < end; i++) {
      for (int j = 0; j < sum_specs.dimension(1); j++) {
        PauliSum p;
        OP_REQUIRES_OK(context, ParseProto(sum_specs(i, j), &p));
        (*p_sums)[i][j] = p;
      }
    }
  };

  // TODO(mbbrough): Determine if this is a good cycle estimate.
  const int cycle_estimate = 1000;
  context->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
      sum_specs.dimension(0), cycle_estimate, DoWork);

  return Status::OK();
}

Status GetSymbolMaps(OpKernelContext* context, std::vector<SymbolMap>* maps) {
  // 1. Convert to dictionary representation for param resolution.
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

  const auto symbol_names = input_names->vec<tensorflow::tstring>();
  const auto symbol_values = input_values->matrix<float>();

  if (symbol_names.dimension(0) != symbol_values.dimension(1)) {
    return Status(tensorflow::error::INVALID_ARGUMENT,
                  "Input symbol names and value sizes do not match.");
  }

  maps->assign(symbol_values.dimension(0), SymbolMap());

  auto DoWork = [&](int start, int end) {
    for (int i = start; i < end; i++) {
      for (int j = 0; j < symbol_values.dimension(1); j++) {
        const std::string& name = symbol_names(j);
        const float value = symbol_values(i, j);
        (*maps)[i][name] = {j, value};
      }
    }
  };

  // TODO(mbbrough): Determine if this is a good cycle estimate.
  const int cycle_estimate = 1000;
  context->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
      symbol_values.dimension(0), cycle_estimate, DoWork);

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

tensorflow::Status GetNumSamples(
    tensorflow::OpKernelContext* context,
    std::vector<std::vector<int>>* parsed_num_samples) {
  const Tensor* input_num_samples;
  Status status = context->input("num_samples", &input_num_samples);
  if (!status.ok()) {
    return status;
  }

  if (input_num_samples->dims() != 2) {
    return Status(tensorflow::error::INVALID_ARGUMENT,
                  absl::StrCat("num_samples must be rank 2. Got rank ",
                               input_num_samples->dims(), "."));
  }

  const auto matrix_num_samples = input_num_samples->matrix<int>();
  parsed_num_samples->reserve(matrix_num_samples.dimension(0));
  for (unsigned int i = 0; i < matrix_num_samples.dimension(0); i++) {
    std::vector<int> sub_parsed_num_samples;
    sub_parsed_num_samples.reserve(matrix_num_samples.dimension(1));
    for (unsigned int j = 0; j < matrix_num_samples.dimension(1); j++) {
      const int num_samples = matrix_num_samples(i, j);
      if (num_samples < 1) {
        return Status(tensorflow::error::INVALID_ARGUMENT,
                      "Each element of num_samples must be greater than 0.");
      }
      sub_parsed_num_samples.push_back(num_samples);
    }
    parsed_num_samples->push_back(sub_parsed_num_samples);
  }

  return Status::OK();
}

}  // namespace tfq
