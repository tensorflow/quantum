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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow_quantum/core/ops/tfq_simulate_utils.h"
#include "tensorflow_quantum/core/proto/pauli_sum.pb.h"
#include "tensorflow_quantum/core/proto/program.pb.h"
#include "tensorflow_quantum/core/src/program_resolution.h"

namespace tfq {
namespace {

using ::tensorflow::OpKernelContext;
using ::tensorflow::Status;
using ::tensorflow::Tensor;
using ::tfq::proto::PauliSum;
using ::tfq::proto::Program;
using ::tfq::proto::ProjectorSum;

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

Status ParsePrograms2D(OpKernelContext* context, const std::string& input_name,
                       std::vector<std::vector<Program>>* programs) {
  const tensorflow::Tensor* input;
  Status status = context->input(input_name, &input);
  if (!status.ok()) {
    return status;
  }

  if (input->dims() != 2) {
    // Never parse anything other than a 1d list of circuits.
    return Status(tensorflow::error::INVALID_ARGUMENT,
                  absl::StrCat("other_programs must be rank 2. Got rank ",
                               input->dims(), "."));
  }

  const auto program_strings = input->matrix<tensorflow::tstring>();
  const int num_programs = program_strings.dimension(0);
  const int num_entries = program_strings.dimension(1);
  programs->assign(num_programs, std::vector<Program>(num_entries, Program()));

  auto DoWork = [&](int start, int end) {
    for (int i = start; i < end; i++) {
      OP_REQUIRES_OK(
          context,
          ParseProto(program_strings(i / num_entries, i % num_entries),
                     &programs->at(i / num_entries).at(i % num_entries)));
    }
  };

  // TODO(mbbrough): Determine if this is a good cycle estimate.
  const int cycle_estimate = 1000;
  context->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
      num_programs * num_entries, cycle_estimate, DoWork);

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
    std::vector<std::vector<PauliSum>>* p_sums /*=nullptr*/,
    std::vector<std::vector<ProjectorSum>>* proj_sums /*=nullptr*/) {
  // 1. Parse input programs
  // 2. (Optional) Parse input PauliSums
  // 2. (Optional) Parse input ProjectorSums.
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
    if (programs->size() != p_sums->size()) {
      return Status(
          tensorflow::error::INVALID_ARGUMENT,
          absl::StrCat("Number of circuits and PauliSums do not match. Got ",
                       programs->size(), " circuits and ", p_sums->size(),
                       " paulisums."));
    }
  }

  if (proj_sums) {
    status = GetProjectorSums(context, proj_sums);
    if (!status.ok()) {
      return status;
    }
    if (programs->size() != proj_sums->size()) {
      return Status(
          tensorflow::error::INVALID_ARGUMENT,
          absl::StrCat("Number of circuits and ProjectorSum do not match. Got ",
                       programs->size(), " circuits and ", proj_sums->size(),
                       " projectorsums."));
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

tensorflow::Status GetProgramsAndNumQubits(
    OpKernelContext* context, std::vector<Program>* programs,
    std::vector<int>* num_qubits,
    std::vector<std::vector<Program>>* other_programs) {
  // 1. Parse input programs
  // 2. Parse other_programs
  // 3. Convert GridQubit locations to integers and ensure exact matching.
  Status status = ParsePrograms(context, "programs", programs);
  if (!status.ok()) {
    return status;
  }

  status = ParsePrograms2D(context, "other_programs", other_programs);
  if (!status.ok()) {
    return status;
  }

  if (programs->size() != other_programs->size()) {
    return Status(tensorflow::error::INVALID_ARGUMENT,
                  absl::StrCat("programs and other_programs batch dimension",
                               " do not match. Foud: ", programs->size(),
                               " and ", other_programs->size()));
  }

  // Resolve qubit ID's in parallel.
  num_qubits->assign(programs->size(), -1);
  auto DoWork = [&](int start, int end) {
    for (int i = start; i < end; i++) {
      Program& program = (*programs)[i];
      unsigned int this_num_qubits;
      OP_REQUIRES_OK(context, ResolveQubitIds(&program, &this_num_qubits,
                                              &(*other_programs)[i]));
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
  const int op_dim = sum_specs.dimension(1);
  auto DoWork = [&](int start, int end) {
    for (int ii = start; ii < end; ii++) {
      const int i = ii / op_dim;
      const int j = ii % op_dim;
      PauliSum p;
      OP_REQUIRES_OK(context, ParseProto(sum_specs(i, j), &p));
      (*p_sums)[i][j] = p;
    }
  };

  // TODO(mbbrough): Determine if this is a good cycle estimate.
  const int cycle_estimate = 1000;
  context->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
      sum_specs.dimension(0) * sum_specs.dimension(1), cycle_estimate, DoWork);

  return Status::OK();
}

Status GetProjectorSums(OpKernelContext* context,
                        std::vector<std::vector<ProjectorSum>>* proj_sums) {
  // 1. Parses ProjectorSum proto.
  const Tensor* input;
  Status status = context->input("projector_sums", &input);
  if (!status.ok()) {
    return status;
  }

  if (input->dims() != 2) {
    return Status(tensorflow::error::INVALID_ARGUMENT,
                  absl::StrCat("projector_sums must be rank 2. Got rank ",
                               input->dims(), "."));
  }

  const auto sum_specs = input->matrix<tensorflow::tstring>();
  proj_sums->assign(
      sum_specs.dimension(0),
      std::vector<ProjectorSum>(sum_specs.dimension(1), ProjectorSum()));
  const int op_dim = sum_specs.dimension(1);
  auto DoWork = [&](int start, int end) {
    for (int ii = start; ii < end; ii++) {
      const int i = ii / op_dim;
      const int j = ii % op_dim;
      ProjectorSum p;
      OP_REQUIRES_OK(context, ParseProto(sum_specs(i, j), &p));
      (*proj_sums)[i][j] = p;
    }
  };

  // TODO(mbbrough): Determine if this is a good cycle estimate.
  const int cycle_estimate = 1000;
  context->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
      sum_specs.dimension(0) * sum_specs.dimension(1), cycle_estimate, DoWork);

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

  const int symbol_dim = symbol_values.dimension(1);
  auto DoWork = [&](int start, int end) {
    for (int i = start; i < end; i++) {
      for (int j = 0; j < symbol_dim; j++) {
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

// used by tfq_simulate_samples.
Status GetIndividualSample(tensorflow::OpKernelContext* context,
                           int* n_samples) {
  const Tensor* input_num_samples;
  Status status = context->input("num_samples", &input_num_samples);
  if (!status.ok()) {
    return status;
  }

  if (input_num_samples->dims() != 1) {
    return Status(tensorflow::error::INVALID_ARGUMENT,
                  absl::StrCat("num_samples must be rank 1. Got rank ",
                               input_num_samples->dims(), "."));
  }

  const auto vector_num_samples = input_num_samples->vec<int>();

  if (vector_num_samples.dimension(0) != 1) {
    return Status(tensorflow::error::INVALID_ARGUMENT,
                  absl::StrCat("num_samples must contain 1 element. Got ",
                               vector_num_samples.dimension(0), "."));
  }

  (*n_samples) = vector_num_samples(0);
  return Status::OK();
}

// used by adj_grad_op.
tensorflow::Status GetPrevGrads(
    tensorflow::OpKernelContext* context,
    std::vector<std::vector<float>>* parsed_prev_grads) {
  const Tensor* input_grads;
  Status status = context->input("downstream_grads", &input_grads);
  if (!status.ok()) {
    return status;
  }

  if (input_grads->dims() != 2) {
    return Status(tensorflow::error::INVALID_ARGUMENT,
                  absl::StrCat("downstream_grads must be rank 2. Got rank ",
                               input_grads->dims(), "."));
  }

  const auto matrix_grads = input_grads->matrix<float>();
  parsed_prev_grads->reserve(matrix_grads.dimension(0));
  for (unsigned int i = 0; i < matrix_grads.dimension(0); i++) {
    std::vector<float> sub_parsed_grads;
    sub_parsed_grads.reserve(matrix_grads.dimension(1));
    for (unsigned int j = 0; j < matrix_grads.dimension(1); j++) {
      const float grad_v = matrix_grads(i, j);
      sub_parsed_grads.push_back(grad_v);
    }
    parsed_prev_grads->push_back(sub_parsed_grads);
  }

  return Status::OK();
}

}  // namespace tfq
