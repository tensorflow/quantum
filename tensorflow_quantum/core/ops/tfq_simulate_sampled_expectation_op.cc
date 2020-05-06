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

#include <memory>
#include <vector>

#include "cirq/google/api/v2/program.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow_quantum/core/ops/parse_context.h"
#include "tensorflow_quantum/core/ops/tfq_simulate_utils.h"
#include "tensorflow_quantum/core/proto/pauli_sum.pb.h"
#include "tensorflow_quantum/core/qsim/mux.h"
#include "tensorflow_quantum/core/qsim/state_space.h"
#include "tensorflow_quantum/core/src/circuit.h"
#include "tensorflow_quantum/core/src/circuit_parser.h"
#include "tensorflow_quantum/core/src/program_resolution.h"

namespace tfq {

using ::cirq::google::api::v2::Program;
using ::tensorflow::Status;
using ::tfq::proto::PauliSum;
using ::tfq::qsim::GetStateSpace;
using ::tfq::qsim::StateSpace;

class TfqSimulateSampledExpectationOp : public tensorflow::OpKernel {
 public:
  explicit TfqSimulateSampledExpectationOp(
      tensorflow::OpKernelConstruction *context)
      : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext *context) override {
    // TODO (mbbrough): add more dimension checks for other inputs here.
    const int num_inputs = context->num_inputs();
    OP_REQUIRES(context, num_inputs == 5,
                tensorflow::errors::InvalidArgument(absl::StrCat(
                    "Expected 5 inputs, got ", num_inputs, " inputs.")));

    // Create the output Tensor.
    const int output_dim_batch_size = context->input(0).dim_size(0);
    const int output_dim_op_size = context->input(3).dim_size(1);
    tensorflow::TensorShape output_shape;
    output_shape.AddDim(output_dim_batch_size);
    output_shape.AddDim(output_dim_op_size);

    tensorflow::Tensor *output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    auto output_tensor = output->matrix<float>();

    std::vector<Program> programs;
    std::vector<int> num_qubits;
    std::vector<std::vector<PauliSum>> pauli_sums;
    OP_REQUIRES_OK(context, GetProgramsAndNumQubits(context, &programs,
                                                    &num_qubits, &pauli_sums));

    std::vector<SymbolMap> maps;
    OP_REQUIRES_OK(context, GetSymbolMaps(context, &maps));

    OP_REQUIRES(context, pauli_sums.size() == programs.size(),
                tensorflow::errors::InvalidArgument(absl::StrCat(
                    "Number of circuits and PauliSums do not match. Got ",
                    programs.size(), " circuits and ", pauli_sums.size(),
                    " paulisums.")));

    std::vector<std::vector<int>> num_samples;
    OP_REQUIRES_OK(context, GetNumSamples(context, &num_samples));

    OP_REQUIRES(context, num_samples.size() == pauli_sums.size(),
                tensorflow::errors::InvalidArgument(absl::StrCat(
                    "Dimension 0 of num_samples and pauli_sums do not match.",
                    "Got ", num_samples.size(), " lists of sample sizes and ",
                    pauli_sums.size(), " lists of pauli sums.")));

    OP_REQUIRES(
        context, num_samples[0].size() == pauli_sums[0].size(),
        tensorflow::errors::InvalidArgument(absl::StrCat(
            "Dimension 1 of num_samples and pauli_sums do not match.", "Got ",
            num_samples[0].size(), " lists of sample sizes and ",
            pauli_sums[0].size(), " lists of pauli sums.")));

    auto DoWork = [&](int start, int end) {
      int old_batch_index = -2;
      int cur_batch_index = -1;
      int old_num_qubits = -2;
      int cur_op_index;
      std::unique_ptr<StateSpace> test_state = GetStateSpace(1, 1);
      std::unique_ptr<StateSpace> scratch_state = GetStateSpace(1, 1);
      for (int i = start; i < end; i++) {
        cur_batch_index = i / output_dim_op_size;
        cur_op_index = i % output_dim_op_size;

        // (#679) Just ignore empty program
        if (programs[cur_batch_index].circuit().moments().empty()) {
          output_tensor(cur_batch_index, cur_op_index) = -2.0;
          continue;
        }

        if (cur_batch_index != old_batch_index) {
          // We've run into a new wavefunction we must compute.
          // Only compute a new wavefunction when we have to.
          Program program = programs[cur_batch_index];
          const int num = num_qubits[cur_batch_index];
          OP_REQUIRES_OK(context,
                         ResolveSymbols(maps[cur_batch_index], &program));

          Circuit circuit;
          OP_REQUIRES_OK(context, CircuitFromProgram(program, num, &circuit));

          // TODO(mbbrough): Update this allocation hack so that a StateSpace
          //  object can grow it's memory dynamically to larger and larger size
          //  without ever having to call free (until very end). This is tricky
          //  to implement because right now certain statespaces can't simulate
          //  all states and we use StateSpaceSlow for smaller circuits.
          if (num != old_num_qubits) {
            test_state = GetStateSpace(num, 1);
            test_state->CreateState();

            // Also re-allocate scratch state for expectation calculations.
            scratch_state = GetStateSpace(num, 1);
            scratch_state->CreateState();
          }
          // no need to update scratch_state since ComputeExpectation
          // will take care of things for us.
          test_state->SetStateZero();
          OP_REQUIRES_OK(context, test_state->Update(circuit));
          old_num_qubits = num;
        }

        float expectation = 0.0;
        OP_REQUIRES_OK(
            context,
            test_state->ComputeSampledExpectation(
                pauli_sums[cur_batch_index][cur_op_index], scratch_state.get(),
                &expectation, num_samples[cur_batch_index][cur_op_index]));

        output_tensor(cur_batch_index, cur_op_index) = expectation;
        old_batch_index = cur_batch_index;
      }
    };

    const int block_size =
        GetBlockSize(context, output_dim_batch_size * output_dim_op_size);
    context->device()
        ->tensorflow_cpu_worker_threads()
        ->workers->TransformRangeConcurrently(
            block_size, output_dim_batch_size * output_dim_op_size, DoWork);
  }
};

REGISTER_KERNEL_BUILDER(
    Name("TfqSimulateSampledExpectation").Device(tensorflow::DEVICE_CPU),
    TfqSimulateSampledExpectationOp);

REGISTER_OP("TfqSimulateSampledExpectation")
    .Input("programs: string")
    .Input("symbol_names: string")
    .Input("symbol_values: float")
    .Input("pauli_sums: string")
    .Input("num_samples: int32")
    .Output("expectations: float")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {
      tensorflow::shape_inference::ShapeHandle programs_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &programs_shape));

      tensorflow::shape_inference::ShapeHandle symbol_names_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &symbol_names_shape));

      tensorflow::shape_inference::ShapeHandle symbol_values_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &symbol_values_shape));

      tensorflow::shape_inference::ShapeHandle pauli_sums_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &pauli_sums_shape));

      tensorflow::shape_inference::ShapeHandle num_samples_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 2, &num_samples_shape));

      tensorflow::shape_inference::DimensionHandle output_rows =
          c->Dim(programs_shape, 0);
      tensorflow::shape_inference::DimensionHandle output_cols =
          c->Dim(pauli_sums_shape, 1);
      c->set_output(0, c->Matrix(output_rows, output_cols));

      return tensorflow::Status::OK();
    });

}  // namespace tfq
